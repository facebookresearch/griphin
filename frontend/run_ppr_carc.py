#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import os
import argparse

import time
from collections import Counter
from typing import List

import torch
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from torch.distributed.rpc import remote, RRef

from utils import extract_hostname_list, extract_master_addr
from ppr import cpp_push_single, cpp_push_batch, python_push_single, python_push_batch, forward_push
from graph import VERTEX_ID_TYPE, GraphDataManager


RUNS = 10
WARMUP = 4

parser = argparse.ArgumentParser()
parser.add_argument('--num_machines', type=int, default=4, help='number of machines (simulated as processes)')
parser.add_argument('--num_processes', type=int, default=1, help='number of processes in each machine used for SSPPR computing')
parser.add_argument('--num_roots', type=int, default=10, help='number of source nodes in each machine')
parser.add_argument('--alpha', type=float, default=0.462, help='teleport probability')
parser.add_argument('--epsilon', type=float, default=1e-6, help='maximum residual')
parser.add_argument('--num_threads', type=int, default=1, help='number of threads used in push operation')
parser.add_argument('--version', type=str, default='cpp_batch', help='version of PPR implementation')
parser.add_argument('--worker_name', type=str, default='worker{}', help='name of workers, formatted by rank')
parser.add_argument('--file_path', type=str, default='', help='path to dataset')
parser.add_argument('--log', action='store_true', help='whether to log breakdown runtime')
parser.add_argument('--machine_rank', type=int)


def run(rank, args, world_size, current_machine_rank, master_hostname):
    os.environ['MASTER_ADDR'] = master_hostname
    os.environ['MASTER_PORT'] = '29500'

    if rank == 0:  # graph data server process
        global_rank = current_machine_rank
    else:          # PPR compute process
        global_rank = args.num_machines-1 + current_machine_rank * args.num_processes + rank

    print(os.getpid(), rank, global_rank)

    rpc.init_rpc(
        args.worker_name.format(global_rank),
        rank=global_rank,
        world_size=world_size,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(num_worker_threads=4),
    )

    print('Init RPC Complete')

    if global_rank == 0:

        ppr_func_dict = {
            'overlap': forward_push,
            'cpp_single': cpp_push_single,
            'cpp_batch': cpp_push_batch,
            'python_single': python_push_single,
            'python_batch': python_push_batch,
        }

        # initialize graph data tensors in shared memory
        data_manager_rrefs: List[RRef] = []
        for machine_rank in range(0, args.num_machines):
            info = rpc.get_worker_info(args.worker_name.format(machine_rank))
            data_manager_rrefs.append(
                remote(info, GraphDataManager, args=(
                    machine_rank, args.file_path, args.worker_name, args.num_machines, args.num_processes
                ))
            )

        # assemble graph shard object rrefs
        rrefs = [None for _ in range(args.num_machines)]
        for machine_rank in range(0, args.num_machines):
            res = data_manager_rrefs[machine_rank].rpc_sync().get_graph_rrefs()
            rrefs[machine_rank] = res[0]
            rrefs += res[1]

        time_counter = Counter()

        tik_ = time.perf_counter()
        for i in range(RUNS + WARMUP):
            if i == WARMUP:
                tik_ = time.perf_counter()

            tik = time.perf_counter()

            source_nodes = torch.arange(args.num_roots, dtype=VERTEX_ID_TYPE)
            num_data = int(args.num_roots / args.num_processes)

            # rpc.rpc_sync(
            #     args.worker_name.format(2),
            #     cpp_push_batch,
            #     args=(0, 0, rrefs, args.num_machines, args.num_threads,
            #           source_nodes[1:2], args.alpha, args.epsilon, args.log)
            # )

            futs = []
            for machine_rank in range(0, args.num_machines):
                for process_rank in range(0, args.num_processes):
                    emit_rank = args.num_machines + machine_rank * args.num_processes + process_rank

                    # slice source nodes
                    start = process_rank * num_data
                    end = args.num_roots if process_rank == args.num_processes - 1 else (process_rank + 1) * num_data

                    if args.log:
                        print(f'emit_rank {emit_rank}, machine_rank {machine_rank}, start {start}, end {end}')

                    futs.append(
                        rpc.rpc_async(
                            args.worker_name.format(emit_rank),
                            ppr_func_dict[args.version],
                            args=(machine_rank, process_rank, rrefs, args.num_machines, args.num_threads,
                                  source_nodes[start:end], args.alpha, args.epsilon, args.log)
                        )
                    )

            c = []
            for j, fut in enumerate(futs):
                res, time_dict = fut.wait()
                c.append(res)
                # for check_ppr.py
                # if j == 0 and i == 0:
                #     res_node_0 = res[0]
                #     torch.save(res, 'temp/data.pt')
                #     torch.save(res_node_0[0], 'temp/node_ids.pt')
                #     torch.save(res_node_0[1], 'temp/shard_ids.pt')
                #     torch.save(res_node_0[2], 'temp/vals.pt')
                if j == 0 and i >= WARMUP:
                    time_counter += Counter(time_dict)

            tok = time.perf_counter()
            print(f'Run {i}, Time = {tok - tik:.3f}s\n')

        tok_ = time.perf_counter()
        avg_time = (tok_ - tik_) / RUNS

        print('##############################')
        for k, v in time_counter.items():
            print(f'Avg {k}: {v / RUNS:.3f}s')
        print(f'Avg Execution time = {avg_time:.3f}s, '
              f'Avg Throughput = {args.num_roots * args.num_machines / avg_time:.1f} nodes/s')
        print('##############################')

    rpc.shutdown()


if __name__ == '__main__':
    args = parser.parse_args()
    assert len(args.file_path) > 0, 'Please specify the path of processed graph partitions'
    # '/scratch1/gangdade/preprocessed/graph_engine/ogbn-product-p{}'.format(args.num_machines))
    # '/scratch1/gangdade/preprocessed/graph_engine/ogbn-papers100M-p{}'.format(args.num_machines))

    world_size = args.num_machines * (args.num_processes + 1)

    # Settings for USC HPC
    # export SLURM_JOB_NODELIST=a01-06,b10-12
    # export CURRENT_HOSTNAME=$(echo `hostname` | cut -d.-f1)
    nodelist_str = os.environ.get('SLURM_JOB_NODELIST')
    current_hostname = os.environ.get('CURRENT_HOSTNAME')
    hostnames = extract_hostname_list(nodelist_str)
    current_machine_rank = hostnames.index(current_hostname)
    master_hostname = hostnames[0]

    print(f'Launch {args.num_machines} machines (partitions) PPR processing task')
    print(f'Spawn {1} graph sever process, {args.num_processes} compute processes')
    print(f'World Size = {world_size}, current machine rank = {current_machine_rank}')
    print(f'Hostnames: {hostnames}')

    t1 = time.time()

    mp.spawn(run, nprocs=args.num_processes+1, args=(
        args, world_size, current_machine_rank, master_hostname
    ), join=True)

    t2 = time.time()
    print(f'\nTotal Execution time = {t2 - t1:.3f}s')
