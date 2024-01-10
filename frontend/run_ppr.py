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

from ppr import cpp_push_single, cpp_push_batch, python_push_single, python_push_batch, forward_push
from utils import get_data_path
from graph import VERTEX_ID_TYPE, GraphDataManager

# RUNS = 10
# WARMUP = 4

RUNS = 1
WARMUP = 0

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ogbn-products')
parser.add_argument('--num_machines', type=int, default=4, help='number of machines (simulated as processes)')
parser.add_argument('--num_processes', type=int, default=1, help='number of processes in each machine used for SSPPR computing')
parser.add_argument('--num_roots', type=int, default=10, help='number of source nodes in each machine')
parser.add_argument('--alpha', type=float, default=0.462, help='teleport probability')
parser.add_argument('--epsilon', type=float, default=1e-6, help='maximum residual')
parser.add_argument('--num_threads', type=int, default=1, help='number of threads used in push operation')
parser.add_argument('--version', type=str, default='overlap', help='version of PPR implementation')
parser.add_argument('--worker_name', type=str, default='worker{}', help='name of workers, formatted by rank')
parser.add_argument('--data_path', type=str, default='', help='path to graph shards')
parser.add_argument('--log', action='store_true', help='whether to log breakdown runtime')

parser.add_argument('--inference_out_path', type=str, default='', help='output path, perform full graph inference if exists')


def run(rank, args, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    rpc.init_rpc(
        args.worker_name.format(rank),
        rank=rank,
        world_size=world_size,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            num_worker_threads=4, rpc_timeout=0)
    )

    if rank == 0:

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
                    machine_rank, args.data_path, args.worker_name, args.num_machines, args.num_processes
                ))
            )

        # assemble graph shard object rrefs
        rrefs = [None for _ in range(args.num_machines)]
        for machine_rank in range(0, args.num_machines):
            res = data_manager_rrefs[machine_rank].rpc_sync().get_graph_rrefs()
            rrefs[machine_rank] = res[0]
            rrefs += res[1]

        print('Assembling graph shard references finished')

        time_counter = Counter()

        tik_ = time.perf_counter()
        for i in range(RUNS + WARMUP):
            if i == WARMUP:
                tik_ = time.perf_counter()

            tik = time.perf_counter()

            output_name = ''
            if len(args.inference_out_path) > 0:
                # Full Graph PPR inference
                shard_ptrs = torch.load(os.path.join(args.data_path, 'partition_book.pt'))
                num_nodes = shard_ptrs[1:] - shard_ptrs[:-1]
                output_name = os.path.join(args.inference_out_path, args.dataset + '_{}_{}.pt')
            else:
                num_nodes = [args.num_roots] * args.num_machines

            futs = []
            for machine_rank in range(0, args.num_machines):
                source_nodes = torch.arange(num_nodes[machine_rank], dtype=VERTEX_ID_TYPE)
                num_slice = int(source_nodes.shape[0] / args.num_processes)

                for process_rank in range(0, args.num_processes):
                    emit_rank = args.num_machines + machine_rank * args.num_processes + process_rank

                    # slice source nodes
                    start = process_rank * num_slice
                    end = source_nodes.shape[0] if process_rank == args.num_processes - 1 else (process_rank + 1) * num_slice

                    if args.log:
                        print(f'emit_rank {emit_rank}, machine_rank {machine_rank}, start {start}, end {end}')

                    futs.append(
                        rpc.rpc_async(
                            args.worker_name.format(emit_rank),
                            ppr_func_dict[args.version],
                            args=(machine_rank, process_rank, rrefs, args.num_machines, args.num_threads,
                                  source_nodes[start:end], args.alpha, args.epsilon, args.log, output_name),
                            timeout=0
                        )
                    )

            c = []
            for j, fut in enumerate(futs):
                res, time_dict = fut.wait()
                print(j, 'finished')
                c += res
                # for check_ppr.py
                # if j == 0 and i == 0:
                #     torch.save(res, 'temp/data.pt')
                #     res_node_0 = res[0]
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
              f'Avg Throughput = {args.num_roots*args.num_machines/avg_time:.1f} nodes/s')
        print('##############################')

        print(f'\n{avg_time:.3f}s\n{args.num_roots*args.num_machines/avg_time:.1f} nodes/s')

    rpc.shutdown()


if __name__ == '__main__':
    args = parser.parse_args()
    if len(args.data_path) == 0:
        args.data_path = os.path.join(get_data_path(), 'ogbn-products-p{}'.format(args.num_machines))

    world_size = args.num_machines * (args.num_processes + 1)

    print(f'Spawn {world_size}-Process to simulate {args.num_machines}-Machine scenario')
    t1 = time.time()

    mp.spawn(run, nprocs=world_size, args=(args, world_size), join=True)

    t2 = time.time()
    print(f'\nTotal Execution time = {t2 - t1:.3f}s')
