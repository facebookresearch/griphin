import os
import argparse

import time
from typing import List

import torch
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from torch.distributed.rpc import remote, RRef

from ppr import cpp_push_single, cpp_push_batch, python_push_single, python_push_batch
from utils import get_data_path
from graph import VERTEX_ID_TYPE, GraphDataManager

RUNS = 10
WARMUP = 3

parser = argparse.ArgumentParser()
parser.add_argument('--num_machines', type=int, default=4, help='number of machines (simulated as processes)')
parser.add_argument('--num_processes', type=int, default=1, help='number of processes in each machine used for SSPPR computing')
parser.add_argument('--num_roots', type=int, default=10, help='number of source nodes in each machine')
parser.add_argument('--alpha', type=float, default=0.462, help='teleport probability')
parser.add_argument('--epsilon', type=float, default=1e-6, help='maximum residual')
parser.add_argument('--num_threads', type=int, default=1, help='num of threads to create in push operation')
parser.add_argument('--version', type=str, default='cpp_batch', help='version of PPR implementation')
parser.add_argument('--worker_name', type=str, default='worker{}', help='name of workers, formatted by rank')
parser.add_argument('--file_path', type=str, default='', help='path to dataset')
parser.add_argument('--log', action='store_true', help='whether to log breakdown runtime')


def run(rank, args, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29503'

    rpc.init_rpc(
        args.worker_name.format(rank),
        rank=rank,
        world_size=world_size,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(num_worker_threads=32)
    )

    if rank == 0:

        ppr_func_dict = {
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

        time_pop = 0
        time_local_fetch = 0
        time_remote_fetch = 0
        time_push = 0

        tik_ = time.perf_counter()
        for i in range(RUNS + WARMUP):
            if i == WARMUP:
                tik_ = time.perf_counter()

            tik = time.perf_counter()

            source_nodes = torch.arange(args.num_roots, dtype=VERTEX_ID_TYPE)
            num_data = int(args.num_roots / args.num_processes)

            # rpc.rpc_sync(
            #     args.worker_name.format(4),
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
                res = fut.wait()
                c.append(res)
                if j == 0 and i >= WARMUP:
                    time_pop += res[1]
                    time_local_fetch += res[2]
                    time_remote_fetch += res[3]
                    time_push += res[4]

            tok = time.perf_counter()
            print(f'Run {i}, Time = {tok - tik:.3f}s\n')

        print(f'Avg Time pop: {time_pop / RUNS:.3f}s, '
              f'Avg Time fetch local: {time_local_fetch / RUNS:.3f}s, '
              f'Avg Time fetch remote: {time_remote_fetch / RUNS:.3f}s, '
              f'Avg Time push: {time_push / RUNS:.3f}s')

        tok_ = time.perf_counter()
        print(f'Avg Execution time = {(tok_ - tik_)/RUNS:.3f}s')

    rpc.shutdown()


if __name__ == '__main__':
    args = parser.parse_args()
    if len(args.file_path) == 0:
        args.file_path = os.path.join(get_data_path(), 'hz-ogbn-product-p{}-pt'.format(args.num_machines))

    world_size = args.num_machines * (args.num_processes + 1)

    print(f'Spawn {world_size}-Process to simulate {args.num_machines}-Machine scenario')
    t1 = time.time()

    mp.spawn(run, nprocs=world_size, args=(args, world_size), join=True)

    t2 = time.time()
    print(f'Total Execution time = {t2 - t1:.3f}s')
