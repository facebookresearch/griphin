#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import os
import argparse

import time
from typing import List

import torch
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from torch.distributed.rpc import remote, RRef

from utils import get_data_path
from graph import GraphDataManager
from random_walk import random_walk, random_walk2


RUNS = 10
WARMUP = 3

parser = argparse.ArgumentParser()
parser.add_argument('--num_machines', type=int, default=4, help='number of machines (simulated as processes)')
parser.add_argument('--num_roots', type=int, default=8192, help='number of root nodes in each machine')
parser.add_argument('--walk_length', type=int, default=15, help='walk length')
parser.add_argument('--num_threads', type=int, default=1, help='number of threads used in sample operation')
parser.add_argument('--worker_name', type=str, default='worker{}', help='name of workers, formatted by rank')
parser.add_argument('--file_path', type=str, default='', help='path to dataset')
parser.add_argument('--version', type=int, default=2, help='version of random walk implementation')
parser.add_argument('--profile', action='store_true', help='whether to use torch.profile to profile program. '
                                                           'Note: this will create overheads and slow down program.')
parser.add_argument('--profile_prefix', type=str, default='tb_log/', help='path to profiling log')
parser.add_argument('--log', action='store_true', help='whether to log breakdown runtime. ')


def run(rank, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=16)

    rpc.init_rpc(args.worker_name.format(rank), rank=rank, world_size=args.num_machines*2, rpc_backend_options=options)

    if rank == 0:

        rw_func_dict = {
            1: random_walk,
            2: random_walk2
        }

        # initialize graph data tensors in shared memory
        data_manager_rrefs: List[RRef] = []
        for machine_rank in range(args.num_machines):
            info = rpc.get_worker_info(args.worker_name.format(machine_rank))
            # rrefs.append(remote(info, GraphShard, args=(args.file_path, machine_rank)))
            data_manager_rrefs.append(
                remote(info, GraphDataManager, args=(
                    # fix num_processes = 1 for random walk
                    machine_rank, args.file_path, args.worker_name, args.num_machines, 1
                ))
            )

        # assemble graph shard object rrefs
        rrefs = [None for _ in range(args.num_machines)]
        for machine_rank in range(0, args.num_machines):
            res = data_manager_rrefs[machine_rank].rpc_sync().get_graph_rrefs()
            rrefs[machine_rank] = res[0]
            rrefs += res[1]

        profile = False
        for i in range(RUNS + WARMUP + 1):
            if i == WARMUP:
                tik_ = time.perf_counter()

            if i == RUNS + WARMUP:
                profile = args.profile
                tok_ = time.perf_counter()

            tik = time.perf_counter()

            futs = []
            for emit_rank in range(args.num_machines, 2*args.num_machines):
                futs.append(
                    rpc.rpc_async(
                        args.worker_name.format(emit_rank),
                        rw_func_dict[args.version],
                        args=(rrefs, args.num_machines, args.num_threads, args.num_roots, args.walk_length,
                              profile, '{}/{}'.format(args.profile_prefix, emit_rank), args.log)
                    )
                )
            c = []
            for fut in futs:
                c.append(fut.wait())

            tok = time.perf_counter()
            print(f'Run {i}, Time = {tok - tik:.3f}s')

        print(f'Random walk summary:\n {torch.cat(c, dim=0)}')
        print(f'Avg Execution time = {(tok_ - tik_)/RUNS:.3f}s')

    rpc.shutdown()


if __name__ == '__main__':
    args = parser.parse_args()
    if args.profile:
        args.profile_prefix = '{}/{}'.format(args.profile_prefix, time.time())
    if 0 == len(args.file_path):
        args.file_path = os.path.join(get_data_path(), 'ogbn-products-p{}'.format(args.num_machines))

    print('Spawn Multi-Process to simulate Multi-Machine scenario')
    start = time.time()

    mp.spawn(run, nprocs=args.num_machines*2, args=(args,), join=True)

    end = time.time()
    print(f'Total Execution time = {end - start:.3}s')
