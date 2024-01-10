#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import os

import time

import torch
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef, remote

from tqdm import tqdm

NUM_MACHINES = 2
NUM_PROCESSES = 1
RUNS = 10
NUM_DATA = 25000000

WORKER_NAME = 'worker{}'


class Walker:
    def __init__(self):
        self.id = rpc.get_worker_info().id
        self.data = torch.rand(NUM_DATA, dtype=torch.float32)  # simulates loading large data

    def step(self):
        return self.data


def random_walk(walker_rrefs):
    rank = rpc.get_worker_info().id
    # print(rank)

    for i in tqdm(range(RUNS)) if rank == 0 else range(RUNS):
        futs = []

        # if rank < 2:
        #     target_ranks = range(2, 4)
        # else:
        #     target_ranks = range(0, 2)

        if rank < 1:
            target_ranks = range(1, 2)
        else:
            target_ranks = range(0, 1)

        for j in target_ranks:
            fut = walker_rrefs[j].rpc_async().step()  # fetch data from remote walker
            futs.append(fut)

        for fut in futs:
            fut.wait()


def run(rank):
    # os.environ['MASTER_ADDR'] = '10.125.132.35'
    # os.environ['MASTER_ADDR'] = 'd05-10'
    os.environ['MASTER_ADDR'] = '10.125.132.32'
    os.environ['MASTER_PORT'] = '29599'
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=4,
    )

    world_size = NUM_MACHINES * NUM_PROCESSES

    rank = rank
    rpc.init_rpc(WORKER_NAME.format(rank), rank=rank, world_size=world_size,
                 rpc_backend_options=options
                 )

    print('rpc init done')

    if rank == 0:
        rrefs = []
        for machine_rank in range(world_size):
            info = rpc.get_worker_info(WORKER_NAME.format(machine_rank))
            rrefs.append(remote(info, Walker))

        tik_ = time.time()

        futs = []
        for rref in rrefs:
            futs.append(
                rpc.rpc_async(
                    rref.owner(),
                    random_walk,
                    args=(rrefs,)
                )
            )

        for fut in futs:
            fut.wait()

        tok_ = time.time()
        print(f'Inner Execution time = {tok_ - tik_:.3f}s')
        print(f'Avg time per step = {(tok_ - tik_)/RUNS*1000:.1f}ms')

    rpc.shutdown()


if __name__ == '__main__':
    print(f"world_size={NUM_MACHINES}, num_data={NUM_DATA * 4/1e6}MB")

    tik = time.time()
    mp.spawn(run, nprocs=NUM_PROCESSES, join=True)
    # run(0)
    tok = time.time()

    print(f'Outer Execution time = {tok - tik:.3}s')
