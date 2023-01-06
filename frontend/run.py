import os
import time

import torch
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef, remote

from graph import GraphShard
from random_walk import random_walk

NUM_MACHINES = 4
NUM_ROOTS = 8192
WALK_LENGTH = 15
WORKER_NAME = 'worker{}'
FILE_PATH = '../engine/ogbn_files_txt_small'


def run(rank):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=4)

    rpc.init_rpc(WORKER_NAME.format(rank), rank=rank, world_size=NUM_MACHINES, rpc_backend_options=options)

    if rank == 0:
        rrefs = []
        for machine_rank in range(NUM_MACHINES):
            info = rpc.get_worker_info(WORKER_NAME.format(machine_rank))
            rrefs.append(remote(info, GraphShard, args=(FILE_PATH, machine_rank)))

        tik_ = time.time()

        futs = []
        for rref in rrefs:
            futs.append(
                rpc.rpc_async(
                    rref.owner(),
                    random_walk,
                    args=(rrefs, NUM_MACHINES, NUM_ROOTS, WALK_LENGTH)
                )
            )

        c = []
        for fut in futs:
            c.append(fut.wait())
        tok_ = time.time()

        print(f'Random walk summary:\n {torch.cat(c, dim=0)}')
        print(f'Inner Execution time = {tok_ - tik_:.3}s')

    rpc.shutdown()


if __name__ == '__main__':
    print('Spawn Multi-Process to simulate Multi-Machine scenario')

    tik = time.time()
    mp.spawn(run, nprocs=NUM_MACHINES, join=True)
    tok = time.time()

    print(f'Outer Execution time = {tok - tik:.3}s')
