import os
import os.path as osp

import time

import torch
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef, remote

NUM_MACHINES = 4
RUNS = 100
NUM_DATA = 50000000

WORKER_NAME = 'worker{}'


class Walker:
    def __init__(self):
        self.id = rpc.get_worker_info().id
        self.data = torch.rand(NUM_DATA, dtype=torch.float32)  # simulates loading large data

    def step(self):
        return self.data


def random_walk(walker_rrefs):
    rank = rpc.get_worker_info().id
    # walker = walker_rrefs[rank].to_here()

    for i in range(RUNS):
        futs = []
        for j in range(NUM_MACHINES):
            if rank == j:
                continue
            fut = walker_rrefs[j].rpc_async().step()  # fetch data from remote walker
            futs.append(fut)

        rets = []
        for fut in futs:
            rets.append(fut.wait())


def run(rank):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=4)

    rpc.init_rpc(WORKER_NAME.format(rank), rank=rank, world_size=NUM_MACHINES, rpc_backend_options=options)

    if rank == 0:
        rrefs = []
        for machine_rank in range(NUM_MACHINES):
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
    mp.spawn(run, nprocs=NUM_MACHINES, join=True)
    tok = time.time()

    print(f'Outer Execution time = {tok - tik:.3}s')
