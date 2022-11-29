import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

NUM_MACHINES = 4
RUNS = 100
NUM_DATA = 5000000
BACK_END = 'gloo'


def run(rank):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    dist.init_process_group(BACK_END, rank=rank, world_size=NUM_MACHINES)
    group = dist.new_group(list(range(NUM_MACHINES)))

    data = torch.rand(NUM_DATA, dtype=torch.float32)  # machine owned data

    if rank == 0:
        tik_ = time.time()

    for i in range(RUNS):
        futs, rets = [], [torch.empty_like(data) for _ in range(dist.get_world_size(group))]
        for j in range(NUM_MACHINES):
            if rank == j:
                fut = dist.gather(data, gather_list=rets, dst=j, group=group, async_op=True)
            else:
                fut = dist.gather(data, dst=j, group=group, async_op=True)
            futs.append(fut)

        for fut in futs:
            fut.wait()

        # print(rank, rets)

    dist.barrier()

    if rank == 0:
        tok_ = time.time()
        print(f'Avg time per step = {(tok_ - tik_) / RUNS * 1000:.1f}ms')


if __name__ == "__main__":
    print(f"world_size={NUM_MACHINES}, num_data={NUM_DATA * 4 / 1e6}MB")

    tik = time.time()
    mp.spawn(run, nprocs=NUM_MACHINES, join=True)
    tok = time.time()

    print(f'Outer Execution time = {tok - tik:.3f}s')
