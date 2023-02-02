import time
import os

import torch
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef, remote

from utils import get_root_path
from graph import GraphShard, VERTEX_ID_TYPE

NUM_MACHINES = 4

ALPHA = 0.462
EPSILON = 1e-3
MAX_DEGREE = -1
NUM_SOURCE = 1000

RUNS = 10
WARMUP = 2

WORKER_NAME = 'worker{}'
FILE_PATH = os.path.join(get_root_path(), 'engine/ogbn_csr_format')


def fetch_neighbor(rrefs):
    rank = rpc.get_worker_info().id
    local_shard: GraphShard = rrefs[rank].to_here()

    max_local_id = 600000
    local_nids = torch.randperm(max_local_id, dtype=VERTEX_ID_TYPE)[:(NUM_MACHINES-1) * NUM_SOURCE]
    remote_nids = torch.randperm(max_local_id, dtype=VERTEX_ID_TYPE)[:NUM_SOURCE]

    # local time
    tik = time.time()
    local_shard.batch_fetch_neighbors(local_nids)
    tok = time.time()

    # remote time
    tik2 = time.time()
    futs = {}
    for j in range(NUM_MACHINES):
        if j == rank:
            continue
        futs[j] = rrefs[j].rpc_async().batch_fetch_neighbors(remote_nids)
    for j, fut in futs.items():
        neighbors = fut.wait()
        # print(j, len(neighbors))
    tok2 = time.time()

    return tok - tik, tok2 - tik2


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

        total_local_time = total_remote_time = 0
        for i in range(RUNS + WARMUP):
            if i == WARMUP:
                total_local_time = total_remote_time = 0

            local_time, remote_time = fetch_neighbor(rrefs)

            print(f'Run {i}, Local Fetch Time = {local_time:.4f}s,'
                  f' Remote Fetch Time = {remote_time:.4f}s')
            total_local_time += local_time
            total_remote_time += remote_time

        print(f'Avg Local Fetch time = {total_local_time/RUNS:.4f}s,'
              f' Avg Remote Fetch time = {total_remote_time/RUNS:.4f}s')

    rpc.shutdown()


if __name__ == '__main__':
    print('Simulates Communication scenario')

    start = time.time()
    mp.spawn(run, nprocs=NUM_MACHINES, join=True)
    end = time.time()

    print(f'Total Execution time = {end - start:.3}s')
