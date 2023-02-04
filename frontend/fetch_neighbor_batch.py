import time
import os

import torch
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef, remote

from utils import get_root_path
from graph import GraphShard, VERTEX_ID_TYPE

NUM_MACHINES = 2
NUM_SOURCE = 1000

RUNS = 100
WARMUP = 2

WORKER_NAME = 'worker{}'
FILE_PATH = os.path.join(get_root_path(), 'data/ogbn_products_{}partitions'.format(NUM_MACHINES))


def fetch_neighbor(rrefs):
    rank = rpc.get_worker_info().id
    local_shard: GraphShard = rrefs[rank].to_here()

    max_local_id = 600000  # make sure every shard has num_core_nodes larger than this value
    local_nids = torch.randperm(max_local_id, dtype=VERTEX_ID_TYPE)[:(NUM_MACHINES-1) * NUM_SOURCE]
    remote_nids = torch.randperm(max_local_id, dtype=VERTEX_ID_TYPE)[:NUM_SOURCE]

    num_local_pushes = num_remote_pushes = 0

    # local time
    tik = time.time()
    neighbors = local_shard.batch_fetch_neighbors(local_nids)
    tok = time.time()
    num_local_pushes += torch.cat(neighbors).numel()

    # remote time
    tik2 = time.time()
    futs = {}
    for j in range(NUM_MACHINES):
        if j == rank:
            continue
        futs[j] = rrefs[j].rpc_async().batch_fetch_neighbors(remote_nids)
    res = []
    for j, fut in futs.items():
        res.append(fut.wait())
    tok2 = time.time()

    for result in res:
        num_remote_pushes += torch.cat(result).numel()

    print(num_local_pushes, num_remote_pushes)
    return tok - tik, tok2 - tik2, num_local_pushes, num_remote_pushes


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

        for i in range(WARMUP):
            fetch_neighbor(rrefs)

        total_local_time = total_remote_time = 0
        total_local_pushes = total_remote_pushes = 0

        for i in range(RUNS):
            res = fetch_neighbor(rrefs)
            print(f'Run {i+1}, Local Fetch Time = {res[0]:.4f}s, Remote Fetch Time = {res[1]:.4f}s')
            total_local_time += res[0]
            total_remote_time += res[1]
            total_local_pushes += res[2]
            total_remote_pushes += res[3]

        print(f'Avg fetch time per local neighbor = {1e6*total_local_time/total_local_pushes:.3f}μs,'
              f' Avg fetch time per remote neighbor = {1e6*total_remote_time/total_remote_pushes:.3f}μs')

    rpc.shutdown()


if __name__ == '__main__':
    print('Simulates Communication scenario')

    start = time.time()
    mp.spawn(run, nprocs=NUM_MACHINES, join=True)
    end = time.time()

    print(f'Total Execution time = {end - start:.3}s')
