import time
import os

import torch
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from random import randrange
from torch.distributed.rpc import RRef, remote

import matplotlib.pyplot as plt
import numpy as np

from utils import get_root_path
from graph import GraphShard, VERTEX_ID_TYPE

NUM_MACHINES = 2

ALPHA = 0.462
EPSILON = 1e-3
MAX_DEGREE = -1
NUM_SOURCE = 1000

RUNS = 3000
WARMUP = 1000

WORKER_NAME = 'worker{}'
FILE_PATH = os.path.join(get_root_path(), 'data/ogbn_products_{}partitions'.format(NUM_MACHINES))


def fetch_neighbor(rrefs):
    rank = rpc.get_worker_info().id
    local_shard: GraphShard = rrefs[rank].to_here()

    max_local_id = 600000
    local_nids = torch.tensor([randrange(max_local_id)], dtype=VERTEX_ID_TYPE)
    remote_nids = torch.tensor([randrange(max_local_id)], dtype=VERTEX_ID_TYPE)

    # local time
    tik = time.time()
    ret = local_shard.batch_fetch_neighbors(local_nids)
    tok = time.time()
    deg1 = ret[0].shape[0]

    # remote time
    tik2 = time.time()
    fut = rrefs[1].rpc_async().batch_fetch_neighbors(remote_nids)
    neighbors = fut.wait()
    tok2 = time.time()
    deg2 = neighbors[0].shape[0]

    return tok - tik, tok2 - tik2, deg1, deg2


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

        d1, d2, t1, t2 = [], [], [], []
        total_local_time = total_remote_time = 0

        for i in range(RUNS):
            local_time, remote_time, deg1, deg2 = fetch_neighbor(rrefs)
            d1.append(deg1)
            d2.append(deg2)
            t1.append(local_time * 1000)
            t2.append(remote_time * 1000)
            print(f'Run {i}, Local Fetch Time = {local_time * 1000:.4f}ms,'
                  f' Remote Fetch Time = {remote_time * 1000:.4f}ms')
            total_local_time += local_time
            total_remote_time += remote_time

        print(f'Avg Local Fetch time = {total_local_time * 1000/RUNS:.4f}ms,'
              f' Avg Remote Fetch time = {total_remote_time * 1000/RUNS:.4f}ms')

        plt.scatter(d1, t1, color='blue', label='Local')
        plt.scatter(d2, t2, color='red', label='Remote')
        plt.title('Single Node Neighbor Fetch Time')
        plt.ylim(0, 1.2)
        plt.xlim(0, 2000)
        plt.xlabel('Node Degree')
        plt.ylabel('Time (ms)')
        plt.legend()
        plt.savefig('fetch_neighbor_single.png', dpi=300)

    rpc.shutdown()


if __name__ == '__main__':
    print('Simulates Communication scenario')

    start = time.time()
    mp.spawn(run, nprocs=NUM_MACHINES, join=True)
    end = time.time()

    print(f'Total Execution time = {end - start:.3}s')

