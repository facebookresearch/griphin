import os
import os.path as osp

import time
from collections import Counter

import torch
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef, remote

from torch_geometric.utils import mask_to_index
from pyg_lib.sampler import random_walk as pyg_random_walk

ENABLE_COMMUNICATION = True
NUM_MACHINES = 4
NUM_ROOTS = 4096
NUM_WALKS = 15
RUNS = 10
WARMUP = 3

WORKER_NAME = 'worker{}'
PROCESSED_DIR = osp.join(os.environ.get('DATA_DIR'), 'ogb', 'ogbn_products', 'processed')


class Walker:
    def __init__(self):
        self.id = rpc.get_worker_info().id
        sub_data = self.load_sub_data()
        indptr, indices, _ = sub_data.data.adj_t.csr()
        rowcount = sub_data.data.adj_t.storage.rowcount().to(torch.float32)

        self.indptr = indptr
        self.indices = indices
        self.n_id = sub_data.n_id
        self._batch_size = sub_data.batch_size
        self.rowcount = rowcount

    @property
    def batch_size(self):
        return self._batch_size

    def load_sub_data(self):
        filename_ = f'partition_{NUM_MACHINES}_{self.id}.pt'
        # filename_ = f'partition_40_{NUM_MACHINES}_{self.id}.pt'
        path_ = osp.join(PROCESSED_DIR, filename_)
        return torch.load(path_)

    def to_global(self, indices):
        return self.n_id[indices]

    def tensor_step(self, src_nodes):
        rowcount_ = self.rowcount[src_nodes]
        indptr_ = self.indptr[src_nodes]
        mask_ = torch.rand(rowcount_.size(0), dtype=torch.float32)
        mask_ = mask_.mul(rowcount_).to(torch.long).add_(indptr_)
        return self.indices[mask_]

    def pyg_step(self, src_nodes):  # pyg_lib
        return pyg_random_walk(self.indptr, self.indices, src_nodes, 1)[:, 1]

    def step(self, src_nodes, rank=None):
        # dst_nodes = self.tensor_step(src_nodes)
        dst_nodes = self.pyg_step(src_nodes)
        # return global id for remote access
        if rank != self.id:
            return self.to_global(dst_nodes)
        else:
            return dst_nodes


def random_walk(walker_rrefs):
    rank = rpc.get_worker_info().id
    walker = walker_rrefs[rank].to_here()

    batch_size = walker.batch_size
    cluster_ptr = torch.tensor([0, 613761, 1236365, 1838296, 2449029])  # TODO: ogbn_products
    # cluster_ptr = torch.tensor([0, 14997, 30413, 46108, 613761])  # TODO: ogbn_products 1/40
    root_nodes = torch.randperm(batch_size)[:NUM_ROOTS]
    # walks_summary = torch.full((NUM_ROOTS, NUM_WALKS + 1), -1)
    walks_summary = torch.empty((NUM_ROOTS, NUM_WALKS + 1), dtype=root_nodes.dtype)
    walks_summary[:, 0] = walker.to_global(root_nodes)

    # source nodes u, target nodes v
    u = root_nodes
    in_batch = u < batch_size
    out_of_batch = ~in_batch

    if rank == 0:
        print(f"Starting Distributed Random Walk:"
              f" world_size={NUM_MACHINES}, num_roots={NUM_ROOTS}, num_walks={NUM_WALKS}")

    # package_sizes = []
    for i in range(NUM_WALKS):
        # print(f'{i}---{rank}', u, out_of_batch)
        u_out_global = u[out_of_batch]
        futs, reverse_idx = [], []

        for j in range(NUM_MACHINES):
            if rank == j:
                continue
            mask = (u_out_global >= cluster_ptr[j]) & (u_out_global < cluster_ptr[j + 1])

            num_data = mask.sum()
            if num_data == 0:
                continue

            # package_sizes.append(num_data.item() * u_out_global.element_size())

            reverse_idx.append(mask_to_index(mask))
            u_out_j = u_out_global[mask] - cluster_ptr[j]  # global -> local

            # remote call
            if ENABLE_COMMUNICATION:
                fut = walker_rrefs[j].rpc_async().step(u_out_j)
                futs.append(fut)

        # Overlap in batch sampling with remote sampling:
        v = torch.empty_like(u)
        # 1. torch_sparse: serial neighbor sampling; Slowest, 10 units of time
        # 2. torch: neighbor count sampling; 1.5 ~ 2 units of time
        # 3. pyg_lib: ATen parallel neighbor sampling; Fastest, 1 unit of time
        v_local = walker.step(u[in_batch], rank)
        walks_summary[in_batch, i + 1] = walker.to_global(v_local)  # storing global index
        out_mask_local = v_local >= batch_size
        v_local[out_mask_local] = walker.to_global(v_local[out_mask_local])  # global index for halo nodes
        v[in_batch] = v_local

        # collect remote results
        v_remote = torch.empty_like(u_out_global)
        if ENABLE_COMMUNICATION:
            rets = []
            for fut in futs:
                rets.append(fut.wait())
            if len(rets) == 0:
                out_of_batch = out_mask_local
                in_batch = ~out_mask_local
                u = v
                continue
            v_remote[torch.cat(reverse_idx)] = torch.cat(rets)
        else:
            v_remote = u_out_global

        walks_summary[out_of_batch, i + 1] = v_remote  # storing global index
        in_mask_remote = (v_remote >= cluster_ptr[rank]) & (v_remote < cluster_ptr[rank + 1])
        v_remote[in_mask_remote] -= cluster_ptr[rank]  # local index
        v[out_of_batch] = v_remote

        # re-allocate masks
        out_of_batch[in_batch] = out_mask_local
        out_of_batch[~in_batch] = ~in_mask_remote
        in_batch = ~out_of_batch
        u = v

    # print(rank, walks_summary)
    # return package_sizes
    return walks_summary


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

        for i in range(RUNS + WARMUP):
            if i == WARMUP:
                tik_ = time.time()

            tik = time.time()

            futs = []
            for rref in rrefs:
                futs.append(
                    rpc.rpc_async(
                        rref.owner(),
                        random_walk,
                        args=(rrefs,)
                    )
                )
            c = []
            for fut in futs:
                c.append(fut.wait())

            tok = time.time()
            print(f'Run {i},  Time = {(tok - tik):.3f}s')

        tok_ = time.time()
        print(f'Random walk summary:\n {torch.cat(c, dim=0)}')
        print(f'Avg Execution time = {(tok_ - tik_)/RUNS:.3f}s')

    rpc.shutdown()


if __name__ == '__main__':
    if ENABLE_COMMUNICATION:
        print('Spawn Multi-Process to simulate Multi-Machine scenario')
    else:
        print('Simulates Non Communication scenario')

    tik = time.time()
    mp.spawn(run, nprocs=NUM_MACHINES, join=True)
    tok = time.time()

    print(f'Total Execution time = {tok - tik:.3}s')
