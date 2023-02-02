import time
import os
import os.path as osp

import torch
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef, remote

NUM_MACHINES = 4

ALPHA = 0.462
EPSILON = 1e-3
MAX_DEGREE = 20
NUM_SOURCE = 1

RUNS = 1
WARMUP = 0

WORKER_NAME = 'worker{}'
PROCESSED_DIR = osp.join('/data/gangda', 'ogb', 'ogbn_products', 'processed')


class GraphShard:
    def __init__(self):
        self.id = rpc.get_worker_info().id
        sub_data = self.load_sub_data()
        indptr, indices, weights = sub_data.data.adj_t.csr()
        self.indptr = indptr
        self.indices = indices
        self.weights = weights
        self.n_id = sub_data.n_id
        self.num_core_nodes = sub_data.batch_size

    def load_sub_data(self):
        filename_ = f'weighted_partition_{NUM_MACHINES}_{self.id}.pt'
        path_ = osp.join(PROCESSED_DIR, filename_)
        return torch.load(path_)

    def fetch_neighbor_list(self, node_id):
        """
        :return: global_neighbor_ids, neighbor_weights
        """
        start, end = self.indptr[node_id], self.indptr[node_id+1]
        if MAX_DEGREE == -1 or MAX_DEGREE > end - start:
            ptr = torch.arange(start, end)
        else:
            ptr = torch.randperm(end-start)[:MAX_DEGREE] + start

        return self.n_id[self.indices[ptr]], self.weights[ptr]

    def batch_fetch_neighbor_list(self, node_ids):
        res = []
        for nid in node_ids:
            res.append(self.fetch_neighbor_list(nid))
        print('batch_fetch', len(res))
        return res


def approx_ppr(graph_rrefs):
    rank = rpc.get_worker_info().id
    local_shard: GraphShard = graph_rrefs[rank].to_here()

    if rank != 0:
        print(rank)
        return

    cluster_ptr = torch.tensor([0, 613761, 1236365, 1838296, 2449029])  # TODO: ogbn_products
    num_nodes = cluster_ptr[-1]

    local_sources = torch.randperm(local_shard.num_core_nodes)[:NUM_SOURCE] + cluster_ptr[rank]
    for epoch, target_id in enumerate([94314]):
        p = torch.zeros(num_nodes)
        r = torch.zeros(num_nodes)
        visited_degrees = torch.zeros(num_nodes)
        r[target_id] = 1

        _, source_neighbor_weights = local_shard.fetch_neighbor_list(target_id-cluster_ptr[rank])
        visited_degrees[target_id] = source_neighbor_weights.sum()

        def push(neighbor_infos, msg, v_ids):
            for info, v_msg, v_id in zip(neighbor_infos, msg, v_ids):
                u_ids, u_weights = info
                v_degree = u_weights.sum()
                r[u_ids] += v_msg * u_weights / v_degree
                # init sampled degree
                if visited_degrees[v_id] == 0:
                    visited_degrees[v_id] = v_degree

        iteration = 0
        while True > 0:
            v_mask = r > EPSILON * visited_degrees
            iteration += 1
            print('iter:', iteration, ', v_mask_sum:', v_mask.sum())
            if v_mask.sum() == 0:
                break

            v_idx = v_mask.nonzero(as_tuple=False).view(-1)
            print(rank, v_idx)

            p[v_idx] += ALPHA * r[v_idx]
            m_v = (1 - ALPHA) * r[v_idx]
            r[v_idx] = 0

            shard_masks = {}
            for j in range(NUM_MACHINES):
                shard_masks[j] = (v_idx >= cluster_ptr[j]) & (v_idx < cluster_ptr[j+1])

            futs = {}
            for j, j_mask in shard_masks.items():
                if rank == j or j_mask.sum() == 0:
                    continue
                v_out_j = v_idx[j_mask] - cluster_ptr[j]
                futs[j] = graph_rrefs[j].rpc_async().batch_fetch_neighbor_list(v_out_j)

            v_out_local = v_idx[shard_masks[rank]] - cluster_ptr[rank]
            print(v_idx[shard_masks[rank]].size(0))
            local_neighbor_infos = local_shard.batch_fetch_neighbor_list(v_out_local)
            push(local_neighbor_infos, m_v[shard_masks[rank]], v_idx[shard_masks[rank]])

            for j, fut in futs.items():
                infos = fut.wait()
                push(infos, m_v[shard_masks[j]], v_idx[shard_masks[j]])

            # for i, v in enumerate(v_idx):
            #     v_host_id = 0
            #     while v_host_id < NUM_MACHINES:
            #         if cluster_ptr[v_host_id] <= v < cluster_ptr[v_host_id+1]:
            #             break
            #         v_host_id += 1
            #
            #     u_idx, u_weights = graph_rrefs[v_host_id].rpc_sync().fetch_neighbor_list(v - cluster_ptr[v_host_id])
            #     print(v_host_id, u_idx)
            #     v_degree = u_weights.sum()
            #     r[u_idx] += m_v[i] * u_weights / v_degree
            #
            #     # update sampled degree
            #     visited_degrees[v] = v_degree

        print(rank, (p > 0).sum())


def run(rank):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=4)

    rpc.init_rpc(WORKER_NAME.format(rank), rank=rank, world_size=NUM_MACHINES, rpc_backend_options=options)

    if rank == 0:
        rrefs = []
        for machine_rank in range(NUM_MACHINES):
            info = rpc.get_worker_info(WORKER_NAME.format(machine_rank))
            rrefs.append(remote(info, GraphShard))

        for i in range(RUNS + WARMUP):
            if i == WARMUP:
                tik_ = time.time()

            tik = time.time()

            futs = []
            for rref in rrefs:
                futs.append(
                    rpc.rpc_async(
                        rref.owner(),
                        approx_ppr,
                        args=(rrefs,),
                        timeout=-1
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
    print('Simulates Communication scenario')

    tik = time.time()
    mp.spawn(run, nprocs=NUM_MACHINES, join=True)
    tok = time.time()

    print(f'Total Execution time = {tok - tik:.3}s')