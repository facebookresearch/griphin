#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import argparse
import time
import os
import os.path as osp

import torch
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef, remote

# Must import!
import sklearn
from torch_geometric_autoscale.loader import SubData

parser = argparse.ArgumentParser()
parser.add_argument('--file_path', type=str, default='/data/gangda/ogb/ogbn_products/processed')
parser.add_argument('--num_partitions', type=int, default=4)
parser.add_argument('--num_roots', type=int, default=10, help='number of source nodes in each machine')
args = parser.parse_args()

NUM_MACHINES = args.num_partitions

ALPHA = 0.462
EPSILON = 1e-6
MAX_DEGREE = -1
NUM_SOURCE = args.num_roots

RUNS = 10
WARMUP = 3
LOG = False

WORKER_NAME = 'worker{}'
FILENAME = 'weighted_partition_{}_{}.pt'
FILENAME2 = 'weighted_degree.pt'
FILENAME3 = 'permuted_graph_data.pt'
FILENAME4 = 'partition_book.pt'
PROCESSED_DIR = args.file_path


class GraphShard:
    def __init__(self):
        self.id = rpc.get_worker_info().id
        # For simplicity, assume each shard owns full degree list
        self.full_degree = torch.load(osp.join(PROCESSED_DIR, FILENAME2))

        sub_data: SubData = self.load_sub_data()
        indptr, indices, weights = sub_data.data.adj_t.csr()

        self.indptr = indptr
        self.indices = indices
        self.weights = weights
        self.n_id = sub_data.n_id.to(torch.int32)
        self.num_core_nodes = sub_data.batch_size

    def load_sub_data(self):
        path_ = osp.join(PROCESSED_DIR, FILENAME.format(NUM_MACHINES, self.id))
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
        data_size = 0
        for nid in node_ids:
            neighborhood = self.fetch_neighbor_list(nid)
            res.append(neighborhood)
            data_size += neighborhood[0].numel() * neighborhood[0].element_size() +\
                         neighborhood[1].numel() * neighborhood[1].element_size()
        if LOG:
            print(f'Rank {self.id} batch_fetch, neighbors: {len(res)}, data_size: {data_size/1e6:.2f}MB')
        return res


def approx_ppr(graph_rrefs):
    rank = rpc.get_worker_info().id
    local_shard: GraphShard = graph_rrefs[rank].to_here()
    full_degree = local_shard.full_degree

    cluster_ptr = torch.load(osp.join(PROCESSED_DIR, FILENAME4))
    num_nodes = cluster_ptr[-1]

    time_pop = 0
    time_mask = 0
    time_fetch_neighbor_local = 0
    time_fetch_neighbor_remote = 0
    time_push = 0

    time_start = time.time()

    ssppr_list = []
    local_sources = torch.randperm(local_shard.num_core_nodes)[:NUM_SOURCE] + cluster_ptr[rank]
    for epoch, target_id in enumerate(local_sources):
        p = torch.zeros(num_nodes)
        r = torch.zeros(num_nodes)
        threshold = EPSILON * full_degree
        r[target_id] = 1

        def push(neighbor_infos, msg):
            for info, v_msg in zip(neighbor_infos, msg):
                u_ids, u_weights = info
                u_ids = u_ids.to(torch.long)
                r[u_ids] += v_msg * u_weights / u_weights.sum()

        iteration = 0
        while True:
            tik = time.time()
            v_mask = r > threshold
            iteration += 1
            if LOG:
                print('iter:', iteration, ', v_mask_sum:', v_mask.sum().item())
            if v_mask.sum() == 0:
                break
            v_idx = v_mask.nonzero(as_tuple=False).view(-1)
            time_pop += time.time() - tik

            tik = time.time()
            p[v_idx] += ALPHA * r[v_idx]
            m_v = (1 - ALPHA) * r[v_idx]
            r[v_idx] = 0

            shard_masks = {}
            for j in range(NUM_MACHINES):
                shard_masks[j] = (v_idx >= cluster_ptr[j]) & (v_idx < cluster_ptr[j+1])

            time_mask += time.time() - tik

            tik_ = time.time()
            futs = {}
            for j, j_mask in shard_masks.items():
                if rank == j or j_mask.sum() == 0:
                    continue
                v_out_j = v_idx[j_mask] - cluster_ptr[j]
                futs[j] = graph_rrefs[j].rpc_async().batch_fetch_neighbor_list(v_out_j)

            tik = time.time()
            v_out_local = v_idx[shard_masks[rank]] - cluster_ptr[rank]
            local_neighbor_infos = local_shard.batch_fetch_neighbor_list(v_out_local)
            time_fetch_neighbor_local += time.time() - tik

            infos = {}
            for j, fut in futs.items():
                infos[j] = fut.wait()
            time_fetch_neighbor_remote += time.time() - tik_

            tik = time.time()
            push(local_neighbor_infos, m_v[shard_masks[rank]])
            for j, info in infos.items():
                push(info, m_v[shard_masks[j]])
            time_push += time.time() - tik

        if LOG:
            print(f'\nEpoch: {epoch}, Rank{rank}, NNZ = {(p > 0).sum().item()} \n')
        ssppr_list.append(p)

    if rank == 0:
        print(f'Time pop: {time_pop:.3f}s, '
              f'Time mask: {time_mask:.3f}s, '
              f'Time fetch local: {time_fetch_neighbor_local:.3f}s, '
              f'Time fetch remote: {time_fetch_neighbor_remote:.3f}s, '
              f'Time push: {time_push:.3f}s, '
              f'Time total: {time.time()-time_start:.3f}s')

    return ssppr_list, time_fetch_neighbor_local, time_fetch_neighbor_remote, time_push


def power_iter_ppr(P_w, target_id_, alpha_, epsilon_, max_iter):
    num_nodes = P_w.size(0)
    s = torch.zeros(num_nodes)
    s[target_id_] = 1
    s = s.view(-1, 1)

    x = s.clone()
    for i in range(max_iter):
        x_last = x
        x = alpha_ * s + (1 - alpha_) * (P_w @ x)
        # check convergence, l1 norm
        if (abs(x - x_last)).sum() < num_nodes * epsilon_:
            print(f'power-iter      Iterations: {i}, NNZ: {(x.view(-1) > 0).sum()}')
            return x.view(-1)

    print(f'Failed to converge with tolerance({epsilon_}) and iter({max_iter})')
    return x.view(-1)


def run(rank):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29502'
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=4, rpc_timeout=600)

    rpc.init_rpc(WORKER_NAME.format(rank), rank=rank, world_size=NUM_MACHINES, rpc_backend_options=options)

    if rank == 0:
        rrefs = []
        for machine_rank in range(NUM_MACHINES):
            info = rpc.get_worker_info(WORKER_NAME.format(machine_rank))
            rrefs.append(remote(info, GraphShard))

        # Compute Ground_truth
        # data = torch.load(osp.join(PROCESSED_DIR, FILENAME3))
        # degree = torch.load(osp.join(PROCESSED_DIR, FILENAME2))
        # norm_adj_t = data.adj_t * degree.pow(-1).view(1, -1)

        time_local_fetch = 0
        time_remote_fetch = 0
        time_push = 0

        for i in range(RUNS + WARMUP):
            if i == WARMUP:
                tik_ = time.time()

            # target_id = torch.randperm(10000)[0]
            # base_p = power_iter_ppr(norm_adj_t, target_id, ALPHA, 1e-10, 100)

            tik = time.time()
            # approx_p = approx_ppr(rrefs)

            futs = []
            for rref in rrefs:
                futs.append(
                    rpc.rpc_async(
                        rref.owner(),
                        approx_ppr,
                        args=(rrefs,)
                    )
                )
            c = []
            for j, fut in enumerate(futs):
                res = fut.wait()
                c.append(res[0])

                if j == 0 and i >= WARMUP:
                    time_local_fetch += res[1]
                    time_remote_fetch += res[2]
                    time_push += res[3]

            tok = time.time()

            print(f'Run {i},  Time = {(tok - tik):.3f}s\n')

            # print(f'Mean power-iter ppr: {base_p.mean():.3e},'
            #       f' MAE: {(abs(approx_p - base_p)).sum().item() / data.num_nodes:.3e}')

        tok_ = time.time()

        print(f'Avg Time fetch local: {time_local_fetch/RUNS:.3f}s, '
              f'Avg Time fetch remote: {time_remote_fetch/RUNS:.3f}s, '
              f'Avg Time push: {time_push/RUNS:.3f}s')

        avg_time = (tok_ - tik_)/RUNS
        print(f'Avg Execution time = {avg_time:.3f}s')
        print(f'Avg Throughput = {args.num_roots*args.num_partitions/avg_time:.3f} nodes/s')

    rpc.shutdown()


if __name__ == '__main__':
    print('Simulates Communication scenario')

    start_time = time.time()
    mp.spawn(run, nprocs=NUM_MACHINES, join=True)
    end_time = time.time()

    print(f'Total Execution time = {end_time - start_time:.3}s')
