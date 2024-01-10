#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import os
import argparse
import time
import torch.multiprocessing as mp

import torch
import dgl.sparse as dglsp

parser = argparse.ArgumentParser()
parser.add_argument('--file_path', type=str, default='data/ogbn-products-p4')
parser.add_argument('--num_partitions', type=int, default=4)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--num_roots', type=int, default=10, help='number of source nodes in each machine')
args = parser.parse_args()

edge_index = torch.load(os.path.join(args.file_path, 'dgl_edge_index.pt'))
edge_weights = torch.load(os.path.join(args.file_path, 'dgl_edge_weights.pt'))
local_id_mapping = torch.load(os.path.join(args.file_path, 'local_id_mapping.pt'))
shard_id_mapping = torch.load(os.path.join(args.file_path, 'shard_id_mapping.pt'))

A = dglsp.spmatrix(edge_index, edge_weights)
D = dglsp.diag(A.sum(dim=1))
P = (D ** -1) @ A
P_t = P.t()


def get_global_id(local_id_, shard_id_):
    global_id = ((local_id_mapping == local_id_) & (shard_id_mapping == shard_id_)).nonzero(as_tuple=False).view(-1)
    return global_id


def power_iter_ppr(P_w, target_id_, alpha_, epsilon_, max_iter):
    num_nodes = P_w.shape[0]
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
    tik_ = time.time()
    for i in range(args.runs):
        tik = time.time()
        for local_root_id in torch.arange(args.num_roots):
            global_root_id = get_global_id(local_root_id, rank)
            power_iter_ppr(P_t, global_root_id, 0.462, 1e-10, 50)
        tok = time.time()
        print(f'Run {i}, Time = {tok - tik:.3f}s\n')

    print(f'\n Rank {rank}, Avg Run time = {(time.time() - tik_) / args.runs:.3f}s')


if __name__ == '__main__':
    print(f'Spawn {args.num_partitions}-Process to simulate {args.num_partitions}-Machine '
          f'full-graph Power Iteration')

    t1 = time.time()
    mp.spawn(run, nprocs=args.num_partitions, join=True)
    t2 = time.time()

    print(f'\nTotal Execution time = {(t2 - t1)/args.runs:.3f}s')
