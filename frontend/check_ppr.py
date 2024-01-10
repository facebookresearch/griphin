#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import torch
import os
import dgl.sparse as dglsp
import numpy as np

from utils import get_data_path, get_root_path


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


root_dir = os.path.join(get_data_path(), 'hz-ogbn-product-p{}-pt'.format(4))

edge_index = torch.load(os.path.join(root_dir, 'dgl_edge_index.pt'))
edge_weights = torch.load(os.path.join(root_dir, 'dgl_edge_weights.pt'))
local_id_mapping = torch.load(os.path.join(root_dir, 'local_id_mapping.pt'))
shard_id_mapping = torch.load(os.path.join(root_dir, 'shard_id_mapping.pt'))

A = dglsp.spmatrix(edge_index, edge_weights)
D = dglsp.diag(A.sum(dim=1))
P = (D ** -1) @ A
P_t = P.t()

def get_global_id(local_id_, shard_id_):
    global_id = ((local_id_mapping == local_id_) & (shard_id_mapping == shard_id_)).nonzero(as_tuple=False).view(-1)
    return global_id


K = 100
datas = torch.load(os.path.join(get_root_path(), 'temp/data.pt'))
for local_root_id, data in enumerate(datas[:]):
    global_root_id = get_global_id(local_root_id, 0)
    base_p = power_iter_ppr(P_t, global_root_id, 0.462, 1e-10, 100)
    _, base_index = torch.sort(base_p, descending=True)
    base_global_top_k = base_index[:K]

    local_ids, shard_ids, p = data
    _, index = torch.sort(p, descending=True)
    index_top_k = index[:K]

    mae = 0.
    global_top_k = []
    for i in index_top_k.tolist():
        gid = get_global_id(local_ids[i], shard_ids[i])
        global_top_k.append(gid)
        mae += (p[i] - base_p[gid]).abs().item()

    K_ = len(index_top_k)
    concur = np.intersect1d(torch.cat(global_top_k), base_global_top_k).shape[0]
    print(f'MEAN Top-{K}: {p[index_top_k].mean():.3e}, MAE Top-{K}: {mae/K_:.3e}, Precision Top-{K}: {concur/K_:.2f}')
