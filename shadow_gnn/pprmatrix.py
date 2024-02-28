#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import argparse

import torch
import os

from dgl.data import CoraGraphDataset
from ogb.nodeproppred import DglNodePropPredDataset
from tqdm import tqdm

def extract_core_global_ids(parts):
    part_core_global_ids = []
    for i in range(len(parts)):
        part = parts[i]  # parts is a dict
        core_mask = part.ndata['inner_node'].type(torch.bool)
        part_global_id = part.ndata['orig_id']
        part_core_global_ids.append(part_global_id[core_mask])
    return part_core_global_ids


def get_global_id(local_ids, shard_ids):
    local_ids = local_ids.to(torch.long)
    global_ids = torch.empty_like(local_ids)
    for j in range(len(dataset['part_core_global_ids'])):
        mask = shard_ids == j
        if mask.sum() == 0: continue
        global_ids[mask] = dataset['part_core_global_ids'][j][local_ids[mask]]
    return global_ids


parser = argparse.ArgumentParser()
parser.add_argument('--K', type=int, default=150)
parser.add_argument('--dataset', type=str, default='ogbn-products')
parser.add_argument('--data_path', type=str, default='data/ogbn-products-p2', help='graph shards dir')
parser.add_argument('--file_path', type=str, default='intermediate')
parser.add_argument('--num_machines', type=int, default=2)
parser.add_argument('--num_processes', type=int, default=8)
args = parser.parse_args()

data_name = args.dataset
data_path = args.data_path
K = args.K
FILE_DIR = args.file_path

# load dataset
if data_name == 'ogbn-products':
    og, y = DglNodePropPredDataset(name='ogbn-products', root='/data/gangda/dgl')[0]
elif data_name == 'cora':
    og = CoraGraphDataset(raw_dir='/data/gangda/dgl', verbose=False)[0]
    y = og.ndata['label']
else:
    raise NotImplementedError
dataset = dict(X=og.ndata['feat'], y=y)
dataset['edge_index'] = torch.load(os.path.join(data_path, 'dgl_edge_index.pt'))
parts = torch.load(os.path.join(data_path, 'metis_partitions.pt'))
dataset['part_core_global_ids'] = extract_core_global_ids(parts)
print('Dataset Loading Finished')

# get global ids
shard_ptrs = torch.load(os.path.join(data_path, 'partition_book.pt'))
num_roots = shard_ptrs[1:] - shard_ptrs[:-1]
vids = torch.cat([torch.arange(nr) for nr in num_roots])  # local vertex IDs
sids = torch.cat([torch.full((nr,), i) for i, nr in enumerate(num_roots)])  # shard IDs
gids = get_global_id(vids, sids)  # global vertex IDs
print('Global Vertex ID Conversion Finished')

# load ppr results
pbar = tqdm(total=args.num_machines * args.num_processes)
pbar.set_description(f'Load PPR results')
res = []
for i in range(args.num_machines):
    for j in range(args.num_processes):
        res += torch.load(os.path.join(FILE_DIR, '{}_{}_{}.pt'.format(data_name, i, j)))
        pbar.update(1)
pbar.close()
print('PPR Results Loading Finished')

global_topk = []
for i, r in tqdm(enumerate(res)):
    val, idx = torch.sort(r[2], descending=True)
    top_k_index = idx[:K]
    global_ids = get_global_id(r[0][top_k_index], r[1][top_k_index])
    vec = torch.full((K,), gids[i])
    vec[:global_ids.shape[0]] = global_ids
    global_topk.append(vec)
ppr_matrix = torch.empty((dataset['X'].shape[0], K), dtype=torch.long)
ppr_matrix[gids] = torch.stack(global_topk)
torch.save(ppr_matrix, os.path.join(FILE_DIR, '{}_ppr_matrix.pt'.format(data_name)))
print('PPR Matrix Construction Finished!')
