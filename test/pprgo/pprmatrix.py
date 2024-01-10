#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import argparse

import numpy as np
import torch
import os
import time

from utils import get_data
from ppr import topk_ppr_matrix

from dgl.data import CoraGraphDataset
from ogb.nodeproppred import DglNodePropPredDataset
import sklearn
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import add_remaining_self_loops, to_scipy_sparse_matrix, from_scipy_sparse_matrix
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ogbn-products')
parser.add_argument('--K', type=int, default=150)
parser.add_argument('--num_machines', type=int, default=2)
parser.add_argument('--num_processes', type=int, default=8)
args = parser.parse_args()

DATA_DIR = '/home/gangda/workspace/graph_engine/intermediate/'

data_name = args.dataset
K = args.K
alpha = 0.261
eps = 1e-5
topk = 150

data, edge_index = None, None
if data_name == 'ogbn-products':
    data = PygNodePropPredDataset('ogbn-products', root='/data/gangda/ogb')[0]
    edge_index = data.edge_index

ppr_adj = None
target_path = os.path.join(DATA_DIR, '{}_pprgo_ppr_adj.pt'.format(data_name))
if os.path.isfile(target_path):
    ppr_adj = torch.load(target_path)
else:
    # edge_index = add_remaining_self_loops(edge_index)[0]
    adj_matrix = to_scipy_sparse_matrix(edge_index)
    adj_matrix = adj_matrix.tocsr().astype(np.float32)
    idx = np.arange(data.num_nodes)

    tik = time.time()
    print('start ppr computing')
    ppr_adj = topk_ppr_matrix(adj_matrix, alpha, eps, idx, topk)
    print(f'PPR compute time: {time.time()-tik:.2f}')
    torch.save(ppr_adj, target_path)

ppr_matrix = torch.empty((data.num_nodes, topk), dtype=torch.long)
for bid in tqdm(range(data.num_nodes)):
    idx, weight = from_scipy_sparse_matrix(ppr_adj[bid])
    vals, order = weight.sort(descending=True)
    idx = idx[1][order]

    ppr_matrix[bid, :idx.shape[0]] = idx
    ppr_matrix[bid, idx.shape[0]:] = bid

torch.save(ppr_matrix, os.path.join(DATA_DIR, '{}_pprgo_ppr_matrix.pt'.format(data_name)))
print('PPR Matrix Finished!')
