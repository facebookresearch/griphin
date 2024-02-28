#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import argparse
import os
import os.path as osp

import torch
import dgl
from ogb.nodeproppred import DglNodePropPredDataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ogbn-products')
parser.add_argument('--input_path', type=str, default='/data/gangda/dgl')
parser.add_argument('--output_path', type=str, default='/data/gangda/graph_engine')
parser.add_argument('--unweighted', action='store_true')
args = parser.parse_args()

if not osp.isdir(osp.join(args.output_path, args.dataset)):
    os.makedirs(osp.join(args.output_path, args.dataset))

og, _ = DglNodePropPredDataset(name=args.dataset, root=args.input_path)[0]

# to bidirected graph, remove duplicated
g = dgl.to_bidirected(og)

# assign random edge weight
g.edata['w'] = torch.ones((g.num_edges())) if args.unweighted else torch.rand((g.num_edges()))

dgl.save_graphs(osp.join(args.output_path, args.dataset, 'dgl_data_processed'), g)
