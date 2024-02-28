#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import argparse
import os
import os.path as osp

import torch
import dgl
from dgl.data import CoraGraphDataset

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='/data/gangda/dgl')
parser.add_argument('--output_path', type=str, default='/data/gangda/graph_engine')
args = parser.parse_args()

dataset = 'cora'

if not osp.isdir(osp.join(args.output_path, dataset)):
    os.makedirs(osp.join(args.output_path, dataset))

og = CoraGraphDataset(raw_dir=args.input_path)[0]

# to bidirected graph, remove duplicated
g = dgl.to_bidirected(og)

# assign random edge weight
g.edata['w'] = torch.rand((g.num_edges()))

dgl.save_graphs(osp.join(args.output_path, dataset, 'dgl_data_processed'), g)
