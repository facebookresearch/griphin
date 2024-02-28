#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import argparse
import os.path as osp

import dgl
import numpy as np
import torch
from ogb.lsc import MAG240MDataset

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='/data/gangda/ogb')
parser.add_argument('--output_path', type=str, default='/data/gangda/dgl')
args = parser.parse_args()

print("Building graph")
dataset = MAG240MDataset(root=args.input_path)
ei_writes = dataset.edge_index("author", "writes", "paper")
ei_cites = dataset.edge_index("paper", "paper")
ei_affiliated = dataset.edge_index("author", "institution")
g = dgl.heterograph(
    {
        ("author", "write", "paper"): (ei_writes[0], ei_writes[1]),
        ("paper", "write-by", "author"): (ei_writes[1], ei_writes[0]),
        ("author", "affiliate-with", "institution"): (
            ei_affiliated[0],
            ei_affiliated[1],
        ),
        ("institution", "affiliate", "author"): (
            ei_affiliated[1],
            ei_affiliated[0],
        ),
        ("paper", "cite", "paper"): (
            np.concatenate([ei_cites[0], ei_cites[1]]),
            np.concatenate([ei_cites[1], ei_cites[0]]),
        ),
    }
)

print("Processing graph")
g = dgl.to_homogeneous(g)
# DGL ensures that nodes with the same type are put together with the order preserved.
# DGL also ensures that the node types are sorted in ascending order.
assert torch.equal(
    g.ndata[dgl.NTYPE],
    torch.cat(
        [
            torch.full((dataset.num_authors,), 0),
            torch.full((dataset.num_institutions,), 1),
            torch.full((dataset.num_papers,), 2),
        ]
    ),
)
assert torch.equal(
    g.ndata[dgl.NID],
    torch.cat(
        [
            torch.arange(dataset.num_authors),
            torch.arange(dataset.num_institutions),
            torch.arange(dataset.num_papers),
        ]
    ),
)
g.edata["etype"] = g.edata[dgl.ETYPE].byte()
del g.edata[dgl.ETYPE]
del g.ndata[dgl.NTYPE]
del g.ndata[dgl.NID]

g = dgl.to_bidirected(g)
g.edata['w'] = torch.rand((g.num_edges()))
dgl.save_graphs(osp.join(args.output_path, 'mag240M', 'dgl_data_processed'), g)

