#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import argparse
import os
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'

import torch
import torch.multiprocessing as mp

import dgl
from dgl.data import CoraGraphDataset
from ogb.nodeproppred import DglNodePropPredDataset

import os.path as osp

from torch_geometric.data import Data
from torch_geometric.utils import subgraph, add_remaining_self_loops
from pyg_lib.sampler import subgraph as libsubgraph
from torch_sparse import SparseTensor
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default='ogbn-products')
parser.add_argument('--dgl_path', type=str, default='/data/gangda/dgl')
parser.add_argument('--data_path', type=str, default='data/ogbn-products-p2')
# parser.add_argument('--file_path', type=str, default='intermediate')
parser.add_argument('--ppr_file', type=str, default='test_dir/ogbn-products_0.261_1e-05_top150.pt')
parser.add_argument('--num_processes', type=int, default=20)
parser.add_argument('--ppr', type=str, default='engine')
args = parser.parse_args()

file_path = os.path.dirname(args.ppr_file)


def to_edge_index(rowptr, col):
    row = torch.arange(rowptr.size(0) - 1, dtype=col.dtype, device=col.device)
    row = row.repeat_interleave(rowptr[1:] - rowptr[:-1])
    return torch.stack([row, col], dim=0)


def run(rank, ppr_matrix, y, rowptr, col):
    num_nodes = y.shape[0]
    source_nodes = torch.arange(num_nodes).split(num_nodes // args.num_processes)
    if len(source_nodes) > args.num_processes:
        last_source = torch.cat([source_nodes[-2], source_nodes[-1]])
        source_nodes = source_nodes[:-2] + (last_source,)
    batch_index = source_nodes[rank]

    # datas = []
    node_offset = [0]
    edge_offset = [0]
    sub_nidx = []
    sub_eidx = []
    ego_idx = []

    if rank == 0:
        batch_index = tqdm(batch_index)
    for i, bid in enumerate(batch_index):
        ppr_vec = ppr_matrix[bid]
        ppr_vec[ppr_vec == -1] = bid
        subset, inv = torch.cat([bid.unsqueeze(dim=0), ppr_vec]).unique(return_inverse=True)
        ego_index = inv[0].item()

        ego_rowptr, ego_col, _ = libsubgraph(rowptr, col, subset, return_edge_id=False)
        sub_edge_index = to_edge_index(ego_rowptr, ego_col)

        node_offset.append(node_offset[i] + subset.shape[0])
        edge_offset.append(edge_offset[i] + sub_edge_index.shape[1])
        sub_nidx.append(subset)
        sub_eidx.append(sub_edge_index)
        ego_idx.append(ego_index)

    node_offset.pop(-1)
    edge_offset.pop(-1)
    assert len(node_offset) == len(edge_offset) == len(batch_index)

    node_offset = torch.tensor(node_offset)
    edge_offset = torch.tensor(edge_offset)
    sub_nidx = torch.cat(sub_nidx)
    sub_eidx = torch.cat(sub_eidx, dim=1)
    ego_idx = torch.tensor(ego_idx)
    datas = {
        'node_offset': node_offset,
        'edge_offset': edge_offset,
        'sub_nidx': sub_nidx,
        'sub_eidx': sub_eidx,
        'ego_idx':  ego_idx
    }
    if args.ppr == 'pprgo':
        torch.save(datas, osp.join(file_path, '{}_pprgo_egograph_datas_{}.pt'.format(args.data_name, rank)))
    else:
        torch.save(datas, osp.join(file_path, '{}_egograph_datas_{}.pt'.format(args.data_name, rank)))
    del datas
    print('Rank', rank, 'finished')


def assemble_data_list():
    # Assemble Data List
    node_offset, edge_offset, sub_nidx, sub_eidx, ego_idx = [], [], [], [], []
    n_ptr, e_ptr = 0, 0
    for r in tqdm(range(args.num_processes)):
        if args.ppr == 'pprgo':
            data = torch.load(osp.join(file_path, '{}_pprgo_egograph_datas_{}.pt'.format(args.data_name, r)))
        else:
            data = torch.load(osp.join(file_path, '{}_egograph_datas_{}.pt'.format(args.data_name, r)))
        node_offset.append(data['node_offset'] + n_ptr)
        edge_offset.append(data['edge_offset'] + e_ptr)
        sub_nidx.append(data['sub_nidx'])
        sub_eidx.append(data['sub_eidx'])
        ego_idx.append(data['ego_idx'])
        n_ptr += data['sub_nidx'].shape[0]
        e_ptr += data['sub_eidx'].shape[1]
    node_offset.append(torch.tensor([n_ptr]))
    edge_offset.append(torch.tensor([e_ptr]))

    return {
        'node_offset': torch.cat(node_offset),
        'edge_offset': torch.cat(edge_offset),
        'sub_nidx': torch.cat(sub_nidx),
        'sub_eidx': torch.cat(sub_eidx, dim=1),
        'ego_idx': torch.cat(ego_idx)
    }


if __name__ == '__main__':
    if args.data_name == 'ogbn-products':
        og, y = DglNodePropPredDataset(name='ogbn-products', root=args.dgl_path)[0]
    elif args.data_name == 'cora':
        og = CoraGraphDataset(raw_dir=args.dgl_path, verbose=False)[0]
        y = og.ndata['label']
    else:
        raise NotImplementedError

    dataset = dict(X=og.ndata['feat'], y=y)
    dataset['edge_index'] = torch.load(osp.join(args.data_path, 'dgl_edge_index.pt'))
    dataset['edge_index'] = add_remaining_self_loops(dataset['edge_index'])[0]
    adj = SparseTensor.from_edge_index(dataset['edge_index'])
    rowptr, col, _ = adj.csr()

    # load ppr matrix
    results = torch.load(args.ppr_file)
    ppr_matrix = results['global_node_ids']

    # extract subgraphs
    mp.spawn(run, nprocs=args.num_processes, args=(ppr_matrix, y, rowptr, col), join=True)
    print('EGO Graph Computing Finished!')

    # assemble data list
    datas = assemble_data_list()
    assert len(datas['node_offset']) == len(datas['edge_offset']) == y.shape[0] + 1
    assert datas['node_offset'][-1] == datas['sub_nidx'].shape[0] \
           and datas['edge_offset'][-1] == datas['sub_eidx'].shape[1]
    print('Data List Assembling Finished!')

    # store assembled datas
    if args.ppr == 'pprgo':
        path = osp.join(file_path, '{}_pprgo_egograph_datas.pt'.format(args.data_name))
        for r in range(args.num_processes):
            os.remove(osp.join(file_path, '{}_pprgo_egograph_datas_{}.pt'.format(args.data_name, r)))
    else:
        path = osp.join(file_path, '{}_egograph_datas.pt'.format(args.data_name))
        for r in range(args.num_processes):
            os.remove(osp.join(file_path, '{}_egograph_datas_{}.pt'.format(args.data_name, r)))
    torch.save(datas, path)
    print('EGO Graph Matrix Processing Finished!')
    print('Saved Path: {}'.format(path))
