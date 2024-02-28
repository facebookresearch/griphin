#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import dgl
import torch
import argparse
import os
from ogb.nodeproppred import DglNodePropPredDataset
import dgl.sparse as dglsp

# Data types in frontend/graph.py, data_generation/gen_engine_data.py,
# and engine/global.h should be consistent
VERTEX_ID_TYPE = torch.int32
EDGE_ID_TYPE = torch.int64
SHARD_ID_TYPE = torch.int8
WEIGHT_TYPE = torch.float32


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='ogbn-products', help='which dataset to use')
parser.add_argument('--num_partition', type=int, default=4, help='how many partitions')
parser.add_argument('--input_path', type=str, default='/data/gangda/graph_engine')
parser.add_argument('--output_path', type=str, default='data', help='output_path')
args = parser.parse_args()

out_path = os.path.join(args.output_path, '{}-p{}'.format(args.data, args.num_partition))
if not os.path.isdir(out_path):
    os.makedirs(out_path)


def to_file(path, fn, d):
    torch.save(d, '{}/{}'.format(path, fn))


"""Load DGL Format Data"""
# if args.data == 'twitter' or args.data == 'friendster' or args.data == 'mag240M':
#     (og,), _ = dgl.load_graphs(os.path.join(args.input_path, args.data, 'dgl_data_processed'))
# else:
#     og, _ = DglNodePropPredDataset(name=args.data, root=args.input_path)[0]
# g = dgl.to_bidirected(og)
# g.edata['w'] = torch.rand((g.num_edges()))

(g,), _ = dgl.load_graphs(os.path.join(args.input_path, args.data, 'dgl_data_processed'))
print(f'dataset: {args.data}, num_nodes: {g.num_nodes()}, num_edges: {g.num_edges()}')

# compute node weighted degree
g_indices = torch.stack(g.edges())
N = g.num_nodes()
A = dglsp.spmatrix(indices=g_indices, val=g.edata['w'], shape=(N, N))
g.ndata['weighted_degree'] = A.sum(dim=1)
# g.update_all(dgl.function.copy_e('w', 'weighted_degree'), dgl.function.sum('weighted_degree', 'weighted_degree'))

print('--- Load DGL Format Data Finished ---')


"""Graph Partition"""
parts = dgl.metis_partition(g, k=args.num_partition, extra_cached_hops=1, reshuffle=True)

# get partition mappings
globalid_to_localid = torch.zeros((g.num_nodes()), dtype=torch.int64)
globalid_to_shardid = torch.zeros((g.num_nodes()), dtype=torch.int64)
partition_book = [0]
for i in range(len(parts)):
    part = parts[i]

    core_mask = part.ndata['inner_node'].type(torch.bool)
    part_global_id = part.ndata['orig_id']
    part_core_global_id = part_global_id[core_mask]

    globalid_to_localid[part_core_global_id] = torch.arange(part_core_global_id.shape[0])
    globalid_to_shardid[part_core_global_id] = i

    partition_book.append(part_core_global_id.shape[0] + partition_book[i])

print('--- Graph Partition Finished ---')


"""Save DGL Format Data"""
to_file(out_path, 'metis_partitions.pt', parts)
to_file(out_path, 'local_id_mapping.pt', globalid_to_localid)
to_file(out_path, 'shard_id_mapping.pt', globalid_to_shardid)
to_file(out_path, 'dgl_edge_index.pt', g_indices)
to_file(out_path, 'dgl_edge_weights.pt', g.edata['w'])

print('--- Save DGL Format Data Finished ---')


"""Save Graph Engine Format Data"""
to_file(out_path, 'partition_book.pt', torch.tensor(partition_book))
for i in range(len(parts)):
    part = parts[i]
    core_mask = part.ndata['inner_node'].type(torch.bool)
    # make sure all core nodes have smaller local id than halo nodes
    assert core_mask[:core_mask.sum()].sum() == core_mask.sum()

    num_core_nodes = core_mask.sum()
    csr = part.adj_tensors('csr')

    csr_indptr = csr[0][:num_core_nodes + 1]
    to_file(out_path, 'p{}_indptr.pt'.format(i), csr_indptr.to(EDGE_ID_TYPE))

    # convert to graph engine format
    csr_indices_localid = csr[1][:csr_indptr[-1]]
    csr_indices_globalid = part.ndata['orig_id'][csr_indices_localid]
    csr_indices_engine_localid = globalid_to_localid[csr_indices_globalid]
    csr_indices_engine_shardid = globalid_to_shardid[csr_indices_globalid]
    to_file(out_path, 'p{}_indices_node_id.pt'.format(i), csr_indices_engine_localid.to(VERTEX_ID_TYPE))
    to_file(out_path, 'p{}_indices_shard_id.pt'.format(i), csr_indices_engine_shardid.to(SHARD_ID_TYPE))

    # get edge weights 
    csr_indices_weights = g.edata['w'][part.edata['orig_id']][csr[2][:csr_indptr[-1]]]
    csr_indices_weighted_degree = g.ndata['weighted_degree'][csr_indices_globalid]
    to_file(out_path, 'p{}_indices_edge_weight.pt'.format(i), csr_indices_weights.to(WEIGHT_TYPE))
    to_file(out_path, 'p{}_indices_weighted_degree.pt'.format(i), csr_indices_weighted_degree.to(WEIGHT_TYPE))

print('--- Save Graph Engine Format Data Finished ---')


# change group owner of written files
# os.system('chown -R :meta_research {}'.format(out_path))
