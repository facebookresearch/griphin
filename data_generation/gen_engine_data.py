import dgl
import torch
import argparse
import os
from ogb.nodeproppred import DglNodePropPredDataset
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='ogbn-products', help='which dataset to use')
parser.add_argument('--num_partition', type=int, default=4, help='how many partitions')
parser.add_argument('--path', type=str, default='data/hz-ogbn-product-p4', help='output_path')
args = parser.parse_args()

if not os.path.isdir(args.path):
    os.makedirs(args.path)

og, _ = DglNodePropPredDataset(name=args.data, root='/data/gangda/dgl')[0]

# assign random edge weight
og.edata['w'] = torch.rand((og.num_edges()))

# add reverse edges
g = dgl.add_reverse_edges(og, copy_edata=True)

# compute node weighted degree
g.update_all(dgl.function.copy_e('w', 'weighted_degree'), dgl.function.sum('weighted_degree', 'weighted_degree'))

parts = dgl.metis_partition(g, k=args.num_partition, extra_cached_hops=1, reshuffle=True)


def to_file(path, fn, d):
    with open('{}/{}'.format(path, fn), 'w') as f:
        for i in tqdm(d.tolist()):
            f.write(str(i) + '\n')


# get partitions
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
to_file(args.path, 'partition_book.txt', torch.tensor(partition_book))

for i in range(len(parts)):
    part = parts[i]
    core_mask = part.ndata['inner_node'].type(torch.bool)
    # make sure all core nodes have smaller local id than halo nodes
    assert core_mask[:core_mask.sum()].sum() == core_mask.sum()

    num_core_nodes = core_mask.sum()
    csr = part.adj_sparse('csr')

    csr_indptr = csr[0][:num_core_nodes + 1]
    to_file(args.path, 'csr_indptr{}.txt'.format(i), csr_indptr)

    # convert to graph engine format
    csr_indices_localid = csr[1][:csr_indptr[-1]]
    csr_indices_globalid = part.ndata['orig_id'][csr_indices_localid]
    csr_indices_engine_localid = globalid_to_localid[csr_indices_globalid]
    csr_indices_engine_shardid = globalid_to_shardid[csr_indices_globalid]
    to_file(args.path, 'csr_indices{}.txt'.format(i), csr_indices_engine_localid)
    to_file(args.path, 'csr_shards{}.txt'.format(i), csr_indices_engine_shardid)

    # get edge weights 
    csr_indices_weights = g.edata['w'][part.edata['orig_id']][csr[2][:csr_indptr[-1]]]
    csr_indices_weighted_degree = g.ndata['weighted_degree'][csr_indices_globalid]
    to_file(args.path, 'csr_edge_weights_p{}.txt'.format(i), csr_indices_weights)
    to_file(args.path, 'csr_weighted_degrees_p{}.txt'.format(i), csr_indices_weighted_degree)
