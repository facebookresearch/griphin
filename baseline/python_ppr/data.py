#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import argparse
import os
import torch
import torch_geometric.transforms as T

import sklearn  # necessary for PygNodePropPredDataset!
from torch_geometric.data import Data
from torch_geometric_autoscale import metis, permute, SubgraphLoader
from ogb.nodeproppred import PygNodePropPredDataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ogbn-products')
parser.add_argument('--file_path', type=str, default='data/ogbn-products-p4')
parser.add_argument('--data_path', type=str, default='/data/gangda/ogb')
parser.add_argument('--num_partitions', type=int, default=4)
args = parser.parse_args()

if not os.path.isdir(args.file_path):
    os.makedirs(args.file_path)

NUM_PARTITION = args.num_partitions
FILENAME = 'weighted_partition_{}_{}.pt'
FILENAME2 = 'weighted_degree.pt'
FILENAME3 = 'permuted_graph_data.pt'
FILENAME4 = 'partition_book.pt'


def load_sub_data(ptr_idx: int, num_partition: int, save_dir: str):
    path_ = os.path.join(save_dir, FILENAME.format(num_partition, ptr_idx))
    return torch.load(path_)


if __name__ == '__main__':
    transform = T.Compose([T.ToUndirected(), T.ToSparseTensor()])

    if args.dataset == 'twitter' or args.dataset == 'friendster' or args.dataset == 'mag240M':
        edge_index = torch.load(os.path.join(args.data_path, args.dataset + '.pt'))
        data = Data(edge_index=edge_index)
        data = transform(data)
    else:
        dataset = PygNodePropPredDataset(name=args.dataset, root=args.data_path, transform=transform)
        data = dataset[0]

    # path = os.path.join(os.environ.get('DATA_DIR'), 'pyg', 'Reddit2')
    # dataset = Reddit2(path, transform=T.ToSparseTensor())
    # data = dataset[0]

    data.adj_t.set_value_(torch.rand(data.num_edges), layout='csc')

    perm, ptr = metis(data.adj_t, NUM_PARTITION, log=True)
    data = permute(data, perm, log=True)
    torch.save(data, os.path.join(args.file_path, FILENAME3))
    torch.save(ptr, os.path.join(args.file_path, FILENAME4))

    # collect degree after permutation
    # For easy implementation, we push messages to in-degree nodes in this python version
    degree = data.adj_t.sum(dim=1).to(torch.float)
    torch.save(degree, os.path.join(args.file_path, FILENAME2))

    data_list = list(SubgraphLoader(data, ptr, batch_size=1, shuffle=False))
    for i, sub_data in enumerate(data_list):
        print(sub_data)
        path = os.path.join(args.file_path, FILENAME.format(NUM_PARTITION, i))
        torch.save(sub_data, path)

    sub_data = load_sub_data(0, NUM_PARTITION, args.file_path)
    print('\n', sub_data)
