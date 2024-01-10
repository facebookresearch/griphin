#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import os
import torch
import torch_geometric.transforms as T

import sklearn  # necessary for PygNodePropPredDataset!
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric_autoscale import metis, permute, SubgraphLoader

NUM_PARTITION = 4


def load_sub_data(ptr_idx: int, num_partition: int, save_dir: str):
    filename_ = f'partition_{num_partition}_{ptr_idx}.pt'
    path_ = os.path.join(save_dir, filename_)
    return torch.load(path_)


if __name__ == '__main__':
    dataset = PygNodePropPredDataset(name='ogbn-products', root='/data/gangda/ogb')
    data = dataset[0]

    # path = os.path.join(os.environ.get('DATA_DIR'), 'pyg', 'Reddit2')
    # dataset = Reddit2(path, transform=T.ToSparseTensor())
    # data = dataset[0]

    # cluster_data = ClusterData(data, num_parts=40)
    # data_list = list(ClusterLoader(cluster_data, batch_size=1, shuffle=False))
    # data = data_list[0]

    transform = T.ToSparseTensor()
    data = transform(data)

    perm, ptr = metis(data.adj_t, NUM_PARTITION, log=True)
    data = permute(data, perm, log=True)
    data_list = list(SubgraphLoader(data, ptr, batch_size=1, shuffle=False))

    for i, sub_data in enumerate(data_list):
        print(sub_data)
        filename = f'partition_{NUM_PARTITION}_{i}.pt'
        path = os.path.join(dataset.processed_dir, filename)
        torch.save(sub_data, path)

    sub_data = load_sub_data(0, NUM_PARTITION, dataset.processed_dir)
    print(sub_data)
