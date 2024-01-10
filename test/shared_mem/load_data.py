#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import os

import torch
import torch.multiprocessing as mp
from dgl.ndarray import create_shared_mem_array


def load_data(rank, dataset, data_dir):
    data = torch.load(os.path.join(data_dir, '{}_egograph_list_{}.pt'.format(dataset, rank)))
    create_shared_mem_array('{}_{}'.format(dataset, rank), data.size())


if __name__ == '__main__':
    DATASET = 'ogbn-products'
    DATA_DIR = '/home/gangda/workspace/graph_engine/intermediate'

    num_processes = 8
    mp.spawn(load_data, nprocs=num_processes, args=(DATASET, DATA_DIR), join=True)

