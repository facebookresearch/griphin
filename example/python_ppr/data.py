import os
import torch
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.loader import ClusterLoader, ClusterData
from torch_geometric_autoscale import metis, permute, SubgraphLoader
from torch_geometric.datasets import Reddit2

NUM_PARTITION = 4
FILENAME = 'weighted_partition_{}_{}.pt'
FILENAME2 = 'weighted_degree.pt'
FILENAME3 = 'permuted_graph_data.pt'


def load_sub_data(ptr_idx: int, num_partition: int, save_dir: str):
    path_ = os.path.join(save_dir, FILENAME.format(num_partition, ptr_idx))
    return torch.load(path_)


if __name__ == '__main__':
    path = os.path.join('/data/gangda', 'ogb')
    dataset = PygNodePropPredDataset(name='ogbn-products', root=path, transform=T.ToSparseTensor())
    data = dataset[0]

    # path = os.path.join(os.environ.get('DATA_DIR'), 'pyg', 'Reddit2')
    # dataset = Reddit2(path, transform=T.ToSparseTensor())
    # data = dataset[0]

    data.adj_t.set_value_(torch.rand(data.num_edges), layout='csc')

    perm, ptr = metis(data.adj_t, NUM_PARTITION, log=True)
    data = permute(data, perm, log=True)
    torch.save(data, os.path.join(dataset.processed_dir, FILENAME3))

    # collect degree after permutation
    # For easy implementation, we push messages to in-degree nodes in this python version
    degree = data.adj_t.sum(dim=1).to(torch.float)
    torch.save(degree, os.path.join(dataset.processed_dir, FILENAME2))

    data_list = list(SubgraphLoader(data, ptr, batch_size=1, shuffle=False))
    for i, sub_data in enumerate(data_list):
        print(sub_data)
        path = os.path.join(dataset.processed_dir, FILENAME.format(NUM_PARTITION, i))
        torch.save(sub_data, path)

    sub_data = load_sub_data(0, NUM_PARTITION, dataset.processed_dir)
    print('\n', sub_data)
