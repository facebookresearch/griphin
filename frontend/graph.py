from collections import OrderedDict

import torch


class GraphShard:
    """
    Front end wrapper for Graph.h
    """
    def __init__(self, shard_id):
        self.id = shard_id
        self.g = None  # TODO: read from C layer

    @property
    def cluster_size(self):
        # return self.g.batch_size
        return 613761  # TODO: ogbn_products[0]

    @property
    def cluster_ptr(self):
        # return self.g.cluster_ptr
        return torch.tensor([0, 613761, 1236365, 1838296, 2449029])  # TODO: ogbn_products

    def to_global(self, indices, shard_id=None):
        if shard_id is None:
            shard_id = self.id
        return indices + self.cluster_ptr[shard_id]

    def step(self, src_nodes) -> (torch.Tensor, OrderedDict):
        return self.g.sampleSingleNeighbor(src_nodes)

