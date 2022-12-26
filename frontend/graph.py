from collections import OrderedDict

import torch
import graph_engine

VERTEX_ID_TYPE = torch.int32


def init_graph(path, shard_id):
    ids_file = 'p{}_ids.txt'
    shards_file = 'p{}_halo_shards.txt'
    rows_file = 'p{}_edge_sources.txt'
    cols_file = 'p{}_edge_dests.txt'

    def _dir(filename):
        return path + filename.format(shard_id)

    return graph_engine.Graph(shard_id, _dir(ids_file), _dir(shards_file), _dir(rows_file), _dir(cols_file))


class GraphShard:
    """
    Front end wrapper for Graph.h
    """
    def __init__(self, path, shard_id):
        self.id = shard_id
        self.g = init_graph(path, shard_id)

    @property
    def cluster_size(self):
        return self.g.num_core_nodes()

    @property
    def cluster_ptr(self):
        return self.g.cluster_ptr()

    def to_global(self, indices, shard_id=None):
        if shard_id is None:
            shard_id = self.id
        return indices + self.cluster_ptr[shard_id]

    def step(self, src_nodes) -> (torch.Tensor, OrderedDict):
        nid, shard_dict = self.g.sample_single_neighbor(src_nodes)
        return nid, shard_dict

