from typing import Tuple, Dict

import torch
import graph_engine
import os.path as osp

from torch import Tensor

VERTEX_ID_TYPE = torch.int32


def init_graph(path, shard_id):
    ids_file = 'p{}_ids.txt'
    shards_file = 'p{}_halo_shards.txt'
    csr_indices_file = 'csr_indices{}.txt'
    csr_shard_indices_file = 'csr_shards{}.txt'
    csr_indptrs_file = 'csr_indptr{}.txt'
    partition_book = osp.join(path, 'partition_book.txt')

    def _dir(filename):
        return osp.join(path, filename.format(shard_id))

    return graph_engine.Graph(
        shard_id, _dir(ids_file), _dir(shards_file), _dir(rows_file), _dir(cols_file), partition_book)

    

class GraphShard:
    """
    Front end wrapper for Graph.h
    """
    def __init__(self, path, shard_id):
        self.id = shard_id
        self.g = init_graph(path, shard_id)

    @property
    def num_core_nodes(self):
        return self.g.num_core_nodes()

    @property
    def cluster_ptr(self):
        return self.g.partition_book()

    def to_global(self, indices, shard_id=None):
        if shard_id is None:
            shard_id = self.id
        return indices + self.cluster_ptr[shard_id]

    def walk_one_step(self, src_nodes: Tensor) -> Tuple[Tensor, Dict[int, Tensor]]:
        """Sample one neighbor for each source node in current graph shard

        :param  src_nodes: Source node local ID tensor.
                Each value represents a core node in the current shard
                and must smaller than `self.num_core_nodes`
        :return:
            - nid: Target node local ID tensor.
              Note that the local IDs could belong to core nodes of local or remote shards.
            - shard_dict: For each pair, the key is a shard ID and the value is an index tensor of `nid`.
              shard_dict is used to assign each target node to a shard.
        """
        nid, shard_dict = self.g.sample_single_neighbor(src_nodes)
        return nid, shard_dict

