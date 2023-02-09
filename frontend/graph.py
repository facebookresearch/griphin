from collections import OrderedDict, defaultdict
from typing import Tuple, Dict, List

import torch
import graph_engine
import os.path as osp

from torch import Tensor

VERTEX_ID_TYPE = torch.int32
SHARD_ID_TYPE = torch.int8


def init_graph(path, shard_id):
    return graph_engine.Graph(shard_id, path)


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

    def walk_one_step2(self, src_nodes: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        local_nid, global_nid, shard_id = self.g.sample_single_neighbor2(src_nodes)
        return local_nid, global_nid, shard_id

    def batch_fetch_neighbors(self, src_nodes: Tensor) -> List[Tensor]:
        return self.g.get_neighbor_lists(src_nodes)

    def batch_fetch_neighbor_infos(self, src_nodes: Tensor) -> List[Tuple[Tensor, Tensor, Tensor, Tensor]]:
        """
        :param src_nodes:
        :return: List of (VertexIDs, ShardIDs, EdgeWeights, Degrees)
        """
        return self.g.get_neighbor_infos(src_nodes)


def key_str(node_id, shard_id):
    return '{}_{}'.format(node_id, shard_id)


class SSPPR:
    """
        Front end wrapper for SSPPR.h
    """
    def __init__(self, target_id, shard_id, alpha, epsilon):
        self.alpha = alpha
        self.epsilon = epsilon

        self.p = defaultdict(float)
        self.r = defaultdict(float)
        self.r[key_str(target_id, shard_id)] = 1

        # self.activated_nodes = OrderedDict({self.key_str.format(target_id, shard_id): (target_id, shard_id)})
        self.activated_nodes = {key_str(target_id, shard_id): (target_id, shard_id)}
        # self.next_node_ids = [target_id]
        # self.next_shard_ids = [shard_id]

    def pop_activated_nodes(self) -> Tuple[Tensor, Tensor]:
        # node_ids, shard_ids = self.next_node_ids, self.next_shard_ids
        # self.next_node_ids, self.next_shard_ids = [], []
        # self.activated_nodes.clear()
        # return torch.tensor(node_ids), torch.tensor(shard_ids)

        node_ids, shard_ids = [], []
        for nid, sid in self.activated_nodes.values():
            node_ids.append(nid)
            shard_ids.append(sid)
        self.activated_nodes.clear()
        return torch.tensor(node_ids), torch.tensor(shard_ids)

    def push(self, neighbor_infos: List, v_ids: Tensor, v_shard_ids: Tensor):
        for u_info, v_id, v_shard_id in zip(neighbor_infos, v_ids, v_shard_ids):
            u_ids, u_shard_ids, u_weights, u_degrees = u_info

            v_key = key_str(v_id, v_shard_id)
            self.p[v_key] += self.alpha * self.r[v_key]
            u_vals = (1 - self.alpha) * self.r[v_key] * u_weights / u_weights.sum()
            self.r[v_key] = 0
            self.activated_nodes.pop(v_key, None)

            for val, u_id, u_shard_id, u_degree in zip(u_vals, u_ids, u_shard_ids, u_degrees):
                u_key = key_str(u_id, u_shard_id)
                # update neighbor node
                self.r[u_key] += val
                # check threshold
                if self.r[u_key] >= self.epsilon * u_degree:
                    if u_key not in self.activated_nodes.keys():
                        self.activated_nodes[u_key] = (u_id, u_shard_id)
                        # self.next_node_ids.append(u_id)
                        # self.next_shard_ids.append(u_shard_id)

