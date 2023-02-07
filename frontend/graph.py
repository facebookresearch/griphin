from typing import Tuple, Dict, List

import torch
import graph_engine
import os.path as osp

from torch import Tensor

VERTEX_ID_TYPE = torch.int32
SHARD_ID_TYPE = torch.int8


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
        shard_id, _dir(ids_file), _dir(shards_file), _dir(csr_indices_file), _dir(csr_shard_indices_file), _dir(csr_indptrs_file), partition_book)


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


    # --- for test only ---
    def get_dict_tensor(self, root_nodes):
        rand = torch.ones_like(root_nodes)
        return rand, {1: rand[:10], 2: rand[10: 40], 3: rand[40: 100], 4: rand[100:]}


class SSPPR:
    """
        Front end wrapper for SSPPR.h
    """
    def __init__(self, num_nodes, target_id, shard_id, cluster_ptr, alpha, epsilon):
        self.cluster_ptr = cluster_ptr
        self.alpha = alpha
        self.epsilon = epsilon

        self.p = torch.zeros(num_nodes)
        self.r = torch.zeros(num_nodes)
        self.r[target_id] = 1

        self.key_str = '{}_{}'
        self.activated_node_dict = {self.key_str.format(target_id, shard_id): (target_id, shard_id)}

    def pop_activated_nodes(self) -> Tuple[Tensor, Tensor]:
        # TODO: is there any efficient implementation?
        node_ids, shard_ids = [], []
        for _, val in self.activated_node_dict.items():
            node_ids.append(val[0])
            shard_ids.append(val[1])
        self.activated_node_dict.clear()
        return torch.tensor(node_ids), torch.tensor(shard_ids)

    def push(self, neighbor_infos: List, v_ids: Tensor, v_shard_ids: Tensor):
        for u_info, v_id, v_shard_id in zip(neighbor_infos, v_ids.tolist(), v_shard_ids.tolist()):
            u_ids, u_shard_ids, u_weights, u_degrees = u_info
            global_v_id = v_id + self.cluster_ptr[v_shard_id]
            self.p[global_v_id] += self.alpha * self.r[global_v_id]
            u_vals = (1 - self.alpha) * self.r[global_v_id] * u_weights / u_weights.sum()
            self.r[global_v_id] = 0

            for val, u_id, u_shard_id, u_degree in zip(u_vals, u_ids, u_shard_ids, u_degrees):
                global_u_id = u_id + self.cluster_ptr[u_shard_id]
                # update neighbor node
                self.r[global_u_id] += val
                # check threshold
                if self.r[global_u_id] >= self.alpha * u_degree:
                    u_key = self.key_str.format(u_id, u_shard_id)
                    if u_key not in self.activated_node_dict.keys():
                        self.activated_node_dict[u_key] = (u_id, u_shard_id)

