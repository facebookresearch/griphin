#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import time
from collections import defaultdict
from typing import Tuple, Dict, List, Union

import torch
import graph_engine
import os.path as osp

from torch import Tensor
from torch.futures import Future

from dgl.utils.shared_mem import create_shared_mem_array, get_shared_mem_array

from torch.distributed import rpc
from torch.distributed.rpc import remote, RRef


# Data types in frontend/graph.py, data_generation/gen_engine_data.py,
# and engine/global.h should be consistent
VERTEX_ID_TYPE = torch.int32
EDGE_ID_TYPE = torch.int64
SHARD_ID_TYPE = torch.int8
WEIGHT_TYPE = torch.float32


class DistGraphStorage:
    def __init__(self, rrefs, machine_rank):
        self.rank = rpc.get_worker_info().id
        self.machine_rank = machine_rank
        self.rrefs = rrefs

    def get_neighbor_infos(self,
                           dst_machine_rank: SHARD_ID_TYPE,
                           vertex_ids: Tensor
                           ) -> Union[List[graph_engine.VertexProp], Future]:
        if dst_machine_rank == self.machine_rank:
            # return self.rrefs[self.rank].to_here().get_neighbor_infos(vertex_ids)
            return self.rrefs[self.rank].to_here().get_neighbor_infos_local(vertex_ids)
        else:
            # return self.rrefs[dst_machine_rank].rpc_async().get_neighbor_infos(vertex_ids)
            return self.rrefs[dst_machine_rank].rpc_async().get_neighbor_infos_remote(vertex_ids)


class SubGraphDataManager:
    def __init__(self, shard_id, *args):
        self.shard_id = shard_id
        self.graph_data = GraphDataManager.get_shared_mem_graph_data(shard_id, *args)

    def get_graph_rref(self):
        return RRef(graph_engine.Graph(self.shard_id, *self.graph_data))


class GraphDataManager:
    # File Names
    indptr_file = 'p{}_indptr.pt'
    indices_file = 'p{}_indices_node_id.pt'
    indices_shard_id_file = 'p{}_indices_shard_id.pt'
    indices_edge_weight_file = 'p{}_indices_edge_weight.pt'
    indices_weighted_degree_file = 'p{}_indices_weighted_degree.pt'
    partition_file = 'partition_book.pt'

    def __init__(self, machine_rank, root_dir, worker_name, num_machines, num_processes):
        self.machine_rank = machine_rank
        self.num_machines = num_machines
        self.num_processes = num_processes
        self.worker_name = worker_name

        res = self.create_shared_mem_graph_data(machine_rank, root_dir)
        self.graph_data = res[0]
        self.indptr_size = res[1]
        self.indices_size = res[2]

        # init data reference on sub processes
        self.sub_manager_rrefs = []
        for rank_ in range(self.subp_start_rank, self.subp_start_rank + self.num_processes):
            self.sub_manager_rrefs.append(
                remote(self.__get_worker_info(rank_), SubGraphDataManager, args=(
                    self.machine_rank, self.num_machines, self.indptr_size, self.indices_size
                )))

    @property
    def subp_start_rank(self):
        """ rank of the first sub process """
        return self.num_machines + self.machine_rank * self.num_processes

    def get_graph_rrefs(self):
        main_rref = RRef(graph_engine.Graph(self.machine_rank, *self.graph_data))
        sub_rrefs = [m.rpc_sync().get_graph_rref() for m in self.sub_manager_rrefs]
        return main_rref, sub_rrefs

    def __get_worker_info(self, rank):
        return rpc.get_worker_info(self.worker_name.format(rank))

    @staticmethod
    def create_shared_mem_graph_data(shard_id, root_dir):
        clazz = GraphDataManager

        def load_data(file_name, dtype, format_required=True):
            name = file_name.format(shard_id) if format_required else file_name
            data = torch.load(osp.join(root_dir, name)).to(dtype)
            # shared_data = data.share_memory_()
            shared_data = create_shared_mem_array(name, data.size(), dtype=dtype)
            shared_data[:] = data
            return shared_data

        graph_data = [
            load_data(clazz.indptr_file, EDGE_ID_TYPE),
            load_data(clazz.indices_file, VERTEX_ID_TYPE),
            load_data(clazz.indices_shard_id_file, SHARD_ID_TYPE),
            load_data(clazz.indices_edge_weight_file, WEIGHT_TYPE),
            load_data(clazz.indices_weighted_degree_file, WEIGHT_TYPE),
            load_data(clazz.partition_file, VERTEX_ID_TYPE, format_required=False),
        ]
        return graph_data, graph_data[0].size(), graph_data[1].size()

    @staticmethod
    def get_shared_mem_graph_data(shard_id, num_machines, indptr_size, indices_size):
        clazz = GraphDataManager

        def _key(file_name):
            return file_name.format(shard_id)

        graph_data = [
            get_shared_mem_array(_key(clazz.indptr_file), indptr_size, EDGE_ID_TYPE),
            get_shared_mem_array(_key(clazz.indices_file), indices_size, VERTEX_ID_TYPE),
            get_shared_mem_array(_key(clazz.indices_shard_id_file), indices_size, SHARD_ID_TYPE),
            get_shared_mem_array(_key(clazz.indices_edge_weight_file), indices_size, WEIGHT_TYPE),
            get_shared_mem_array(_key(clazz.indices_weighted_degree_file), indices_size, WEIGHT_TYPE),
            get_shared_mem_array(clazz.partition_file, (num_machines + 1,), VERTEX_ID_TYPE),
        ]

        return graph_data


class GraphShard:
    """
        Deprecated. Frontend wrapper for Graph.h
    """

    def __init__(self, shard_id, path):
        self.id = shard_id
        graph_data, indptr_size, indices_size = GraphDataManager.create_shared_mem_graph_data(shard_id, path)
        self.graph_tensor_arr = graph_data

        tik = time.time()
        self.g = graph_engine.Graph(shard_id, *self.graph_tensor_arr)
        tok = time.time()
        print(f'Graph {shard_id} loading time: {(tok - tik):.2f}s')

    @property
    def num_core_nodes(self):
        return self.g.num_core_nodes()

    @property
    def cluster_ptr(self):
        return self.g.partition_book()

    def partition_book(self):
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

    def get_neighbor_infos(self, src_nodes: Tensor) -> List[Tuple[Tensor, Tensor, Tensor, Tensor]]:
        """
        :param src_nodes:
        :return: List of (VertexIDs, ShardIDs, EdgeWeights, Degrees)
        """
        return self.g.get_neighbor_infos(src_nodes)


class PPR:
    """
        Deprecated. Frontend wrapper for PPR.h
    """

    def __init__(self, target_id, shard_id, alpha, epsilon, num_threads):
        self.ppr = graph_engine.PPR(target_id, shard_id, alpha, epsilon, num_threads)

    def pop_activated_nodes(self) -> Tuple[Tensor, Tensor]:
        """
        after this operation, activated nodes is reset 

        :return:
            - node_ids: node ids of the activated nodes are returned
            - shards_ids: shard ids of the activated nodes are returned
        """
        node_ids, shard_ids = self.ppr.pop_activated_nodes()
        return node_ids, shard_ids

    def push(self, neighbor_infos: List, v_ids: Tensor, v_shard_ids: Tensor):
        """
        :param neighbor_infos: a list of list holding neighbor information
            of v_ids [indices, shards, edge_weight, weighted_degree]
        :param v_ids: ids of the nodes to be pushed
        :param v_shard_ids: corresponding shard ids of v_ids
        """
        self.ppr.push(neighbor_infos, v_ids, v_shard_ids)

    def get_p(self):
        """
        :return: 
            -self.ppr.get_p() is the current p values of PPR class
        """
        return self.ppr.get_p()


class SSPPR:
    """
        Python version of PPR.h
    """

    def __init__(self, target_id, shard_id, alpha, epsilon):
        self.alpha = alpha
        self.epsilon = epsilon

        self.p = defaultdict(float)
        self.r = defaultdict(float)
        self.r[self.key_str(target_id, shard_id)] = 1

        self.activated_nodes = {self.key_str(target_id, shard_id): (target_id, shard_id)}

    @staticmethod
    def key_str(node_id, shard_id):
        return '{}_{}'.format(node_id, shard_id)

    def pop_activated_nodes(self) -> Tuple[Tensor, Tensor]:

        node_ids, shard_ids = [], []
        for nid, sid in self.activated_nodes.values():
            node_ids.append(nid)
            shard_ids.append(sid)
        self.activated_nodes.clear()
        return torch.tensor(node_ids, dtype=VERTEX_ID_TYPE), torch.tensor(shard_ids)

    def push(self, neighbor_infos: List, v_ids: Tensor, v_shard_ids: Tensor):
        for u_info, v_id, v_shard_id in zip(neighbor_infos, v_ids, v_shard_ids):
            u_ids, u_shard_ids, u_weights, u_degrees = u_info

            v_key = self.key_str(v_id, v_shard_id)
            self.p[v_key] += self.alpha * self.r[v_key]
            u_vals = (1 - self.alpha) * self.r[v_key] * u_weights / u_weights.sum()
            self.r[v_key] = 0
            self.activated_nodes.pop(v_key, None)

            for val, u_id, u_shard_id, u_degree in zip(u_vals, u_ids, u_shard_ids, u_degrees):
                u_key = self.key_str(u_id, u_shard_id)
                # update neighbor node
                self.r[u_key] += val
                # check threshold
                if self.r[u_key] >= self.epsilon * u_degree:
                    if u_key not in self.activated_nodes.keys():
                        self.activated_nodes[u_key] = (u_id, u_shard_id)
