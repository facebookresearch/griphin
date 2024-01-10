#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import os
import sys
import time
import types
from pathlib import Path

import torch
import graph_engine

from utils import get_root_path, get_data_path
from graph import GraphShard, VERTEX_ID_TYPE, GraphDataManager

import torch.multiprocessing as mp

import dgl.sparse as dglsp


def test1():
    p0_ids_file = '../engine/files/p0_ids.txt'
    p0_shards_file = '../engine/files/p0_halo_shards.txt'
    p0_rows_file = '../engine/files/p0_edge_sources.txt'
    p0_cols_file = '../engine/files/p0_edge_dests.txt'

    partition_book_file = '../engine/files/partition_book.txt'

    g = graph_engine.Graph(0, p0_ids_file, p0_shards_file, p0_rows_file, p0_cols_file, partition_book_file)
    print(g.num_core_nodes())
    print(g.partition_book())
    print(g.sample_single_neighbor(torch.arange(100, dtype=torch.int32)))


def test2():
    t1 = time.time()
    path = os.path.join(get_root_path(), 'engine/ogbn_small_csr_format')
    gs = GraphShard(path, 0)
    t2 = time.time()
    print(f'Graph loading time: {(t2-t1):.3f}')

    g = gs.g
    print('Num core nodes: ', g.num_core_nodes())
    print('Cluster ptr: ', g.partition_book())

    num_roots = 8192
    epochs = 1000
    roots = [torch.randperm(gs.num_core_nodes, dtype=VERTEX_ID_TYPE)[:num_roots] for _ in range(epochs)]

    t1 = time.perf_counter()
    for i in range(epochs):
        g.sample_single_neighbor2(roots[i])
    t2 = time.perf_counter()
    print(f'Overall Run Time: {(t2-t1):.3f}')


def test3():
    current_dir = Path(__file__)
    project_dir = [p for p in current_dir.parents if p.parts[-1] == 'graph_engine'][0]
    print(project_dir)


def test4():
    root1 = os.path.abspath(__file__)
    root2 = os.path.dirname(root1)
    root3 = os.path.dirname(root2)
    print(root1)
    print(root2)
    print(root3)


def test5():
    # graph_engine.omp_add()
    b = torch.empty(1, 20, 256, 256, dtype=torch.bool)
    print(sys.getsizeof(b.storage()))  # 1310776 (bytes)
    a = torch.empty(1, 20, 256, 256, dtype=torch.uint8)
    print(sys.getsizeof(a.storage()))  # 1310776 (bytes)


def test6():
    tik = time.perf_counter()
    for i in range(1000):
        # torch.full((8192, 15), -1)
        torch.empty(8192, 15)
    tok = time.perf_counter()
    print(f'Time: {tok-tik:.3f}')


def test7():
    # path = os.path.join(get_data_path(), 'ogbn_products_{}partitions'.format(4))
    path = os.path.join(get_data_path(), 'hz-ogbn-product-p{}'.format(4))
    t1 = time.time()
    gs = GraphShard(path, 0)
    # print(gs.cluster_ptr)
    t2 = time.time()
    print(f'Graph loading time: {(t2-t1):.3f}')
    print(gs.get_neighbor_infos(torch.tensor([1, 2], dtype=VERTEX_ID_TYPE)))


def load_graph(rank, *args):
    g = GraphShard(*args)
    print(rank, g.partition_book())
    time.sleep(1)

    if rank == 0:
        print(g.get_neighbor_infos(torch.tensor([1], dtype=VERTEX_ID_TYPE)))


def test8():
    path = os.path.join(get_data_path(), 'hz-ogbn-product-p{}-pt'.format(4))
    graph_data, indptr_size, indices_size = GraphDataManager.create_shared_mem_graph_data(0, path)

    with mp.Pool(10) as pool:
        futs = [pool.apply_async(load_graph, args=(i, 0, 4, indptr_size, indices_size)) for i in range(10)]
        results = [fut.get() for fut in futs]


def test9():
    root_dir = os.path.join(get_data_path(), 'hz-ogbn-product-p{}-pt'.format(4))

    edge_index = torch.load(os.path.join(root_dir, 'dgl_edge_index.pt'))
    edge_weights = torch.load(os.path.join(root_dir, 'dgl_edge_weights.pt'))
    local_id_mapping = torch.load(os.path.join(root_dir, 'local_id_mapping.pt'))
    shard_id_mapping = torch.load(os.path.join(root_dir, 'shard_id_mapping.pt'))

    A = dglsp.spmatrix(edge_index, edge_weights)
    D = dglsp.diag(A.sum(dim=1))
    P = (D ** -1) @ A
    P_t = P.t()

    def power_iter_ppr(P_w, target_id_, alpha_, epsilon_, max_iter):
        num_nodes = P_w.shape[0]
        s = torch.zeros(num_nodes)
        s[target_id_] = 1
        s = s.view(-1, 1)

        x = s.clone()
        for i in range(max_iter):
            x_last = x
            x = alpha_ * s + (1 - alpha_) * (P_w @ x)
            # check convergence, l1 norm
            if (abs(x - x_last)).sum() < num_nodes * epsilon_:
                print(f'power-iter      Iterations: {i}, NNZ: {(x.view(-1) > 0).sum()}')
                return x.view(-1)

        print(f'Failed to converge with tolerance({epsilon_}) and iter({max_iter})')
        return x.view(-1)

    global_id = ((local_id_mapping == 2) & (shard_id_mapping == 0)).nonzero(as_tuple=False).view(-1)
    print(global_id)

    ppr_val = power_iter_ppr(P_t, global_id, 0.462, 1e-10, 100)
    print(ppr_val)


def test10():
    graph_data = GraphDataManager.create_shared_mem_graph_data(0, 'data/ogbn-products-p4')[0]
    g = graph_engine.Graph(0, *graph_data)

    src = torch.arange(2).to(VERTEX_ID_TYPE)
    is_local = True

    def _get_neighbor_infos(self, vertex_ids):
        if is_local:
            return g.get_neighbor_infos_local(vertex_ids)
        else:
            return g.get_neighbor_infos_remote(vertex_ids)

    # graph_engine.Graph.get_neighbor_infos = _get_neighbor_infos

    # funcType = type(graph_engine.Graph.get_neighbor_infos)
    graph_engine.Graph.get_neighbor_infos = _get_neighbor_infos
    g.get_neighbor_infos2 = _get_neighbor_infos.__get__(g)
    # g.get_neighbor_infos2 = types.MethodType(_get_neighbor_infos, g)
    print(g.get_neighbor_infos)
    v = g.get_neighbor_infos(src)
    print(v)


if __name__ == '__main__':
    test10()

