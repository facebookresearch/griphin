import os
import sys
import time
from pathlib import Path

import torch
import graph_engine

from utils import get_root_path
from graph import GraphShard, VERTEX_ID_TYPE


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
    tik_ = time.time()
    path = os.path.join(get_root_path(), 'engine/ogbn_files_txt_small')
    gs = GraphShard(path, 0)

    tok = time.time()
    print(tok - tik_)

    # g = gs.g
    # # print(g.num_core_nodes())
    # # print(g.partition_book())
    #
    # src = torch.arange(g.num_core_nodes()-8192, g.num_core_nodes(), dtype=VERTEX_ID_TYPE)
    # # print(g.sample_single_neighbor2(src))
    #
    # num_roots = 8192
    # epochs = 1000
    # roots = [torch.randperm(gs.num_core_nodes, dtype=VERTEX_ID_TYPE)[:num_roots] for _ in range(epochs)]
    #
    # t1 = time.perf_counter()
    # for i in range(epochs):
    #     g.sample_single_neighbor2(roots[i])
    # t2 = time.perf_counter()
    # print(f'Overall Run Time: {(t2-t1):.3f}')


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


if __name__ == '__main__':
    test2()

