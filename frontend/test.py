import os
from pathlib import Path

import torch
import graph_engine

from graph import GraphShard


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
    gs = GraphShard('../engine/ogbn_files_txt_small', 0)
    g = gs.g
    print(g.num_core_nodes())
    print(g.partition_book())
    src = torch.arange(g.num_core_nodes()-100, g.num_core_nodes(), dtype=torch.int32)
    print(src)
    print(g.sample_single_neighbor(src))
    print(gs.num_core_nodes)


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


if __name__ == '__main__':
    test3()

