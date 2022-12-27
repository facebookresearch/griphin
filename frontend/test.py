from collections import defaultdict

import torch
import graph_engine

from frontend.graph import GraphShard


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
    gs = GraphShard('../engine/files', 0)
    print(gs.num_core_nodes)


def test3():
    d = torch.tensor([0, 1, 2, 4, 3, 5])
    # t = torch.tensor([0, 1, 4, 5, 2, 3])
    # d = d.index_select(0, t)
    print(d)


def test4():
    d = {2: 1, 1: 0}
    # d = defaultdict(list)
    # d[1].append(123)
    for k in sorted(d.items()):
        print(k)


if __name__ == '__main__':
    test1()

