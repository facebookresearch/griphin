import os
from pathlib import Path
import time

import torch
import graph_engine

from graph import GraphShard


def test1():
    p0_ids_file = "../engine/ogbn_csr_format/p0_ids.txt";
    p0_halo_shards_file = "../engine/ogbn_csr_format/p0_halo_shards.txt";
    p0_csr_indices_file = "../engine/ogbn_csr_format/csr_indices0.txt";
    p0_csr_shard_indices_file = "../engine/ogbn_csr_format/csr_shards0.txt";
    p0_csr_indptrs_file = "../engine/ogbn_csr_format/csr_indptr0.txt";
    partition_book_file = '../engine/files/partition_book.txt'

    g = graph_engine.Graph(0, p0_ids_file, p0_halo_shards_file, p0_csr_indices_file, p0_csr_shard_indices_file, p0_csr_indptrs_file, partition_book_file)
    print(g.num_core_nodes())
    print(g.partition_book())
    print(g.sample_single_neighbor(torch.arange(100, dtype=torch.int32)))


def test2():
    tik = time.time()
    gs = GraphShard('../engine/ogbn_csr_format', 0)
    tok = time.time()

    print("time: ", tok-tik)

    g = gs.g
    print(g.num_core_nodes())
    print(g.partition_book())
    src = torch.arange(g.num_core_nodes()-150, g.num_core_nodes() - 50, dtype=torch.int32)
    print(src)
    print(gs.num_core_nodes)
    print(g.sample_single_neighbor(src))


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
    test2()

