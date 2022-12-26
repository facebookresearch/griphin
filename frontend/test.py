import torch
import graph_engine

if __name__ == '__main__':
    p0_ids_file = '../engine/files/p0_ids.txt'
    p0_shards_file = '../engine/files/p0_halo_shards.txt'
    p0_rows_file = '../engine/files/p0_edge_sources.txt'
    p0_cols_file = '../engine/files/p0_edge_dests.txt'

    g = graph_engine.Graph(0, p0_ids_file, p0_shards_file, p0_rows_file, p0_cols_file)
    print(g.num_core_nodes())
    print(g.sample_single_neighbor(torch.arange(100, dtype=torch.int32)))

