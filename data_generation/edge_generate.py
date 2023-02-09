import dgl
import torch

from ogb.nodeproppred import DglNodePropPredDataset
from dgl.dataloading import GraphDataLoader

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix

from tqdm import tqdm
import matplotlib.pyplot as plt


def edge_weights_generate():
    dataset = DglNodePropPredDataset(name='ogbn-products', root='/data/gangda/dgl')
    g, labels = dataset[0]
    
    edge_weight = torch.rand(g.num_edges())

    g_csr_format = csr_matrix((edge_weight, (g.edges()[0], g.edges()[1])), shape=(g.num_nodes(), g.num_nodes()))

    weighted_degrees = []

    for i in tqdm(range(len(g_csr_format.indptr) - 1)):
        start = g_csr_format.indptr[i]
        end = g_csr_format.indptr[i+1]

        weighted = 0
        for j in range(start, end):
            u = g_csr_format.indices[j]
            weighted += g_csr_format.data[u]

        weighted_degrees.append(weighted)

    neigbor_weighted_degrees = []

    for i in tqdm(range(len(g_csr_format.indptr) - 1)):
        start = g_csr_format.indptr[i]
        end = g_csr_format.indptr[i+1]

        for j in range(start, end):
            u = g_csr_format.indices[j]
            neigbor_weighted_degrees.append(weighted_degrees[u])
            
    partitions = dgl.metis_partition(g, k=4, extra_cached_hops=1, reshuffle=False, balance_edges=False, mode='k-way')    
    
    weighted_degrees_p0 = []
    for i in tqdm(range(len(partitions[0].ndata['_ID']))):
        node = partitions[0].ndata['_ID'][i]
        weighted_degrees_p0.append(weighted_degrees[node])

    weighted_degrees_p1 = []
    for i in tqdm(range(len(partitions[1].ndata['_ID']))):
        node = partitions[1].ndata['_ID'][i]
        weighted_degrees_p1.append(weighted_degrees[node])

    weighted_degrees_p2 = []
    for i in tqdm(range(len(partitions[2].ndata['_ID']))):
        node = partitions[2].ndata['_ID'][i]
        weighted_degrees_p2.append(weighted_degrees[node])

    weighted_degrees_p3 = []
    for i in tqdm(range(len(partitions[3].ndata['_ID']))):
        node = partitions[3].ndata['_ID'][i]
        weighted_degrees_p3.append(weighted_degrees[node])
        
        
    edge_weights_p0 = []
    for i in tqdm(range(len(partitions[0].edata['_ID']))):
        edge = partitions[0].edata['_ID'][i]
        edge_weights_p0.append(edge_weight[edge])

    edge_weights_p1 = []
    for i in tqdm(range(len(partitions[1].edata['_ID']))):
        edge = partitions[1].edata['_ID'][i]
        edge_weights_p1.append(edge_weight[edge])

    edge_weights_p2 = []
    for i in tqdm(range(len(partitions[2].edata['_ID']))):
        edge = partitions[2].edata['_ID'][i]
        edge_weights_p2.append(edge_weight[edge])

    edge_weights_p3 = []
    for i in tqdm(range(len(partitions[3].edata['_ID']))):
        edge = partitions[3].edata['_ID'][i]
        edge_weights_p3.append(edge_weight[edge])
        
    part0_src, part0_dst = partitions[0].edges()[1], partitions[0].edges()[0]

    part1_src, part1_dst = partitions[1].edges()[1], partitions[1].edges()[0]

    part2_src, part2_dst = partitions[2].edges()[1], partitions[2].edges()[0]

    part3_src, part3_dst = partitions[3].edges()[1], partitions[3].edges()[0]
    
     
    part0_csr_format = csr_matrix((edge_weights_p0, (part0_src, part0_dst)), shape = (partitions[0].num_nodes(),  partitions[0].num_nodes()))
    part1_csr_format = csr_matrix((edge_weights_p1, (part1_src, part1_dst)), shape = (partitions[1].num_nodes(), partitions[1].num_nodes()))
    part2_csr_format = csr_matrix((edge_weights_p2, (part2_src, part2_dst)), shape = (partitions[2].num_nodes(), partitions[2].num_nodes()))
    part3_csr_format = csr_matrix((edge_weights_p3, (part3_src, part3_dst)), shape = (partitions[3].num_nodes(), partitions[3].num_nodes()))
    
    part0_edge_weights = part0_csr_format.data
    with open('ogbn_products_4partitions/csr_edge_weights_p0.txt', 'w') as fp:
        for item in tqdm(part0_edge_weights):
            fp.write("%f\n" % item)

    part1_edge_weights = part1_csr_format.data
    with open('ogbn_products_4partitions/csr_edge_weights_p1.txt', 'w') as fp:
        for item in tqdm(part1_edge_weights):
            fp.write("%f\n" % item)

    part2_edge_weights = part2_csr_format.data
    with open('ogbn_products_4partitions/csr_edge_weights_p2.txt', 'w') as fp:
        for item in tqdm(part2_edge_weights):
            fp.write("%f\n" % item)

    part3_edge_weights = part3_csr_format.data
    with open('ogbn_products_4partitions/csr_edge_weights_p3.txt', 'w') as fp:
        for item in tqdm(part3_edge_weights):
            fp.write("%f\n" % item)



    with open('ogbn_products_4partitions/weighted_degrees_p0.txt', 'w') as fp:
        for item in tqdm(weighted_degrees_p0):
            fp.write("%f\n" % item)

    with open('ogbn_products_4partitions/weighted_degrees_p1.txt', 'w') as fp:
        for item in tqdm(weighted_degrees_p1):
            fp.write("%f\n" % item)

    with open('ogbn_products_4partitions/weighted_degrees_p2.txt', 'w') as fp:
        for item in tqdm(weighted_degrees_p2):
            fp.write("%f\n" % item)

    with open('ogbn_products_4partitions/weighted_degrees_p3.txt', 'w') as fp:
        for item in tqdm(weighted_degrees_p3):
            fp.write("%f\n" % item)
            
            
if __name__ == "__main__":
    edge_weights_generate()




















