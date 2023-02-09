import dgl
import torch

from ogb.nodeproppred import DglNodePropPredDataset
from dgl.dataloading import GraphDataLoader
from dgl.data import CoraGraphDataset

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix

from tqdm import tqdm
import matplotlib.pyplot as plt
    
#############################    
#############################
# I copied from jupyter notebook, there are some error messages right now.
# I will fix after the meeting.
#############################
#############################

def get_ogbn_graph():
    dataset = DglNodePropPredDataset(name='ogbn-products')

    g, l = dataset[0]
    return g

def get_weighted_degrees(g_csr_format):
    print("Started weighted degree calculation.")
    
    weighted_degrees = []

    for i in tqdm(range(len(g_csr_format.indptr) - 1)):
        start = g_csr_format.indptr[i]
        end = g_csr_format.indptr[i+1]

        weighted = 0
        for j in range(start, end):
            u = g_csr_format.indices[j]
            weighted += g_csr_format.data[u]

        weighted_degrees.append(weighted)
        
    print("Done with weighted degree calculation.")
    
    return weighted_degrees


def get_neigbor_weighted_degrees(g_csr_format, weighted_degrees):
    neigbor_weighted_degrees = []

    for i in tqdm(range(len(g_csr_format.indptr) - 1)):
        start = g_csr_format.indptr[i]
        end = g_csr_format.indptr[i+1]

        for j in range(start, end):
            u = g_csr_format.indices[j]
            neigbor_weighted_degrees.append(weighted_degrees[u])
            
    return neigbor_weighted_degrees

def partition_graph(g, num_part=4):
    partitions = dgl.metis_partition(g, k=num_part, extra_cached_hops=1, reshuffle=False, balance_ntypes=None, balance_edges=False, mode='k-way')    
    
    return partitions

def get_partition_weighted_degrees(partitions, weighted_degrees, par_num=4):
    weighted_degrees = []
    
    for idx in range(par_num):
        weighted_degrees_p = []
        for i in tqdm(range(len(partitions[idx].ndata['_ID']))):
            node = partitions[idx].ndata['_ID'][i]
            weighted_degrees_p.append(weighted_degrees[node])
            
        weighted_degrees.append(weighted_degrees_p)
        
    return weighted_degrees

def get_partition_edge_weights(partitions, edge_weight, num_part=4):
    edge_weights = []
    for idx in range(num_part):
        
        edge_weights_p = []
        for i in tqdm(range(len(partitions[idx].edata['_ID']))):
            edge = partitions[idx].edata['_ID'][i]
            edge_weights_p.append(edge_weight[edge])
            
        edge_weights.append(edge_weights_p)
        
    return edge_weights

def get_partition_book(partitions, num_par=4):
    
    core_counts = []
    indices = [0]
    index_p = 0
    
    for idx in range(num_par):
        core_p = len(partitions[idx].ndata['part_id'][partitions[idx].ndata['part_id'] == idx]) 
        index_p += core_p
        core_counts.append(core_p)
        indices.append(index_p)
        
    return core_counts, indices

def partition_local_and_global_ids(partitions, core_counts, par_num = 4):
    local_partitions = []
    global_partitions = []
    
    for i in range(par_num):
        partition_local = partitions[i].nodes()[:core_counts[i]]
        partition_global = partitions[i].ndata['_ID'][:core_counts[i]]
        
        local_partitions.append(partition_local)
        global_partitions.append(partition_global)
        
    return local_partitions, global_partitions

def halo_node_local_id_extract(partitions, core_counts, local_partitions, global_partitions, par_num=4,):
    local_ids = []

    for idx in range(par_num):
        local_ids_p = []
        for i in tqdm(range(core_counts[idx], partitions[idx].num_nodes())):
            part_id = partitions[idx].ndata['part_id'][i]

            local_id = partitions[idx].nodes()[i]
            global_id = partitions[idx].ndata['_ID'][i]

            index = (global_partitions[part_id] == global_id).nonzero(as_tuple=True)[0]
            local_ids_p.append(local_partitions[part_id][index])
            
        local_ids.append(torch.tensor(local_ids_p))
        
    return local_ids

def get_core_local_ids(partitions, core_counts, num_par=4):
    core_local_ids = []
    
    for idx in range(num_par):
        core_local_ids.append(partitions[idx].nodes()[:core_counts[idx]])
        
    return core_local_ids

def get_new_local_ids(core_local_ids, halo_local_ids, par_num=4):
    new_local_ids = []
    
    for idx in range(par_num):
        new_local_ids.append(torch.cat((core_local_ids[idx], halo_local_ids[idx])).numpy())
    
    return new_local_ids

def get_halo_shards(partitions, core_counts, num_par=4):
    halo_shards = []
    
    for idx in range(num_par):
        halo_shards_p = partitions[idx].ndata['part_id'][core_counts[idx]:]
        halo_shards.append(halo_shards_p.numpy())
    
    return halo_shards

def get_shards(partitions, num_par=4):
    shards = []
    
    for idx in range(num_par):
        shards_p = partitions[idx].ndata['part_id']
        shards.append(shards_p)
        
    return shards

def get_partition_csr_formats(partitions, edge_weights, num_par=4):
        
    csr_formats = []
    
    for idx in range(num_par):
        src = partitions[idx].edges()[1]
        dst = partitions[idx].edges()[0]
    
        size = partitions[idx].num_nodes()
        csr_format = csr_matrix((edge_weights[idx], (src, dst)), shape=(size, size))

        csr_formats.append(csr_format)
        
    return csr_formats

def adjust_ids_for_indices_and_shards(csr_formats, new_local_ids, core_counts, halo_shards, num_par=4):
    csr_adjusted_indices = []
    csr_adjusted_shards = []

    for idx in range(num_par):
        csr_adjusted_indices_p = []
        csr_adjusted_shards_p = []
        for i in tqdm(range(len(csr_formats[idx].indices))):
            old_index = csr_formats[idx].indices[i]
            new_index = new_local_ids[idx][old_index]

            csr_adjusted_indices_p.append(new_index)

            if old_index < core_counts[idx]:
                csr_adjusted_shards_p.append(idx)
            else:
                csr_adjusted_shards_p.append(halo_shards[idx][old_index-core_counts[idx]])

    return csr_adjusted_indices, csr_adjusted_shards

def get_csr_indptrs(partition_csr_formats, num_par=4):
    csr_indptrs = []
    
    for idx in range(num_par):
        csr_indptrs.append(partition_csr_formats[idx].indptr)
        
    return csr_indptrs

def get_csr_indices(partition_csr_formats, num_par=4):
    csr_indices = []
    
    for idx in range(num_par):
        csr_indices.append(partition_csr_formats[idx].indices)
        
    return csr_indices

def get_partition_edge_weights_from_csr(partition_csr_formats, num_par=4):
    partition_edge_weights = []
    
    for idx in range(num_par):
        partition_edge_weights.append(partition_csr_formats[idx].data)

    return partition_edge_weights

def save_np_to_txt(np_file, file_type="ids", dtype='%i', num_par=4):
    
    for idx in range(num_par):
        fil_dir = "ogbn_products_{}partitions/p{}_{}.txt".format(num_par, idx, file_type)
        np.savetxt(fil_dir, np_file[idx], delimiter='\n', fmt=dtype)

def save_to_txt(array, file_type="csr_indices", dtype='%d\n', num_par=4):
    
    for idx in range(num_par):
        fil_dir = "ogbn_products_{}partitions/p{}_{}.txt".format(num_par, idx, file_type)
        
        with open(fil_dir, 'w') as fp:
            for item in tqdm(array[idx]):
                fp.write(dtype % item)



def main():
    p = 4
    
    g = get_ogbn_graph()
    
    edge_weight = torch.rand(g.num_edges())
    g.edata['w'] = edge_weight
    
    g_csr_format = csr_matrix((edge_weight, (g.edges()[0], g.edges()[1])), shape=(g.num_nodes(), g.num_nodes()))
    
    weighted_degrees = get_weighted_degrees(g_csr_format)
    
    neigbor_weighted_degrees = get_neigbor_weighted_degrees(g_csr_format, weighted_degrees)
    
    partitions = partition_graph(g, p)
    
    partition_weighted_degrees = get_partition_weighted_degrees(partitions, weighted_degrees, p)
    
    partition_edge_weights = get_partition_edge_weights(partitions, edge_weight, p)
    
    core_counts, indices = get_partition_book(partitions)
    
    local_partitions, global_partitions = partition_local_and_global_ids(partitions, core_counts, p)
    
    halo_local_ids = halo_node_local_id_extract(partitions, core_counts, local_partitions, global_partitions, p)

    core_local_ids = get_core_local_ids(partitions, core_counts, p)
    
    new_local_ids = get_new_local_ids(core_local_ids, halo_local_ids, p)
    
    halo_shards = get_halo_shards(partitions, core_counts, p)
    
    shards = get_shards(partitions, p)
    
    partition_csr_formats = get_partition_csr_formats(partitions, partition_edge_weights, p)
    
    csr_adjusted_indices, csr_adjusted_shards = adjust_ids_for_indices_and_shards(partition_csr_formats, new_local_ids, core_counts, halo_shards, p)
    
    csr_indptrs = get_csr_indptrs(partition_csr_formats, p)
    
    #csr_indices = get_csr_indices(partition_csr_formats, p)
    
    partition_edge_weights = get_partition_edge_weights_from_csr(partition_csr_formats)
    
    save_np_to_txt(new_local_ids, file_type="p{}_ids", dtype='%i', num_par=p)
    
    save_to_txt(csr_adjusted_indices, file_type="csr_indices", dtype='%d\n', num_par=p)
    
    save_to_txt(csr_adjusted_shards, file_type="csr_shards", dtype='%d\n', num_par=p)
    
    save_np_to_txt(csr_indptrs, file_type="csr_indptr", dtype='%i', num_par=p)
    
    save_np_to_txt(halo_shards, file_type="halo_shards", dtype='%i', num_par=p)
    
    np.savetxt('ogbn_products_4partitions/partition_book.txt', np.array(indices), delimiter='\n', fmt='%i')
    
    save_to_txt(partition_edge_weights, file_type="csr_edge_weights", dtype='%f\n', num_par=p)
    
    save_to_txt(partition_weighted_degrees, file_type="weighted_degrees", dtype='%f\n', num_par=p)
    
    
if __name__ == "__main__":
    main()