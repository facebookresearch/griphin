import dgl
import torch
from ogb.nodeproppred import DglNodePropPredDataset
from scipy.sparse import csr_matrix
from tqdm import tqdm


dataset = DglNodePropPredDataset(name='ogbn-products', root='/data/gangda/dgl')
g, labels = dataset[0]

edge_weight = torch.rand(g.num_edges())
g_csr_format = csr_matrix((edge_weight, (g.edges()[0], g.edges()[1])), shape=(g.num_nodes(), g.num_nodes()))

weighted_degrees = []

for i in tqdm(range(len(g_csr_format.indptr) - 1)):
    start = g_csr_format.indptr[i]
    end = g_csr_format.indptr[i + 1]

    weighted = 0
    for j in range(start, end):
        u = g_csr_format.indices[j]
        weighted += g_csr_format.data[u]

    weighted_degrees.append(weighted)

partitions = dgl.metis_partition(g, k=4, extra_cached_hops=1, reshuffle=False, balance_edges=False, mode='k-way')

weighted_degrees = torch.tensor(weighted_degrees)
weighted_degrees_p0 = weighted_degrees[partitions[0].ndata['_ID']]
weighted_degrees_p1 = weighted_degrees[partitions[1].ndata['_ID']]
weighted_degrees_p2 = weighted_degrees[partitions[2].ndata['_ID']]
weighted_degrees_p3 = weighted_degrees[partitions[3].ndata['_ID']]

edge_weights_p0 = edge_weight[partitions[0].edata['_ID']]
edge_weights_p1 = edge_weight[partitions[1].edata['_ID']]
edge_weights_p2 = edge_weight[partitions[2].edata['_ID']]
edge_weights_p3 = edge_weight[partitions[3].edata['_ID']]

part0_src, part0_dst = partitions[0].edges()[1], partitions[0].edges()[0]
part1_src, part1_dst = partitions[1].edges()[1], partitions[1].edges()[0]
part2_src, part2_dst = partitions[2].edges()[1], partitions[2].edges()[0]
part3_src, part3_dst = partitions[3].edges()[1], partitions[3].edges()[0]

part0_csr_format = csr_matrix((edge_weights_p0, (part0_src, part0_dst)),
                              shape=(partitions[0].num_nodes(), partitions[0].num_nodes()))
part1_csr_format = csr_matrix((edge_weights_p1, (part1_src, part1_dst)),
                              shape=(partitions[1].num_nodes(), partitions[1].num_nodes()))
part2_csr_format = csr_matrix((edge_weights_p2, (part2_src, part2_dst)),
                              shape=(partitions[2].num_nodes(), partitions[2].num_nodes()))
part3_csr_format = csr_matrix((edge_weights_p3, (part3_src, part3_dst)),
                              shape=(partitions[3].num_nodes(), partitions[3].num_nodes()))

p0_indices = torch.from_numpy(part0_csr_format.indices).to(torch.long)
p1_indices = torch.from_numpy(part1_csr_format.indices).to(torch.long)
p2_indices = torch.from_numpy(part2_csr_format.indices).to(torch.long)
p3_indices = torch.from_numpy(part3_csr_format.indices).to(torch.long)

print(p0_indices.max(), part0_csr_format.indptr.size)

csr_weighted_degrees_p0 = weighted_degrees[p0_indices].tolist()
csr_weighted_degrees_p1 = weighted_degrees[p1_indices].tolist()
csr_weighted_degrees_p2 = weighted_degrees[p2_indices].tolist()
csr_weighted_degrees_p3 = weighted_degrees[p3_indices].tolist()


part0_edge_weights = part0_csr_format.data
with open('data/ogbn_products_4partitions/csr_edge_weights_p0.txt', 'w') as fp:
    for item in tqdm(part0_edge_weights):
        fp.write("%f\n" % item)

part1_edge_weights = part1_csr_format.data
with open('data/ogbn_products_4partitions/csr_edge_weights_p1.txt', 'w') as fp:
    for item in tqdm(part1_edge_weights):
        fp.write("%f\n" % item)

part2_edge_weights = part2_csr_format.data
with open('data/ogbn_products_4partitions/csr_edge_weights_p2.txt', 'w') as fp:
    for item in tqdm(part2_edge_weights):
        fp.write("%f\n" % item)

part3_edge_weights = part3_csr_format.data
with open('data/ogbn_products_4partitions/csr_edge_weights_p3.txt', 'w') as fp:
    for item in tqdm(part3_edge_weights):
        fp.write("%f\n" % item)

with open('data/ogbn_products_4partitions/csr_weighted_degrees_p0.txt', 'w') as fp:
    for item in tqdm(csr_weighted_degrees_p0):
        fp.write("%f\n" % item)

with open('data/ogbn_products_4partitions/csr_weighted_degrees_p1.txt', 'w') as fp:
    for item in tqdm(csr_weighted_degrees_p1):
        fp.write("%f\n" % item)

with open('data/ogbn_products_4partitions/csr_weighted_degrees_p2.txt', 'w') as fp:
    for item in tqdm(csr_weighted_degrees_p2):
        fp.write("%f\n" % item)

with open('data/ogbn_products_4partitions/csr_weighted_degrees_p3.txt', 'w') as fp:
    for item in tqdm(csr_weighted_degrees_p3):
        fp.write("%f\n" % item)
