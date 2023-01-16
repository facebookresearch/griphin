#include <iostream>
#include <string>
#include <fstream> 
#include <ctime>
#include <stdint.h>
#include "Graph.cpp"
#include "EdgeProp.h"
#include "VertexProp.h"

int main(){
    char p0_ids_file[] = "ogbn_small_csr_format/p0_ids.txt";
    char p0_halo_shards_file[] = "ogbn_small_csr_format/p0_halo_shards.txt";
    char p0_csr_indices_file[] = "ogbn_small_csr_format/csr_indices0.txt";
    char p0_csr_shard_indices_file[] = "ogbn_small_csr_format/csr_shards0.txt";
    char p0_csr_indptrs_file[] = "ogbn_small_csr_format/csr_indptr0.txt";

    char p1_ids_file[] = "ogbn_small_csr_format/p1_ids.txt";
    char p1_halo_shards_file[] = "ogbn_small_csr_format/p1_halo_shards.txt";
    char p1_csr_indices_file[] = "ogbn_small_csr_format/csr_indices1.txt";
    char p1_csr_shard_indices_file[] = "ogbn_small_csr_format/csr_shards1.txt";
    char p1_csr_indptrs_file[] = "ogbn_small_csr_format/csr_indptr1.txt";

    char p2_ids_file[] = "ogbn_small_csr_format/p2_ids.txt";
    char p2_halo_shards_file[] = "ogbn_small_csr_format/p2_halo_shards.txt";
    char p2_csr_indices_file[] = "ogbn_small_csr_format/csr_indices2.txt";
    char p2_csr_shard_indices_file[] = "ogbn_small_csr_format/csr_shards2.txt";
    char p2_csr_indptrs_file[] = "ogbn_small_csr_format/csr_indptr2.txt";

    char p3_ids_file[] = "ogbn_small_csr_format/p3_ids.txt";
    char p3_halo_shards_file[] = "ogbn_small_csr_format/p3_halo_shards.txt";
    char p3_csr_indices_file[] = "ogbn_small_csr_format/csr_indices3.txt";
    char p3_csr_shard_indices_file[] = "ogbn_small_csr_format/csr_shards3.txt";
    char p3_csr_indptrs_file[] = "ogbn_small_csr_format/csr_indptr3.txt";
    
    char partition_book_file[] = "ogbn_small_csr_format/partition_book.txt";

    Graph<VertexProp, EdgeProp> shard0(0, p0_ids_file, p0_halo_shards_file, p0_csr_indices_file, p0_csr_shard_indices_file, p0_csr_indptrs_file, partition_book_file);
    Graph<VertexProp, EdgeProp> shard1(1, p1_ids_file, p1_halo_shards_file, p1_csr_indices_file, p1_csr_shard_indices_file, p1_csr_indptrs_file, partition_book_file);
    Graph<VertexProp, EdgeProp> shard2(2, p2_ids_file, p2_halo_shards_file, p2_csr_indices_file, p2_csr_shard_indices_file, p2_csr_indptrs_file, partition_book_file);
    Graph<VertexProp, EdgeProp> shard3(3, p3_ids_file, p3_halo_shards_file, p3_csr_indices_file, p3_csr_shard_indices_file, p3_csr_indptrs_file, partition_book_file);

    // float data[] = { 1, 2, 3,4, 5, 6 };
    // auto a = shard0.sampleSingleNeighbor(torch::from_blob(data, {6}));

    shard0.getNumOfCoreVertices();
    int x = shard0.getNumOfVertices();
    x = shard0.getNumOfHaloVertices();
    VertexProp v = shard0.findVertex(10);
    printf("%d - %d\n", v.getNodeId(), v.getShardId());
    printf("neighbor size - %d\n", v.getSize());
    printf("neighbor i - %d\n", v.getIndex(10));
    printf("neighbor ptr - %p\n", (void *) v.getIndicesPtr());
    printf("neighbor i - %d\n", v.getIndicesPtr()[10]);
    // std::vector<VertexType> neighbors = shard0.getNeighbors(10);
    // for(int i = 0; i < neighbors.size(); i ++){
    //     printf("%d - ", neighbors[i]);
    // }

    // printf("\n\n");

    // x = shard1.getNumOfVertices();
    // x = shard1.getNumOfHaloVertices();
    // v = shard1.findVertex(120);
    // neighbors = shard1.getNeighbors(120);
    // for(int i = 0; i < neighbors.size(); i ++){
    //     printf("%d - ", neighbors[i]);
    // }
    // printf("\n\n");

    // x = shard2.getNumOfVertices();
    // x = shard2.getNumOfHaloVertices();
    // v =shard2.findVertex(330);
    // neighbors = shard2.getNeighbors(330);
    // for(int i = 0; i < neighbors.size(); i ++){
    //     printf("%d - ", neighbors[i]);
    // }

    // printf("\n\n");

    // x = shard3.getNumOfVertices();
    // x = shard3.getNumOfHaloVertices();
    // v = shard3.findVertex(240);
    // neighbors = shard3.getNeighbors(240);
    // for(int i = 0; i < neighbors.size(); i ++){
    //     printf("%d - ", neighbors[i]);
    // }
    // printf("\n\n");

    return 0;
}