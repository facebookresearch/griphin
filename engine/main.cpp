#include <iostream>
#include <string>
#include <fstream> 
#include <ctime>
#include <stdint.h>
#include "Graph.cpp"
#include "EdgeProp.h"
#include "VertexProp.h"

int main(){
    char p0_ids_file[] = "files/p0_ids.txt";
    char p0_shards_file[] = "files/p0_halo_shards.txt";
    char p0_rows_file[] = "files/p0_edge_sources.txt";
    char p0_cols_file[] = "files/p0_edge_dests.txt";

    char p1_ids_file[] = "files/p1_ids.txt";
    char p1_shards_file[] = "files/p1_halo_shards.txt";
    char p1_rows_file[] = "files/p1_edge_sources.txt";
    char p1_cols_file[] = "files/p1_edge_dests.txt";

    char p2_ids_file[] = "files/p2_ids.txt";
    char p2_shards_file[] = "files/p2_halo_shards.txt";
    char p2_rows_file[] = "files/p2_edge_sources.txt";
    char p2_cols_file[] = "files/p2_edge_dests.txt";

    char p3_ids_file[] = "files/p3_ids.txt";
    char p3_shards_file[] = "files/p3_halo_shards.txt";
    char p3_rows_file[] = "files/p3_edge_sources.txt";
    char p3_cols_file[] = "files/p3_edge_dests.txt";

    char partition_book_file[] = "files/p3_edge_dests.txt";

    Graph<VertexProp, EdgeProp> shard0(0, p0_ids_file, p0_shards_file, p0_rows_file, p0_cols_file, partition_book_file);
    Graph<VertexProp, EdgeProp> shard1(1, p1_ids_file, p1_shards_file, p1_rows_file, p1_cols_file, partition_book_file);
    Graph<VertexProp, EdgeProp> shard2(2, p2_ids_file, p2_shards_file, p2_rows_file, p2_cols_file, partition_book_file);
    Graph<VertexProp, EdgeProp> shard3(3, p3_ids_file, p3_shards_file, p3_rows_file, p3_cols_file, partition_book_file);

    int x = shard0.getNumOfVertices();
    x = shard0.getNumOfHaloVertices();
    VertexProp v = shard0.findVertex(10);
    v.getNeighbors();
    v.getShardsOfNeighbors();

    x = shard1.getNumOfVertices();
    x = shard1.getNumOfHaloVertices();
    v = shard1.findVertex(120);
    v.getNeighbors();
    v.getShardsOfNeighbors();

    x = shard2.getNumOfVertices();
    x = shard2.getNumOfHaloVertices();
    v =shard2.findVertex(330);
    v.getNeighbors();
    v.getShardsOfNeighbors();

    x = shard3.getNumOfVertices();
    x = shard3.getNumOfHaloVertices();
    v = shard3.findVertex(240);
    v.getNeighbors();
     v.getShardsOfNeighbors();

    return 0;
}