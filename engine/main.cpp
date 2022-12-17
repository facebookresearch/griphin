#include <iostream>
#include <string>
#include <fstream> 
#include <ctime>
#include <stdint.h>
#include "Graph.cpp"
#include "EdgeProp.h"
#include "VertexProp.h"

int main(){
    char p0_ids_file[] = "p0_ids.txt";
    char p0_shards_file[] = "p0_halo_shards.txt";
    char p0_rows_file[] = "p0_edge_sources.txt";
    char p0_cols_file[] = "p0_edge_dests.txt";

    char p1_ids_file[] = "p1_ids.txt";
    char p1_shards_file[] = "p1_halo_shards.txt";
    char p1_rows_file[] = "p1_edge_sources.txt";
    char p1_cols_file[] = "p1_edge_dests.txt";

    char p2_ids_file[] = "p2_ids.txt";
    char p2_shards_file[] = "p2_halo_shards.txt";
    char p2_rows_file[] = "p2_edge_sources.txt";
    char p2_cols_file[] = "p2_edge_dests.txt";

    char p3_ids_file[] = "p3_ids.txt";
    char p3_shards_file[] = "p3_halo_shards.txt";
    char p3_rows_file[] = "p3_edge_sources.txt";
    char p3_cols_file[] = "p3_edge_dests.txt";

    Graph<VertexProp, EdgeProp> shard0(0, p0_ids_file, p0_shards_file, p0_rows_file, p0_cols_file);
    Graph<VertexProp, EdgeProp> shard1(0, p1_ids_file, p1_shards_file, p1_rows_file, p1_cols_file);
    Graph<VertexProp, EdgeProp> shard2(0, p2_ids_file, p2_shards_file, p2_rows_file, p2_cols_file);
    Graph<VertexProp, EdgeProp> shard3(0, p3_ids_file, p3_shards_file, p3_rows_file, p3_cols_file);

    int x = shard0.getNumOfVertices();
    x = shard0.getNumOfHaloVertices();
    shard0.findVertex(30);

    x = shard1.getNumOfVertices();
    x = shard1.getNumOfHaloVertices();
    shard1.findVertex(830);

    x = shard2.getNumOfVertices();
    x = shard2.getNumOfHaloVertices();
    shard2.findVertex(30);

    x = shard3.getNumOfVertices();
    x = shard3.getNumOfHaloVertices();
    shard3.findVertex(2400);
  

    return 0;
}