// Copyright (c) Meta Platforms, Inc. and affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <iostream>
#include <string>
#include <fstream> 
#include <ctime>
#include <stdint.h>
#include "Graph.cpp"
#include "EdgeProp.h"
#include "VertexProp.h"
#include <random>

int main(){

    char path[] = "../../../../data/omer/meta_data/hz-ogbn-product-p4";

    Graph<VertexProp, EdgeProp> shard0(0, path);
    Graph<VertexProp, EdgeProp> shard1(1, path);
    Graph<VertexProp, EdgeProp> shard2(2, path);
    Graph<VertexProp, EdgeProp> shard3(3, path);


    printf("num of cores: %ld\n", shard0.getNumOfCoreVertices());
    printf("num of cores: %ld\n", shard1.getNumOfCoreVertices());
    printf("num of cores: %ld\n", shard2.getNumOfCoreVertices());
    printf("num of cores: %ld\n", shard3.getNumOfCoreVertices());

    VertexProp v = shard0.findVertex(10);
    printf("weighted degree %f\n", v.getWeightedDegree());
    printf("num neighbors %d\n", v.getNeighborCount());
    printf("weighted degree %f\n", v.getWeightedDegreesPtr()[1]);
    printf("%d - %d\n", v.getNodeID(), v.getShardID());
    printf("neighbor ptr - %p\n", (void *) v.getIndicesPtr());
    printf("neighbor i - %d\n", v.getIndicesPtr()[10]);
    printf("neighbor i - %d\n", v.getNeighborVertexID(10));
    printf("shard i - %d\n", v.getNeighborShardID(10));
    printf("w deg i - %f\n", v.getNeighborWeightedDegree(10));

    std::vector<VertexType> partitionBookVec = shard1.getPartitionBook();

    for (VertexType i: partitionBookVec)
        std::cout << i << ' ';
    
    return 0;
}