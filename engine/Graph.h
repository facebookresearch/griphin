// Copyright (c) Meta Platforms, Inc. and affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#ifndef GRAPH_H
#define GRAPH_H

#include <iostream>
#include <string>
#include <fstream> 
#include <ctime>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <map>
#include <torch/extension.h>
#include "global.h"
#include "EdgeProp.h"
#include "VertexProp.h"

template <class VertexProp, class EdgeProp> class Graph {
    private:
        ShardType shardID;
        ShardType numPartition;

        int64_t numNodes;
        int64_t numCoreNodes;
        int64_t numHaloNodes;
        int64_t numEdges;
        int64_t indicesLen;
        int64_t indptrLen;

        EdgeType* csrIndptrs;
        VertexType* csrIndices;
        ShardType* csrShardIndices;
        WeightType * edgeWeights;
        WeightType * csrWeightedDegrees;
        VertexType* partitionBook;

        VertexProp* vertexProps;

    public:
        Graph(ShardType shardID_, torch::Tensor indptrs_, torch::Tensor indices_, torch::Tensor shardIndices_,
              torch::Tensor edgeWeightIndices_, torch::Tensor weightedDegreeIndices_, torch::Tensor partition_book_);
        ~Graph();

        // Query
        std::vector<VertexType> getPartitionBook();
        int64_t getNumOfVertices();
        int64_t getNumOfCoreVertices();
        int64_t getNumOfHaloVertices();
        ShardType getShardID();

        VertexProp findVertex(VertexType vertexID);            // returns the vertex properties of the given local vertex ID
        bool findVertexLocking(VertexType localVertexID);      // returns true if the given node is locked
        VertexProp findVertexProp(VertexType localVertexID);   // returns the vertex properties of given local vertex ID
        //EdgeProp findEdgeProp(VertexType localEdgeID);       // returns the edge properties of the given edge ID

        // Neighborhood Fetching
        std::vector<torch::Tensor>getNeighborLists(const torch::Tensor &srcVertexIDs_);
        std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> getNeighborInfos(const torch::Tensor &srcVertexIDs_);
        std::vector<VertexProp> getNeighborInfos2(const torch::Tensor &srcVertexIDs_);
        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> getNeighborInfos3(const torch::Tensor &srcVertexIDs_);

        std::tuple<torch::Tensor, std::map<ShardType, torch::Tensor>> sampleSingleNeighbor(const torch::Tensor &srcVertexIDs_);  // return {localIDs, shardIndexMap}
        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sampleSingleNeighbor2(const torch::Tensor &srcVertexIDs_, size_t num_threads);


        // Graph Mutation
        bool addVertex(VertexProp vertex);                                      // adds vertex to the shard
        bool addVertexLocking(VertexType localVertexID);                        // locks the given vertex
        bool addBatchVertexLocking(const std::vector<VertexType>& localVertexIDs);     // locks all the vertices in the vector

        bool findVertexAndUpdatePropertyLocking(VertexType globalVertexID, bool lock=1);     // uses findVertex and updates the locking

        bool addEdgeIntern(VertexType localVertexID1, VertexType localVertexID2);           // connects two nodes in the given shard
        bool addEdgeExtern(VertexType localVertexID, VertexType globalVertexID);            // connects one internal and one external node
        bool addEdgeInternLocking(EdgeType localEdgeID);                                    // locks given edge
        //bool addOrUpdateEdgePropertyLocking();

        bool deleteVertex(VertexType localVertexID);
        bool deleteEdge(VertexType localVertexID1, VertexType localVertexID2);
        bool deleteEdge(EdgeType localEdgeID);

};

#endif