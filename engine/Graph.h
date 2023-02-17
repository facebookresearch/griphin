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

template <class VertexProp, class EdgeProp> class Graph{
    private:
        ShardType shardID;

        VertexProp* vertexProps;
        //std::vector<EdgeProp> edgeProps;

        int64_t numNodes;
        int64_t numCoreNodes;
        int64_t numHaloNodes;
        int64_t numEdges;
        int64_t indicesLen;
        int64_t indptrLen;

        VertexType* csrIndices;
        ShardType* csrShardIndices;
        EdgeType* csrIndptrs;
        VertexType* partitionBook;
        float* edgeWeights;
        float* csrWeightedDegrees;

    public:
        Graph(ShardType shardID_, const char *path);  // takes shards as the argument
        ~Graph();

        std::vector<VertexType> getPartitionBook();

        // Query
        int64_t getNumOfVertices();
        int64_t getNumOfCoreVertices();
        int64_t getNumOfHaloVertices();
        std::vector<torch::Tensor>getNeighborLists(const torch::Tensor &srcVertexIDs_);
        std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>getNeighborInfos(const torch::Tensor &srcVertexIDs_);
        VertexProp findVertex(VertexType vertexID);          // returns local id in the current shard based on given global id

        bool findVertexLocking(VertexType localVertexID);          // i did not understand what are the locks used for but i am assuming this function returns true if the given node is locked
        VertexProp findVertexProp(VertexType localVertexID);       // returns the vertex properties of given local vertex ID
        //VertexProp findEdgeProp(VertexType localEdgeID);           // returns the edge properties of the given edge ID

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

        // Sampling
        std::tuple<torch::Tensor, std::map<ShardType, torch::Tensor>> sampleSingleNeighbor(const torch::Tensor &srcVertexIDs_);  // return {localIDs, shardIndexMap}
        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sampleSingleNeighbor2(const torch::Tensor &srcVertexIDs_);
};

#endif