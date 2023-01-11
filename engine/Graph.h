#ifndef GRAPH_H
#define GRAPH_H

#include <iostream>
#include <string>
#include <fstream> 
#include <ctime>
#include <vector>
#include <algorithm>
#include <stdint.h>
#include <map>
#include <torch/extension.h>
#include "EdgeProp.h"
#include "VertexProp.h"

typedef int VertexType;
typedef int EdgeType;
#define SIZE 100

template <class VertexProp, class EdgeProp> class Graph{
    private:
        int shardID;

        std::vector<VertexProp> vertexProps;
        //std::vector<EdgeProp> edgeProps;

        VertexType numNodes;
        VertexType numCoreNodes;
        VertexType numHaloNodes;
        EdgeType numEdges;

        std::vector<VertexType> nodeIDs;
        std::vector<int> haloNodeShards;
        std::vector<VertexType> csrIndices;
        std::vector<VertexType> csrShardIndices;
        std::vector<VertexType> csrIndptrs;
        std::vector<VertexType> partitionBook;


    public:
        Graph(int shardID_, char *idsList, char *haloShardsList, char *csrIndicesFile, char *csrShardIndicesFile, char *csrIndPtrsFile, char *partitionBookFile);  // takes shards as the argument
        ~Graph();

        std::vector<VertexType> getPartitionBook();

        // Query
        int getNumOfVertices();
        int getNumOfCoreVertices();
        int getNumOfHaloVertices();
        VertexProp findVertex(VertexType vertexID);          // returns local id in the current shard based on given global id

        std::vector<int> getNeighbors(VertexType vertexID);
        std::vector<int> getNeighborShards(VertexType vertexID);

        bool findVertexLocking(VertexType localVertexID);          // i did not understand what are the locks used for but i am assuming this function returns true if the given node is locked
        VertexProp findVertexProp(VertexType localVertexID);       // returns the vertex properties of given local vertex ID
        //VertexProp findEdgeProp(VertexType localEdgeID);           // returns the edge properties of the given edge ID

        // Graph Mutation
        bool addVertex(VertexProp vertex);                                      // adds vertex to the shard
        bool addVertexLocking(VertexType localVertexID);                        // locks the given vertex
        bool addBatchVertexLocking(std::vector<VertexType> localVertexIDs);     // locks all the vertices in the vector

        bool findVertexAndUpdatePropertyLocking(VertexType globalVertexID, bool lock=1);     // uses findVertex and updates the locking

        bool addEdgeIntern(VertexType localVertexID1, VertexType localVertexID2);           // connects two nodes in the given shard
        bool addEdgeExtern(VertexType localVertexID, VertexType globalVertexID);            // connects one internal and one external node
        bool addEdgeInternLocking(EdgeType localEdgeID);                                    // locks given edge
        //bool addOrUpdateEdgePropertyLocking();

        bool deleteVertex(VertexType localVertexID);
        bool deleteEdge(VertexType localVertexID1, VertexType localVertexID2);
        bool deleteEdge(EdgeType localEdgeID);

        // Sampling
        std::tuple<torch::Tensor, std::map<int, torch::Tensor>> sampleSingleNeighbor(const torch::Tensor &srcVertexIDs_);  // return {localIDs, shardIndexMap}
};

#endif