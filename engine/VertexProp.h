#ifndef VERTEXPROP_H
#define VERTEXPROP_H

#include <iostream>
#include <vector>
#include <string>
#include "global.h"

class VertexProp{
    public:
        VertexProp(VertexType vertexID_, ShardType shardID_, EdgeType neighborStartIndex_, EdgeType neighborEndIndex_);
        VertexType vertexID;
        ShardType shardID;
        int neighborCount;
        EdgeType neighborStartIndex;
        EdgeType neighborEndIndex;
        bool isLocked;
        std::vector<float> vertexData;
        // std::vector<VertexType> *neighborVertices = new std::vector<VertexType>;
        // std::vector<int> *neighborVerticeShards = new std::vector<int>;
        // std::vector<EdgeType> neighborEdges;
        VertexType getNodeId();
        ShardType getShardId();
        EdgeType getNeighborStartIndex();
        EdgeType getNeighborEndIndex();
        bool addNeighbor(VertexType neighborId, int neighborShardId);
        void getNeighbors();
        void getShardsOfNeighbors();
        void setVertexProp(std::vector<float> vertexData, std::vector<VertexType> neighborVertices, std::vector<EdgeType> neighborEdges);
        bool getLocking();
        bool setLocking(bool b);
};

#endif