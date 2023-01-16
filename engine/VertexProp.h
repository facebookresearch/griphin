#ifndef VERTEXPROP_H
#define VERTEXPROP_H

#include <iostream>
#include <vector>
#include <string>
#include "global.h"
#include "utils.h"
#include "SharedMemVector.h"

class VertexProp{
    public:
        VertexProp(VertexType vertexID_, ShardType shardID_, EdgeType neighborStartIndex_, EdgeType neighborEndIndex_, VertexType* csrIndicesPtr_, ShardType* csrShardIndicesPtr_);
        VertexType vertexID;
        ShardType shardID;

        EdgeType neighborCount;
        
        SharedMemoryVector* neighborVector;

        bool isLocked;
        std::vector<float> vertexData;
        
        VertexType getNodeId();
        ShardType getShardId();
        
        // bool addNeighbor(VertexType neighborId, int neighborShardId);
        
        VertexType* getIndicesPtr();
        ShardType* getShardsPointer();
        VertexType getNeighbor(int index);
        ShardType getShard(int index);
        
        EdgeType getNeighborCount();

        bool getLocking();
        bool setLocking(bool b);
};

#endif