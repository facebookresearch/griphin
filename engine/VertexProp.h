#ifndef VERTEXPROP_H
#define VERTEXPROP_H

#include <iostream>
#include <vector>
#include <string>
#include "global.h"
#include "SharedMemoryVector.h"

class VertexProp{
    public:
        VertexProp(VertexType vertexID_, ShardType shardID_, EdgeType neighborStartIndex_, EdgeType neighborEndIndex_, VertexType* csrIndicesPtr_, ShardType* csrShardIndicesPtr_);
        VertexType vertexID;
        ShardType shardID;

        SharedMemoryVector* neighborVector;

        bool isLocked;
        std::vector<float> vertexData;
        VertexType getNodeId();
        ShardType getShardId();

        VertexType* getIndicesPtr();
        ShardType* getShardsPointer();
        VertexType getNeighbor(int index);
        ShardType getShard(int index);
        
        EdgeType getNeighborCount();

        bool getLocking();
        bool setLocking(bool b);
};

#endif