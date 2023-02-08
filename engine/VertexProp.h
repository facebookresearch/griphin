#ifndef VERTEXPROP_H
#define VERTEXPROP_H

#include <iostream>
#include <vector>
#include <string>
#include "global.h"
#include "SharedMemoryVector.h"

class VertexProp{
    public:
        VertexProp(VertexType vertexID_, ShardType shardID_, EdgeType neighborStartIndex_, EdgeType neighborEndIndex_, float weightedDegree_,  float* csrWeightedDegrees_,  float* edgeWeights_, VertexType* csrIndicesPtr_, ShardType* csrShardIndicesPtr_);
        VertexType vertexID;
        ShardType shardID;

        SharedMemoryVector* neighborVector;

        float weightedDegree;

        bool isLocked;
        std::vector<float> vertexData;
        VertexType getNodeId();
        ShardType getShardId();
        float getWeightedDegree();
        
        VertexType* getIndicesPtr();
        ShardType* getShardsPointer();
        float* getWeightedDegreesPtr();
        
        VertexType getNeighbor(int index);
        ShardType getShard(int index);
        float getNeighborWeightedDegree(int index);
        
        EdgeType getNeighborCount();

        bool getLocking();
        bool setLocking(bool b);
};

#endif