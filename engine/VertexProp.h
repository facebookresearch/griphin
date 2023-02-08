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
        VertexType getNodeId() const;
        ShardType getShardId() const;
        float getWeightedDegree() const;
        
        VertexType* getIndicesPtr() const;
        ShardType* getShardsPtr() const;
        float* getWeightedDegreesPtr() const;
        float* getEdgeWeightsPtr() const;

        VertexType getNeighbor(int index) const;
        ShardType getShard(int index) const;
        float getNeighborWeightedDegree(int index) const;
        float getEdgeWeight(int index) const;

        EdgeType getNeighborCount() const;

        bool getLocking() const;
        bool setLocking(bool b);
};

#endif