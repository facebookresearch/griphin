#ifndef SHARE_MEM_VECTOR_H
#define SHARE_MEM_VECTOR_H

#include "global.h"

class SharedMemoryVector{
    private:
        EdgeType neighborStartIndex;
        EdgeType neighborEndIndex;  
        EdgeType size;
        VertexType* csrIndices;
        ShardType* csrShardIndices;
        float* csrWeightedDegrees;
        float* edgeWeights;
        
    public:
        SharedMemoryVector(EdgeType neighborStartIndex_, EdgeType neighborEndIndex_, float* csrWeightedDegrees_, float* edgeWeights_, VertexType* csrIndices_, ShardType* csrShardIndices_);
        
        VertexType* getIndicesPtr();
        ShardType* getShardsPtr();
        float* getWeightedDegreesPtr();

        VertexType getIndex(int index);
        ShardType getShardIndex(int index);
        float getWeightedDegreeIndex(int index);

        VertexType getSize();

};

#endif