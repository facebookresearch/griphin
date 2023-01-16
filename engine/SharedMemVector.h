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
        
    public:
        SharedMemoryVector(EdgeType neighborStartIndex_, EdgeType neighborEndIndex_, VertexType* csrIndices_, ShardType* csrShardIndices_);
        
        VertexType* getIndicesPtr();
        ShardType* getShardsPtr();

        VertexType getIndex(int index);
        ShardType getShardIndex(int index);

        VertexType getSize();

};

#endif