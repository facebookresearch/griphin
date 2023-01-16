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

SharedMemoryVector::SharedMemoryVector(EdgeType neighborStartIndex_, EdgeType neighborEndIndex_, VertexType* csrIndices_, ShardType* csrShardIndices_){
    neighborStartIndex = neighborStartIndex_;
    neighborEndIndex = neighborEndIndex_;
    size = neighborEndIndex - neighborStartIndex;
    csrIndices = csrIndices_;
    csrShardIndices = csrShardIndices_;
}

VertexType* SharedMemoryVector::getIndicesPtr(){
    return &csrIndices[neighborStartIndex];
}

ShardType* SharedMemoryVector::getShardsPtr(){
    return &csrShardIndices[neighborStartIndex];
}

VertexType SharedMemoryVector::getIndex(int index){
    return csrIndices[neighborStartIndex + index];
}

ShardType SharedMemoryVector::getShardIndex(int index){
    return csrShardIndices[neighborStartIndex + index];
}

EdgeType SharedMemoryVector::getSize(){
    return size;
}

#endif