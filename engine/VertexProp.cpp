#include <iostream>
#include "VertexProp.h"

VertexProp::VertexProp(VertexType vertexID_, ShardType shardID_, EdgeType neighborStartIndex_, EdgeType neighborEndIndex_, VertexType* csrIndicesPtr_, ShardType* csrShardIndicesPtr_) {
    vertexID = vertexID_;
    shardID = shardID_;
    neighborVector = new SharedMemoryVector(neighborStartIndex_, neighborEndIndex_, csrIndicesPtr_, csrShardIndicesPtr_);
    isLocked = false;
}

VertexType VertexProp::getNodeId(){
    return vertexID;
}

ShardType VertexProp::getShardId(){
    return shardID;
}


VertexType* VertexProp::getIndicesPtr(){
    return neighborVector->getIndicesPtr();
}

ShardType* VertexProp::getShardsPointer(){
    return neighborVector->getShardsPtr();
}

VertexType VertexProp::getNeighbor(int index){
    return neighborVector->getIndex(index);
}

ShardType VertexProp::getShard(int index){
    return neighborVector->getShardIndex(index);
}

VertexType VertexProp::getNeighborCount(){
    return neighborVector->getSize();
}

bool VertexProp::getLocking(){
    return isLocked;
}

bool VertexProp::setLocking(bool b=true){
    isLocked = true;
    return isLocked;
}