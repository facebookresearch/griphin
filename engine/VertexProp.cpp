#include <iostream>
#include "VertexProp.h"

VertexProp::VertexProp(VertexType vertexID_, ShardType shardID_, EdgeType neighborStartIndex_, EdgeType neighborEndIndex_, float weightedDegree_, float* csrWeightedDegrees_, float* edgeWeights_, VertexType* csrIndicesPtr_, ShardType* csrShardIndicesPtr_) {
    vertexID = vertexID_;
    shardID = shardID_;
    neighborVector = new SharedMemoryVector(neighborStartIndex_, neighborEndIndex_, csrWeightedDegrees_, edgeWeights_, csrIndicesPtr_, csrShardIndicesPtr_);
    isLocked = false;
    weightedDegree = weightedDegree_;
}

VertexType VertexProp::getNodeId(){
    return vertexID;
}

ShardType VertexProp::getShardId(){
    return shardID;
}

float VertexProp::getWeightedDegree(){
    return weightedDegree;
}

VertexType* VertexProp::getIndicesPtr(){
    return neighborVector->getIndicesPtr();
}

ShardType* VertexProp::getShardsPointer(){
    return neighborVector->getShardsPtr();
}

float* VertexProp::getWeightedDegreesPtr(){
    return neighborVector->getWeightedDegreesPtr();
}

VertexType VertexProp::getNeighbor(int index){
    return neighborVector->getIndex(index);
}

ShardType VertexProp::getShard(int index){
    return neighborVector->getShardIndex(index);
}

float VertexProp::getNeighborWeightedDegree(int index){
    return neighborVector->getWeightedDegreeIndex(index);
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