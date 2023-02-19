#include <iostream>
#include "VertexProp.h"

VertexProp::VertexProp(VertexType vertexID_,
                       ShardType shardID_,
                       EdgeType neighborStartIndex_,
                       EdgeType neighborEndIndex_,
                       float weightedDegree_,
                       float** csrWeightedDegrees_,
                       float** edgeWeights_,
                       VertexType** csrIndicesPtr_,
                       ShardType** csrShardIndicesPtr_
                       ) {
    vertexID = vertexID_;
    shardID = shardID_;
    neighborVector = new SharedMemoryVector(neighborStartIndex_,
                                            neighborEndIndex_,
                                            csrWeightedDegrees_,
                                            edgeWeights_,
                                            csrIndicesPtr_,
                                            csrShardIndicesPtr_);
    isLocked = false;
    weightedDegree = weightedDegree_;
}

VertexType VertexProp::getNodeId() const{
    return vertexID;
}

ShardType VertexProp::getShardId() const{
    return shardID;
}

float VertexProp::getWeightedDegree() const{
    return weightedDegree;
}

VertexType* VertexProp::getIndicesPtr() const{
    return neighborVector->getIndicesPtr();
}

ShardType* VertexProp::getShardsPtr() const{
    return neighborVector->getShardsPtr();
}

float* VertexProp::getWeightedDegreesPtr() const{
    return neighborVector->getWeightedDegreesPtr();
}

float* VertexProp::getEdgeWeightsPtr() const{
    return neighborVector->getEdgeWeightsPtr();
}

VertexType VertexProp::getNeighbor(int index) const{
    return neighborVector->getIndex(index);
}

ShardType VertexProp::getShard(int index) const{
    return neighborVector->getShardIndex(index);
}

float VertexProp::getNeighborWeightedDegree(int index) const{
    return neighborVector->getWeightedDegreeIndex(index);
}

float VertexProp::getEdgeWeight(int index) const{
    return neighborVector->getEdgeWeightIndex(index);
}

VertexType VertexProp::getNeighborCount() const{
    return neighborVector->getSize();
}

bool VertexProp::getLocking() const{
    return isLocked;
}

bool VertexProp::setLocking(bool b=true){
    isLocked = true;
    return isLocked;
}