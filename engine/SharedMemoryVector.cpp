#include "SharedMemoryVector.h"

SharedMemoryVector::SharedMemoryVector(EdgeType neighborStartIndex_,
                                       EdgeType neighborEndIndex_,
                                       float* csrWeightedDegrees_,
                                       float* edgeWeights_,
                                       VertexType* csrIndices_,
                                       ShardType* csrShardIndices_){
    neighborStartIndex = neighborStartIndex_;
    neighborEndIndex = neighborEndIndex_;
    size = neighborEndIndex - neighborStartIndex;
    csrIndices = csrIndices_;
    csrShardIndices = csrShardIndices_;
    csrWeightedDegrees = csrWeightedDegrees_;
    edgeWeights = edgeWeights_;
}

VertexType* SharedMemoryVector::getIndicesPtr(){
    return &csrIndices[neighborStartIndex];
}

ShardType* SharedMemoryVector::getShardsPtr(){
    return &csrShardIndices[neighborStartIndex];
}

float* SharedMemoryVector::getWeightedDegreesPtr(){
    return &csrWeightedDegrees[neighborStartIndex];
}

float* SharedMemoryVector::getEdgeWeightsPtr(){
    return &edgeWeights[neighborStartIndex];
}

VertexType SharedMemoryVector::getIndex(int index){
    return csrIndices[neighborStartIndex + index];
}

ShardType SharedMemoryVector::getShardIndex(int index){
    return csrShardIndices[neighborStartIndex + index];
}

float SharedMemoryVector::getWeightedDegreeIndex(int index){
    return csrWeightedDegrees[neighborStartIndex + index];
}

float SharedMemoryVector::getEdgeWeightIndex(int index){
    return edgeWeights[neighborStartIndex + index];
}

EdgeType SharedMemoryVector::getSize(){
    return size;
}