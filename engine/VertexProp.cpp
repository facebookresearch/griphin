// Copyright (c) Meta Platforms, Inc. and affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <iostream>
#include "VertexProp.h"

VertexProp::VertexProp(VertexType vertexID_,
                       ShardType shardID_,
                       EdgeType neighborStartIndex_,
                       EdgeType neighborEndIndex_,
                       WeightType** csrWeightedDegrees_,
                       WeightType** edgeWeights_,
                       VertexType** csrIndicesPtr_,
                       ShardType** csrShardIndicesPtr_
                       ) {
    vertexID = vertexID_;
    shardID = shardID_;

    neighborVertexIDs = new SharedIndices<VertexType>(neighborStartIndex_, neighborEndIndex_, csrIndicesPtr_);
    neighborShardIDs = new SharedIndices<ShardType>(neighborStartIndex_, neighborEndIndex_, csrShardIndicesPtr_);
    edgeWeights = new SharedIndices<WeightType>(neighborStartIndex_, neighborEndIndex_, edgeWeights_);
    neighborWeightedDegrees = new SharedIndices<WeightType>(neighborStartIndex_, neighborEndIndex_, csrWeightedDegrees_);

    isLocked = false;
}

VertexType VertexProp::getNodeID() const{
    return vertexID;
}

ShardType VertexProp::getShardID() const{
    return shardID;
}

EdgeType VertexProp::getNeighborCount() const {
    return neighborVertexIDs->size();
}

VertexType* VertexProp::getIndicesPtr() const{
    return neighborVertexIDs->getPtr();
}

ShardType* VertexProp::getShardsPtr() const{
    return neighborShardIDs->getPtr();
}

WeightType* VertexProp::getWeightedDegreesPtr() const{
    return neighborWeightedDegrees->getPtr();
}

WeightType* VertexProp::getEdgeWeightsPtr() const{
    return edgeWeights->getPtr();
}

VertexType VertexProp::getNeighborVertexID(int index) const{
    return neighborVertexIDs->getVal(index);
}

ShardType VertexProp::getNeighborShardID(int index) const{
    return neighborShardIDs->getVal(index);
}

WeightType VertexProp::getNeighborWeightedDegree(int index) const{
    return neighborWeightedDegrees->getVal(index);
}

WeightType VertexProp::getEdgeWeight(int index) const{
    return edgeWeights->getVal(index);
}

bool VertexProp::getLocking() const{
    return isLocked;
}

bool VertexProp::setLocking(bool b=true){
    isLocked = true;
    return isLocked;
}