// Copyright (c) Meta Platforms, Inc. and affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#ifndef VERTEXPROP_H
#define VERTEXPROP_H

#include <iostream>
#include <vector>
#include <string>
#include "global.h"
#include "SharedIndices.h"

class VertexProp{
private:
    // self vertex infos
    VertexType vertexID;
    ShardType shardID;

    // neighboring vertices infos, each array has the same size as out-degrees
    SharedIndices<VertexType>* neighborVertexIDs;
    SharedIndices<ShardType>* neighborShardIDs;
    SharedIndices<WeightType>* edgeWeights;
    SharedIndices<WeightType>* neighborWeightedDegrees;

    bool isLocked;

public:
    VertexProp() {};
    VertexProp(VertexType vertexID_, ShardType shardID_,
               EdgeType neighborStartIndex_, EdgeType neighborEndIndex_,
               WeightType** csrWeightedDegrees_,  WeightType** edgeWeights_,
               VertexType** csrIndicesPtr_, ShardType** csrShardIndicesPtr_);

    VertexType getNodeID() const;
    ShardType getShardID() const;
    EdgeType getNeighborCount() const;

    VertexType* getIndicesPtr() const;
    ShardType* getShardsPtr() const;
    WeightType* getWeightedDegreesPtr() const;
    WeightType* getEdgeWeightsPtr() const;

    VertexType getNeighborVertexID(int index) const;
    ShardType getNeighborShardID(int index) const;
    WeightType getNeighborWeightedDegree(int index) const;
    WeightType getEdgeWeight(int index) const;

    bool getLocking() const;
    bool setLocking(bool b);
};

#endif