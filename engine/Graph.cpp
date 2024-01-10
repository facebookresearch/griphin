// Copyright (c) Meta Platforms, Inc. and affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <iostream>
#include <vector>
#include <ctime>
#include <cstdio>
#include <omp.h>
#include "Graph.h"
#include "utils.h"

template <class VertexProp, class EdgeProp>
Graph<VertexProp, EdgeProp>::Graph(ShardType shardID_, torch::Tensor indptrs_, torch::Tensor indices_,
                                   torch::Tensor shardIndices_, torch::Tensor edgeWeightIndices_,
                                   torch::Tensor weightedDegreeIndices_, torch::Tensor partition_book_) {
    csrIndptrs = indptrs_.contiguous().data_ptr<EdgeType>();
    csrIndices = indices_.contiguous().data_ptr<VertexType>();
    csrShardIndices = shardIndices_.contiguous().data_ptr<ShardType>();
    edgeWeights = edgeWeightIndices_.contiguous().data_ptr<WeightType>();
    csrWeightedDegrees = weightedDegreeIndices_.contiguous().data_ptr<WeightType>();
    partitionBook = partition_book_.contiguous().data_ptr<VertexType>();

    shardID = shardID_;
    numPartition = static_cast<ShardType>(partition_book_.numel() - 1);

    indptrLen = indptrs_.numel();
    indicesLen = indices_.numel();
    numEdges = indices_.numel();
    numCoreNodes = partitionBook[shardID+1] - partitionBook[shardID];
    numHaloNodes = 0;  // TODO:
    numNodes = 0;      // TODO:

    vertexProps = (VertexProp *) malloc(numCoreNodes * sizeof(VertexProp));

    for(VertexType vertexID = 0; vertexID < numCoreNodes; vertexID++){
        auto neighborStartIndex = csrIndptrs[vertexID];
        auto neighborEndIndex = csrIndptrs[vertexID + 1];

        vertexProps[vertexID] = VertexProp(
                vertexID, shardID,
                neighborStartIndex, neighborEndIndex,
                &csrWeightedDegrees, &edgeWeights,
                &csrIndices, &csrShardIndices
                );
    }
}

template<class VertexProp, class EdgeProp>
std::vector<VertexType> Graph<VertexProp, EdgeProp>::getPartitionBook() {
    std::vector<VertexType> partitionBookVec(numPartition + 1, 0);
    for(int i = 0; i < numPartition + 1; i++)
        partitionBookVec[i] = partitionBook[i];
    return partitionBookVec;
}

template <class VertexProp, class EdgeProp> 
VertexProp Graph<VertexProp, EdgeProp>::findVertex(VertexType vertexID){
    return vertexProps[vertexID];
} 

template <class VertexProp, class EdgeProp>
Graph<VertexProp, EdgeProp>::~Graph() = default;

template <class VertexProp, class EdgeProp>
int64_t Graph<VertexProp, EdgeProp>::getNumOfVertices(){
//    return numNodes;
    std::cout << "Function not yet implemented" << std::endl;
    abort();
}

template <class VertexProp, class EdgeProp>
int64_t Graph<VertexProp, EdgeProp>::getNumOfCoreVertices(){
    return numCoreNodes;
}

template <class VertexProp, class EdgeProp>
int64_t Graph<VertexProp, EdgeProp>::getNumOfHaloVertices(){
//    return numHaloNodes;
    std::cout << "Function not yet implemented" << std::endl;
    abort();
}

template<class VertexProp, class EdgeProp>
ShardType Graph<VertexProp, EdgeProp>::getShardID() {
    return shardID;
}

template <class VertexProp, class EdgeProp>
bool Graph<VertexProp, EdgeProp>::findVertexLocking(VertexType localVertexID){
    return vertexProps[localVertexID].getLocking();
}

template <class VertexProp, class EdgeProp> 
VertexProp Graph<VertexProp, EdgeProp>::findVertexProp(VertexType localVertexID){
    return vertexProps[localVertexID];
}

template <class VertexProp, class EdgeProp>
bool Graph<VertexProp, EdgeProp>::addVertex(VertexProp vertex){
//    return true;
    std::cout << "Function not yet implemented" << std::endl;
    abort();
}

template <class VertexProp, class EdgeProp> 
bool Graph<VertexProp, EdgeProp>::addVertexLocking(VertexType localVertexID){
    vertexProps[localVertexID].setLocking();
    return true;
}

template <class VertexProp, class EdgeProp> 
bool Graph<VertexProp, EdgeProp>::addBatchVertexLocking(const std::vector<VertexType>& localVertexIDs){
    for(auto localVertexID: localVertexIDs){
        vertexProps[localVertexID].setLocking();
    }
    return true;
}

template<class VertexProp, class EdgeProp>
std::tuple<torch::Tensor, std::map<ShardType, torch::Tensor>>
Graph<VertexProp, EdgeProp>::sampleSingleNeighbor(const torch::Tensor& srcVertexIDs_) {
    int64_t len = srcVertexIDs_.numel();
    torch::Tensor srcVertexIDs = srcVertexIDs_.contiguous();
    const VertexType* srcVertexPtr = srcVertexIDs.data_ptr<VertexType>();

    auto* sampledVertices_ = new VertexType[len];  // to avoid copy, we need to allocate memory for sampled Vertices
    std::map<ShardType, std::vector<int64_t>*> shardIndexMap_;

    std::random_device r;
    std::default_random_engine e(r());

    for (int64_t i=0; i < len; i++) {
        VertexProp prop = findVertex(srcVertexPtr[i]);
        auto size = prop.getNeighborCount();

        VertexType neighborID;
        ShardType neighborShardID;

        if (prop.getNeighborCount() == 0) {
            neighborID = prop.getNodeID();
            neighborShardID = prop.getShardID();
        }
        else{
            std::uniform_int_distribution<int> uniform_dist(0, size-1);
            auto rand = uniform_dist(e);

            neighborID = prop.getNeighborVertexID(rand);
            neighborShardID = prop.getNeighborShardID(rand);
        }

        sampledVertices_[i] = neighborID;

        if (shardIndexMap_.find(neighborShardID) == shardIndexMap_.end()) {
            shardIndexMap_[neighborShardID] = new std::vector<int64_t>();  // allocate memory
        }
        shardIndexMap_[neighborShardID]->push_back(i);
    }

    auto opts = srcVertexIDs_.options();
    torch::Tensor sampledVertices = torch::from_blob(sampledVertices_, {len}, opts);
    std::map<ShardType, torch::Tensor> shardIndexMap;
    for (auto & it : shardIndexMap_) {
        shardIndexMap[it.first] = torch::from_blob(
                it.second->data(), {(int64_t)it.second->size()}, torch::kInt64);
    }

    return {sampledVertices, shardIndexMap};
}

template<class VertexProp, class EdgeProp>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
Graph<VertexProp, EdgeProp>::sampleSingleNeighbor2(const torch::Tensor& srcVertexIDs_, size_t num_threads) {
    int64_t len = srcVertexIDs_.numel();
    torch::Tensor srcVertexIDs = srcVertexIDs_.contiguous();
    const VertexType* srcVertexPtr = srcVertexIDs.data_ptr<VertexType>();

    // allocate memory for sampled vertices to avoid copy
    auto* localVertexIDs_ = new VertexType[len];
    auto* globalVertexIDs_ = new VertexType[len];
    auto* shardIDs_ = new ShardType[len];

    size_t num_threads_ = len > 2000 ? num_threads : 1;
#pragma omp parallel default(shared) num_threads(num_threads_)
    {
//        std::random_device dev;
//        std::mt19937_64 rng(dev());
        std::mt19937_64 rng((omp_get_thread_num() + 1) * time(nullptr));

        #pragma omp for schedule(static)
        for (int64_t i=0; i < len; i++) {
            VertexProp prop = findVertex(srcVertexPtr[i]);
            auto size = prop.getNeighborCount();

            VertexType neighborID;
            ShardType neighborShardID;

            if (size == 0) {
                neighborID = prop.getNodeID();
                neighborShardID = prop.getShardID();
            }
            else {
                std::uniform_int_distribution<int> uniform_dist(0, size-1);
                auto rand = uniform_dist(rng);

                neighborID = prop.getNeighborVertexID(rand);
                neighborShardID = prop.getNeighborShardID(rand);
            }

            localVertexIDs_[i] = neighborID;
            globalVertexIDs_[i] = neighborID + partitionBook[neighborShardID];
            shardIDs_[i] = neighborShardID;
        }
    }

    torch::Tensor localVertexIDs = torch::from_blob(localVertexIDs_, {len}, tensorVertexType);
    torch::Tensor globalVertexIDs = torch::from_blob(globalVertexIDs_, {len}, tensorVertexType);
    torch::Tensor shardIDs = torch::from_blob(shardIDs_, {len}, tensorShardType);

    return {localVertexIDs, globalVertexIDs, shardIDs};
}

template<class VertexProp, class EdgeProp>
std::vector<torch::Tensor> Graph<VertexProp, EdgeProp>::getNeighborLists(const torch::Tensor &srcVertexIDs_) {
    int64_t len = srcVertexIDs_.numel();
    torch::Tensor srcVertexIDs = srcVertexIDs_.contiguous();
    const VertexType* srcVertexPtr = srcVertexIDs.data_ptr<VertexType>();

    std::vector<torch::Tensor> res(len, torch::Tensor());
    for (auto i=0; i<len; i++) {
        auto prop = findVertex(srcVertexPtr[i]);
        res[i] = torch::from_blob(prop.getIndicesPtr(), {prop.getNeighborCount()}, tensorVertexType);
    }

    return res;
}

template<class VertexProp, class EdgeProp>
std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>
Graph<VertexProp, EdgeProp>::getNeighborInfos(const torch::Tensor &srcVertexIDs_) {
    int64_t len = srcVertexIDs_.numel();
    torch::Tensor srcVertexIDs = srcVertexIDs_.contiguous();
    const VertexType* srcVertexPtr = srcVertexIDs.data_ptr<VertexType>();

    std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> res(
            len,{torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()}
            );

//    int time1 = 0;
//    int time2 = 0;

#pragma omp parallel for default(shared) schedule(runtime) num_threads(len > 100 ? 5 : 1)
    for (auto i=0; i<len; i++) {
//        auto start = std::chrono::high_resolution_clock::now();

        auto prop = findVertex(srcVertexPtr[i]);
        auto size = prop.getNeighborCount();

//        auto t = std::make_tuple(prop.getIndicesPtr(), prop.getShardsPtr(),
//                                 prop.getEdgeWeightsPtr(), prop.getWeightedDegreesPtr());
//
//        auto end = std::chrono::high_resolution_clock::now();
//        time1 += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
//        start = std::chrono::high_resolution_clock::now();

        res[i] = std::make_tuple(torch::from_blob(prop.getIndicesPtr(), {size}, tensorVertexType),
                                 torch::from_blob(prop.getShardsPtr(), {size}, tensorShardType),
                                 torch::from_blob(prop.getEdgeWeightsPtr(), {size}, tensorWeightType),
                                 torch::from_blob(prop.getWeightedDegreesPtr(), {size}, tensorWeightType));

//        res[i] = std::make_tuple(torch::from_blob(std::get<0>(t), {size}, tensorVertexType),
//                                 torch::from_blob(std::get<1>(t), {size}, tensorShardType),
//                                 torch::from_blob(std::get<2>(t), {size}, tensorWeightType),
//                                 torch::from_blob(std::get<3>(t), {size}, tensorWeightType));
//
//        end = std::chrono::high_resolution_clock::now();
//        time2 += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
//    std::cout << "    load/wrap time: " << time1 << "/" << time2 << std::endl;
    return res;
}

template<class VertexProp, class EdgeProp>
std::vector<VertexProp>
Graph<VertexProp, EdgeProp>::getNeighborInfos2 (const torch::Tensor &srcVertexIDs_) {
    int64_t len = srcVertexIDs_.numel();
    torch::Tensor srcVertexIDs = srcVertexIDs_.contiguous();
    const VertexType* srcVertexPtr = srcVertexIDs.data_ptr<VertexType>();

    std::vector<VertexProp> res;
    res.resize(len);

#pragma omp parallel for default(shared) schedule(runtime) num_threads(1)
    for (auto i=0; i<len; i++) {
        res[i] = findVertex(srcVertexPtr[i]);
    }
    return res;
}

template<class VertexProp, class EdgeProp>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
Graph<VertexProp, EdgeProp>::getNeighborInfos3(const torch::Tensor &srcVertexIDs_) {
    int64_t len = srcVertexIDs_.numel();
    torch::Tensor srcVertexIDs = srcVertexIDs_.contiguous();
    const VertexType* srcVertexPtr = srcVertexIDs.data_ptr<VertexType>();

    auto* indptr = new EdgeType[len+1];

    std::vector<VertexType>* indices = new std::vector<VertexType>;
    std::vector<ShardType>* shardIndices = new std::vector<ShardType>;
    std::vector<WeightType>* edgeWeightIndices = new std::vector<WeightType>;
    std::vector<WeightType>* degreeIndices = new std::vector<WeightType>;

    indices->reserve(len);
    shardIndices->reserve(len);
    edgeWeightIndices->reserve(len);
    degreeIndices->reserve(len);

    indptr[0] = 0;
    for (auto i=0; i<len; i++) {
        auto prop = findVertex(srcVertexPtr[i]);
        auto size = prop.getNeighborCount();

        indptr[i+1] = indptr[i] + size;

        indices->insert(indices->end(), prop.getIndicesPtr(), prop.getIndicesPtr()+size);
        shardIndices->insert(shardIndices->end(), prop.getShardsPtr(), prop.getShardsPtr()+size);
        edgeWeightIndices->insert(edgeWeightIndices->end(), prop.getEdgeWeightsPtr(), prop.getEdgeWeightsPtr()+size);
        degreeIndices->insert(degreeIndices->end(), prop.getWeightedDegreesPtr(), prop.getWeightedDegreesPtr()+size);
    }

//    for(auto i=0; i<indices->size(); i++){
//        std::cout << indices[i] << std::endl;
//    }

    long long edgeSize = indices->size();
    return std::make_tuple(torch::from_blob(indptr, {len+1}, tensorEdgeType),
                           torch::from_blob(indices->data(), {edgeSize}, tensorVertexType),
                           torch::from_blob(shardIndices->data(), {edgeSize}, tensorShardType),
                           torch::from_blob(edgeWeightIndices->data(), {edgeSize}, tensorWeightType),
                           torch::from_blob(degreeIndices->data(), {edgeSize}, tensorWeightType));
}