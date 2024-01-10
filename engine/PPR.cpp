// Copyright (c) Meta Platforms, Inc. and affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "PPR.h"
#include <omp.h>

PPR::PPR(VertexType targetId_, ShardType shardId_, WeightType alpha_, WeightType epsilon_, size_t num_threads_){
    targetId = targetId_;
    shardId = shardId_;
    alpha = alpha_;
    epsilon = epsilon_;

    if (r.subcnt() % num_threads_ == 0) {
        num_threads = num_threads_;
    }
    else {
        std::cout << "Warning: num_threads should be the factor of the number of submaps " << r.subcnt();
        std::cout << ". Set num_threads to 1" << std::endl;
        num_threads = 1;
    }

    r[std::make_pair(targetId, shardId)] = 1.;
    activatedNodes.insert(std::make_pair(targetId_, shardId_));
}

VertexType PPR::getTargetId(){
    return targetId;
}

ShardType PPR::getShardId(){
    return shardId;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> PPR::getP(){
    std::vector<VertexType>* nodeIds = new std::vector<VertexType>();
    std::vector<ShardType>* shardIds = new std::vector<ShardType>();
    std::vector<WeightType>* values = new std::vector<WeightType>();

    for(auto it = p.begin(); it != p.end(); ++it) {
        nodeIds->push_back(std::get<0>(it->first));
        shardIds->push_back(std::get<1>(it->first));
        values->push_back(it->second);
    }

    int64_t len = nodeIds->size();
    torch::Tensor torchNodeIds = torch::from_blob(nodeIds->data(), {len}, tensorVertexType);
    torch::Tensor torchShardIds = torch::from_blob(shardIds->data(), {len}, tensorShardType);
    torch::Tensor torchValues = torch::from_blob(values->data(), {len}, tensorWeightType);

    return std::make_tuple(torchNodeIds, torchShardIds, torchValues);
}

std::tuple<torch::Tensor, torch::Tensor> PPR::popActivatedNodes(){
    int64_t len = activatedNodes.size();
    auto* nodeIds = new VertexType[len];
    auto* shardIds = new ShardType [len];

    auto i = 0;
    for(auto it = activatedNodes.begin(); it != activatedNodes.end(); ++it) {
        nodeIds[i] = it->first;
        shardIds[i] = it->second;
        i++;
    }
    activatedNodes.clear();

    torch::Tensor torchNodeIds = torch::from_blob(nodeIds, {len}, tensorVertexType);
    torch::Tensor torchShardIds = torch::from_blob(shardIds, {len}, tensorShardType);

    return std::make_tuple(torchNodeIds, torchShardIds);
}


/**
 * @Deprecated For test only
 */
void PPR::push(std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> neighborInfos_,
               torch::Tensor v_ids_, torch::Tensor v_shard_ids_){

    VertexType vSize = neighborInfos_.size();
    auto edgeSize = 0;
    std::vector<WeightType> rvVals(vSize, 0);

    VertexType* vIdsPtr = v_ids_.data_ptr<VertexType>();
    ShardType * vShardIdsPtr = v_shard_ids_.data_ptr<ShardType>();
    for (int i = 0; i < vSize; i++) {
        auto vKey = std::make_pair(vIdsPtr[i], vShardIdsPtr[i]);
        rvVals[i] = r[vKey];
        r[vKey] = 0;
        p.try_emplace_l(vKey, [&](MapPR::value_type &vVal) { vVal.second += alpha * rvVals[i]; }, alpha * rvVals[i]);
        activatedNodes.erase(vKey);

        edgeSize += std::get<0>(neighborInfos_[i]).numel();;
    }

    size_t num_threads_ = edgeSize > 2000 ? num_threads : 1;
#pragma omp parallel for default(shared) schedule(static) num_threads(num_threads_)
    for (size_t thread_idx = 0; thread_idx < num_threads_; thread_idx++)
    {
        size_t modulo = r.subcnt() / num_threads_;

        for (int i = 0; i < vSize; i++) {
            VertexType* uIds = std::get<0>(neighborInfos_[i]).data_ptr<VertexType>();
            ShardType* uShardIds = std::get<1>(neighborInfos_[i]).data_ptr<ShardType>();
            WeightType* uDegrees = std::get<3>(neighborInfos_[i]).data_ptr<WeightType>();

            torch::Tensor uWeights = std::get<2>(neighborInfos_[i]);
            WeightType* uWeightsPtr = uWeights.data_ptr<WeightType>();
            auto uSize = uWeights.sizes()[0];

            WeightType weight_sum = 0;
            for (auto j = 0; j < uSize; j++) {
                weight_sum += uWeightsPtr[j];
            }
            WeightType vVal = (1 - alpha) * rvVals[i] / weight_sum;

            // bottleneck
            for(int64_t j = 0; j < uSize; j++){
                auto val = uWeightsPtr[j] * vVal;
                auto uId = uIds[j];
                auto uShardId = uShardIds[j];
                auto uDegree = uDegrees[j];

                auto uKey = std::make_pair(uId, uShardId);
                size_t hash_val = r.hash(uKey);
                size_t idx = r.subidx(hash_val);
                if (idx / modulo == thread_idx) {
                    r.try_emplace_l(uKey, [&](MapPR::value_type& uVal) { uVal.second += val; }, val);
                    if(r[uKey] >= epsilon * uDegree){
                        activatedNodes.insert(uKey);  // insert uKey to the same submap idx as r
                    }
                }
            }
        }
    }
}


void PPR::push(std::vector<VertexProp> neighborInfos_, torch::Tensor v_ids_, torch::Tensor v_shard_ids_){
//    int compute_time = 0;
//    int update_time = 0;
//    auto start = std::chrono::high_resolution_clock::now();

    VertexType vSize = neighborInfos_.size();
    auto edgeSize = 0;
    std::vector<WeightType> rvVals(vSize, 0);

    VertexType* vIdsPtr = v_ids_.data_ptr<VertexType>();
    ShardType * vShardIdsPtr = v_shard_ids_.data_ptr<ShardType>();
    for (int i = 0; i < vSize; i++) {
        auto vKey = std::make_pair(vIdsPtr[i], vShardIdsPtr[i]);
        rvVals[i] = r[vKey];
        r[vKey] = 0;
        p.try_emplace_l(vKey, [&](MapPR::value_type &vVal) { vVal.second += alpha * rvVals[i]; }, alpha * rvVals[i]);
        activatedNodes.erase(vKey);

        edgeSize += neighborInfos_[i].getNeighborCount();
    }

//    auto end = std::chrono::high_resolution_clock::now();
//    compute_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
//    start = std::chrono::high_resolution_clock::now();

//    std::cout << edgeSize << " " << vSize << std::endl;
//    size_t num_threads_ = vSize > 20 ? num_threads : 1;
    size_t num_threads_ = edgeSize > 2000 ? num_threads : 1;
#pragma omp parallel for default(shared) schedule(static) num_threads(num_threads_)
    for (size_t thread_idx = 0; thread_idx < num_threads_; thread_idx++)
    {
        size_t modulo = r.subcnt() / num_threads_;

        for (int i = 0; i < vSize; i++) {
            VertexType* uIds = neighborInfos_[i].getIndicesPtr();
            ShardType* uShardIds = neighborInfos_[i].getShardsPtr();
            WeightType* uWeightsPtr = neighborInfos_[i].getEdgeWeightsPtr();
            WeightType* uDegrees = neighborInfos_[i].getWeightedDegreesPtr();
            auto uSize = neighborInfos_[i].getNeighborCount();

            WeightType weight_sum = 0;
            for (auto j = 0; j < uSize; j++) {
                weight_sum += uWeightsPtr[j];
            }
            WeightType vVal = (1 - alpha) * rvVals[i] / weight_sum;

            // bottleneck
            for(int64_t j = 0; j < uSize; j++){
                auto val = uWeightsPtr[j] * vVal;
                auto uId = uIds[j];
                auto uShardId = uShardIds[j];
                auto uDegree = uDegrees[j];

                auto uKey = std::make_pair(uId, uShardId);
                size_t hash_val = r.hash(uKey);
                size_t idx = r.subidx(hash_val);
                if (idx / modulo == thread_idx) {
                    r.try_emplace_l(uKey, [&](MapPR::value_type& uVal) { uVal.second += val; }, val);
                    if(r[uKey] >= epsilon * uDegree){
                        activatedNodes.insert(uKey);  // insert uKey to the same submap idx as r
                    }
                }
            }
        }
    }
//    end = std::chrono::high_resolution_clock::now();
//    update_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
//    std::cout << "    compute/update time: " << compute_time << "/" << update_time << std::endl;
}


void PPR::push(std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> neighborInfos_,
               torch::Tensor v_ids_, torch::Tensor v_shard_ids_){
    auto vSize = std::get<0>(neighborInfos_).numel() - 1;

    auto* indptr = std::get<0>(neighborInfos_).data_ptr<EdgeType>();
    auto* indices = std::get<1>(neighborInfos_).data_ptr<VertexType>();
    auto* shardIndices = std::get<2>(neighborInfos_).data_ptr<ShardType>();
    auto* edgeWeightIndices = std::get<3>(neighborInfos_).data_ptr<WeightType>();
    auto* degreeIndices = std::get<4>(neighborInfos_).data_ptr<WeightType>();

    VertexType* vIdsPtr = v_ids_.data_ptr<VertexType>();
    ShardType * vShardIdsPtr = v_shard_ids_.data_ptr<ShardType>();

    std::vector<VertexProp> vertexProps;
    vertexProps.resize(vSize);

    //assemble vertexProps
    for(int i = 0; i < vSize; i++) {
        auto neighborStartIndex = indptr[i];
        auto neighborEndIndex = indptr[i + 1];
        vertexProps[i] = VertexProp(
                vIdsPtr[i], vShardIdsPtr[i],
                neighborStartIndex, neighborEndIndex,
                &degreeIndices, &edgeWeightIndices,
                &indices, &shardIndices
        );
    }

    push(vertexProps, v_ids_, v_shard_ids_);
}