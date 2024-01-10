// Copyright (c) Meta Platforms, Inc. and affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#ifndef PPR_H
#define PPR_H

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include "global.h"
#include <torch/extension.h>

#include "parallel_hashmap/phmap.h"
#include "VertexProp.h"

typedef phmap::parallel_flat_hash_map<
            std::pair<VertexType, ShardType>,
            WeightType,
            phmap::priv::hash_default_hash<std::pair<VertexType, ShardType>>,
            phmap::priv::hash_default_eq<std::pair<VertexType, ShardType>>,
            phmap::priv::Allocator<phmap::priv::Pair<
                const std::pair<VertexType, ShardType>, WeightType>>,
            4,
//            std::mutex
            phmap::NullMutex
        > MapPR;

typedef phmap::parallel_flat_hash_set<
            std::pair<VertexType, ShardType>,
            phmap::priv::hash_default_hash<std::pair<VertexType, ShardType>>,
            phmap::priv::hash_default_eq<std::pair<VertexType, ShardType>>,
            phmap::priv::Allocator<std::pair<VertexType, ShardType>>,
            4,
//            std::mutex
            phmap::NullMutex
        > SetActNodes;

class PPR{
    private:
        VertexType targetId;
        ShardType shardId;
        WeightType alpha;
        WeightType epsilon;
        size_t num_threads;

        MapPR p;
        MapPR r;
        SetActNodes activatedNodes;

    public:
        PPR(VertexType targetId_, ShardType shardId_, WeightType alpha_, WeightType epsilon_, size_t num_threads_);

        VertexType getTargetId();
        ShardType getShardId();
        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> getP();
        // std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> getR();
        // std::tuple<torch::Tensor, torch::Tensor> getActivatedNodes();

        std::tuple<torch::Tensor, torch::Tensor> popActivatedNodes();
        void push(std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> neighborInfos_,
                  torch::Tensor v_ids_, torch::Tensor v_shard_ids_);
        void push(std::vector<VertexProp> neighborInfos_, torch::Tensor v_ids_, torch::Tensor v_shard_ids_);
        void push(std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> neighborInfos_,
                  torch::Tensor v_ids_, torch::Tensor v_shard_ids_);
};

#endif