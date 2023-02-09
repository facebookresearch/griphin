#ifndef PPR_H
#define PPR_H

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include "global.h"
#include <torch/extension.h>

class PPR{
    private:
        VertexType targetId;
        ShardType shardId;
        float alpha;
        float epsilon;
        std::map<std::pair<VertexType,ShardType>, float> p;
        std::map<std::pair<VertexType,ShardType>, float> r;
        std::map<std::pair<VertexType, ShardType>, std::tuple<VertexType, ShardType>> activatedNodes;
        // std::vector<VertexType> nextNodeIds;
        // std::vector<ShardType> nextShardIds;

    public:
        PPR(VertexType targetId_, ShardType shardId_, float alpha_, float epsilon_);

        VertexType getTargetId();
        ShardType getShardId();
        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> getP();
        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> getR();
        std::tuple<torch::Tensor, torch::Tensor> getActivatedNodes();

        std::tuple<torch::Tensor, torch::Tensor> pop_activated_nodes();
        void push(std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> neighborInfos_, torch::Tensor v_ids_, torch::Tensor v_shard_ids_);
};

#endif