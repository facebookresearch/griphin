#ifndef PPR
#define PPR

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
        std::map<std::pair<VertexType,ShardType>>, float> p;
        std::map<std::pair<VertexType,ShardType>>, float> r;
        std::vector<std::pair<VertexType,ShardType>> activatedNodes;
        std::vector<VertexType> nextNodeIds;
        std::vector<ShardType> nextShardIds;

    public:
        PPR(VertexType targetId_, ShardType shardId_, float alpha_, float epsilon_);
        void push(std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> neighborInfos_, torch::Tensor v_ids_, torch::Tensor v_shard_ids_);
};

#endif