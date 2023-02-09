#include "ppr.h"

PPR::PPR(VertexType targetId_, ShardType shardId_, float alpha_, float epsilon_){
    targetId = targetId_;
    shardId = shardId_;
    alpha = alpha_;
    epsilon = epsilon_;

    r[std::make_pair(targetId, shardId)] = 1.;
    activatedNodes.push_back(std::make_pair(targetId, shardId));
    nextNodeIds.push_back(targetId);
    nextShardIds.push_back(shardId);
}

void PPR::push(std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> neighborInfos_, torch::Tensor v_ids_, torch::Tensor v_shard_ids_){
    VertexType size = neighborInfos_.size();
    for(int i = 0; i < size; i++){
        torch::Tensor uIds std::get<0>(neighborInfos_[i]);
        torch::Tensor uShardIds std::get<1>(neighborInfos_[i]);
        torch::Tensor uWeights std::get<2>(neighborInfos_[i]);
        torch::Tensor uDegrees std::get<3>(neighborInfos_[i]);

        auto vId = v_ids_[i];
        auto vShardId = v_shard_ids_[i];

        auto vKey = std::make_pair(vId, vShardId);
        p[vKey] += alpha * r[vKey];
        torch::Tensor uVals = (1 - alpha) * r[vKey] * uWeights / // uDegrees.sum()
        r[vKey] = 0.;

        auto uSize = uIds.sizes()[0];
        for(int j = 0; j < uSize; j++){
            auto val = uVals[j];
            auto uId = uIds[j];
            auto uShardId = uIds[j];
            auto uDegree = uDegrees[j];

            auto uKey = std::make_pair(uId, uShardId);

            r[uKey] += val;

            if(r[uKey] >= epsilon * uDegree){
                auto it = std::find(activatedNodes.begin(), activatedNodes.end(), uKey);
                if(it == activatedNodes.end()){
                    activatedNodes.push_back(uKey);
                    nextNodeIds.push_back(uId);
                    nextShardIds.push_back(uShardId);
                }
            }
        }

    }
}