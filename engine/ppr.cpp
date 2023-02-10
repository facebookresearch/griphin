#include "ppr.h"

PPR::PPR(VertexType targetId_, ShardType shardId_, float alpha_, float epsilon_){
    targetId = targetId_;
    shardId = shardId_;
    alpha = alpha_;
    epsilon = epsilon_;

    r[std::make_pair(targetId, shardId)] = 1.;
    activatedNodes[std::make_pair(targetId, shardId)] = std::make_tuple(targetId, shardId);
    // nextNodeIds.push_back(targetId);
    // nextShardIds.push_back(shardId);
}

VertexType PPR::getTargetId(){
    return targetId;
}

ShardType PPR::getShardId(){
    return shardId;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> PPR::getP(){
    std::vector<VertexType> nodeIds;
    std::vector<ShardType> shardIds;
    std::vector<float> values;

    for(std::map<std::pair<VertexType,ShardType>, float>::iterator it = p.begin(); it != p.end(); ++it) {
        nodeIds.push_back(std::get<0>(it->first));
        shardIds.push_back(std::get<1>(it->first));
        values.push_back(it->second);
    }

    int64_t len = nodeIds.size();

    auto opts = torch::TensorOptions().dtype(torch::kInt32);
    torch::Tensor torchNodeIds = torch::from_blob(nodeIds.data(), {len}, opts);
    torch::Tensor torchShardIds = torch::from_blob(shardIds.data(), {len}, opts);

    auto opts = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor torchValues = torch::from_blob(values.data(), {len}, opts);

    return std::make_tuple(torchNodeIds, torchShardIds, torchValues);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> PPR::getR(){
    std::vector<VertexType> nodeIds;
    std::vector<ShardType> shardIds;
    std::vector<float> values;

    for(std::map<std::pair<VertexType,ShardType>>, float>::iterator it = r.begin(); it != r.end(); ++it) {
        nodeIds.push_back(std::get<0>(it->first));
        shardIds.push_back(std::get<1>(it->first));
        values.push_back(it->second);
    }

    int64_t len = nodeIds.size();

    auto opts = torch::TensorOptions().dtype(torch::kInt32);
    torch::Tensor torchNodeIds = torch::from_blob(nodeIds.data(), {len}, opts);
    torch::Tensor torchShardIds = torch::from_blob(shardIds.data(), {len}, opts);

    auto opts = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor torchValues = torch::from_blob(values.data(), {len}, opts);

    return std::make_tuple(torchNodeIds, torchShardIds, torchValues);
}

std::tuple<torch::Tensor, torch::Tensor> PPR::getActivatedNodes(){
    std::vector<VertexType> nodeIds;
    std::vector<ShardType> shardIds;

    for(std::map<std::pair<VertexType, ShardType>, std::tuple<VertexType, ShardType>>::iterator it = activatedNodes.begin(); it != activatedNodes.end(); ++it) {
        nodeIds.push_back(std::get<0>(it->second));
        shardIds.push_back(std::get<1>(it->second));
    }

    int64_t len = nodeIds.size();
    auto opts = torch::TensorOptions().dtype(torch::kInt32);

    torch::Tensor torchNodeIds = torch::from_blob(nodeIds.data(), {len}, opts);
    torch::Tensor torchShardIds = torch::from_blob(shardIds.data(), {len}, opts);

    return std::make_tuple(torchNodeIds, torchShardIds);
}


std::tuple<torch::Tensor, torch::Tensor> PPR::pop_activated_nodes(){
    std::vector<VertexType> nodeIds;
    std::vector<ShardType> shardIds;

    for(std::map<std::pair<VertexType, ShardType>, std::tuple<VertexType, ShardType>>::iterator it = activatedNodes.begin(); it != activatedNodes.end(); ++it) {
        nodeIds.push_back(std::get<0>(it->second));
        shardIds.push_back(std::get<1>(it->second));
    }

    activatedNodes.clear();
    //nextNodeIds.clear();
    //nextShardIds.clear();

    int64_t len = nodeIds.size();
    auto opts = torch::TensorOptions().dtype(torch::kInt32);

    torch::Tensor torchNodeIds = torch::from_blob(nodeIds.data(), {len}, opts);
    torch::Tensor torchShardIds = torch::from_blob(shardIds.data(), {len}, opts);

    return std::make_tuple(torchNodeIds, torchShardIds);
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
        torch::Tensor uVals = (1 - alpha) * r[vKey] * uWeights / uWeights.sum(std::vector<int64_t>({0, uWeights.sizes()[0]}))
        r[vKey] = 0.;
        activatedNodes.erase(std::find(activatedNodes.begin(),activatedNodes.end(), vKey));

        auto uSize = uIds.sizes()[0];

        for(int j = 0; j < uSize; j++){
            auto val = uVals[j];
            auto uId = uIds[j];
            auto uShardId = uShardIds[j];
            auto uDegree = uDegrees[j];

            auto uKey = std::make_pair(uId, uShardId);

            if(j == 0)
                r[uKey] = 0;

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