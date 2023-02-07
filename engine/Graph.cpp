#include <iostream>
#include <vector>
#include <ctime>
#include <omp.h>
#include "Graph.h"
#include "utils.h"

template <class VertexProp, class EdgeProp>
    Graph<VertexProp, EdgeProp>::Graph(const ShardType shardID_,
                                       const char *idsListFile,
                                       const char *haloShardsListFile,
                                       const char *csrIndicesFile,
                                       const char *csrShardIndicesFile,
                                       const char *csrIndPtrsFile,
                                       const char *partitionBookFile
                                       ){
    shardID = shardID_;

    numCoreNodes = 0;
    numHaloNodes = 0;
    numNodes = 0;
    numEdges = 0;

    int64_t dummy = 0;

    std::string line;

    readFile(partitionBookFile, &partitionBook, &dummy);

    readFile(idsListFile, &nodeIDs, &numNodes);

    // reads the shard ids of the halo vertices.
    readFile(haloShardsListFile, &haloNodeShards, &numHaloNodes);

    numCoreNodes = numNodes - numHaloNodes;

    // read the csr indices file
    readFile(csrIndicesFile, &csrIndices, &numEdges);

    // read the csr shard indices file
    readFile(csrShardIndicesFile, &csrShardIndices, &dummy);
    
    // read the csr indptrs file
    readFile(csrIndPtrsFile, &csrIndptrs, &dummy);

    std::random_device rd;

    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(0, 1);

    for (int n = 0; n < numEdges; ++n) {
        edgeWeights.push_back(dist(e2));
    }

    for(VertexType i = 0; i < numCoreNodes; i++){
        auto neighborStartIndex = csrIndptrs[i];
        auto neighborEndIndex = csrIndptrs[i+1];

        float temp = 0;
        for(int n = neighborStartIndex; n < neighborEndIndex; n++){
            temp += edgeWeights[n];
        }
        weightedDegrees.push_back(temp);
        vertexProps.push_back(VertexProp(i, shardID, neighborStartIndex, neighborEndIndex, temp, edgeWeights.data(), csrIndices.data(), csrShardIndices.data()));
    }
}

template<class VertexProp, class EdgeProp>
std::vector<VertexType> Graph<VertexProp, EdgeProp>::getPartitionBook() {
    return partitionBook;
}

template <class VertexProp, class EdgeProp> 
VertexProp Graph<VertexProp, EdgeProp>::findVertex(VertexType vertexID){
    return vertexProps[vertexID];
} 

template <class VertexProp, class EdgeProp>
Graph<VertexProp, EdgeProp>::~Graph(){
}

template <class VertexProp, class EdgeProp>
int64_t Graph<VertexProp, EdgeProp>::getNumOfVertices(){
    return numNodes;
}

template <class VertexProp, class EdgeProp>
int64_t Graph<VertexProp, EdgeProp>::getNumOfCoreVertices(){
    return numCoreNodes;
}

template <class VertexProp, class EdgeProp>
int64_t Graph<VertexProp, EdgeProp>::getNumOfHaloVertices(){
    return numHaloNodes;
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
    return true; // to be implemented
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
            neighborID = prop.getNodeId();
            neighborShardID = prop.shardID;
        }
        else{
            std::uniform_int_distribution<int> uniform_dist(0, size-1);
            auto rand = uniform_dist(e);

            neighborID = prop.getNeighbor(rand);
            neighborShardID = prop.getShard(rand);
        }

        sampledVertices_[i] = neighborID;

        //printf("chosen neighbor for i = %d - %d - %d \n\n", i, neighborID, neighborShardID);

        if (shardIndexMap_.find(neighborShardID) == shardIndexMap_.end()) {
            shardIndexMap_[neighborShardID] = new std::vector<int64_t>();  // allocate memory
            //printf("shard not found so creating \n\n");
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
Graph<VertexProp, EdgeProp>::sampleSingleNeighbor2(const torch::Tensor& srcVertexIDs_) {
    int64_t len = srcVertexIDs_.numel();
    torch::Tensor srcVertexIDs = srcVertexIDs_.contiguous();
    const VertexType* srcVertexPtr = srcVertexIDs.data_ptr<VertexType>();

    // allocate memory for sampled vertices to avoid copy
    auto* localVertexIDs_ = new VertexType[len];
    auto* globalVertexIDs_ = new VertexType[len];
    auto* shardIDs_ = new ShardType[len];

    // TODO: Dynamically control num_threads will lead to poor performance
    #pragma omp parallel default(none) shared(len, srcVertexPtr, localVertexIDs_, globalVertexIDs_, shardIDs_)
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
                neighborID = prop.getNodeId();
                neighborShardID = prop.shardID;
            }
            else {
                std::uniform_int_distribution<int> uniform_dist(0, size-1);
                auto rand = uniform_dist(rng);

                neighborID = prop.getNeighbor(rand);
                neighborShardID = prop.getShard(rand);
            }

            localVertexIDs_[i] = neighborID;
            globalVertexIDs_[i] = neighborID + partitionBook[neighborShardID];
            shardIDs_[i] = neighborShardID;
        }
    }

    auto opts = srcVertexIDs_.options();
    torch::Tensor localVertexIDs = torch::from_blob(localVertexIDs_, {len}, opts);
    torch::Tensor globalVertexIDs = torch::from_blob(globalVertexIDs_, {len}, opts);
    torch::Tensor shardIDs = torch::from_blob(shardIDs_, {len}, torch::kInt8);  // hard code

    return {localVertexIDs, globalVertexIDs, shardIDs};
}

template<class VertexProp, class EdgeProp>
std::vector<torch::Tensor> Graph<VertexProp, EdgeProp>::getNeighborLists(const torch::Tensor &srcVertexIDs_) {
    int64_t len = srcVertexIDs_.numel();
    torch::Tensor srcVertexIDs = srcVertexIDs_.contiguous();
    const VertexType* srcVertexPtr = srcVertexIDs.data_ptr<VertexType>();

    std::vector<torch::Tensor> res(len, torch::Tensor());
    auto opts = srcVertexIDs_.options();
    for (auto i=0; i<len; i++) {
        auto prop = findVertex(srcVertexPtr[i]);
        res[i] = torch::from_blob(prop.getIndicesPtr(), {prop.getNeighborCount()}, opts);
    }

    return res;
}
