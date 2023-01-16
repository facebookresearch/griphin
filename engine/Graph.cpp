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

    /*
       idsListFile contains the numbers with the following format:
       for shard 0 -> from 0 to |core vertices in current shard| - 1 and then halo vertices with their original ids in their original shards
       for shard 0 -> from |core vertices in prev. shard| - 1 to |core vertices in current shard| and then halo vertices with their original ids in their original shards
       we initially planned to start from 0 in each shard. However, it required us to keep <shard_id, vertex_id> pair for edges as there might be vertices with the same vertex_id
       however, we can easily change it as current format keeps all the necessary information.
    */
    readFile(partitionBookFile, &partitionBook, &dummy);

    readFile(idsListFile, &nodeIDs, &numNodes);

    // reads the shard ids of the halo vertices.
    readFile(haloShardsListFile, &haloNodeShards, &numHaloNodes);

    numCoreNodes = numNodes - numHaloNodes;

    // read the csr indices file
    readFile(csrIndicesFile, &csrIndices, &dummy);

    // read the csr shard indices file
    readFile(csrShardIndicesFile, &csrShardIndices, &dummy);
    
    // read the csr indptrs file
    readFile(csrIndPtrsFile, &csrIndptrs, &dummy);

    for(VertexType i = 0; i < numCoreNodes; i++){
        auto neighborStartIndex = csrIndptrs[i];
        auto neighborEndIndex = csrIndptrs[i+1];
        vertexProps.push_back(VertexProp(i, shardID, neighborStartIndex, neighborEndIndex, csrIndices.data(), csrShardIndices.data()));
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
//    printf("Num of Nodes: %d\n", numNodes);
    return numNodes;
}

template <class VertexProp, class EdgeProp>
int64_t Graph<VertexProp, EdgeProp>::getNumOfCoreVertices(){
//    printf("Num of Core Nodes: %d\n", numCoreNodes);
    return numCoreNodes;
}

template <class VertexProp, class EdgeProp>
int64_t Graph<VertexProp, EdgeProp>::getNumOfHaloVertices(){
//    printf("Num of Halo Nodes: %d\n", numHaloNodes);
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

        printf("chosen neighbor for i = %d - %d - %d \n\n", i, neighborID, neighborShardID);

        if (shardIndexMap_.find(neighborShardID) == shardIndexMap_.end()) {
            shardIndexMap_[neighborShardID] = new std::vector<int64_t>();  // allocate memory
            printf("shard not found so creating \n\n");
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

    allocate memory for sampled vertices to avoid copy
    auto* localVertexIDs_ = new VertexType[len];
    auto* globalVertexIDs_ = new VertexType[len];
    auto* shardIDs_ = new ShardType[len];

    TODO:
    int numThreads;
    if (len > 200) {
        numThreads = 1;
    } else {
        numThreads = 1;
    }

    #pragma omp parallel num_threads(numThreads) default(none) shared(len, srcVertexPtr, localVertexIDs_, globalVertexIDs_, shardIDs_)
    {
       std::random_device dev;
       std::mt19937_64 rng(dev());
        std::mt19937_64 rng((omp_get_thread_num() + 1) * time(nullptr));

        #pragma omp for
        for (int64_t i=0; i < len; i++) {
            VertexProp prop = findVertex(srcVertexPtr[i]);

            VertexType neighborID;
            ShardType neighborShardID;

            if (prop.getNeighborCount() == 0) {
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
