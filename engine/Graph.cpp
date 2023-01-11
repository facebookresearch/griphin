#include <iostream>
#include <vector>
#include <omp.h>
#include "Graph.h"
#include "utils.h"

template <class VertexProp, class EdgeProp>
    Graph<VertexProp, EdgeProp>::Graph(int shardID_, char *idsList, char *haloShardsList, char *csrIndicesFile, char *csrShardIndicesFile, char *csrIndPtrsFile, char *partitionBookFile){  // takes shards as the argument
    shardID = shardID_;

    numCoreNodes = 0;
    numHaloNodes = 0;
    numNodes = 0;
    numEdges = 0;

    int dummy = 0;

    std::string line;

    /*
       idsList contains the numbers with the following format:
       for shard 0 -> from 0 to |core vertices in current shard| - 1 and then halo vertices with their original ids in their original shards
       for shard 0 -> from |core vertices in prev. shard| - 1 to |core vertices in current shard| and then halo vertices with their original ids in their original shards
       we initially planned to start from 0 in each shard. However, it required us to keep <shard_id, vertex_id> pair for edges as there might be vertices with the same vertex_id
       however, we can easily change it as current format keeps all the necessary information.
    */
    readFile(partitionBookFile, &partitionBook, &dummy);

    readFile(idsList, &nodeIDs, &numNodes);

    // reads the shard ids of the halo vertices.
    readFile(haloShardsList, &haloNodeShards, &numHaloNodes);

    numCoreNodes = numNodes - numHaloNodes; 

    //int offset = partitionBook[shardID];

    // read the csr indices file
    readFile(csrIndicesFile, &csrIndices, &dummy);

    // read the csr shard indices file
    readFile(csrShardIndicesFile, &csrShardIndices, &dummy);
    
    // read the csr indptrs file
    readFile(csrIndPtrsFile, &csrIndptrs, &dummy);

    for(int i = 0; i < numCoreNodes; i++){
        int neighborStartIndex = csrIndptrs[i];
        int neighborEndIndex = csrIndptrs[i+1];
        vertexProps.push_back(VertexProp(i, shardID, neighborStartIndex, neighborEndIndex));
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
std::vector<int> Graph<VertexProp, EdgeProp>::getNeighbors(VertexType vertexID){
    std::vector<int> neighbors;
    int neighborStartIndex = csrIndptrs[vertexID];
    int neighborEndIndex = csrIndptrs[vertexID+1];

    for(int i = neighborStartIndex; i < neighborEndIndex; i++){
        neighbors.push_back(csrIndices[i]);
    }
    return neighbors;
}

template <class VertexProp, class EdgeProp>
std::vector<int> Graph<VertexProp, EdgeProp>::getNeighborShards(VertexType vertexID){
    std::vector<int> neighborShards;
    int neighborStartIndex = csrIndptrs[vertexID];
    int neighborEndIndex = csrIndptrs[vertexID+1];

    for(int i = neighborStartIndex; i < neighborEndIndex; i++){
        neighborShards.push_back(csrShardIndices[i]);
    }
    return neighborShards;
}

template <class VertexProp, class EdgeProp>
Graph<VertexProp, EdgeProp>::~Graph(){
}

template <class VertexProp, class EdgeProp>
int Graph<VertexProp, EdgeProp>::getNumOfVertices(){
//    printf("Num of Nodes: %d\n", numNodes);
    return numNodes;
}

template <class VertexProp, class EdgeProp>
int Graph<VertexProp, EdgeProp>::getNumOfCoreVertices(){
//    printf("Num of Core Nodes: %d\n", numCoreNodes);
    return numCoreNodes;
}

template <class VertexProp, class EdgeProp>
int Graph<VertexProp, EdgeProp>::getNumOfHaloVertices(){
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
bool Graph<VertexProp, EdgeProp>::addBatchVertexLocking(std::vector<VertexType> localVertexIDs){
    for(auto it = localVertexIDs.begin(); it != localVertexIDs.end(); ++it){
        vertexProps[*it].setLocking();
    }
    return true;
}

template<class VertexProp, class EdgeProp>
std::tuple<torch::Tensor, std::map<int, torch::Tensor>>
Graph<VertexProp, EdgeProp>::sampleSingleNeighbor(const torch::Tensor& srcVertexIDs_) {
    int64_t len = srcVertexIDs_.numel();
    torch::Tensor srcVertexIDs = srcVertexIDs_.contiguous();
    const VertexType* srcVertexPtr = srcVertexIDs.data_ptr<VertexType>();

    auto* sampledVertices_ = new VertexType[len];  // to avoid copy, we need to allocate memory for sampled Vertices
    std::map<int, std::vector<int64_t>*> shardIndexMap_;

    std::random_device r;
    std::default_random_engine e(r());

    // TODO: fine grain parallelization
    for (int64_t i=0; i < len; i++) {
        //printf("source: %d - num of core: %d\n", srcVertexPtr[i], getNumOfCoreVertices());
        VertexProp prop = findVertex(srcVertexPtr[i]);
        int neighborStartIndex = prop.getNeighborStartIndex();
        int neighborEndIndex = prop.getNeighborEndIndex();
        int size = neighborEndIndex - neighborStartIndex;
        //printf("neighbor size: %d \n", neighbors.size());
        // for(int i = 0; i < neighbors.size(); i++){
        //     printf("%d - %d ** ", neighbors[i], neighborShards[i]);
        // }
        //printf("\n");


        int neighborShardID;

        if (size == 0) {
            VertexType neighborID = prop.getNodeId();
            sampledVertices_[i] = neighborID;
            neighborShardID = prop.shardID;
        }
        else{
            std::uniform_int_distribution<int> uniform_dist(0, size-1);
            int rand = uniform_dist(e);

            VertexType neighborID = csrIndices[neighborStartIndex + rand];
            sampledVertices_[i] = neighborID;

            neighborShardID = csrShardIndices[neighborStartIndex + rand];
        }

        //printf("chosen neighbor for i = %d - %d - %d \n\n", i, neighborID, neighborShardID);

        if (shardIndexMap_.find(neighborShardID) == shardIndexMap_.end()) {
            shardIndexMap_[neighborShardID] = new std::vector<int64_t>();  // allocate memory
            //printf("shard not found so creating \n\n");
        }
        shardIndexMap_[neighborShardID]->push_back(i);
    }

    auto opts = srcVertexIDs_.options();
    torch::Tensor sampledVertices = torch::from_blob(
            sampledVertices_, {len}, opts);  // from_blob() does not make a copy
    std::map<int, torch::Tensor> shardIndexMap;
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

    // to avoid copy, we need to allocate memory for sampled Vertices
    auto* localVertexIDs_ = new VertexType[len];
    auto* globalVertexIDs_ = new VertexType[len];
    auto* shardIDs_ = new ShardType[len];

    std::random_device r;
    std::default_random_engine e(r());

    #pragma omp parallel for schedule(static) default(none) shared(e, len, srcVertexPtr, localVertexIDs_, globalVertexIDs_, shardIDs_)
    for (int64_t i=0; i < len; i++) {
        VertexProp prop = findVertex(srcVertexPtr[i]);
        int neighborStartIndex = prop.getNeighborStartIndex();
        int neighborEndIndex = prop.getNeighborEndIndex();
        int size = neighborEndIndex - neighborStartIndex;

        VertexType neighborID;
        ShardType neighborShardID;

        if (size == 0) {
            neighborID = prop.getNodeId();
            neighborShardID = prop.shardID;
        }
        else {
            std::uniform_int_distribution<int> uniform_dist(0, size-1);
            int rand = uniform_dist(e);
//            auto rand = uniform_randint((int)prop.neighborVertices->size());
            neighborID = csrIndices[neighborStartIndex + rand];;
            neighborShardID = csrShardIndices[neighborStartIndex + rand];
        }

        localVertexIDs_[i] = neighborID;
        globalVertexIDs_[i] = neighborID + partitionBook[neighborShardID];
        shardIDs_[i] = neighborShardID;
    }

    auto opts = srcVertexIDs_.options();
    torch::Tensor localVertexIDs = torch::from_blob(localVertexIDs_, {len}, opts);
    torch::Tensor globalVertexIDs = torch::from_blob(globalVertexIDs_, {len}, opts);
    torch::Tensor shardIDs = torch::from_blob(shardIDs_, {len}, opts);

    return {localVertexIDs, globalVertexIDs, shardIDs};
}
