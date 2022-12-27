#include <iostream>
#include <vector>
#include "Graph.h"
#include "utils.h"

template <class VertexProp, class EdgeProp>
Graph<VertexProp, EdgeProp>::Graph(int shardID_, char *idsList, char *haloShardsList,  char *pathToCooRow, char *pathToCooColumn, char *partitionBookFile){  // takes shards as the argument
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
    readFile(idsList, &nodeIDs, &numNodes);

    readFile(partitionBookFile, &partitionBook, &dummy);

    // reads the shard ids of the vertices.
    // i didn't store the shard id of the core ones as shardID variable holds it.
    readFile(haloShardsList, &haloNodeShards, &numHaloNodes);

    numCoreNodes = numNodes - numHaloNodes; 

    int offset = partitionBook[shardID];

    for(int i = numCoreNodes; i < numNodes; i++){
        int tempShard = haloNodeShards[i-numCoreNodes];
        int tempOffset = partitionBook[tempShard];
        haloNodeRemoteLocalID.push_back(nodeIDs[i] - tempOffset);
    }

    for(int i=0; i < numCoreNodes; i++){
        vertexProps.push_back(VertexProp(i, shardID));
    }

    /*
    for(int i=0; i < nodeIDs.size(); i++){
        if(i < numCoreNodes)
            vertexProps.push_back(VertexProp(nodeIDs[i], shardID));
        else    
            vertexProps.push_back(VertexProp(nodeIDs[i], haloNodeShards[i-numCoreNodes]));
    }
    */

    // read the source nodes file
    readFile(pathToCooRow, &cooRow, &dummy);

    // read the dest nodes file    
    readFile(pathToCooColumn, &cooCol, &numEdges);
    
    for(int i = 0; i < numEdges; i++){
        for(int j = 0; j < numCoreNodes; j++){
            if(cooRow[i] == nodeIDs[j]){
                if(cooCol[i] >= nodeIDs[0] && cooCol[i] <= nodeIDs[numCoreNodes-1]){
                    //printf("if 1\n");
                    vertexProps[j].addNeighbor(cooCol[i] - offset, shardID);
                }
                else{
                    std::vector<int>::iterator it; 
                    it = std::find(nodeIDs.begin(), nodeIDs.end(), cooCol[i]);
                    int index = it - nodeIDs.begin() - numCoreNodes;
                    int tempShard = haloNodeShards[index];
                    int tempOffset = partitionBook[tempShard];
                    vertexProps[j].addNeighbor(cooCol[i] - tempOffset, tempShard);
                }
            }
        }
    }

/*
    std::ifstream vertexDataFile(pathToVertexData);         // data format in txt file is 2d array
    if(vertexDataFile.is_open()){
        for(int i = 0; i < numCoreNodes; i++){
            std::vector<float> vertexData;
            for(int j = 0; j < SIZE; j++){
                vertexData.push_back(std::atoi(file));
            }
            int startIndex = indptr[i];
            int endIndex = indptr[i+1];

            std::vector<int> neighborVertices_;
            std::vector<int>::iterator it;
            for(int k = startIndex; k < endIndex; k++){
                int neighborGlobal = indices[k];
                auto it = std::find (nodeGlobalIDs.begin(), nodeGlobalIDs.end(), neighborGlobal);
                neighborVertices_.push_back(it - nodeGlobalIDs.begin());
            }
            vertexProps.push_back(VertexProp(i, vertexData, neighborVertices_));
        }
        indicesFile.close();
    }   
    else{
        std::cout << "Unable to open vertex data file" << std::endl; 
    }
*/
}

template<class VertexProp, class EdgeProp>
std::vector<VertexType> Graph<VertexProp, EdgeProp>::getPartitionBook() {
    return partitionBook;
}

template <class VertexProp, class EdgeProp> 
VertexProp Graph<VertexProp, EdgeProp>::findVertex(int vertexID){
    // std::vector<int>::iterator it;
    // it = std::find(nodeIDs.begin(), nodeIDs.end(), vertexID);
    // int index = it - nodeIDs.begin();
    // printf("index %d\n", index);
    return vertexProps[vertexID];
} 

template <class VertexProp, class EdgeProp>
Graph<VertexProp, EdgeProp>::~Graph(){
}

template <class VertexProp, class EdgeProp>
int Graph<VertexProp, EdgeProp>::getNumOfVertices(){
    printf("Num of Nodes: %d\n", numNodes);
    return numNodes;
}

template <class VertexProp, class EdgeProp>
int Graph<VertexProp, EdgeProp>::getNumOfCoreVertices(){
    printf("Num of Core Nodes: %d\n", numCoreNodes);
    return numCoreNodes;
}

template <class VertexProp, class EdgeProp>
int Graph<VertexProp, EdgeProp>::getNumOfHaloVertices(){
    printf("Num of Halo Nodes: %d\n", numHaloNodes);
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

    // TODO: fine grain parallelization
    for (int64_t i=0; i < len; i++) {
        VertexProp prop = findVertex(srcVertexPtr[i]);
        auto rand = uniform_randint((int)prop.neighborVertices->size());

        VertexType neighborID = (*prop.neighborVertices)[rand];
        sampledVertices_[i] = neighborID;

        auto neighborShardID = (*prop.neighborVerticeShards)[rand];
        if (shardIndexMap_.find(neighborShardID) == shardIndexMap_.end()) {
            shardIndexMap_[neighborShardID] = new std::vector<int64_t>();  // allocate memory
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

