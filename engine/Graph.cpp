#include <iostream>
#include "Graph.h"

template <class VertexProp, class EdgeProp> 
void Graph<VertexProp, EdgeProp>::readFile(char *fileName, std::vector<VertexType> *vec, int *counter){
    std::string line;

    std::ifstream file(fileName);
    if(file.is_open()){
        while(getline(file, line)){
            (*counter) ++;
            (*vec).push_back(std::atoi(line.c_str()));
        }
        file.close();
    }   
    else{
        std::cout << "Unable to open the file!" << std::endl; 
    }
}

template <class VertexProp, class EdgeProp> 
Graph<VertexProp, EdgeProp>::Graph(int shardID_, char *idsList, char *haloShardsList,  char *pathToCooRow, char *pathToCooColumn){  // takes shards as the argument
    shardID = shardID_;

    numCoreNodes = 0;
    numHaloNodes = 0;
    numNodes = 0;

    std::string line;

    /*
       idsList contains the numbers with the following format: 
       for shard 0 -> from 0 to |core vertices in current shard| - 1 and then halo vertices with their original ids in their original shards
       for shard 0 -> from |core vertices in prev. shard| - 1 to |core vertices in current shard| and then halo vertices with their original ids in their original shards
       we initially planned to start from 0 in each shard. However, it required us to keep <shard_id, vertex_id> pair for edges as there might be vertices with the same vertex_id 
       however, we can easily change it as current format keeps all the necessary information.
    */
    readFile(idsList, &nodeIDs, &numNodes);
    
    // reads the shard ids of the vertices.
    // i didn't store the shard id of the core ones as shardID variable holds it.
    readFile(haloShardsList, &haloNodeShards, &numHaloNodes);

    numCoreNodes = numNodes - numHaloNodes; 

    for(int i=0; i < nodeIDs.size(); i++){
        if(i < numCoreNodes)
            vertexProps.push_back(VertexProp(nodeIDs[i], shardID));
        else    
            vertexProps.push_back(VertexProp(nodeIDs[i], haloNodeShards[i-numCoreNodes]));
    }

    // read the source nodes file
    int dummy = 0;
    readFile(pathToCooRow, &cooRow, &dummy);

    // read the dest nodes file    
    readFile(pathToCooColumn, &cooCol, &numEdges);
    
    for(int i = 0; i < numEdges; i++){
        for(int j = 0; j < numNodes; j++){
            if(cooRow[i] == nodeIDs[j]){
                if(cooCol[i] >= nodeIDs[0] && cooCol[i] <= nodeIDs[numCoreNodes-1]){
                    vertexProps[j].addNeighbor(cooCol[i], shardID);
                }
                else{
                    std::vector<int>::iterator it; 
                    it = std::find(nodeIDs.begin(), nodeIDs.end(), cooCol[i]);
                    int index = it - nodeIDs.begin() - numCoreNodes;
                    vertexProps[j].addNeighbor(cooCol[i], haloNodeShards[index]);
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


template <class VertexProp, class EdgeProp> 
VertexProp Graph<VertexProp, EdgeProp>::findVertex(int vertexID){
    std::vector<int>::iterator it; 
    it = std::find(nodeIDs.begin(), nodeIDs.end(), vertexID);
    int index = it - nodeIDs.begin();
    return vertexProps[index];
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

