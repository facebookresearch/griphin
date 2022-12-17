#include <iostream>
#include "Graph.h"

template <class VertexProp, class EdgeProp> 
Graph<VertexProp, EdgeProp>::Graph(int shardID_, char *idsList, char *haloShardsList,  char *pathToCooRow, char *pathToCooColumn){  // takes shards as the argument
    shardID = shardID_;

    numCoreNodes = 0;
    numHaloNodes = 0;
    numNodes = 0;

    
    std::string line;

    std::ifstream idsFile1(idsList);
    if(idsFile1.is_open()){
        while(getline(idsFile1, line)){
            numNodes ++;
        }
        idsFile1.close();
    }   
    else{
        std::cout << "Unable to open unique IDs file" << std::endl; 
    }

    std::ifstream shardsFile(haloShardsList);
    if(shardsFile.is_open()){
        while(getline(shardsFile, line)){
            numHaloNodes ++;
            int num = std::atoi(line.c_str());
            haloNodeShards.push_back(num);
        }
        shardsFile.close();
    }   
    else{
        std::cout << "Unable to open shards file" << std::endl; 
    }

    numCoreNodes = numNodes - numHaloNodes; 

    std::ifstream idsFile2(idsList);
    if(idsFile2.is_open()){
        while(getline(idsFile2, line)){
            int num = std::atoi(line.c_str());
            nodeIDs.push_back(num);
        }
        idsFile2.close();
    }   
    else{
        std::cout << "Unable to open unique IDs file" << std::endl; 
    }

    std::ifstream uniqueIDsFile(idsList);
    if(uniqueIDsFile.is_open()){
        while(getline(uniqueIDsFile, line)){
            int num = std::atoi(line.c_str());
            nodeIDs.push_back(num);
        }
        uniqueIDsFile.close();
    }   
    else{
        std::cout << "Unable to open unique IDs file" << std::endl; 
    }

    for(int i=0; i < nodeIDs.size(); i++){
        if(i < numCoreNodes)
            vertexProps.push_back(VertexProp(nodeIDs[i], shardID));
        else    
            vertexProps.push_back(VertexProp(nodeIDs[i], haloNodeShards[i-numCoreNodes]));
    }


    std::ifstream cooRowFile(pathToCooRow);
    if(cooRowFile.is_open()){
        while(getline(cooRowFile, line)){
            int num = std::atoi(line.c_str());
            cooRow.push_back(num);
        }
        cooRowFile.close();
    }   
    else{
        std::cout << "Unable to open indptr file" << std::endl; 
    }

    
    std::ifstream cooColFile(pathToCooColumn);
    if(cooColFile.is_open()){
        while(getline(cooColFile, line)){
            int num = std::atoi(line.c_str());
            cooCol.push_back(num);
            numEdges ++;            // each index means one edge
        }
        cooColFile.close();
    }   
    else{
        std::cout << "Unable to open indices file" << std::endl; 
    }

    for(int i = 0; i < numEdges; i++){
        for(int j = 0; j < numNodes; j++){
            if(cooRow[i] == nodeIDs[j]){
                vertexProps[j].addNeighbor(cooCol[i]);
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
int Graph<VertexProp, EdgeProp>::findVertex(int vertexID){
    std::vector<int>::iterator it; 
    it = std::find(nodeIDs.begin(), nodeIDs.end(), vertexID);
    if(it == nodeIDs.end()){
        printf("ID %d is not in this shard.\n", vertexID);
        return -1;
    }
    printf("ID %d is found.\n", vertexID);
    int index = it - nodeIDs.begin();
    return vertexID;
} 



template <class VertexProp, class EdgeProp> 
Graph<VertexProp, EdgeProp>::~Graph(){
}

template <class VertexProp, class EdgeProp> 
int Graph<VertexProp, EdgeProp>::getNumOfHaloVertices(){
    printf("Num of Halo Nodes: %d\n", numHaloNodes);
    return numHaloNodes;
}

template <class VertexProp, class EdgeProp> 
int Graph<VertexProp, EdgeProp>::getNumOfVertices(){
    printf("Num of Nodes: %d\n", numNodes);
    return numNodes;
}

template <class VertexProp, class EdgeProp> 
bool Graph<VertexProp, EdgeProp>::findVertexLocking(VertexType localVertexID){
    vertexProps[localVertexID].getLocking();
    return true;
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

