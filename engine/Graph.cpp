#include <iostream>
#include "Graph.h"

template <class VertexProp, class EdgeProp> 
Graph<VertexProp, EdgeProp>::Graph(int shardID_, char *uniqueIDsList, char *pathToCsrIndPtr, char *pathToCsrIndices, char *pathToVertexData, int coreCount, int haloCount){
    shardID = shardID_;

    numCoreNodes = coreCount;
    numHaloNodes = haloCount;
    numNodes = coreCount + haloCount;

    for(int i = 0; i < numNodes; i++){
        nodeLocalIDs.push_back(i);
    }
    
    string line;

    std::ifstream uniqueIDsFile(uniqueIDsList);
    if(uniqueIDsFile.is_open()){
        while(getline(uniqueIDsFile, line)){
            int num = std::atoi(line);
            nodeGlobalIDs.push_back(num);
        }
        uniqueIDsFile.close();
    }   
    else{
        std::cout << "Unable to open unique IDs file" << std::endl; 
    }

    std::ifstream indPtrFile(pathToCsrIndPtr);
    if(indPtrFile.is_open()){
        while(getline(indPtrFile, line)){
            int num = std::atoi(line);
            indptr.push_back(num);
        }
        indPtrFile.close();
    }   
    else{
        std::cout << "Unable to open indptr file" << std::endl; 
    }

    
    std::ifstream indicesFile(pathToCsrIndices);
    if(indicesFile.is_open()){
        while(getline(indicesFile, line)){
            int num = std::atoi(line);
            indices.push_back(num);
            numEdges ++;            // each index means one edge
        }
        indicesFile.close();
    }   
    else{
        std::cout << "Unable to open indices file" << std::endl; 
    }

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
}

template <class VertexProp, class EdgeProp> 
VertexType Graph<VertexProp, EdgeProp>::findVertex(VertexType globalVertexID){
    auto it = std::find (nodeGlobalIDs.begin(), nodeGlobalIDs.end(), globalVertexID);
    VertexType globalIndex = neighborVertices_.push_back(it - nodeGlobalIDs.begin());
    return localVertexID[globalIndex];
}  


template <class VertexProp, class EdgeProp> 
bool Graph<VertexProp, EdgeProp>::findVertexLocking(VertexType localVertexID){
    vertexProps[localVertexID].getLocking();
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
}

template <class VertexProp, class EdgeProp> 
bool Graph<VertexProp, EdgeProp>::addBatchVertexLocking(std::vector<VertexType> localVertexIDs){
    for(auto it = localVertexIDs.begin(); it != localVertexIDs.end(); ++it){
        vertexProps[*it].setLocking();
    }
}

