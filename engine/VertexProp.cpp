#include <iostream>
#include "VertexProp.h"

VertexProp::VertexProp(int vertexID_, int shardID_) {
    vertexID = vertexID_;
    shardID = shardID_;
    neighborCount = 0;
    isLocked = false;
}

/*
VertexProp::VertexProp(VertexType id, std::vector<float> vertexData_=NULL, std::vector<int> neighborVertices_=NULL) : vertexId(id){
    isLocked = false;
    vertexData = vertexData_;
    neighborVertices = neighborVertices_;
}
*/

VertexType VertexProp::getNodeId(){
    return vertexID;
}

int VertexProp::getShard(){
    return shardID;
}

bool VertexProp::addNeighbor(VertexType neighborId, int neighborShardId){
    neighborVertices->push_back(neighborId);
    neighborVerticeShards->push_back(neighborShardId);
    neighborCount ++;
    return true;
}

void VertexProp::getNeighbors(){
    printf("Neighbors of node %d: ", vertexID);
    for(int i = 0; i < neighborCount; i++){
        printf("%d ", (*neighborVertices)[i]);
    }
    printf("\n");
}

void VertexProp::getShardsOfNeighbors(){
    printf("Shards of neighbors of node %d: ", vertexID);
    for(int i = 0; i < neighborCount; i++){
        printf("%d ", (*neighborVerticeShards)[i]);
    }
    printf("\n\n");
}


bool VertexProp::getLocking(){
    return isLocked;
}

bool VertexProp::setLocking(bool b=true){
    isLocked = true;
    return isLocked;
}