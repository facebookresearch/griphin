#include <iostream>
#include "VertexProp.h"

VertexProp::VertexProp(int nodeId, int shard_) {
    vertexID = nodeId;
    shard = shard_;
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

VertexType VertexProp::getShard(){
    return shard;
}

bool VertexProp::addNeighbor(VertexType neighborId){
    neighborVertices->push_back(neighborId);
    return true;
}

void VertexProp::getNeighbors(){
    for(int i = 0; i < neighborCount; i++){
        printf("%d ", (*neighborVertices)[i]);
    }
    printf("\n");
}



bool VertexProp::getLocking(){
    return isLocked;
}

bool VertexProp::setLocking(bool b=true){
    isLocked = true;
    return isLocked;
}