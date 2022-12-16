#include <iostream>
#include "VertexProp.h"

VertexProp::VertexProp(VertexType id, std::vector<float> vertexData_, std::vector<int> neighborVertices_) : vertexID(id){
    isLocked = false;
    vertexData = vertexData_;
    neighborVertices = neighborVertices_;
}

bool VertexProp::getLocking(){
    return isLocked;
}

bool VertexProp::setLocking(bool b=true){
    isLocked = true;
    return isLocked;
}
