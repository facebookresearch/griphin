#ifndef VERTEXPROP_H
#define VERTEXPROP_H

#include <iostream>
#include <string>

typedef int VertexType;    
typedef int EdgeType;    
#define SIZE 200

class VertexProp{
    public:
        VertexProp(VertexType id, std::vector<float> vertexData_, std::vector<int> neighborVertices_);
        VertexType vertexID;
        bool isLocked;
        std::vector<float> vertexData;
        std::vector<VertexType> neighborVertices;
        std::vector<EdgeType> neighborEdges;
        void setVertexProp(std::vector<float> vertexData, std::vector<VertexType> neighborVertices, std::vector<EdgeType> neighborEdges=NULL);
        bool getLocking();
        bool setLocking(bool b=True);
};


#endif