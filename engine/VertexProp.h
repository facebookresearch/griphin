#ifndef VERTEXPROP_H
#define VERTEXPROP_H

#include <iostream>
#include <vector>
#include <string>

typedef int VertexType;    
typedef int EdgeType;    
#define PROP_SIZE 200

class VertexProp{
    public:
        VertexProp(int localId, int globalId);
        //VertexProp(VertexType id, std::vector<float> vertexData_, std::vector<int> neighborVertices_);
        VertexType vertexID;
        int shard;
        int neighborCount;
        bool isLocked;
        std::vector<float> vertexData;
        std::vector<VertexType> *neighborVertices = new std::vector<VertexType>;
        std::vector<VertexType> shardOfneighborVertices;
        std::vector<EdgeType> neighborEdges;
        VertexType getNodeId();
        VertexType getShard();
        bool addNeighbor(VertexType neighborId);
        void getNeighbors();
        void setVertexProp(std::vector<float> vertexData, std::vector<VertexType> neighborVertices, std::vector<EdgeType> neighborEdges);
        bool getLocking();
        bool setLocking(bool b);
};


#endif