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
        VertexProp(int vertexID_, int shardID_); 
        //VertexProp(VertexType id, std::vector<float> vertexData_, std::vector<int> neighborVertices_);
        VertexType vertexID;
        int shardID;
        int neighborCount;
        bool isLocked;
        std::vector<float> vertexData;
        std::vector<VertexType> *neighborVertices = new std::vector<VertexType>;
        std::vector<int> *neighborVerticeShards = new std::vector<int>;
        std::vector<VertexType> shardOfneighborVertices;
        std::vector<EdgeType> neighborEdges;
        VertexType getNodeId();
        int getShard();
        bool addNeighbor(VertexType neighborId, int neighborShardId);
        void getNeighbors();
        void getShardsOfNeighbors();
        void setVertexProp(std::vector<float> vertexData, std::vector<VertexType> neighborVertices, std::vector<EdgeType> neighborEdges);
        bool getLocking();
        bool setLocking(bool b);
};


#endif