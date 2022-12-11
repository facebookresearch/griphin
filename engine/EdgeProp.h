#ifndef EDGEPROP_H
#define EDGEPROP_H

#include <iostream>
#include <string>
#include <ctime>
#include <stdint.h>

typedef int EdgeType;   
typedef int VertexType;   

class EdgeProp{
    public:
        EdgeType edgeID;
        std::vector<float> edgeData;
        std::vector<VertexType> neighborNodes;
};


#endif 