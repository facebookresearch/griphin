#ifndef EDGEPROP_H
#define EDGEPROP_H

#include <iostream>
#include <string>
#include <ctime>
#include <vector>
#include <stdint.h>

typedef int EdgeType;   
typedef int VertexType;   

class EdgeProp{
    public:
        EdgeProp();
        EdgeType edgeID;
        std::vector<float> edgeData;
        std::vector<VertexType> neighborNodes;
};


#endif 