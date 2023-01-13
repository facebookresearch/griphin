#ifndef EDGEPROP_H
#define EDGEPROP_H

#include <iostream>
#include <string>
#include <ctime>
#include <vector>
#include <cstdint>
#include "global.h"

class EdgeProp{
    public:
        EdgeProp();
        EdgeType edgeID;
        std::vector<float> edgeData;
        std::vector<VertexType> neighborNodes;
};

#endif 