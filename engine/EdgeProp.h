// Copyright (c) Meta Platforms, Inc. and affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

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