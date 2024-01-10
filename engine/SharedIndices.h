// Copyright (c) Meta Platforms, Inc. and affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#ifndef SHARE_MEM_VECTOR2_H
#define SHARE_MEM_VECTOR2_H

#include "global.h"

template<class T>
class SharedIndices{
private:
    EdgeType startIndex;
    EdgeType endIndex;
    T* indicesPtr;
        
public:
    SharedIndices(EdgeType startIndex_, EdgeType endIndex_, T** indicesPtr_);

    T* getPtr();
    T getVal(int index);
    EdgeType size();
};

template<class T>
SharedIndices<T>::SharedIndices(EdgeType startIndex_, EdgeType endIndex_, T **indicesPtr_) {
    startIndex = startIndex_;
    endIndex = endIndex_;
    indicesPtr = *indicesPtr_;
}

template<class T>
T* SharedIndices<T>::getPtr(){
    return &indicesPtr[startIndex];
}

template<class T>
T SharedIndices<T>::getVal(int index){
    return indicesPtr[startIndex + index];
}

template<class T>
EdgeType SharedIndices<T>::size(){
    return endIndex - startIndex;
}

#endif