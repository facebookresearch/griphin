// Copyright (c) Meta Platforms, Inc. and affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#ifndef ENGINE_UTILS_H
#define ENGINE_UTILS_H

#include <random>

inline int uniform_randint(int low, int high) {
    std::random_device r;
    std::default_random_engine e(r());
    std::uniform_int_distribution<int> uniform_dist(low, high-1);
    return uniform_dist(e);
}

inline int uniform_randint(int high) {
    return uniform_randint(0, high);
}

inline void countLineNumber(const char *fileName, int64_t *counter){
    std::string line;

    std::ifstream file(fileName);
    if(file.is_open()){
        while(getline(file, line)){
            (*counter) ++;
        }
        file.close();
    }
    else{
        std::cout << "Unable to open the file:" << fileName << std::endl;
    }
}

template <class T>
inline void readFile(const char *fileName, T **vec, int mode = 0){ // mode 0 is int, mode 1 is float
    std::string line;
    int i = 0;

    std::ifstream file(fileName);
    if(file.is_open()){
        while(getline(file, line)){
            if(mode == 0)
                (*vec)[i] = std::atoi(line.c_str());
            else
                (*vec)[i] = std::atof(line.c_str());
            i ++;
        }
        file.close();
    }
    else{
        std::cout << "Unable to open the file:" << fileName << std::endl;
    }
}

#endif //ENGINE_UTILS_H
