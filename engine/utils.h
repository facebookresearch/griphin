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

inline void readFile(char *fileName, std::vector<VertexType> *vec, int *counter){
    std::string line;

    std::ifstream file(fileName);
    if(file.is_open()){
        while(getline(file, line)){
            (*counter) ++;
            (*vec).push_back(std::atoi(line.c_str()));
        }
        file.close();
    }
    else{
        std::cout << "Unable to open the file!" << std::endl;
    }
}

#endif //ENGINE_UTILS_H
