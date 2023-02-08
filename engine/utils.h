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

template <class T>
inline void readFile(const char *fileName, std::vector<T> *vec, int64_t *counter, int mode = 0){ // mode 0 is int, mode 1 is float
    std::string line;

    std::ifstream file(fileName);
    if(file.is_open()){
        while(getline(file, line)){
            (*counter) ++;
            if(mode == 0)
                (*vec).push_back(std::atoi(line.c_str()));
            else
                (*vec).push_back(std::atof(line.c_str()));
        }
        file.close();
    }
    else{
        std::cout << "Unable to open the file!" << std::endl;
    }
}

#endif //ENGINE_UTILS_H
