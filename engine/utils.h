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
    return uniform_randint(0, high)
}


#endif //ENGINE_UTILS_H
