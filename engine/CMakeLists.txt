cmake_minimum_required(VERSION 3.23)
project(engine)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_executable(engine
        EdgeProp.cpp
        EdgeProp.h
        Graph.cpp
        Graph.h
        VertexProp.cpp
        VertexProp.h
        global.h
        SharedIndices.h
        ppr.h
        ppr.cpp
        bindings.cpp
        )


set(CMAKE_PREFIX_PATH /Users/synapse/opt/anaconda3/envs/pytorch-graph/lib/python3.9/site-packages/torch/share/cmake)
find_package(Torch REQUIRED)

include_directories(.)

target_link_libraries(engine "${TORCH_LIBRARIES}")