cmake_minimum_required(VERSION 3.23)
project(engine)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(engine
        EdgeProp.cpp
        EdgeProp.h
        Graph.cpp
        Graph.h
        VertexProp.cpp
        VertexProp.h)
