#ifndef ENGINE_GLOBAL_H
#define ENGINE_GLOBAL_H

#include <cstdint>
#include <torch/types.h>

typedef int32_t VertexType;
typedef int32_t EdgeType;
typedef int8_t ShardType;
typedef float_t WeightType;

constexpr auto tensorVertexType = torch::kInt32;
constexpr auto tensorEdgeType = torch::kInt32;
constexpr auto tensorShardType = torch::kInt8;
constexpr auto tensorWeightType = torch::kFloat32;

#endif //ENGINE_GLOBAL_H
