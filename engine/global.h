// Copyright (c) Meta Platforms, Inc. and affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#ifndef ENGINE_GLOBAL_H
#define ENGINE_GLOBAL_H

#include <cstdint>
#include <torch/types.h>

// Data types in frontend/graph.py, data_generation/gen_engine_data.py,
// and engine/global.h should be consistent

typedef int32_t VertexType;
typedef int64_t EdgeType;
typedef int8_t ShardType;
typedef float_t WeightType;

constexpr auto tensorVertexType = torch::kInt32;
constexpr auto tensorEdgeType = torch::kInt64;
constexpr auto tensorShardType = torch::kInt8;
constexpr auto tensorWeightType = torch::kFloat32;

#endif //ENGINE_GLOBAL_H
