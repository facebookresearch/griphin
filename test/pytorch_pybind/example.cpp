// Copyright (c) Meta Platforms, Inc. and affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/extension.h>  // already import pybind
#include <iostream>
#include <omp.h>

namespace py = pybind11;

torch::Tensor tensor_add(torch::Tensor i, torch::Tensor j) {
    return i + j;
}

PYBIND11_MODULE(python_example, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("tensor_add", &tensor_add, "A function that adds two tensors");
}