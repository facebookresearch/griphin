#include <torch/extension.h>  // already import pybind
#include <iostream>

namespace py = pybind11;

torch::Tensor add(torch::Tensor i, torch::Tensor j) {
    return i + j;
}

PYBIND11_MODULE(python_example, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function that adds two numbers");
}