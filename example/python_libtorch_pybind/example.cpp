#include <torch/extension.h>  // already import pybind
#include <iostream>
#include <omp.h>

namespace py = pybind11;

torch::Tensor tensor_add(torch::Tensor i, torch::Tensor j) {
    return i + j;
}

int omp_add(int a, int b){
    omp_set_num_threads(4);
    #pragma omp parallel for
    for (int i = 0; i < 10; i ++) {
        a += b;
    }
    return a;
}

PYBIND11_MODULE(python_example, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("tensor_add", &tensor_add, "A function that adds two tensors");
    m.def("omp_add", &omp_add, "A function that adds b in to a for 10 times");
}