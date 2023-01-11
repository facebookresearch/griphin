//#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "Graph.cpp"
#include "EdgeProp.cpp"
#include "VertexProp.cpp"

namespace py = pybind11;

PYBIND11_MODULE(graph_engine, m) {
    py::class_<Graph<VertexProp, EdgeProp>>(m, "Graph")
    .def(py::init<int, char*, char*, char*, char*, char*>())
    .def("num_core_nodes", &Graph<VertexProp, EdgeProp>::getNumOfCoreVertices)
    .def("sample_single_neighbor", &Graph<VertexProp, EdgeProp>::sampleSingleNeighbor)
    .def("partition_book", &Graph<VertexProp, EdgeProp>::getPartitionBook);
}
