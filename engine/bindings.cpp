//#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "Graph.cpp"
#include "EdgeProp.cpp"
#include "VertexProp.cpp"
#include "SharedMemoryVector.cpp"
#include "PPR.cpp"
#include <vector>

namespace py = pybind11;

PYBIND11_MODULE(graph_engine, m) {
    py::class_<Graph<VertexProp, EdgeProp>>(m, "Graph")
    .def(py::init<ShardType, char*>())
    .def("num_core_nodes", &Graph<VertexProp, EdgeProp>::getNumOfCoreVertices)
    .def("sample_single_neighbor", &Graph<VertexProp, EdgeProp>::sampleSingleNeighbor)
    .def("sample_single_neighbor2", &Graph<VertexProp, EdgeProp>::sampleSingleNeighbor2)
    .def("partition_book", &Graph<VertexProp, EdgeProp>::getPartitionBook)
    .def("get_neighbor_lists", &Graph<VertexProp, EdgeProp>::getNeighborLists)
    .def("get_neighbor_infos", &Graph<VertexProp, EdgeProp>::getNeighborInfos, py::call_guard<py::gil_scoped_release>());

    py::class_<PPR>(m, "PPR")
    .def(py::init<VertexType, ShardType, float, float>())
    .def("pop_activated_nodes", &PPR::popActivatedNodes)
    // Release GIL before calling into (potentially long-running) C++ code
    .def("push", &PPR::push, py::call_guard<py::gil_scoped_release>())
    .def("get_p", &PPR::getP);
}
