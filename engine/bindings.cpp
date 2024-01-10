// Copyright (c) Meta Platforms, Inc. and affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "Graph.cpp"
#include "EdgeProp.cpp"
#include "VertexProp.cpp"
#include "PPR.cpp"
#include <vector>

namespace py = pybind11;

/* NOTE
 * 1. When creating new .cpp files, include these files in bindings.cpp
 * 2. Release GIL in bindings.cpp before calling into (potentially long-running) C++ code
 */
PYBIND11_MODULE(graph_engine, m) {
    // Graph Storage Class
    py::class_<Graph<VertexProp, EdgeProp>>(m, "Graph")
    .def(py::init<ShardType, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>())

    .def("num_core_nodes", &Graph<VertexProp, EdgeProp>::getNumOfCoreVertices,
         "A function that returns the number of core nodes in current shard")

    .def("sample_single_neighbor", &Graph<VertexProp, EdgeProp>::sampleSingleNeighbor,
         "This function is deprecated")

    .def("sample_single_neighbor2", &Graph<VertexProp, EdgeProp>::sampleSingleNeighbor2,
         "A function that samples single neighbor for each input source node",
         py::call_guard<py::gil_scoped_release>())

    .def("partition_book", &Graph<VertexProp, EdgeProp>::getPartitionBook,
         "A function that returns cluster pointers")

    .def("get_shard_id", &Graph<VertexProp, EdgeProp>::getShardID,
         "A function that returns graph shard ID")

    .def("get_neighbor_lists", &Graph<VertexProp, EdgeProp>::getNeighborLists,
         "This function is for test only")

    .def("get_neighbor_infos", &Graph<VertexProp, EdgeProp>::getNeighborInfos,
         "A function that returns the whole neighborhood for each input source node",
         py::call_guard<py::gil_scoped_release>())

    .def("get_neighbor_infos_local", &Graph<VertexProp, EdgeProp>::getNeighborInfos2,
         "A function that returns the neighborhood for local request",
         py::call_guard<py::gil_scoped_release>())

    .def("get_neighbor_infos_remote", &Graph<VertexProp, EdgeProp>::getNeighborInfos3,
         "A function that returns the whole neighborhood for remote request",
         py::call_guard<py::gil_scoped_release>());

    py::class_<VertexProp, std::shared_ptr<VertexProp>>(m, "VertexProp");

    // PPR Computation Class
    py::class_<PPR>(m, "PPR")
    .def(py::init<VertexType, ShardType, WeightType, WeightType, size_t>())

    .def("pop_activated_nodes", &PPR::popActivatedNodes,
         "A function that returns the activated nodes and clear its corresponding set")

    .def("push", py::overload_cast<
            std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>,
            torch::Tensor,
            torch::Tensor
            >(&PPR::push),
         "A function that updates the Residual and PPR maps according to input source nodes",
         py::call_guard<py::gil_scoped_release>())

    .def("push", py::overload_cast<
                 std::vector<VertexProp>,
                 torch::Tensor,
                 torch::Tensor
         >(&PPR::push),
         "A function that updates the Residual and PPR maps according to input source nodes",
         py::call_guard<py::gil_scoped_release>())

    .def("push", py::overload_cast<
                 std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>,
                 torch::Tensor,
                 torch::Tensor
         >(&PPR::push),
         "A function that updates the Residual and PPR maps according to input source nodes",
         py::call_guard<py::gil_scoped_release>())

    .def("get_p", &PPR::getP,
         "A function that returns the current PPR values");
}
