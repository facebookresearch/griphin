//#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "Graph.cpp"
#include "EdgeProp.cpp"
#include "VertexProp.cpp"
#include "SharedMemoryVector.cpp"
#include <vector>
#include <omp.h>

namespace py = pybind11;

void omp_test(){
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> prod;

    for (int i = 0; i < 5000000; i++) {
        double r1 = ((double)rand() / double(RAND_MAX)) * 5;
        double r2 = ((double)rand() / double(RAND_MAX)) * 5;
        x.push_back(r1);
        y.push_back(r2);
    }

    int len = x.size();
    double t1 = omp_get_wtime();
    #pragma omp parallel for
    for (int i = 0; i < len; i++)
        for (int j=0; j < 1000; j++)
            x[i] = x[i] * y[i];
    double t2 = omp_get_wtime();
    std::cout << "TIME FOR OMP_ADD: " << t2 - t1 << std::endl;
    return;
}

PYBIND11_MODULE(graph_engine, m) {
    py::class_<Graph<VertexProp, EdgeProp>>(m, "Graph")
    .def(py::init<ShardType, char*, char*, char*, char*, char*, char*>())
    .def("num_core_nodes", &Graph<VertexProp, EdgeProp>::getNumOfCoreVertices)
    .def("sample_single_neighbor", &Graph<VertexProp, EdgeProp>::sampleSingleNeighbor)
    .def("sample_single_neighbor2", &Graph<VertexProp, EdgeProp>::sampleSingleNeighbor2)
    .def("partition_book", &Graph<VertexProp, EdgeProp>::getPartitionBook)
    .def("get_neighbor_lists", &Graph<VertexProp, EdgeProp>::getNeighborLists)
    .def("get_neighbor_infos", &Graph<VertexProp, EdgeProp>::getNeighborInfos);

    m.def("omp_add", &omp_test, "A function that adds b in to a for 10 times");
}
