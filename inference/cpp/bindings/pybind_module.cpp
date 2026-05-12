// inference/cpp/bindings/pybind_module.cpp

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../kernels/blas.hpp"

#include "../core/tensor.hpp"
#include "../core/mmap_loader.hpp"

namespace py = pybind11;

using namespace axion;

PYBIND11_MODULE(axion_cpp, m) {

    py::class_<Tensor>(m, "Tensor")

        .def(py::init<>())

        .def_readwrite(
            "name",
            &Tensor::name
        )

        .def_readwrite(
            "shape",
            &Tensor::shape
        )

        .def(
            "numel",
            &Tensor::numel
        )
        .def_readwrite(
            "data",
            &Tensor::data
        )

        .def(
            "print_info",
            &Tensor::print_info
        );


    py::class_<MMapLoader>(m, "MMapLoader")

        .def(py::init<>())

        .def(
            "load_file",
            &MMapLoader::load_file
        )
        .def(
            "load_tensor_data",
            &MMapLoader::load_tensor_data
        )

        .def(
            "load_tensor",
            &MMapLoader::load_tensor
        );
        m.def(
            "matmul",
            &matmul
        );
        
}