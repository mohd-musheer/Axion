// inference/cpp/bindings/pybind_module.cpp

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../kernels/blas.hpp"
#include "../kernels/rmsnorm.hpp"
#include "../core/tensor.hpp"
#include "../runtime/transformer_layer.hpp"
#include "../core/mmap_loader.hpp"
#include "../kernels/transpose.hpp"
#include "../kernels/attention.hpp"
#include "../kernels/softmax.hpp"
#include "../kernels/silu.hpp"
#include "../kernels/elementwise.hpp"
#include "../kernels/mlp.hpp"
#include "../kernels/attention_output.hpp"
#include "../kernels/rope.hpp"
#include "../kernels/residual.hpp"
#include "../runtime/generation.hpp"
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
            "list_tensors",
            &MMapLoader::list_tensors
        )        

        .def(
            "load_tensor",
            &MMapLoader::load_tensor
        );
        m.def(
            "matmul",
            &matmul
        );
        m.def(
            "rmsnorm",
            &rmsnorm
        );
        m.def(
            "apply_rope",
            &apply_rope
        );
        
        m.def(
            "transpose",
            &transpose
        );
            
        m.def(
            "attention_scores",
            &attention_scores
        );

        m.def(
            "softmax",
            &softmax
        );
        
        m.def(
            "attention_output",
            &attention_output
        );

        m.def(
            "residual_add",
            &residual_add
        );
        
        m.def(
            "silu",
            &silu
        );

        m.def(
            "elementwise_mul",
            &elementwise_mul
        );

        m.def(
            "mlp_block",
            &mlp_block
        );

        m.def(
            "transformer_layer",
            &transformer_layer
        );

        m.def(
            "generate_tokens",
            &generate_tokens
        );
        
}