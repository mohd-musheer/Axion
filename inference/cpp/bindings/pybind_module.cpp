// inference/cpp/bindings/pybind_module.cpp

#include <pybind11/pybind11.h>
#include "../kernels/layernorm.hpp"
#include "../runtime/paged_kv.hpp"
#include "../runtime/last_token.hpp"
#include "../gguf/gguf.hpp"
#include "../quantization/q8.hpp"
#include <pybind11/stl.h>
#include "../runtime/single_position.hpp"
#include "../runtime/token_embedding.hpp"
#include "../core/arena.hpp"
#include "../core/tensor_factory.hpp"
#include "../runtime/incremental_forward.hpp"
#include "../kernels/blas.hpp"
#include "../runtime/residual.hpp"
#include "../kernels/add.hpp"
#include "../quantization/q4.hpp"
#include "../runtime/cached_attention.hpp"
#include "../kernels/multihead_attention.hpp"
#include "../runtime/linear.hpp"
#include "../kernels/rmsnorm.hpp"
#include "../runtime/kv_state.hpp"
#include "../runtime/kv_append.hpp"
#include "../runtime/full_forward.hpp"
#include "../runtime/position_embedding.hpp"
#include "../runtime/logits.hpp"
#include "../runtime/embedding.hpp"
#include "../runtime/fused_qkv.hpp"
#include "../runtime/kv_cache.hpp"
#include "../runtime/weight_lookup.hpp"
#include "../runtime/real_attention.hpp"
#include "../runtime/execution_graph.hpp"
#include "../core/tensor.hpp"
#include "../runtime/layer_scheduler.hpp"
#include "../kernels/causal_mask.hpp"
#include "../kernels/multihead.hpp"
#include "../runtime/transformer_layer.hpp"
#include "../core/mmap_loader.hpp"
#include "../runtime/gpt2_ln.hpp"
#include "../kernels/transpose.hpp"
#include "../kernels/attention.hpp"
#include "../kernels/gelu.hpp"
#include "../kernels/softmax.hpp"
#include "../kernels/silu.hpp"
#include "../core/tensor_view.hpp"
#include "../kernels/elementwise.hpp"
#include "../kernels/mlp.hpp"
#include "../kernels/attention_output.hpp"
#include "../kernels/rope.hpp"
#include "../runtime/transformer_stack.hpp"
#include "../runtime/final_norm.hpp"
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

        // --------------------------------
        // PYTHON DATA ACCESS
        // --------------------------------

        .def_property(
            "data",

            [](Tensor& t) {

                std::vector<float> out(
                    t.data(),
                    t.data() + t.numel()
                );

                return out;
            },

            
            [](Tensor& t,
            const std::vector<float>& v) {

                if ((int64_t)v.size() != t.numel()) {

                    throw std::runtime_error(
                        "Tensor size mismatch"
                    );
                }

                // --------------------------------
                // OWNED TENSOR
                // --------------------------------

                if (t.owns_data()) {

                    t.owned_data = v;
                    return;
                }

                // --------------------------------
                // POINTER-BASED TENSOR
                // --------------------------------

                if (t.data_ptr != nullptr) {

                    for (int64_t i = 0;
                        i < t.numel();
                        i++) {

                        t.data_ptr[i] = v[i];
                    }

                    return;
                }

                throw std::runtime_error(
                    "Tensor has no writable storage"
                );
            }


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

        m.def(
            "causal_mask",
            &causal_mask
        );

    py::class_<KVCache>(m, "KVCache")

        .def(py::init<>())

        .def(
            "add",
            &KVCache::add
        )

        .def(
            "get_all_keys",
            &KVCache::get_all_keys
        )

        .def(
            "get_all_values",
            &KVCache::get_all_values
        )

        .def(
            "clear",
            &KVCache::clear
        );

        m.def(
            "split_heads",
            &split_heads
        );

        m.def(
            "merge_heads",
            &merge_heads
        );

        m.def(
            "multihead_attention",
            &multihead_attention
        );

        m.def(
            "discover_layers",
            &discover_layers
        );

        m.def(
            "find_tensor_by_suffix",
            &find_tensor_by_suffix
        );

        m.def(
            "execute_model",
            &execute_model
        );

        m.def(
            "embedding_lookup",
            &embedding_lookup
        );

        m.def(
            "compute_logits",
            &compute_logits
        );

        m.def(
            "argmax",
            &argmax
        );
        m.def(
            "real_attention",
            &real_attention
        );


    py::class_<QKV>(m, "QKV")

        .def_readwrite(
            "Q",
            &QKV::Q
        )

        .def_readwrite(
            "K",
            &QKV::K
        )

        .def_readwrite(
            "V",
            &QKV::V
        );
        m.def(
            "split_fused_qkv",
            &split_fused_qkv
        );
        m.def(
            "transformer_stack",
            &transformer_stack
        );
       m.def(
            "final_norm",
            &final_norm
        );
        m.def(
            "full_forward",
            &full_forward
        );
        m.def(
            "linear",
            &linear
        );
        m.def(
            "gpt2_ln",
            &gpt2_ln
        );
        m.def(
            "gelu",
            &gelu
        );
        m.def(
            "last_token",
            &last_token
        );
        m.def(
            "position_embedding_lookup",
            &position_embedding_lookup
        );
        m.def(
            "add",
            &add
        );
    py::class_<LayerKVCache>(
        m,
        "LayerKVCache"
    )

        .def(py::init<>())

        .def_readwrite(
            "keys",
            &LayerKVCache::keys
        )

        .def_readwrite(
            "values",
            &LayerKVCache::values
        );

        py::class_<KVState>(
            m,
            "KVState"
        )

        .def(
            py::init<int>()
        )

        .def_readwrite(
            "layers",
            &KVState::layers
        );

        m.def(
            "append_kv_cache",
            &append_kv_cache
        );
        m.def(
            "cached_attention",
            &cached_attention
        );
        m.def(
            "token_embedding",
            &token_embedding
        );
        m.def(
            "single_position_embedding",
            &single_position_embedding
        );
        m.def(
            "incremental_forward",
            &incremental_forward
        );

        m.def(
            "layernorm",
            &layernorm,
            py::arg("input"),
            py::arg("weight"),
            py::arg("bias"),
            py::arg("eps") = 1e-5f
        );
        m.def(
            "tensor_view",
            &tensor_view
        );
    py::class_<Arena>(m, "Arena")

        .def(
            py::init<int64_t>()
        )

        .def(
            "reset",
            &Arena::reset
        );

        
    py::enum_<DType>(m, "DType")

        .value(
            "FLOAT16",
            DType::FLOAT16
        )

        .value(
            "FLOAT32",
            DType::FLOAT32
        )

        .value(
            "UNKNOWN",
            DType::UNKNOWN
        );
        m.def(
        "create_tensor",
        &create_tensor,

        py::arg("arena"),
        py::arg("shape"),
        py::arg("dtype") =
            DType::FLOAT32
    );
        m.def(
        "create_tensor",
        &create_tensor,

        py::arg("arena"),
        py::arg("shape"),
        py::arg("dtype") =
            DType::FLOAT32
        );

        m.def(
            "create_owned_tensor",
            &create_owned_tensor,

            py::arg("shape"),
            py::arg("dtype") =
                DType::FLOAT32
        );



        m.def(
            "matmul_arena",
            &matmul_arena
        );
        py::class_<Q8Tensor>(
        m,
        "Q8Tensor"
        )

        .def(py::init<>())

        .def_readwrite(
            "data",
            &Q8Tensor::data
        )

        .def_readwrite(
            "shape",
            &Q8Tensor::shape
        )

        .def_readwrite(
            "scale",
            &Q8Tensor::scale
        );
        m.def(
        "quantize_q8",
        &quantize_q8
        );

        m.def(
            "dequantize_q8",
            &dequantize_q8
        );

        py::class_<Q4Tensor>(
        m,
        "Q4Tensor"
    )

    .def(py::init<>())

    .def_readwrite(
        "data",
        &Q4Tensor::data
    )

    .def_readwrite(
        "shape",
        &Q4Tensor::shape
    )

    .def_readwrite(
        "scale",
        &Q4Tensor::scale
    );
    m.def(
    "quantize_q4",
    &quantize_q4
    );

    m.def(
        "dequantize_q4",
        &dequantize_q4
    );
    py::class_<GGUFLoader>(
    m,
    "GGUFLoader"
    )

    .def(py::init<>())

    .def(
        "load_file",
        &GGUFLoader::load_file
    )

    .def(
        "tensor_names",
        &GGUFLoader::tensor_names
    );
    py::class_<PagedKVCache>(
        m,
        "PagedKVCache"
    )

    .def(py::init<>())

    .def(
        "initialize",
        &PagedKVCache::initialize
    )

    .def(
        "append",
        &PagedKVCache::append
    )

    .def(
        "materialize_keys",
        &PagedKVCache::materialize_keys
    )

    .def(
        "materialize_values",
        &PagedKVCache::materialize_values
    )

    .def_readwrite(
        "page_size",
        &PagedKVCache::page_size
    )

    .def_readwrite(
        "hidden_size",
        &PagedKVCache::hidden_size
    );

}
