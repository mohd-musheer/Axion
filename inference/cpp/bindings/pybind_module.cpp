// inference/cpp/bindings/pybind_module.cpp
//
// ARCHITECTURE RULE:
//   RuntimeMemoryScheduler is INTERNAL to C++ runtime only.
//   It is NOT exposed to Python. All scheduler parameters are
//   hidden behind lambda wrappers that call the C++ functions
//   with nullptr, keeping the Python API clean and stable.

#include <pybind11/pybind11.h>
#include "../kernels/layernorm.hpp"
#include "../runtime/paged_kv.hpp"
#include "../runtime/last_token.hpp"
#include "../gguf/gguf.hpp"
#include "../quantization/q8.hpp"
#include "../runtime/streaming_executor.hpp"
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
#include "../runtime/model_runner.hpp"
#include "../runtime/sampler.hpp"
#include "../runtime/quantized_matmul.hpp"
#include <memory>

namespace py = pybind11;

using namespace axion;

PYBIND11_MODULE(axion_cpp, m) {


    // ================================================================
    // Tensor
    // ================================================================

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


    // ================================================================
    // MMapLoader
    // ================================================================

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


    // ================================================================
    // 1. MATMUL
    //    C++ signature: matmul(a, b, scheduler=nullptr)
    //    Python API:    matmul(a, b)
    // ================================================================
    m.def(
        "matmul",

        [](const Tensor& a,
           const Tensor& b) {

            return matmul(a, b, nullptr);
        },

        py::arg("a"),
        py::arg("b")
    );


    // ================================================================
    // 2. RMSNORM
    //    C++ signature: rmsnorm(input, weight, eps, scheduler=nullptr)
    //    Python API:    rmsnorm(input, weight, eps=1e-6)
    // ================================================================
    m.def(
        "rmsnorm",

        [](const Tensor& input,
           const Tensor& weight,
           float eps) {

            return rmsnorm(input, weight, eps, nullptr);
        },

        py::arg("input"),
        py::arg("weight"),
        py::arg("eps") = 1e-6f
    );


    // ================================================================
    // APPLY ROPE
    //    No scheduler parameter — direct binding.
    // ================================================================
    m.def(
        "apply_rope",
        &apply_rope
    );


    // ================================================================
    // 3. TRANSPOSE
    //    C++ signature: transpose(input, scheduler=nullptr)
    //    Python API:    transpose(input)
    // ================================================================
    m.def(
        "transpose",

        [](const Tensor& input) {

            return transpose(input, nullptr);
        },

        py::arg("input")
    );


    // ================================================================
    // 4. ATTENTION SCORES
    //    C++ signature: attention_scores(Q, K, scheduler=nullptr)
    //    Python API:    attention_scores(Q, K)
    // ================================================================
    m.def(
        "attention_scores",

        [](const Tensor& Q,
           const Tensor& K) {

            return attention_scores(Q, K, nullptr);
        },

        py::arg("Q"),
        py::arg("K")
    );


    // ================================================================
    // 5. SOFTMAX
    //    C++ signature: softmax(input, scheduler=nullptr)
    //    Python API:    softmax(input)
    // ================================================================
    m.def(
        "softmax",

        [](const Tensor& input) {

            return softmax(input, nullptr);
        },

        py::arg("input")
    );


    // ================================================================
    // 6. ATTENTION OUTPUT
    //    C++ signature: attention_output(attention_probs, V, scheduler=nullptr)
    //    Python API:    attention_output(attention_probs, V)
    // ================================================================
    m.def(
        "attention_output",

        [](const Tensor& attention_probs,
           const Tensor& V) {

            return attention_output(attention_probs, V, nullptr);
        },

        py::arg("attention_probs"),
        py::arg("V")
    );


    // ================================================================
    // 7. RESIDUAL ADD
    //    C++ signature: residual_add(x, y, scheduler=nullptr)
    //    Python API:    residual_add(x, y)
    // ================================================================
    m.def(
        "residual_add",

        [](const Tensor& x,
           const Tensor& y) {

            return residual_add(x, y, nullptr);
        },

        py::arg("x"),
        py::arg("y")
    );


    // ================================================================
    // 8. SILU
    //    C++ signature: silu(input, scheduler=nullptr)
    //    Python API:    silu(input)
    // ================================================================
    m.def(
        "silu",

        [](const Tensor& input) {

            return silu(input, nullptr);
        },

        py::arg("input")
    );


    // ================================================================
    // 9. ELEMENTWISE MUL
    //    C++ signature: elementwise_mul(a, b, scheduler=nullptr)
    //    Python API:    elementwise_mul(a, b)
    // ================================================================
    m.def(
        "elementwise_mul",

        [](const Tensor& a,
           const Tensor& b) {

            return elementwise_mul(a, b, nullptr);
        },

        py::arg("a"),
        py::arg("b")
    );


    // ================================================================
    // 10. MLP BLOCK
    //    C++ signature: mlp_block(gate, up, scheduler=nullptr)
    //    Python API:    mlp_block(gate, up)
    // ================================================================
    m.def(
        "mlp_block",

        [](const Tensor& gate,
           const Tensor& up) {

            return mlp_block(gate, up, nullptr);
        },

        py::arg("gate"),
        py::arg("up")
    );


    // ================================================================
    // 11. TRANSFORMER LAYER
    //    C++ signature: transformer_layer(input, scheduler=nullptr)
    //    Python API:    transformer_layer(input)
    // ================================================================
    m.def(
        "transformer_layer",

        [](const Tensor& input) {

            return transformer_layer(input, nullptr);
        },

        py::arg("input")
    );


    // ================================================================
    // GENERATE TOKENS
    //    No scheduler parameter — direct binding.
    // ================================================================
    m.def(
        "generate_tokens",
        &generate_tokens
    );


    // ================================================================
    // 12. CAUSAL MASK
    //    C++ signature: causal_mask(scores, scheduler=nullptr)
    //    Python API:    causal_mask(scores)
    // ================================================================
    m.def(
        "causal_mask",

        [](const Tensor& scores) {

            return causal_mask(scores, nullptr);
        },

        py::arg("scores")
    );


    // ================================================================
    // KVCache
    // ================================================================

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


    // ================================================================
    // SPLIT HEADS
    //    No scheduler parameter — direct binding.
    // ================================================================
    m.def(
        "split_heads",
        &split_heads
    );


    // ================================================================
    // 13. MERGE HEADS
    //    C++ signature: merge_heads(const std::vector<Tensor>&, scheduler=nullptr)
    //    Python API:    merge_heads(heads)
    // ================================================================
    m.def(
        "merge_heads",

        [](const std::vector<Tensor>& heads) {

            return merge_heads(heads, nullptr);
        },

        py::arg("heads")
    );


    // ================================================================
    // 14. MULTIHEAD ATTENTION
    //    C++ signature: multihead_attention(Q, K, V, num_heads, scheduler=nullptr)
    //    Python API:    multihead_attention(Q, K, V, num_heads)
    // ================================================================
    m.def(
        "multihead_attention",

        [](const Tensor& Q,
           const Tensor& K,
           const Tensor& V,
           int num_heads) {

            return multihead_attention(Q, K, V, num_heads, nullptr);
        },

        py::arg("Q"),
        py::arg("K"),
        py::arg("V"),
        py::arg("num_heads")
    );


    // ================================================================
    // DISCOVER LAYERS / FIND TENSOR / EXECUTE MODEL / LOGITS / ARGMAX
    //    No scheduler parameters — direct bindings.
    // ================================================================
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


    // ================================================================
    // 15. REAL ATTENTION
    //    C++ signature: real_attention(input, loader,
    //                       q_weight_name, k_weight_name, v_weight_name,
    //                       num_heads, scheduler=nullptr)
    //    Python API:    real_attention(input, loader,
    //                       q_weight_name, k_weight_name, v_weight_name,
    //                       num_heads)
    // ================================================================
    m.def(
        "real_attention",

        [](const Tensor& input,
           MMapLoader& loader,
           const std::string& q_weight_name,
           const std::string& k_weight_name,
           const std::string& v_weight_name,
           int num_heads) {

            return real_attention(
                input,
                loader,
                q_weight_name,
                k_weight_name,
                v_weight_name,
                num_heads,
                nullptr
            );
        },

        py::arg("input"),
        py::arg("loader"),
        py::arg("q_weight_name"),
        py::arg("k_weight_name"),
        py::arg("v_weight_name"),
        py::arg("num_heads")
    );


    // ================================================================
    // QKV
    // ================================================================

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


    // ================================================================
    // SPLIT FUSED QKV
    //    No scheduler parameter — direct binding.
    // ================================================================
    m.def(
        "split_fused_qkv",
        &split_fused_qkv
    );


    // ================================================================
    // 16. TRANSFORMER STACK
    //    C++ signature: transformer_stack(input, loader,
    //                       num_layers, num_heads, scheduler=nullptr)
    //    Python API:    transformer_stack(input, loader,
    //                       num_layers, num_heads)
    // ================================================================
    m.def(
        "transformer_stack",

        [](const Tensor& input,
           MMapLoader& loader,
           int num_layers,
           int num_heads) {

            return transformer_stack(
                input,
                loader,
                num_layers,
                num_heads,
                nullptr
            );
        },

        py::arg("input"),
        py::arg("loader"),
        py::arg("num_layers"),
        py::arg("num_heads")
    );


    // ================================================================
    // 17. FINAL NORM
    //    C++ signature: final_norm(input, weight, scheduler=nullptr)
    //    Python API:    final_norm(input, weight)
    // ================================================================
    m.def(
        "final_norm",

        [](const Tensor& input,
           const Tensor& weight) {

            return final_norm(input, weight, nullptr);
        },

        py::arg("input"),
        py::arg("weight")
    );


    // ================================================================
    // 18. FULL FORWARD
    //    C++ signature: full_forward(input_ids, loader,
    //                       num_layers, num_heads,
    //                       final_norm_weight, scheduler=nullptr)
    //    Python API:    full_forward(input_ids, loader,
    //                       num_layers, num_heads, final_norm_weight)
    // ================================================================
    m.def(
        "full_forward",

        [](const Tensor& input_ids,
           MMapLoader& loader,
           int num_layers,
           int num_heads,
           const Tensor& final_norm_weight) {

            return full_forward(
                input_ids,
                loader,
                num_layers,
                num_heads,
                final_norm_weight,
                nullptr
            );
        },

        py::arg("input_ids"),
        py::arg("loader"),
        py::arg("num_layers"),
        py::arg("num_heads"),
        py::arg("final_norm_weight")
    );


    // ================================================================
    // 19. LINEAR
    //    C++ signature: linear(input, weight, scheduler=nullptr)
    //    Python API:    linear(input, weight)
    // ================================================================
    m.def(
        "linear",

        [](const Tensor& input,
           const Tensor& weight) {

            return linear(input, weight, nullptr);
        },

        py::arg("input"),
        py::arg("weight")
    );


    // ================================================================
    // 20. GPT2 LN
    //    C++ signature: gpt2_ln(input, weight, scheduler=nullptr)
    //    Python API:    gpt2_ln(input, weight)
    // ================================================================
    m.def(
        "gpt2_ln",

        [](const Tensor& input,
           const Tensor& weight) {

            return gpt2_ln(input, weight, nullptr);
        },

        py::arg("input"),
        py::arg("weight")
    );


    // ================================================================
    // 21. GELU
    //    C++ signature: gelu(input, scheduler=nullptr)
    //    Python API:    gelu(input)
    // ================================================================
    m.def(
        "gelu",

        [](const Tensor& input) {

            return gelu(input, nullptr);
        },

        py::arg("input")
    );


    // ================================================================
    // LAST TOKEN / POSITION EMBEDDING
    //    No scheduler parameters — direct bindings.
    // ================================================================
    m.def(
        "last_token",
        &last_token
    );

    m.def(
        "position_embedding_lookup",
        &position_embedding_lookup
    );


    // ================================================================
    // 22. ADD
    //    C++ signature: add(a, b, scheduler=nullptr)
    //    Python API:    add(a, b)
    // ================================================================
    m.def(
        "add",

        [](const Tensor& a,
           const Tensor& b) {

            return add(a, b, nullptr);
        },

        py::arg("a"),
        py::arg("b")
    );


    // ================================================================
    // LayerKVCache
    // ================================================================

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


    // ================================================================
    // KVState
    // ================================================================

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


    // ================================================================
    // KV CACHE OPS / TOKEN / POSITION / INCREMENTAL
    //    No scheduler parameters — direct bindings.
    // ================================================================
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


    // ================================================================
    // LAYERNORM
    //    C++ signature: layernorm(input, weight, bias, eps, scheduler=nullptr)
    //    Python API:    layernorm(input, weight, bias, eps=1e-5)
    // ================================================================
    m.def(
        "layernorm",

        [](const Tensor& input,
           const Tensor& weight,
           const Tensor& bias,
           float eps) {

            return layernorm(input, weight, bias, eps, nullptr);
        },

        py::arg("input"),
        py::arg("weight"),
        py::arg("bias"),
        py::arg("eps") = 1e-5f
    );


    // ================================================================
    // TENSOR VIEW
    //    No scheduler parameter — direct binding.
    // ================================================================
    // Explicit cast pins the Python binding to the unmanaged
    // overload; the scheduler-managed overload is C++-internal.
    m.def(
        "tensor_view",
        static_cast<
            Tensor (*)(
                const Tensor&,
                int64_t,
                std::vector<int64_t>
            )
        >(&tensor_view)
    );


    // ================================================================
    // Arena
    // ================================================================

    py::class_<Arena>(m, "Arena")

        .def(
            py::init<int64_t>()
        )

        .def(
            "reset",
            &Arena::reset
        );


    // ================================================================
    // DType
    // ================================================================

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


    // ================================================================
    // TENSOR FACTORY
    // ================================================================

    m.def(
        "create_tensor",
        &create_tensor,
        py::arg("arena"),
        py::arg("shape"),
        py::arg("dtype") = DType::FLOAT32
    );

    m.def(
        "create_owned_tensor",
        &create_owned_tensor,
        py::arg("shape"),
        py::arg("dtype") = DType::FLOAT32
    );

    m.def(
        "matmul_arena",
        &matmul_arena
    );


    // ================================================================
    // Q8Tensor
    // ================================================================

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


    // ================================================================
    // Q4Tensor
    // ================================================================

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


    // ================================================================
    // GGUFLoader
    // ================================================================

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
        )

        .def(
            "load_tensor",
            &GGUFLoader::load_tensor
        )

        .def(
            "architecture",
            &GGUFLoader::architecture
        )

        .def(
            "get_u32",
            &GGUFLoader::get_u32,
            py::arg("key"), py::arg("fallback") = 0
        )

        .def(
            "get_f32",
            &GGUFLoader::get_f32,
            py::arg("key"), py::arg("fallback") = 0.0f
        )

        .def(
            "get_str",
            &GGUFLoader::get_str,
            py::arg("key"), py::arg("fallback") = std::string()
        );


    // ================================================================
    // PagedKVCache
    // ================================================================

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


    // ================================================================
    // StreamingExecutor
    //
    // NOTE: RuntimeMemoryScheduler is NOT exposed to Python.
    //   Scheduler is internal C++ runtime state only.
    //   Python callers never see or touch the scheduler.
    //   C++ ops internally receive nullptr and manage their
    //   own scheduler lifecycle via StreamingExecutor.
    // ================================================================

    py::class_<StreamingExecutor>(
        m,
        "StreamingExecutor"
    )

        .def(
            py::init<GGUFLoader*>()
        )

        .def(
            "forward",
            [](StreamingExecutor& self,
               const Tensor& input,
               int num_layers) {
                return self.forward(input, num_layers, LayerConfig{});
            },
            py::arg("input"),
            py::arg("num_layers")
        );


    // ================================================================
    // Quantized runtime (MR10)
    // ================================================================
    py::enum_<QuantType>(m, "QuantType")
        .value("FP32", QuantType::FP32)
        .value("Q4_0", QuantType::Q4_0)
        .value("Q8_0", QuantType::Q8_0)
        .value("Q8_1", QuantType::Q8_1);

    py::class_<QuantizedWeight>(m, "QuantizedWeight")
        .def(py::init<>())
        .def_readonly("type", &QuantizedWeight::type)
        .def_readonly("K", &QuantizedWeight::K)
        .def_readonly("N", &QuantizedWeight::N)
        .def_readonly("scale", &QuantizedWeight::scale)
        .def("payload_bytes", &QuantizedWeight::payload_bytes);

    m.def("quantize_weight", &quantize_weight,
          py::arg("weight"), py::arg("type"));
    m.def("quantized_matmul", &quantized_matmul,
          py::arg("input"), py::arg("weight"));


    // ================================================================
    // Sampling (MR9)
    // ================================================================
    py::class_<SamplingParams>(m, "SamplingParams")
        .def(py::init<>())
        .def_readwrite("temperature", &SamplingParams::temperature)
        .def_readwrite("top_k", &SamplingParams::top_k)
        .def_readwrite("top_p", &SamplingParams::top_p)
        .def_readwrite("seed", &SamplingParams::seed);


    // ================================================================
    // High-level ModelRunner API (MR6-MR9): the supported entrypoint.
    // ================================================================
    py::class_<ModelConfig>(m, "ModelConfig")
        .def_readonly("arch", &ModelConfig::arch)
        .def_readonly("n_layers", &ModelConfig::n_layers)
        .def_readonly("n_heads", &ModelConfig::n_heads)
        .def_readonly("hidden", &ModelConfig::hidden)
        .def_readonly("vocab", &ModelConfig::vocab)
        .def_readonly("rms_eps", &ModelConfig::rms_eps);

    py::class_<ModelRunner> runner(m, "ModelRunner");

    py::class_<ModelRunner::GenerationParams>(runner, "GenerationParams")
        .def(py::init<>())
        .def_readwrite("max_new_tokens",
                       &ModelRunner::GenerationParams::max_new_tokens)
        .def_readwrite("sampling",
                       &ModelRunner::GenerationParams::sampling);

    runner
        .def(py::init<GGUFLoader*>(), py::keep_alive<1, 2>())
        .def("config", &ModelRunner::config,
             py::return_value_policy::reference_internal)
        .def("forward_logits", &ModelRunner::forward_logits,
             py::arg("tokens"))
        .def("predict_next_token", &ModelRunner::predict_next_token,
             py::arg("tokens"))
        .def("generate", &ModelRunner::generate,
             py::arg("prompt"), py::arg("params"));


    // ================================================================
    // load_model(path): construct loader + runner, keep loader alive.
    //
    // Returns a tuple (loader, runner). The loader must outlive the
    // runner; returning both keeps the loader referenced from Python.
    // ================================================================
    m.def(
        "load_model",
        [](const std::string& path) {
            // Heap-allocate so the objects outlive this call; ownership
            // transfers to Python (take_ownership). The wrapper module
            // (AxionModel) holds both, and keep_alive on ModelRunner's
            // ctor ties the loader to the runner.
            GGUFLoader* loader = new GGUFLoader();
            if (!loader->load_file(path)) {
                delete loader;
                throw std::runtime_error("failed to load GGUF: " + path);
            }
            ModelRunner* rn = new ModelRunner(loader);
            py::object py_loader = py::cast(
                loader, py::return_value_policy::take_ownership);
            py::object py_runner = py::cast(
                rn, py::return_value_policy::take_ownership);
            return py::make_tuple(py_loader, py_runner);
        },
        py::arg("path")
    );

}
