
#include "real_attention.hpp"

#include "../kernels/blas.hpp"
#include "../kernels/rope.hpp"
#include "../kernels/multihead_attention.hpp"

#include "fused_qkv.hpp"

namespace axion {

Tensor real_attention(
    const Tensor& hidden,
    MMapLoader& loader,
    const std::string& q_name,
    const std::string& k_name,
    const std::string& v_name,
    int num_heads,
    RuntimeMemoryScheduler* scheduler
) {

    // -------------------------
    // LOAD FUSED WEIGHT
    // GPT2:
    // [768, 2304]
    // -------------------------

    Tensor fused_weight =
        loader.load_tensor_data(q_name);

    // -------------------------
    // PROJECT
    // [B,768] @ [768,2304]
    // -> [B,2304]
    // -------------------------

    Tensor fused_qkv =
        matmul(
            hidden,
            fused_weight
        );

    // -------------------------
    // SPLIT ACTIVATIONS
    // -------------------------

    QKV qkv =
        split_fused_qkv(
            fused_qkv
        );

    Tensor Q = qkv.Q;
    Tensor K = qkv.K;
    Tensor V = qkv.V;

    // -------------------------
    // ROPE
    // -------------------------

    int hidden_size =
        Q.shape[1];

    int head_dim =
        hidden_size / num_heads;

    apply_rope(
        Q,
        K,
        0,
        head_dim
    );

    // -------------------------
    // ATTENTION
    // -------------------------

    Tensor output =
        multihead_attention(
            Q,
            K,
            V,
            num_heads
        );

    output.name =
        "real_attention_output";

    return output;
}

}
