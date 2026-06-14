
#include "cached_attention.hpp"

#include "fused_qkv.hpp"
#include "kv_append.hpp"

#include "../kernels/blas.hpp"
#include "../kernels/transpose.hpp"
#include "../kernels/rope.hpp"
#include "../kernels/multihead_attention.hpp"

namespace axion {

Tensor cached_attention(
    const Tensor& hidden,
    MMapLoader& loader,
    LayerKVCache& cache,
    const std::string& fused_name,
    int num_heads
) {

    // -------------------------
    // LOAD FUSED QKV
    // -------------------------

    Tensor fused =
        loader.load_tensor_data(
            fused_name
        );

    // -------------------------
    // SPLIT QKV
    // -------------------------

    QKV qkv =
        split_fused_qkv(
            fused
        );

    Tensor Wq =
        qkv.Q;

    Tensor Wk =
        qkv.K;

    Tensor Wv =
        qkv.V;

    // -------------------------
    // TRANSPOSE
    // -------------------------

    Tensor Wq_t =
        transpose(Wq);

    Tensor Wk_t =
        transpose(Wk);

    Tensor Wv_t =
        transpose(Wv);

    // -------------------------
    // QKV PROJECTIONS
    // -------------------------

    Tensor Q =
        matmul(
            hidden,
            Wq_t
        );

    Tensor K =
        matmul(
            hidden,
            Wk_t
        );

    Tensor V =
        matmul(
            hidden,
            Wv_t
        );

    // -------------------------
    // ROPE
    // -------------------------

    int hidden_size =
        Q.shape[1];

    int head_dim =
        hidden_size / num_heads;

    int position =
        cache.keys.shape.empty()
        ? 0
        : cache.keys.shape[0];

    apply_rope(
        Q,
        K,
        position,
        head_dim
    );

    // -------------------------
    // APPEND KV CACHE
    // -------------------------

    append_kv_cache(
        cache,
        K,
        V
    );

    // -------------------------
    // ATTENTION
    // -------------------------

    Tensor output =
        multihead_attention(
            Q,
            cache.keys,
            cache.values,
            num_heads
        );

    output.name =
        "cached_attention_output";

    return output;
}

}
