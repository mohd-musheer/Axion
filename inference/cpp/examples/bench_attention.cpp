// Attention benchmark (Phase 14 / MR13): old single-head full-dim vs
// new MHA+GQA+RoPE on a synthetic sequence. Reports wall time and the
// allocator peak so the cost of correct attention is visible.

#include "../core/tensor.hpp"
#include "../core/tensor_factory.hpp"
#include "../core/allocator.hpp"
#include "../runtime/mha.hpp"
#include "../kernels/attention.hpp"
#include "../kernels/blas.hpp"
#include "../kernels/softmax.hpp"

#include <chrono>
#include <cstdio>
#include <limits>

using namespace axion;

static Tensor rand_tensor(int64_t r, int64_t c, unsigned seed) {
    Tensor t = create_owned_tensor({r, c}, DType::FLOAT32);
    unsigned x = seed;
    for (int64_t i = 0; i < r * c; i++) {
        x = x * 1103515245u + 12345u;
        t.data()[i] = ((x >> 16) & 0x7FFF) / 32768.0f - 0.5f;
    }
    return t;
}

// old path: single-head full-dim causal attention
static Tensor old_attention(const Tensor& q, const Tensor& k, const Tensor& v) {
    int64_t seq = q.shape[0];
    Tensor scores = attention_scores(q, k);
    for (int64_t i = 0; i < seq; i++)
        for (int64_t j = i + 1; j < seq; j++)
            scores.data()[i * seq + j] = -std::numeric_limits<float>::infinity();
    Tensor w = softmax(scores);
    return matmul(w, v);
}

int main(int argc, char** argv) {
    int seq      = argc > 1 ? atoi(argv[1]) : 128;
    int n_head   = argc > 2 ? atoi(argv[2]) : 8;
    int n_kv     = argc > 3 ? atoi(argv[3]) : 2;
    int head_dim = argc > 4 ? atoi(argv[4]) : 64;
    int hidden   = n_head * head_dim;
    int kv_width = n_kv * head_dim;

    Tensor q  = rand_tensor(seq, hidden, 1);
    Tensor k  = rand_tensor(seq, hidden, 2);
    Tensor v  = rand_tensor(seq, hidden, 3);
    Tensor kk = rand_tensor(seq, kv_width, 2);
    Tensor vv = rand_tensor(seq, kv_width, 3);

    auto t0 = std::chrono::high_resolution_clock::now();
    Tensor o_old = old_attention(q, k, v);
    auto t1 = std::chrono::high_resolution_clock::now();

    MHAConfig cfg; cfg.n_head = n_head; cfg.n_kv_head = n_kv;
    cfg.head_dim = head_dim; cfg.rope_theta = 10000.0f;
    rope_apply_rows(q,  n_head, head_dim, cfg.rope_theta);
    rope_apply_rows(kk, n_kv,   head_dim, cfg.rope_theta);
    auto t2 = std::chrono::high_resolution_clock::now();
    Tensor o_new = mha_attention(q, kk, vv, cfg);
    auto t3 = std::chrono::high_resolution_clock::now();

    auto ms = [](auto a, auto b){
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    std::printf("seq=%d n_head=%d n_kv=%d head_dim=%d hidden=%d\n",
                seq, n_head, n_kv, head_dim, hidden);
    std::printf("old single-head attention : %.3f ms (out %lldx%lld)\n",
                ms(t0, t1), (long long)o_old.shape[0], (long long)o_old.shape[1]);
    std::printf("new MHA+GQA+RoPE          : %.3f ms (out %lldx%lld)\n",
                ms(t2, t3), (long long)o_new.shape[0], (long long)o_new.shape[1]);

    AllocatorStats st = SystemAllocator::instance().stats();
    std::printf("allocator peak bytes      : %lld\n", (long long)st.peak_bytes);
    return 0;
}
