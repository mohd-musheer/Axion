// Axion runtime validation harness.
//
// Phase 11: ownership, scheduler, eviction, KV correctness.
// Phase 12: allocator foundation, TensorBuffer shared ownership.
//
// No Python, no model files: every check is self-contained.

#include "../core/tensor.hpp"
#include "../core/tensor_view.hpp"
#include "../core/tensor_factory.hpp"
#include "../core/tensor_buffer.hpp"
#include "../core/allocator.hpp"
#include "../core/scheduler.hpp"
#include "../runtime/tensor_evictor.hpp"
#include "../runtime/kv_append.hpp"
#include "../runtime/kv_cache.hpp"
#include "../runtime/paged_kv.hpp"
#include "../core/mmap_loader.hpp"
#include "../gguf/gguf.hpp"
#include "../runtime/logits.hpp"
#include "../runtime/model_runner.hpp"
#include "../runtime/sampler.hpp"
#include "../runtime/quantized_matmul.hpp"
#include "../runtime/mha.hpp"
#include "../kernels/simd_dot.hpp"
#include "../core/allocator.hpp"
#include <vector>
#include <random>
#include <cmath>

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>

static int failures = 0;

#define CHECK(cond, msg)                                  \
    do {                                                  \
        if (cond) {                                       \
            std::printf("PASS  %s\n", msg);              \
        } else {                                          \
            std::printf("FAIL  %s\n", msg);              \
            failures++;                                   \
        }                                                 \
    } while (0)

using namespace axion;

static void fill_seq(Tensor& t) {
    for (int64_t i = 0; i < t.numel(); i++) {
        t.data()[i] = static_cast<float>(i);
    }
}

int main() {

    std::printf("=== PHASE 11: scheduler ===\n");

    // 1. block reuse
    {
        RuntimeMemoryScheduler s;
        Tensor a = s.request_tensor("a", {32});
        float* p0 = a.data_ptr;
        s.release_tensor(a.name);
        Tensor b = s.request_tensor("b", {32});
        CHECK(b.data_ptr == p0, "scheduler reuses released block");
    }

    // 2. pin blocks release; unpin drains the deferred queue
    {
        RuntimeMemoryScheduler s;
        Tensor a = s.request_tensor("a", {64});
        float* p0 = a.data_ptr;
        s.pin_tensor(a.name);
        s.release_tensor(a.name);              // must defer
        Tensor b = s.request_tensor("b", {64});
        CHECK(b.data_ptr != p0, "pinned block is not reused while pinned");
        s.unpin_tensor(a.name);                // deferred release fires
        Tensor c = s.request_tensor("c", {64});
        CHECK(c.data_ptr == p0, "deferred release drains on unpin");
    }

    // 3. no monotonic block growth across a generation-like loop
    {
        RuntimeMemoryScheduler s;
        for (int i = 0; i < 100; i++) {
            Tensor t = s.request_tensor("step", {16, 16});
            s.release_tensor(t.name);
        }
        CHECK(s.peak_memory_bytes() == 16 * 16 * 4,
              "peak stays at one block across 100 iterations");
        CHECK(s.current_memory_bytes() == 16 * 16 * 4,
              "pool does not grow monotonically");
    }

    // 4. reset correctness (previously threw)
    {
        RuntimeMemoryScheduler s;
        Tensor a = s.request_tensor("a", {8});
        std::string name = a.name;
        s.reset();
        bool threw = false;
        try { s.release_tensor(name); } catch (...) { threw = true; }
        CHECK(!threw, "release after reset does not throw");
    }

    std::printf("=== PHASE 11: views ===\n");

    // 5. view data correctness over an owned base
    {
        Tensor base = create_owned_tensor({2, 4});
        fill_seq(base);
        Tensor v = tensor_view(base, 4, {1, 4});
        CHECK(v.numel() == 4, "view numel");
        CHECK(v.data()[0] == 4.0f && v.value(3) == 7.0f,
              "view reads correct slice");
        bool threw = false;
        try {
            Tensor bad = tensor_view(base, 6, {1, 4});
            (void)bad;
        } catch (...) { threw = true; }
        CHECK(threw, "out-of-range view throws");
    }

    // 6. managed view pins parent; release_view unpins
    {
        RuntimeMemoryScheduler s;
        Tensor t = s.request_tensor("t", {16});
        float* p0 = t.data_ptr;
        Tensor v = tensor_view(t, 0, {16}, &s);
        s.release_tensor(t.name);              // deferred: view pin held
        Tensor x = s.request_tensor("x", {16});
        CHECK(x.data_ptr != p0, "viewed block survives release attempt");
        v.release_view(&s);                    // unpin -> deferred release fires
        Tensor y = s.request_tensor("y", {16});
        CHECK(y.data_ptr == p0, "block reclaimed after view release");
    }

    std::printf("=== PHASE 11: eviction ===\n");

    // 7. evictor safety
    {
        TensorEvictor e;
        Tensor owned = create_owned_tensor({4});
        fill_seq(owned);
        Tensor pinned = create_owned_tensor({4});
        fill_seq(pinned);
        pinned.pin_count = 1;
        e.register_tensor(&owned);
        e.register_tensor(&pinned);
        e.evict_all();
        CHECK(owned.owned_data.empty(), "unpinned tensor evicted");
        CHECK(pinned.owned_data.size() == 4, "pinned tensor survives eviction");
    }

    std::printf("=== PHASE 11: KV cache ===\n");

    // 8. KV append from a VIEW source (C1 regression)
    {
        Tensor kf = create_owned_tensor({2, 8});
        fill_seq(kf);
        Tensor vf = create_owned_tensor({2, 8});
        fill_seq(vf);
        Tensor krow = tensor_view(kf, 8, {1, 8});
        Tensor vrow = tensor_view(vf, 8, {1, 8});
        LayerKVCache c;
        append_kv_cache(c, krow, vrow);
        append_kv_cache(c, krow, vrow);
        CHECK(c.keys.shape[0] == 2 && c.keys.owned_data.size() == 16,
              "KV grows with real data, not just shape");
        CHECK(c.keys.value(0) == 8.0f && c.keys.value(15) == 15.0f,
              "KV contents correct from view source");
    }

    // 9. KVCache concatenation from view sources
    {
        Tensor kf = create_owned_tensor({2, 4});
        fill_seq(kf);
        KVCache cache;
        cache.add(tensor_view(kf, 0, {1, 4}), tensor_view(kf, 0, {1, 4}));
        cache.add(tensor_view(kf, 4, {1, 4}), tensor_view(kf, 4, {1, 4}));
        Tensor all = cache.get_all_keys();
        CHECK(all.shape[0] == 2 && all.value(4) == 4.0f && all.value(7) == 7.0f,
              "KVCache concatenation correct");
    }

    // 10. PagedKV multi-row append + page split (H1 regression)
    {
        PagedKVCache p;
        p.initialize(8, 4);
        Tensor K = create_owned_tensor({6, 8});
        fill_seq(K);
        Tensor V = create_owned_tensor({6, 8});
        fill_seq(V);
        p.append(K, V);
        CHECK(p.pages.size() == 2 && p.pages[0].used == 4 && p.pages[1].used == 2,
              "rows split across pages");
        Tensor mk = p.materialize_keys();
        bool ok = (mk.shape[0] == 6);
        for (int64_t i = 0; ok && i < mk.numel(); i++) {
            ok = (mk.value(i) == static_cast<float>(i));
        }
        CHECK(ok, "materialized keys match all appended rows");
    }

    // 11. 1D strided value() (H5 regression)
    {
        static float buf[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        Tensor t;
        t.shape = {4};
        t.is_view = true;
        t.is_strided = true;
        t.stride = 2;
        t.data_ptr = buf;
        CHECK(t.value(0) == 0.0f && t.value(3) == 6.0f,
              "1D strided view indexes by element stride");
    }

    std::printf("=== PHASE 12: allocator ===\n");

    // 12. alignment + statistics + free-on-last-reference
    {
        auto& sys = SystemAllocator::instance();
        AllocatorStats before = sys.stats();
        {
            Tensor t = create_buffer_tensor({16, 16});
            CHECK((reinterpret_cast<uintptr_t>(t.data()) % 64) == 0,
                  "buffer tensor is 64-byte aligned");
            AllocatorStats during = sys.stats();
            CHECK(during.active_allocations == before.active_allocations + 1,
                  "allocator tracks active allocation");
            t.data()[0] = 42.0f;
            CHECK(t.value(0) == 42.0f,
                  "buffer tensor read/write through Tensor API");
        }
        AllocatorStats after = sys.stats();
        CHECK(after.active_allocations == before.active_allocations,
              "buffer freed when last reference dies");
    }

    // 13. view outlives base (C2 regression proof)
    {
        Tensor v;
        {
            Tensor base = create_buffer_tensor({8});
            fill_seq(base);
            v = tensor_view(base, 2, {4});
        } // base destroyed; storage kept alive by the view's buffer ref
        CHECK(v.value(0) == 2.0f && v.value(3) == 5.0f,
              "view keeps buffer alive after base destruction");
    }

    // 14. buffer pin blocks eviction; unpinned buffer is evicted
    {
        Tensor t = create_buffer_tensor({4});
        t.data()[0] = 7.0f;
        t.buffer->pin();
        TensorEvictor e1;
        e1.register_tensor(&t);
        e1.evict_all();
        CHECK(t.buffer != nullptr && t.value(0) == 7.0f,
              "pinned buffer survives eviction");
        t.buffer->unpin();
        TensorEvictor e2;
        e2.register_tensor(&t);
        e2.evict_all();
        CHECK(t.buffer == nullptr, "unpinned buffer evicted");
    }

    std::printf("=== PHASE 12 MR4: GGUF sizing ===\n");

    // 15. exact ggml block sizes (regression for audit-found UB)
    {
        CHECK(gguf_tensor_byte_size(GGUFType::F16, 32) == 64,
              "F16 sizing");
        CHECK(gguf_tensor_byte_size(GGUFType::Q4_0, 32) == 18,
              "Q4_0 block is 18 bytes (was 16: heap overread)");
        CHECK(gguf_tensor_byte_size(GGUFType::Q8_0, 32) == 34,
              "Q8_0 block is 34 bytes (was 32: heap overread)");
        CHECK(gguf_tensor_byte_size(GGUFType::Q8_1, 32) == 36,
              "Q8_1 block is 36 bytes");
        CHECK(gguf_tensor_byte_size(GGUFType::Q2_K, 256) == 84,
              "Q2_K super-block is 84 bytes");
        CHECK(gguf_tensor_byte_size(GGUFType::Q4_K, 256) == 144,
              "Q4_K super-block is 144 bytes");
        CHECK(gguf_tensor_byte_size(GGUFType::Q6_K, 256) == 210,
              "Q6_K super-block is 210 bytes (was 209)");
        CHECK(gguf_tensor_byte_size(GGUFType::Q8_K, 256) == 292,
              "Q8_K super-block is 292 bytes");
    }

    std::printf("=== PHASE 12 MR4: true mmap ===\n");

    // 16. mapped loader: views into the mapping, page release, caching
    {
        const char* path = "axion_mmap_test.safetensors";
        {
            std::string header =
                "{\"a\":{\"dtype\":\"F32\",\"shape\":[2,2],"
                "\"data_offsets\":[0,16]}}";
            uint64_t hsize = header.size();
            float payload[4] = {1.0f, 2.0f, 3.0f, 4.0f};
            std::FILE* f = std::fopen(path, "wb");
            std::fwrite(&hsize, sizeof(hsize), 1, f);
            std::fwrite(header.data(), 1, header.size(), f);
            std::fwrite(payload, sizeof(float), 4, f);
            std::fclose(f);
        }

        MMapLoader loader;
        bool opened = loader.load_file(path);
        CHECK(opened, "mapped safetensors file opens");

        if (opened) {
            Tensor t = loader.load_tensor_data("a");
            const uint8_t* base = loader.mapped_base();
            const uint8_t* ptr =
                reinterpret_cast<const uint8_t*>(t.data_ptr);
            CHECK(ptr >= base && ptr < base + loader.mapped_bytes(),
                  "tensor data points INTO the mapping (no heap copy)");
            CHECK(t.value(0) == 1.0f && t.value(3) == 4.0f,
                  "mapped tensor values correct");

            // Page release: mapping stays valid; pages refault.
            loader.release_tensor_pages("a");
            CHECK(t.value(0) == 1.0f && t.value(3) == 4.0f,
                  "values intact after page release (refault from file)");

            // Cached directory: repeated loads return identical
            // pointers without reparsing the JSON header.
            Tensor t2 = loader.load_tensor_data("a");
            CHECK(t2.data_ptr == t.data_ptr,
                  "directory cached across loads");
        }

        std::remove(path);
    }

    // 17. mapped tensor OUTLIVES the loader (TensorBuffer keepalive)
    {
        const char* path = "axion_mmap_test2.safetensors";
        {
            std::string header =
                "{\"a\":{\"dtype\":\"F32\",\"shape\":[2,2],"
                "\"data_offsets\":[0,16]}}";
            uint64_t hsize = header.size();
            float payload[4] = {5.0f, 6.0f, 7.0f, 8.0f};
            std::FILE* f = std::fopen(path, "wb");
            std::fwrite(&hsize, sizeof(hsize), 1, f);
            std::fwrite(header.data(), 1, header.size(), f);
            std::fwrite(payload, sizeof(float), 4, f);
            std::fclose(f);
        }

        {
            Tensor t;
            {
                MMapLoader loader;
                if (loader.load_file(path)) {
                    t = loader.load_tensor_data("a");
                }
            } // loader destroyed; mapping kept alive by t.buffer
            CHECK(t.buffer != nullptr &&
                  t.value(0) == 5.0f && t.value(3) == 8.0f,
                  "mapped tensor outlives loader via TensorBuffer ownership");
        } // last reference dies here; file is unmapped before removal

        std::remove(path);
    }

    // ===== PHASE 13 / MR7: logits projection =====
    {
        // hidden: [seq=2, hidden=3]
        Tensor hidden = create_owned_tensor({2, 3}, DType::FLOAT32);
        float hv[6] = {1, 2, 3,   4, 5, 6};
        for (int i = 0; i < 6; i++) hidden.data()[i] = hv[i];

        // embedding / LM head: [vocab=2, hidden=3]
        Tensor emb = create_owned_tensor({2, 3}, DType::FLOAT32);
        float ev[6] = {1, 0, 0,   0, 1, 1};
        for (int i = 0; i < 6; i++) emb.data()[i] = ev[i];

        // logits = hidden . emb^T  -> [2, 2]
        //   row0: [1*1+2*0+3*0, 1*0+2*1+3*1] = [1, 5]
        //   row1: [4, 11]
        Tensor logits = compute_logits(hidden, emb);

        CHECK(logits.shape.size() == 2 &&
              logits.shape[0] == 2 && logits.shape[1] == 2,
              "logits shape is [seq, vocab]");

        CHECK(logits.value(0) == 1.0f && logits.value(1) == 5.0f &&
              logits.value(2) == 4.0f && logits.value(3) == 11.0f,
              "logits match hand-computed reference");

        // argmax uses the LAST row -> max(4, 11) = index 1
        CHECK(argmax(logits) == 1,
              "argmax selects max-logit index of last row");
    }

    // LM-head selection: untied head wins, else tied embeddings.
    {
        std::vector<std::string> with_head =
            { "token_embd.weight", "output.weight", "output_norm.weight" };
        std::vector<std::string> tied =
            { "token_embd.weight", "output_norm.weight" };

        CHECK(select_output_weight_name(with_head) == "output.weight",
              "LM head: separate output.weight selected when present");
        CHECK(select_output_weight_name(tied) == "token_embd.weight",
              "LM head: tied token_embd.weight selected when no output.weight");
    }

    // ===== PHASE 13 / MR8: single-step decode contract =====
    {
        // logits [seq=2, vocab=4]; last row max is index 2 (value 9).
        Tensor logits = create_owned_tensor({2, 4}, DType::FLOAT32);
        float lv[8] = { 0, 0, 0, 0,    1, 3, 9, 2 };
        for (int i = 0; i < 8; i++) logits.data()[i] = lv[i];

        // predict_next_token() greedily takes argmax of the last row;
        // exercise the same argmax it uses.
        CHECK(argmax(logits) == 2,
              "single-step decode: argmax of last row picks token 2");
    }

    // ===== PHASE 13 / MR9: sampler =====
    {
        // vocab = 5; index 3 is the clear maximum.
        float row[5] = { 0.1f, 0.2f, 0.5f, 4.0f, 0.3f };
        std::mt19937_64 rng(123);

        // Greedy (temperature 0) -> argmax.
        {
            SamplingParams p; p.temperature = 0.0f;
            CHECK(sample_token(row, 5, p, rng) == 3,
                  "sampler: temperature 0 is greedy argmax");
        }

        // top_k == 1 -> only the max survives, regardless of temp.
        {
            SamplingParams p; p.temperature = 1.5f; p.top_k = 1;
            CHECK(sample_token(row, 5, p, rng) == 3,
                  "sampler: top_k=1 forces the max-logit token");
        }

        // top_p very small -> nucleus collapses to the single best.
        {
            SamplingParams p; p.temperature = 1.0f; p.top_p = 0.0001f;
            CHECK(sample_token(row, 5, p, rng) == 3,
                  "sampler: tiny top_p collapses to the best token");
        }

        // Reproducibility: same seed + params -> same draw.
        {
            SamplingParams p; p.temperature = 1.0f; p.seed = 42;
            std::mt19937_64 a(p.seed), b(p.seed);
            int x = sample_token(row, 5, p, a);
            int y = sample_token(row, 5, p, b);
            CHECK(x == y, "sampler: fixed seed is reproducible");
        }
    }

    // ===== PHASE 14 / MR10: quantized matmul =====
    {
        // input [M=2, K=4], weight [K=4, N=3]
        Tensor in  = create_owned_tensor({2, 4}, DType::FLOAT32);
        Tensor w   = create_owned_tensor({4, 3}, DType::FLOAT32);
        for (int i = 0; i < 8;  i++) in.data()[i] = (float)((i % 5) - 2) * 0.5f;
        for (int i = 0; i < 12; i++) w.data()[i]  = (float)((i % 7) - 3) * 0.3f;

        // FP32 reference via the quantized path (type FP32).
        QuantizedWeight wf = quantize_weight(w, QuantType::FP32);
        QuantizedWeight w8 = quantize_weight(w, QuantType::Q8_0);
        QuantizedWeight w4 = quantize_weight(w, QuantType::Q4_0);

        Tensor rf = quantized_matmul(in, wf);
        Tensor r8 = quantized_matmul(in, w8);
        Tensor r4 = quantized_matmul(in, w4);

        CHECK(rf.shape[0] == 2 && rf.shape[1] == 3,
              "quantized_matmul output shape [M, N]");

        double e8 = 0.0, e4 = 0.0;
        for (int64_t i = 0; i < rf.numel(); i++) {
            e8 += std::fabs(r8.value(i) - rf.value(i));
            e4 += std::fabs(r4.value(i) - rf.value(i));
        }
        e8 /= rf.numel();
        e4 /= rf.numel();

        std::printf("      [quant] mean abs err  Q8_0=%.5f  Q4_0=%.5f\n",
                    e8, e4);

        CHECK(e8 <= e4 + 1e-6,
              "quant error: Q8_0 no worse than Q4_0");
        CHECK(e8 < 0.05,
              "quant error: Q8_0 close to FP32 reference");

        // Q8_1 (bias-corrected) on a positively-biased weight should
        // reconstruct at least as well as Q8_0.
        Tensor wb = create_owned_tensor({4, 3}, DType::FLOAT32);
        for (int i = 0; i < 12; i++) wb.data()[i] = 5.0f + (float)(i % 3) * 0.2f;
        QuantizedWeight wb8  = quantize_weight(wb, QuantType::Q8_0);
        QuantizedWeight wb81 = quantize_weight(wb, QuantType::Q8_1);
        QuantizedWeight wbf  = quantize_weight(wb, QuantType::FP32);
        Tensor rbf  = quantized_matmul(in, wbf);
        Tensor rb8  = quantized_matmul(in, wb8);
        Tensor rb81 = quantized_matmul(in, wb81);
        double eb8 = 0.0, eb81 = 0.0;
        for (int64_t i = 0; i < rbf.numel(); i++) {
            eb8  += std::fabs(rb8.value(i)  - rbf.value(i));
            eb81 += std::fabs(rb81.value(i) - rbf.value(i));
        }
        CHECK(eb81 <= eb8 + 1e-6,
              "quant error: Q8_1 (bias) no worse than Q8_0 on biased weight");

        // Memory comparison report.
        std::printf("      [quant] weight bytes  FP32=%lld  Q8=%lld  Q4=%lld\n",
                    (long long)wf.payload_bytes(),
                    (long long)w8.payload_bytes(),
                    (long long)w4.payload_bytes());
        CHECK(w8.payload_bytes() * 4 == wf.payload_bytes(),
              "quant memory: Q8 is 4x smaller than FP32");
        CHECK(w4.payload_bytes() * 8 <= wf.payload_bytes() + 8,
              "quant memory: Q4 is ~8x smaller than FP32");

        // Allocator statistics snapshot.
        AllocatorStats st = SystemAllocator::instance().stats();
        std::printf("      [alloc] peak=%lld active=%lld count=%lld\n",
                    (long long)st.peak_bytes,
                    (long long)st.active_allocations,
                    (long long)st.allocation_count);
        CHECK(st.peak_bytes >= 0, "allocator stats are queryable");
    }

    // ===== PHASE 14 / MR13: RoPE + MHA + GQA =====
    {
        // RoPE at position 0 is the identity.
        Tensor x0 = create_owned_tensor({1, 4}, DType::FLOAT32);
        float xv[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        for (int i = 0; i < 4; i++) x0.data()[i] = xv[i];
        rope_apply_rows(x0, 1, 4, 10000.0f);
        CHECK(std::fabs(x0.value(0) - 1.0f) < 1e-6 &&
              std::fabs(x0.value(1) - 2.0f) < 1e-6 &&
              std::fabs(x0.value(2) - 3.0f) < 1e-6 &&
              std::fabs(x0.value(3) - 4.0f) < 1e-6,
              "RoPE: position 0 is identity");

        // RoPE preserves the norm of each rotated pair.
        Tensor x1 = create_owned_tensor({2, 2}, DType::FLOAT32);
        x1.data()[0] = 0.0f; x1.data()[1] = 0.0f;   // pos 0
        x1.data()[2] = 1.0f; x1.data()[3] = 0.0f;   // pos 1
        rope_apply_rows(x1, 1, 2, 10000.0f);
        float n_in  = 1.0f;
        float n_out = std::sqrt(x1.value(2)*x1.value(2) +
                                x1.value(3)*x1.value(3));
        CHECK(std::fabs(n_out - n_in) < 1e-5,
              "RoPE: rotation preserves pair norm");

        // Known angle: pos 1, head_dim 2 -> freq=1, angle=1 rad.
        CHECK(std::fabs(x1.value(2) - std::cos(1.0f)) < 1e-5 &&
              std::fabs(x1.value(3) - std::sin(1.0f)) < 1e-5,
              "RoPE: known-angle pair matches cos/sin");

        // split_heads / merge_heads round-trip.
        Tensor h = create_owned_tensor({2, 6}, DType::FLOAT32);
        for (int i = 0; i < 12; i++) h.data()[i] = (float)i;
        std::vector<Tensor> hs = {
            split_head(h, 3, 2, 0),
            split_head(h, 3, 2, 1),
            split_head(h, 3, 2, 2)
        };
        Tensor remerged = merge_heads(hs);
        bool rt = (remerged.shape[0] == 2 && remerged.shape[1] == 6);
        for (int i = 0; i < 12 && rt; i++)
            rt = std::fabs(remerged.value(i) - (float)i) < 1e-6;
        CHECK(rt, "split_heads/merge_heads round-trip identity");

        // MHA shape: n_head=4, head_dim=2 -> q width 8.
        Tensor q = create_owned_tensor({3, 8}, DType::FLOAT32);
        Tensor k = create_owned_tensor({3, 8}, DType::FLOAT32);
        Tensor v = create_owned_tensor({3, 8}, DType::FLOAT32);
        for (int i = 0; i < 24; i++) {
            q.data()[i] = 0.01f * i;
            k.data()[i] = 0.02f * i;
            v.data()[i] = 0.03f * i;
        }
        MHAConfig mc; mc.n_head = 4; mc.n_kv_head = 4; mc.head_dim = 2;
        Tensor ctx = mha_attention(q, k, v, mc);
        CHECK(ctx.shape[0] == 3 && ctx.shape[1] == 8,
              "MHA: output shape [seq, n_head*head_dim]");

        // GQA: n_kv_head=1 broadcasts to all 4 query heads.
        Tensor kg = create_owned_tensor({3, 2}, DType::FLOAT32);
        Tensor vg = create_owned_tensor({3, 2}, DType::FLOAT32);
        for (int i = 0; i < 6; i++) { kg.data()[i] = 0.02f*i; vg.data()[i] = 0.03f*i; }
        MHAConfig gc; gc.n_head = 4; gc.n_kv_head = 1; gc.head_dim = 2;
        Tensor gctx = mha_attention(q, kg, vg, gc);
        CHECK(gctx.shape[0] == 3 && gctx.shape[1] == 8,
              "GQA: n_kv_head=1 broadcasts; output [seq, n_head*head_dim]");

        // First-token attention attends only to itself, so context row 0
        // equals V row 0 for every head (within each kv group).
        bool first_ok = true;
        for (int hh = 0; hh < 4 && first_ok; hh++) {
            first_ok = std::fabs(gctx.value(hh*2 + 0) - vg.value(0)) < 1e-4 &&
                       std::fabs(gctx.value(hh*2 + 1) - vg.value(1)) < 1e-4;
        }
        CHECK(first_ok, "GQA: first token attends only to itself (== V[0])");
    }

    // ===== PHASE 14 / MR13.1: incremental KV reuse equivalence =====
    {
        // Full MHA over [seq, *]; incremental over the same cache must
        // reproduce the LAST row exactly (MHA case).
        int seq = 4, n_head = 2, head_dim = 4;
        int width = n_head * head_dim;
        Tensor q = create_owned_tensor({seq, width}, DType::FLOAT32);
        Tensor k = create_owned_tensor({seq, width}, DType::FLOAT32);
        Tensor v = create_owned_tensor({seq, width}, DType::FLOAT32);
        for (int i = 0; i < seq * width; i++) {
            q.data()[i] = 0.013f * ((i * 7) % 11 - 5);
            k.data()[i] = 0.017f * ((i * 5) % 13 - 6);
            v.data()[i] = 0.019f * ((i * 3) % 9  - 4);
        }

        MHAConfig mc; mc.n_head = n_head; mc.n_kv_head = n_head; mc.head_dim = head_dim;
        Tensor full = mha_attention(q, k, v, mc);

        // q_row = last query row of q as [1, width]; cache = full k/v.
        Tensor q_last = create_owned_tensor({1, width}, DType::FLOAT32);
        for (int j = 0; j < width; j++)
            q_last.data()[j] = q.value((seq - 1) * width + j);

        Tensor inc = mha_attention_incremental(q_last, k, v, mc);

        bool eq = (inc.shape[0] == 1 && inc.shape[1] == width);
        for (int j = 0; j < width && eq; j++) {
            eq = std::fabs(inc.value(j) - full.value((seq - 1) * width + j)) < 1e-4;
        }
        CHECK(eq, "incremental attention == last row of full MHA");

        // GQA case: n_kv_head = 1.
        Tensor kg = create_owned_tensor({seq, head_dim}, DType::FLOAT32);
        Tensor vg = create_owned_tensor({seq, head_dim}, DType::FLOAT32);
        for (int i = 0; i < seq * head_dim; i++) {
            kg.data()[i] = 0.017f * ((i * 5) % 13 - 6);
            vg.data()[i] = 0.019f * ((i * 3) % 9  - 4);
        }
        MHAConfig gc; gc.n_head = n_head; gc.n_kv_head = 1; gc.head_dim = head_dim;
        Tensor gfull = mha_attention(q, kg, vg, gc);
        Tensor ginc  = mha_attention_incremental(q_last, kg, vg, gc);
        bool geq = true;
        for (int j = 0; j < width && geq; j++) {
            geq = std::fabs(ginc.value(j) - gfull.value((seq - 1) * width + j)) < 1e-4;
        }
        CHECK(geq, "incremental GQA attention == last row of full GQA");
    }

    // ===== PHASE 14.5: extended GGUF block sizing =====
    {
        CHECK(gguf_tensor_byte_size(GGUFType::Q4_1, 32) == 20, "Q4_1 block is 20 bytes");
        CHECK(gguf_tensor_byte_size(GGUFType::Q5_0, 32) == 22, "Q5_0 block is 22 bytes");
        CHECK(gguf_tensor_byte_size(GGUFType::Q5_1, 32) == 24, "Q5_1 block is 24 bytes");
        CHECK(gguf_tensor_byte_size(GGUFType::Q3_K, 256) == 110, "Q3_K super-block is 110 bytes");
        CHECK(gguf_tensor_byte_size(GGUFType::Q5_K, 256) == 176, "Q5_K super-block is 176 bytes");
    }

    // ===== PHASE 14.5: SIMD dot == scalar dot =====
    {
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (int trial = 0; trial < 4; trial++) {
            int64_t len = 1 + trial * 37;   // tails: 1, 38, 75, 112
            std::vector<float> a(len), b(len);
            for (int64_t i = 0; i < len; i++) { a[i] = dist(rng); b[i] = dist(rng); }
            double refd = 0.0;
            for (int64_t i = 0; i < len; i++) refd += (double)a[i] * (double)b[i];
            float got = simd_dot_f32(a.data(), b.data(), len);
            CHECK(std::fabs((double)got - refd) < 1e-3,
                  "simd_dot_f32 matches scalar reference");
        }
        std::printf("      [simd] avx2 path active = %d\n", simd_dot_has_avx2());
    }

    // ===== PHASE 14.5: NEOX RoPE correctness + regression =====
    {
        // NEOX rotates split-half pairs (i, i+half). Position 0 identity.
        Tensor x0 = create_owned_tensor({1, 4}, DType::FLOAT32);
        float xv[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        for (int i = 0; i < 4; i++) x0.data()[i] = xv[i];
        rope_apply_rows(x0, 1, 4, 10000.0f, RopeType::NEOX);
        CHECK(std::fabs(x0.value(0) - 1.0f) < 1e-6 &&
              std::fabs(x0.value(1) - 2.0f) < 1e-6 &&
              std::fabs(x0.value(2) - 3.0f) < 1e-6 &&
              std::fabs(x0.value(3) - 4.0f) < 1e-6,
              "NEOX RoPE: position 0 is identity");

        // head_dim=2, pos=1: half=1, pair (0,1), freq exp 0 -> angle=1.
        Tensor x1 = create_owned_tensor({2, 2}, DType::FLOAT32);
        x1.data()[0] = 0.0f; x1.data()[1] = 0.0f;
        x1.data()[2] = 1.0f; x1.data()[3] = 0.0f;
        rope_apply_rows(x1, 1, 2, 10000.0f, RopeType::NEOX);
        CHECK(std::fabs(x1.value(2) - std::cos(1.0f)) < 1e-5 &&
              std::fabs(x1.value(3) - std::sin(1.0f)) < 1e-5,
              "NEOX RoPE: known-angle half-pair matches cos/sin");

        float nrm = std::sqrt(x1.value(2)*x1.value(2) + x1.value(3)*x1.value(3));
        CHECK(std::fabs(nrm - 1.0f) < 1e-5,
              "NEOX RoPE: rotation preserves half-pair norm");

        // Regression: NEOX and NORM must DIVERGE on a head_dim>=4 input
        // at a NON-ZERO position, proving the convention is actually
        // selectable (the bug was hard-coded interleaved rotation for
        // every model). At position 0 both are the identity (angle = 0),
        // so we compare row 1 (position 1), where NEOX pairs (i, i+half)
        // and NORM pairs (2i, 2i+1) genuinely produce different results.
        Tensor xn = create_owned_tensor({2, 4}, DType::FLOAT32);
        Tensor xo = create_owned_tensor({2, 4}, DType::FLOAT32);
        for (int i = 0; i < 8; i++) { xn.data()[i] = 1.0f + i; xo.data()[i] = 1.0f + i; }
        rope_apply_rows(xn, 1, 4, 10000.0f, RopeType::NEOX);
        rope_apply_rows(xo, 1, 4, 10000.0f, RopeType::NORM);
        bool differ = false;
        for (int i = 0; i < 4; i++)   // row 1 only: elements 4..7
            differ = differ || std::fabs(xn.value(4 + i) - xo.value(4 + i)) > 1e-4;
        CHECK(differ, "RoPE NEOX != NORM at position 1 for head_dim=4 (convention selectable)");

        // Incremental NEOX row matches full NEOX last row.
        int seq = 3, hd = 4;
        Tensor q = create_owned_tensor({seq, hd}, DType::FLOAT32);
        for (int i = 0; i < seq * hd; i++) q.data()[i] = 0.1f * (i + 1);
        Tensor qfull = create_owned_tensor({seq, hd}, DType::FLOAT32);
        for (int i = 0; i < seq * hd; i++) qfull.data()[i] = q.value(i);
        rope_apply_rows(qfull, 1, hd, 10000.0f, RopeType::NEOX);
        Tensor qrow = create_owned_tensor({1, hd}, DType::FLOAT32);
        for (int j = 0; j < hd; j++) qrow.data()[j] = q.value((seq - 1) * hd + j);
        rope_apply_row_at(qrow, 1, hd, 10000.0f, seq - 1, RopeType::NEOX);
        bool inc_ok = true;
        for (int j = 0; j < hd && inc_ok; j++)
            inc_ok = std::fabs(qrow.value(j) - qfull.value((seq - 1) * hd + j)) < 1e-5;
        CHECK(inc_ok, "NEOX incremental row == full last row");
    }

    // ===== PHASE 14.5: GQA grouping regression =====
    {
        // n_head=4, n_kv_head=2 -> group=2. Heads 0,1 use kv head 0;
        // heads 2,3 use kv head 1. First-token context per head equals
        // its kv group's V[0].
        int hd = 2;
        Tensor q = create_owned_tensor({1, 4 * hd}, DType::FLOAT32);
        Tensor k = create_owned_tensor({1, 2 * hd}, DType::FLOAT32);
        Tensor v = create_owned_tensor({1, 2 * hd}, DType::FLOAT32);
        for (int i = 0; i < 4 * hd; i++) q.data()[i] = 0.01f * i;
        for (int i = 0; i < 2 * hd; i++) { k.data()[i] = 0.02f * i; v.data()[i] = 0.5f + 0.3f * i; }
        MHAConfig gc; gc.n_head = 4; gc.n_kv_head = 2; gc.head_dim = hd;
        Tensor ctx = mha_attention(q, k, v, gc);
        bool ok = true;
        for (int h = 0; h < 4 && ok; h++) {
            int kv = h / 2;
            for (int d = 0; d < hd && ok; d++)
                ok = std::fabs(ctx.value(h * hd + d) - v.value(kv * hd + d)) < 1e-4;
        }
        CHECK(ok, "GQA grouping: query heads map to correct kv group (first token)");
    }

    // ===== PHASE 14.5: synthetic GGUF dequant round-trip =====
    {
        const char* path = "axion_gguf_dequant_test.gguf";

        auto f32_to_f16 = [](float fv) -> uint16_t {
            uint32_t x; std::memcpy(&x, &fv, 4);
            uint32_t sign = (x >> 16) & 0x8000;
            int32_t  exp  = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
            uint32_t mant = x & 0x7FFFFF;
            if (exp <= 0) return (uint16_t)sign;
            if (exp >= 31) return (uint16_t)(sign | 0x7C00);
            return (uint16_t)(sign | (exp << 10) | (mant >> 13));
        };

        std::FILE* f = std::fopen(path, "wb");
        auto wr = [&](const void* d, size_t nbytes) { std::fwrite(d, 1, nbytes, f); };
        auto wstr = [&](const std::string& s) {
            uint64_t len = s.size(); wr(&len, 8); wr(s.data(), s.size());
        };

        const char magic[4] = {'G','G','U','F'};
        uint32_t version = 3;
        uint64_t n_tensors = 4;
        uint64_t n_meta = 0;
        wr(magic, 4); wr(&version, 4); wr(&n_tensors, 8); wr(&n_meta, 8);

        struct TD { const char* name; uint32_t type; uint64_t off; };
        uint64_t off_f16 = 0;
        uint64_t off_q8  = 64;
        uint64_t off_q4  = 64 + 34;
        uint64_t off_q81 = 64 + 34 + 18;
        TD dir[4] = {
            {"f16",  1, off_f16},
            {"q8_0", 8, off_q8},
            {"q4_0", 2, off_q4},
            {"q8_1", 9, off_q81},
        };
        for (int t = 0; t < 4; t++) {
            wstr(dir[t].name);
            uint32_t ndim = 1; wr(&ndim, 4);
            uint64_t dim = 32; wr(&dim, 8);
            wr(&dir[t].type, 4);
            wr(&dir[t].off, 8);
        }

        long pos = std::ftell(f);
        long aligned = ((pos + 31) / 32) * 32;
        for (long i = pos; i < aligned; i++) { char z = 0; wr(&z, 1); }

        // F16: values 0..31
        for (int i = 0; i < 32; i++) { uint16_t h = f32_to_f16((float)i); wr(&h, 2); }
        // Q8_0: d=0.5, q[i]=i-16 -> (i-16)*0.5
        { uint16_t d = f32_to_f16(0.5f); wr(&d, 2);
          for (int i = 0; i < 32; i++) { int8_t q = (int8_t)(i - 16); wr(&q, 1); } }
        // Q4_0: d=1.0, nibble=i%16 -> (nibble-8)
        { uint16_t d = f32_to_f16(1.0f); wr(&d, 2);
          for (int i = 0; i < 16; i++) {
              uint8_t lo = (uint8_t)((2*i)   % 16);
              uint8_t hi = (uint8_t)((2*i+1) % 16);
              uint8_t byte = (uint8_t)(lo | (hi << 4)); wr(&byte, 1);
          } }
        // Q8_1: d=0.25, m=1.0, q[i]=i-16 -> (i-16)*0.25 + 1
        { uint16_t d = f32_to_f16(0.25f); uint16_t m = f32_to_f16(1.0f);
          wr(&d, 2); wr(&m, 2);
          for (int i = 0; i < 32; i++) { int8_t q = (int8_t)(i - 16); wr(&q, 1); } }

        std::fclose(f);

        GGUFLoader gl;
        bool opened = false;
        try { opened = gl.load_file(path); } catch (...) { opened = false; }
        CHECK(opened, "synthetic GGUF opens and parses");

        if (opened) {
            Tensor f16t = gl.load_tensor("f16");
            bool f16_ok = true;
            for (int i = 0; i < 32 && f16_ok; i++)
                f16_ok = std::fabs(f16t.value(i) - (float)i) < 1e-2;
            CHECK(f16_ok, "GGUF F16 dequant matches 0..31");

            Tensor q8t = gl.load_tensor("q8_0");
            bool q8_ok = true;
            for (int i = 0; i < 32 && q8_ok; i++)
                q8_ok = std::fabs(q8t.value(i) - ((float)(i - 16) * 0.5f)) < 1e-3;
            CHECK(q8_ok, "GGUF Q8_0 dequant matches (i-16)*0.5");

            Tensor q4t = gl.load_tensor("q4_0");
            bool q4_ok = true;
            for (int i = 0; i < 32 && q4_ok; i++) {
                float expect = (float)((i % 16) - 8);
                q4_ok = std::fabs(q4t.value(i) - expect) < 1e-3;
            }
            CHECK(q4_ok, "GGUF Q4_0 dequant matches (nibble-8)");

            Tensor q81t = gl.load_tensor("q8_1");
            bool q81_ok = true;
            for (int i = 0; i < 32 && q81_ok; i++) {
                float expect = (float)(i - 16) * 0.25f + 1.0f;
                q81_ok = std::fabs(q81t.value(i) - expect) < 1e-3;
            }
            CHECK(q81_ok, "GGUF Q8_1 dequant matches (i-16)*0.25 + 1");
        }

        std::remove(path);
    }

    // ===== END OF CHECKS (anchor for later phases; keep this marker) =====

    std::printf("\n%s (%d failure%s)\n",
                failures == 0 ? "AXION RUNTIME VALIDATION PASSED"
                              : "AXION RUNTIME VALIDATION FAILED",
                failures,
                failures == 1 ? "" : "s");

    return failures == 0 ? 0 : 1;
}
