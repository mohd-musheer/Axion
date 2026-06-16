#pragma once

// --------------------------------------------------------------------
// Lightweight, env-gated profiling for the decode path.
//
// Enable at runtime with:   AXION_PROFILE=1
//
// When disabled (default) every macro/accessor compiles down to a
// trivial branch on a cached flag, so there is no measurable overhead
// in normal runs. Uses only <chrono> and <cstdio>; no external
// frameworks, no SIMD, no CUDA.
//
// Usage:
//
//   {
//       ScopedTimer t(prof::Phase::LOAD);   // accumulates into LOAD
//       ... work ...
//   }                                        // stops on scope exit
//
// The accumulators are global and thread-unaware on purpose: the
// decode path is single-threaded at the layer level (OpenMP only
// parallelizes inside matmul), and we want one total per phase.
// --------------------------------------------------------------------

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>

namespace axion {
namespace prof {

enum class Phase : int {
    LOAD = 0,        // GGUF file read (seek + read raw bytes)
    DEQUANT,         // dequantization to FP32
    TRANSPOSE,       // weight transpose
    ATTENTION,       // attention sub-block (norm, qkv, rope, mha, out)
    FFN,             // feed-forward sub-block (norm, gate/up, silu, down)
    DECODE_LAYER,    // one full decode_layer call
    DECODE_STEP,     // one full decode_step call (all layers)
    GENERATION,      // whole generation loop
    COUNT
};

inline const char* phase_name(Phase p) {
    switch (p) {
        case Phase::LOAD:         return "load";
        case Phase::DEQUANT:      return "dequant";
        case Phase::TRANSPOSE:    return "transpose";
        case Phase::ATTENTION:    return "attention";
        case Phase::FFN:          return "ffn";
        case Phase::DECODE_LAYER: return "decode_layer";
        case Phase::DECODE_STEP:  return "decode_step";
        case Phase::GENERATION:   return "generation";
        default:                  return "unknown";
    }
}

// Per-process state. Header-only via inline accessors.
struct State {
    bool     enabled = false;
    bool     resolved = false;
    double   ns[(int)Phase::COUNT] = {0};   // accumulated nanoseconds
    uint64_t calls[(int)Phase::COUNT] = {0};
};

inline State& state() {
    static State s;
    return s;
}

inline bool enabled() {
    State& s = state();
    if (!s.resolved) {
        const char* v = std::getenv("AXION_PROFILE");
        s.enabled  = (v != nullptr && std::strcmp(v, "0") != 0 && v[0] != '\0');
        s.resolved = true;
    }
    return s.enabled;
}

inline void add(Phase p, double nanos) {
    State& s = state();
    s.ns[(int)p]    += nanos;
    s.calls[(int)p] += 1;
}

inline double seconds(Phase p) {
    return state().ns[(int)p] / 1e9;
}

inline uint64_t calls(Phase p) {
    return state().calls[(int)p];
}

inline void reset() {
    State& s = state();
    for (int i = 0; i < (int)Phase::COUNT; i++) {
        s.ns[i] = 0;
        s.calls[i] = 0;
    }
}

// Accumulating scoped timer. Adds elapsed time to a phase on destruction.
class ScopedTimer {
public:
    explicit ScopedTimer(Phase p)
        : phase(p),
          active(enabled()) {
        if (active) {
            start = std::chrono::steady_clock::now();
        }
    }

    ~ScopedTimer() {
        if (active) {
            auto end = std::chrono::steady_clock::now();
            double nanos =
                std::chrono::duration<double, std::nano>(end - start).count();
            add(phase, nanos);
        }
    }

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    Phase phase;
    bool  active;
    std::chrono::steady_clock::time_point start;
};

}  // namespace prof
}  // namespace axion
