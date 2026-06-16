#pragma once

#include "../core/tensor.hpp"
#include "../core/scheduler.hpp"
namespace axion {

// Final RMSNorm before the LM head. `eps` defaults to the LLaMA-family
// value but callers should pass the model's resolved
// attention.layer_norm_rms_epsilon (TinyLlama uses 1e-5, not 1e-6).
Tensor final_norm(
    const Tensor& input,
    const Tensor& weight,
    float eps,
     RuntimeMemoryScheduler* scheduler = nullptr
);

}