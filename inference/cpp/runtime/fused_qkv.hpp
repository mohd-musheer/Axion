
#pragma once

#include "../core/tensor.hpp"

namespace axion {

struct QKV {

    Tensor Q;
    Tensor K;
    Tensor V;
};

QKV split_fused_qkv(
    const Tensor& fused
);

}