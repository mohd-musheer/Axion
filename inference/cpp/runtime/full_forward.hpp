#pragma once

#include "../core/tensor.hpp"
#include "../core/mmap_loader.hpp"

namespace axion {

Tensor full_forward(
    const Tensor& embeddings,
    MMapLoader& loader,
    int num_layers,
    int num_heads,
    const Tensor& embedding_matrix
);

}