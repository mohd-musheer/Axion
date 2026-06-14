#pragma once

#include "tensor.hpp"

namespace axion {

class RuntimeMemoryScheduler;

// Unmanaged view: the caller guarantees that the base tensor's
// storage outlives the view.
Tensor tensor_view(
    const Tensor& base,
    int64_t offset,
    std::vector<int64_t> shape
);

// Managed view: pins `base.name` in the scheduler so the parent
// cannot be released while the view is alive. Pair with
// Tensor::release_view(scheduler) to unpin.
Tensor tensor_view(
    const Tensor& base,
    int64_t offset,
    std::vector<int64_t> shape,
    RuntimeMemoryScheduler* scheduler
);

}
