
#include "linear.hpp"

#include "../kernels/blas.hpp"

namespace axion {

Tensor linear(
    const Tensor& input,
    const Tensor& weight,
    RuntimeMemoryScheduler* scheduler
) {

    return matmul(
        input,
        weight
    );
}

}
