#include "final_norm.hpp"

#include "../kernels/rmsnorm.hpp"

namespace axion {

Tensor final_norm(
    const Tensor& input,
    const Tensor& weight,
    float eps,
    RuntimeMemoryScheduler* scheduler
) {

    Tensor output =
        rmsnorm(
            input,
            weight,
            eps,
            scheduler
        );

    output.name =
        "final_norm_output";

    return output;
}

}