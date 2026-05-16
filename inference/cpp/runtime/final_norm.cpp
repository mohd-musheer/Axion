#include "final_norm.hpp"

#include "../kernels/rmsnorm.hpp"

namespace axion {

Tensor final_norm(
    const Tensor& input,
    const Tensor& weight,
    RuntimeMemoryScheduler* scheduler
) {

    Tensor output =
        rmsnorm(
            input,
            weight,
            1e-6f,
            scheduler
        );

    output.name =
        "final_norm_output";

    return output;
}

}