#include "mlp.hpp"

#include "silu.hpp"
#include "elementwise.hpp"

namespace axion {

Tensor mlp_block(
    const Tensor& gate,
    const Tensor& up,
    RuntimeMemoryScheduler* scheduler
) {

    Tensor activated =
        silu(
            gate,
            scheduler
        );

    Tensor output =
        elementwise_mul(
            activated,
            up,
            scheduler
        );

    if (scheduler != nullptr) {

        scheduler->release_tensor(
            activated.name
        );
    }

    output.name =
        "mlp_output";

    return output;
}

}