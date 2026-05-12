
#include "mlp.hpp"

#include "silu.hpp"
#include "elementwise.hpp"

namespace axion {

Tensor mlp_block(
    const Tensor& gate,
    const Tensor& up
) {

    Tensor activated =
        silu(gate);

    Tensor output =
        elementwise_mul(
            activated,
            up
        );

    output.name =
        "mlp_output";

    return output;
}

}
