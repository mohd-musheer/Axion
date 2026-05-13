#include "final_norm.hpp"

#include "../kernels/rmsnorm.hpp"

namespace axion {

Tensor final_norm(
    const Tensor& input
) {

    Tensor weight;

    weight.name =
        "final_norm_weight";

    weight.shape = {
        input.shape[1]
    };

    weight.dtype =
        input.dtype;

    weight.data.resize(
        input.shape[1],
        1.0f
    );

    Tensor output =
        rmsnorm(
            input,
            weight
        );

    output.name =
        "final_norm_output";

    return output;
}

}