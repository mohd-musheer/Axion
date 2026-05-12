
#include "generation.hpp"

#include "transformer_layer.hpp"

#include <iostream>

namespace axion {

Tensor generate_tokens(
    const Tensor& input,
    int steps
) {

    Tensor hidden =
        input;

    for (int i = 0;
         i < steps;
         i++) {

        std::cout
            << "Generation step: "
            << i + 1
            << std::endl;

        hidden =
            transformer_layer(
                hidden
            );
    }

    hidden.name =
        "generated_hidden_state";

    return hidden;
}

}
