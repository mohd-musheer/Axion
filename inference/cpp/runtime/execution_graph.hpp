
#pragma once

#include "../core/tensor.hpp"

#include <string>

namespace axion {

Tensor execute_model(
    const std::string& safetensor_path,
    const Tensor& input
);

}
