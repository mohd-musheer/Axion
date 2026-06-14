
#pragma once

#include <string>
#include <vector>

namespace axion {

std::vector<std::string> discover_layers(
    const std::vector<std::string>& tensor_names
);

}
