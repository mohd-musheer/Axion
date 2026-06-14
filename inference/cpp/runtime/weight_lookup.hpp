
#pragma once

#include <string>
#include <vector>

namespace axion {

std::string find_tensor_by_suffix(
    const std::vector<std::string>& tensor_names,
    const std::string& layer_name,
    const std::string& suffix
);

}
