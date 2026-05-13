
#include "weight_lookup.hpp"

#include <stdexcept>

namespace axion {

std::string find_tensor_by_suffix(
    const std::vector<std::string>& tensor_names,
    const std::string& layer_name,
    const std::string& suffix
) {

    for (const auto& name : tensor_names) {

        // must belong to layer

        if (name.find(layer_name)
            == std::string::npos) {

            continue;
        }

        // must contain suffix

        if (name.find(suffix)
            == std::string::npos) {

            continue;
        }

        return name;
    }

    throw std::runtime_error(
        "Tensor not found: " +
        layer_name +
        " :: " +
        suffix
    );
}

}
