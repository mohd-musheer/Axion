
#include "layer_scheduler.hpp"

#include <set>

namespace axion {

std::vector<std::string> discover_layers(
    const std::vector<std::string>& tensor_names
) {

    std::set<std::string> unique_layers;

    for (const auto& name : tensor_names) {

        size_t pos =
            name.find("layers.");

        if (pos == std::string::npos) {
            continue;
        }

        size_t end =
            name.find('.', pos + 7);

        if (end == std::string::npos) {
            continue;
        }

        std::string layer =
            name.substr(
                pos,
                end - pos
            );

        unique_layers.insert(layer);
    }

    return std::vector<std::string>(
        unique_layers.begin(),
        unique_layers.end()
    );
}

}
