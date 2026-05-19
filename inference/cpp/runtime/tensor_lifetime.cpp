#include "tensor_lifetime.hpp"

namespace axion {

void TensorLifetimeGraph::register_tensor(
    const std::string& tensor_name
) {

    TensorLifetimeNode node;

    node.tensor_name =
        tensor_name;

    nodes[tensor_name] =
        node;
}

void TensorLifetimeGraph::add_dependency(
    const std::string& parent,
    const std::string& child
) {

    nodes[parent]
        .dependencies
        .push_back(child);
}

bool TensorLifetimeGraph::can_release(
    const std::string& tensor_name
) {

    auto& node =
        nodes[tensor_name];

    for (auto& dep : node.dependencies) {

        if (!nodes[dep].released) {
            return false;
        }
    }

    return true;
}

void TensorLifetimeGraph::release_tensor(
    const std::string& tensor_name
) {

    nodes[tensor_name]
        .released = true;
}

}