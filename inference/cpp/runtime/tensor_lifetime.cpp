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

    auto it = nodes.find(tensor_name);

    if (it == nodes.end()) {
        return true;
    }

    for (auto& dep : it->second.dependencies) {

        auto d = nodes.find(dep);

        // Unregistered dependencies must not create
        // permanent released=false ghost nodes that
        // block release forever.
        if (d != nodes.end() &&
            !d->second.released) {

            return false;
        }
    }

    return true;
}

void TensorLifetimeGraph::release_tensor(
    const std::string& tensor_name
) {

    auto it = nodes.find(tensor_name);

    if (it != nodes.end()) {

        it->second.released = true;
    }
}

void TensorLifetimeGraph::reset() {

    nodes.clear();
}

}
