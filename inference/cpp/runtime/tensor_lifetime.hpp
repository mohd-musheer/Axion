#pragma once

#include <string>
#include <vector>
#include <unordered_map>

namespace axion {

struct TensorLifetimeNode {

    std::string tensor_name;

    std::vector<std::string> dependencies;

    bool released = false;
};

class TensorLifetimeGraph {

public:

    void register_tensor(
        const std::string& tensor_name
    );

    void add_dependency(
        const std::string& parent,
        const std::string& child
    );

    void release_tensor(
        const std::string& tensor_name
    );

    bool can_release(
        const std::string& tensor_name
    );

    void reset();

private:

    std::unordered_map<
        std::string,
        TensorLifetimeNode
    > nodes;
};

}
