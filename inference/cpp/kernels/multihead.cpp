
#include "multihead.hpp"
#include "../core/tensor_factory.hpp"
#include <stdexcept>

namespace axion {

std::vector<Tensor> split_heads(
    const Tensor& input,
    int num_heads
) {

    if (input.shape.size() != 2) {

        throw std::runtime_error(
            "split_heads expects 2D tensor"
        );
    }

    int64_t seq_len =
        input.shape[0];

    int64_t hidden_dim =
        input.shape[1];

    if (hidden_dim % num_heads != 0) {

        throw std::runtime_error(
            "hidden_dim not divisible by num_heads"
        );
    }

    int64_t head_dim =
        hidden_dim / num_heads;

    std::vector<Tensor> heads;

    for (int h = 0;
         h < num_heads;
         h++) {

        Tensor head;

        head.name =
            "head_" +
            std::to_string(h);

        head.dtype =
            input.dtype;

        head.shape = {
            seq_len,
            head_dim
        };

   

        // -------------------------
        // STORAGE ROUTING
        // -------------------------

        if (input.owns_data()) {

            head.parent_owned_data =
                const_cast<std::vector<float>*>(
                    &input.owned_data
                );
        }
        else {

            head.data_ptr =
                const_cast<float*>(
                    input.data()
                );
        }

        head.fp16_ptr =
            input.fp16_ptr;

        head.is_fp16 =
            input.is_fp16;

        // -------------------------
        // STRIDED VIEW
        // -------------------------

        head.is_view = true;

        head.is_strided = true;

        head.view_offset =
            h * head_dim;

        head.stride =
            hidden_dim;

        head.view_numel =
            seq_len * head_dim;

        heads.push_back(head);
    }

    return heads;
}

Tensor merge_heads(
    const std::vector<Tensor>& heads,
    RuntimeMemoryScheduler* scheduler
) {

    if (heads.empty()) {

        throw std::runtime_error(
            "No heads to merge"
        );
    }

    int64_t seq_len =
        heads[0].shape[0];

    int64_t head_dim =
        heads[0].shape[1];

    int64_t num_heads =
        heads.size();

    int64_t hidden_dim =
        head_dim * num_heads;

    Tensor output;

    if (scheduler != nullptr) {

        output =
            scheduler->request_tensor(
                "merge_heads_output",
                {seq_len, hidden_dim},
                heads[0].dtype
            );
    }
    else {

        output =
            create_owned_tensor(
                {seq_len, hidden_dim},
                heads[0].dtype
            );
    }

    output.name =
        "merge_heads_output";

    for (int64_t t = 0;
         t < seq_len;
         t++) {

        for (int64_t h = 0;
             h < num_heads;
             h++) {

            for (int64_t d = 0;
                 d < head_dim;
                 d++) {

                int64_t dst_idx =
                    t * hidden_dim +
                    h * head_dim +
                    d;

                int64_t src_idx =
                    t * head_dim + d;

                output.data()[dst_idx] =
                    heads[h].value(src_idx);
            }
        }
    }

    return output;
}

}
