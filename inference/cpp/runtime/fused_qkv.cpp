#include "fused_qkv.hpp"

#include <stdexcept>

namespace axion {

QKV split_fused_qkv(
    const Tensor& fused
) {

    if (fused.shape.size() != 2) {

        throw std::runtime_error(
            "Expected 2D fused tensor"
        );
    }

    int64_t rows =
        fused.shape[0];

    int64_t cols =
        fused.shape[1];

    if (cols % 3 != 0) {

        throw std::runtime_error(
            "Fused QKV not divisible by 3"
        );
    }

    int64_t hidden =
        cols / 3;

    Tensor Q;
    Tensor K;
    Tensor V;

    Q.shape = { rows, hidden };
    K.shape = { rows, hidden };
    V.shape = { rows, hidden };

    Q.owned_data.resize(rows * hidden);
    K.owned_data.resize(rows * hidden);
    V.owned_data.resize(rows * hidden);

    for (int64_t r = 0; r < rows; r++) {

        for (int64_t c = 0; c < hidden; c++) {

            Q.data()[
                r * hidden + c
            ] =
            fused.data()    [
                r * cols + c
            ];

            K.data()[
                r * hidden + c
            ] =
            fused.data()[
                r * cols + hidden + c
            ];

            V.data()[
                r * hidden + c
            ] =
            fused.data()[
                r * cols + (2 * hidden) + c
            ];
        }
    }

    Q.name = "Q_weight";
    K.name = "K_weight";
    V.name = "V_weight";

    return {
        Q,
        K,
        V
    };
}

}