#include "rope.hpp"

#include <cmath>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace axion {

void apply_rope(
    Tensor& q,
    Tensor& k,
    int position,
    int head_dim,
    float theta
) {

    if (q.data.size() != k.data.size()) {

        throw std::runtime_error(
            "Q and K size mismatch"
        );
    }

    if (head_dim % 2 != 0) {

        throw std::runtime_error(
            "head_dim must be even"
        );
    }

    int64_t total = q.data.size();

    if (total % head_dim != 0) {

    throw std::runtime_error(
        "tensor size not divisible by head_dim"
    );
}

    #pragma omp parallel for
    for (int64_t i = 0; i < total; i += head_dim) {

        for (int d = 0; d < head_dim; d += 2) {

           
            int idx1 = i + d;
            int idx2 = i + d + 1;

            float freq =
                1.0f / std::pow(
                    theta,
                    (float)d / (float)head_dim
                );

            float angle =
                position * freq;

            float cos_val =
                std::cos(angle);

            float sin_val =
                std::sin(angle);

            // ----- Q -----

            float q1 = q.data[idx1];
            float q2 = q.data[idx2];

            q.data[idx1] =
                q1 * cos_val -
                q2 * sin_val;

            q.data[idx2] =
                q1 * sin_val +
                q2 * cos_val;

            // ----- K -----

            float k1 = k.data[idx1];
            float k2 = k.data[idx2];

            k.data[idx1] =
                k1 * cos_val -
                k2 * sin_val;

            k.data[idx2] =
                k1 * sin_val +
                k2 * cos_val;
        }
    }
}

}