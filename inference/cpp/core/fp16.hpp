
// #pragma once

// #include <cstdint>

// namespace axion {

// inline float fp16_to_fp32(uint16_t h) {

//     uint32_t sign =
//         (h & 0x8000) << 16;

//     uint32_t exp =
//         (h & 0x7C00) >> 10;

//     uint32_t mant =
//         (h & 0x03FF);

//     uint32_t f;

//     if (exp == 0) {

//         if (mant == 0) {

//             f = sign;
//         }
//         else {

//             exp = 1;

//             while ((mant & 0x0400) == 0) {
//                 mant <<= 1;
//                 exp--;
//             }

//             mant &= 0x03FF;

//             exp =
//                 exp + (127 - 15);

//             mant <<= 13;

//             f =
//                 sign |
//                 (exp << 23) |
//                 mant;
//         }
//     }
//     else if (exp == 31) {

//         f =
//             sign |
//             0x7F800000 |
//             (mant << 13);
//     }
//     else {

//         exp =
//             exp + (127 - 15);

//         mant <<= 13;

//         f =
//             sign |
//             (exp << 23) |
//             mant;
//     }
    

//     return *reinterpret_cast<float*>(&f);
// }

// }


#pragma once

#include <cstdint>

namespace axion {

float fp16_to_fp32(
    uint16_t h
);

}