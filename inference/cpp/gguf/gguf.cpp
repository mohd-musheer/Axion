#include "gguf.hpp"
#include "../core/tensor_factory.hpp"
#include <iostream>
#include <stdexcept>
#include <cstring>

namespace axion {

static std::string read_gguf_string(
    std::ifstream& file
) {

    uint64_t len;

    file.read(
        reinterpret_cast<char*>(&len),
        sizeof(len)
    );

    std::string s;

    s.resize(len);

    file.read(
        s.data(),
        len
    );

    return s;
}

uint64_t GGUFLoader::aligned_offset(
    uint64_t offset,
    uint64_t alignment
) {

    return
        ((offset + alignment - 1)
        / alignment)
        * alignment;
}

bool GGUFLoader::load_file(
    const std::string& path
) {

    file.open(
        path,
        std::ios::binary
    );

    if (!file.is_open()) {

        throw std::runtime_error(
            "Failed to open GGUF"
        );
    }

    // -------------------------
    // MAGIC
    // -------------------------

    char magic[4];

    file.read(
        magic,
        4
    );

    if (
        magic[0] != 'G' ||
        magic[1] != 'G' ||
        magic[2] != 'U' ||
        magic[3] != 'F'
    ) {

        throw std::runtime_error(
            "Invalid GGUF file"
        );
    }

    // -------------------------
    // VERSION
    // -------------------------

    file.read(
        reinterpret_cast<char*>(&version),
        sizeof(version)
    );

    // -------------------------
    // COUNTS
    // -------------------------

    file.read(
        reinterpret_cast<char*>(&tensor_count),
        sizeof(tensor_count)
    );

    file.read(
        reinterpret_cast<char*>(&metadata_count),
        sizeof(metadata_count)
    );

    std::cout
        << "GGUF VERSION: "
        << version
        << std::endl;

    std::cout
        << "TENSORS: "
        << tensor_count
        << std::endl;

    std::cout
        << "METADATA KV: "
        << metadata_count
        << std::endl;

    // -------------------------
    // PARSE
    // -------------------------

    parse_metadata();

    parse_tensor_directory();

    return true;
}

void GGUFLoader::parse_metadata() {

    for (uint64_t i = 0;
         i < metadata_count;
         i++) {

        std::string key =
            read_gguf_string(file);

        uint32_t value_type;

        file.read(
            reinterpret_cast<char*>(
                &value_type
            ),
            sizeof(value_type)
        );

        metadata[key] = {
            key,
            value_type
        };

        // --------------------------------
        // TEMPORARY SKIP
        // --------------------------------

        switch (value_type) {

            // uint8
            case 0: {

                uint8_t v;

                file.read(
                    reinterpret_cast<char*>(&v),
                    sizeof(v)
                );

                break;
            }

            // int8
            case 1: {

                int8_t v;

                file.read(
                    reinterpret_cast<char*>(&v),
                    sizeof(v)
                );

                break;
            }

            // uint16
            case 2: {

                uint16_t v;

                file.read(
                    reinterpret_cast<char*>(&v),
                    sizeof(v)
                );

                break;
            }

            // int16
            case 3: {

                int16_t v;

                file.read(
                    reinterpret_cast<char*>(&v),
                    sizeof(v)
                );

                break;
            }

            // uint32
            case 4: {

                uint32_t v;

                file.read(
                    reinterpret_cast<char*>(&v),
                    sizeof(v)
                );

                break;
            }

            // int32
            case 5: {

                int32_t v;

                file.read(
                    reinterpret_cast<char*>(&v),
                    sizeof(v)
                );

                break;
            }

            // float32
            case 6: {

                float v;

                file.read(
                    reinterpret_cast<char*>(&v),
                    sizeof(v)
                );

                break;
            }

            // bool
            case 7: {

                bool v;

                file.read(
                    reinterpret_cast<char*>(&v),
                    sizeof(v)
                );

                break;
            }

            // string
            case 8: {

                std::string s =
                    read_gguf_string(file);

                break;
            }

            // array
            case 9: {

                uint32_t array_type;

                uint64_t array_len;

                file.read(
                    reinterpret_cast<char*>(
                        &array_type
                    ),
                    sizeof(array_type)
                );

                file.read(
                    reinterpret_cast<char*>(
                        &array_len
                    ),
                    sizeof(array_len)
                );

                // skip array contents

                for (uint64_t j = 0;
                    j < array_len;
                    j++) {

                    switch (array_type) {

                        case 0: {
                            uint8_t v;
                            file.read(reinterpret_cast<char*>(&v), sizeof(v));
                            break;
                        }

                        case 1: {
                            int8_t v;
                            file.read(reinterpret_cast<char*>(&v), sizeof(v));
                            break;
                        }

                        case 2: {
                            uint16_t v;
                            file.read(reinterpret_cast<char*>(&v), sizeof(v));
                            break;
                        }

                        case 3: {
                            int16_t v;
                            file.read(reinterpret_cast<char*>(&v), sizeof(v));
                            break;
                        }

                        case 4: {
                            uint32_t v;
                            file.read(reinterpret_cast<char*>(&v), sizeof(v));
                            break;
                        }

                        case 5: {
                            int32_t v;
                            file.read(reinterpret_cast<char*>(&v), sizeof(v));
                            break;
                        }

                        case 6: {
                            float v;
                            file.read(reinterpret_cast<char*>(&v), sizeof(v));
                            break;
                        }

                        case 7: {
                            bool v;
                            file.read(reinterpret_cast<char*>(&v), sizeof(v));
                            break;
                        }

                        case 8: {
                            read_gguf_string(file);
                            break;
                        }

                        default:

                            throw std::runtime_error(
                                "Unsupported GGUF array type"
                            );
                    }
                }

                break;
            }

            // uint64
            case 10: {

                uint64_t v;

                file.read(
                    reinterpret_cast<char*>(&v),
                    sizeof(v)
                );

                break;
            }

            // int64
            case 11: {

                int64_t v;

                file.read(
                    reinterpret_cast<char*>(&v),
                    sizeof(v)
                );

                break;
            }

            // float64
            case 12: {

                double v;

                file.read(
                    reinterpret_cast<char*>(&v),
                    sizeof(v)
                );

                break;
            }

            default: {

                std::cout
                    << "UNSUPPORTED METADATA TYPE: "
                    << value_type
                    << std::endl;

                throw std::runtime_error(
                    "Unsupported GGUF metadata type"
                );
            }
        }
    }
}

void GGUFLoader::parse_tensor_directory() {

    for (uint64_t i = 0;
         i < tensor_count;
         i++) {

        GGUFTensorInfo info;

        info.name =
            read_gguf_string(file);

        uint32_t n_dims;

        file.read(
            reinterpret_cast<char*>(&n_dims),
            sizeof(n_dims)
        );

        info.shape.resize(n_dims);

        info.element_count = 1;

        for (uint32_t d = 0;
             d < n_dims;
             d++) {

            uint64_t dim;

            file.read(
                reinterpret_cast<char*>(&dim),
                sizeof(dim)
            );

            info.shape[d] =
                dim;

            info.element_count *=
                dim;
        }

        uint32_t ggml_type;

        file.read(
            reinterpret_cast<char*>(&ggml_type),
            sizeof(ggml_type)
        );

        info.type =
            static_cast<GGUFType>(
                ggml_type
            );

        file.read(
            reinterpret_cast<char*>(
                &info.offset
            ),
            sizeof(info.offset)
        );

        // -------------------------
        // BYTE SIZE
        // -------------------------

        switch (info.type) {

            case GGUFType::F32:

                info.byte_size =
                    info.element_count * 4;
                break;

            case GGUFType::F16:

                info.byte_size =
                    info.element_count * 2;
                break;

            case GGUFType::Q8_0:

                info.byte_size =
                    info.element_count;
                break;

            case GGUFType::Q4_0:

                info.byte_size =
                    info.element_count / 2;
                break;
            case GGUFType::Q4_1:

                // 32-element blocks: 4 bytes scale+min + 16 bytes data
                info.byte_size =
                    (info.element_count / 32) * 20;
                break;

            case GGUFType::Q5_0:

                // 32-element blocks: 2 bytes scale + 16 bytes data + 4 bytes high bits
                info.byte_size =
                    (info.element_count / 32) * 22;
                break;

            case GGUFType::Q5_1:

                // 32-element blocks: 4 bytes scale+min + 16 bytes data + 4 bytes high bits
                info.byte_size =
                    (info.element_count / 32) * 24;
                break;

            case GGUFType::Q8_1:

                // 32-element blocks: 4 bytes scale + 4 bytes sum + 32 bytes data
                info.byte_size =
                    (info.element_count / 32) * 40;
                break;

            case GGUFType::Q2_K:

                // 256-element super-blocks
                info.byte_size =
                    (info.element_count / 256) * (64 + 64 + 2 + 2);
                break;

            case GGUFType::Q3_K:

                // 256-element super-blocks
                info.byte_size =
                    (info.element_count / 256) * (32 + 64 + 12 + 2);
                break;

            case GGUFType::Q4_K:

                // 256-element super-blocks: 2+2 byte d/dmin + 12 byte scales + 128 byte data
                info.byte_size =
                    (info.element_count / 256) * (2 + 2 + 12 + 128);
                break;

            case GGUFType::Q5_K:

                // 256-element super-blocks
                info.byte_size =
                    (info.element_count / 256) * (2 + 2 + 12 + 32 + 128);
                break;

            case GGUFType::Q6_K:

                // 256-element super-blocks: 128 + 64 + 16 + 1 bytes per block
                info.byte_size =
                    (info.element_count / 256) * (128 + 64 + 16 + 1);
                break;

            case GGUFType::Q8_K:

                // 256-element super-blocks: 256 + 32 + 2 bytes per block
                info.byte_size =
                    (info.element_count / 256) * (256 + 32 + 2);
                break;

            default: {

                std::cout
                    << "UNSUPPORTED GGML TENSOR TYPE: "
                    << ggml_type
                    << std::endl;

                std::cout
                    << "TENSOR NAME: "
                    << info.name
                    << std::endl;

                throw std::runtime_error(
                    "Unsupported GGUF tensor type"
                );
            }
        }

        tensors[info.name] =
            info;
    }

    tensor_data_offset =
        aligned_offset(
            file.tellg(),
            32
        );

    std::cout
        << "Parsed tensor directory"
        << std::endl;
}
Tensor GGUFLoader::load_tensor(
    const std::string& name
) {

    if (!tensors.count(name)) {

        throw std::runtime_error(
            "Tensor not found"
        );
    }

    auto& info =
        tensors[name];

    uint64_t absolute_offset =
        tensor_data_offset +
        info.offset;

    file.seekg(
        absolute_offset,
        std::ios::beg
    );

    // -------------------------
    // READ RAW BYTES
    // -------------------------

    std::vector<uint8_t> raw(info.byte_size);

    file.read(
        reinterpret_cast<char*>(raw.data()),
        info.byte_size
    );

    // -------------------------
    // DEQUANTIZE TO FLOAT32
    // -------------------------

    Tensor t =
        create_owned_tensor(
            info.shape,
            DType::FLOAT32
        );

    t.name = info.name;

    float* out = t.data();
    uint64_t n  = info.element_count;

    switch (info.type) {

        // --------------------------------------------------
        // F32 — no conversion needed, copy directly
        // --------------------------------------------------
        case GGUFType::F32: {

            std::memcpy(out, raw.data(), n * 4);
            break;
        }

        // --------------------------------------------------
        // F16 — convert each half to float
        // --------------------------------------------------
        case GGUFType::F16: {

            const uint16_t* src =
                reinterpret_cast<const uint16_t*>(raw.data());

            for (uint64_t i = 0; i < n; i++) {

                uint16_t h = src[i];

                uint32_t sign     = (h >> 15) & 0x1;
                uint32_t exponent = (h >> 10) & 0x1F;
                uint32_t mantissa =  h        & 0x3FF;

                uint32_t f;

                if (exponent == 0) {
                    if (mantissa == 0) {
                        f = sign << 31;
                    } else {
                        // Subnormal
                        exponent = 1;
                        while (!(mantissa & 0x400)) {
                            mantissa <<= 1;
                            exponent--;
                        }
                        mantissa &= 0x3FF;
                        f = (sign << 31)
                            | ((exponent + 127 - 15) << 23)
                            | (mantissa << 13);
                    }
                } else if (exponent == 31) {
                    // Inf or NaN
                    f = (sign << 31) | (0xFF << 23) | (mantissa << 13);
                } else {
                    f = (sign << 31)
                        | ((exponent + 127 - 15) << 23)
                        | (mantissa << 13);
                }

                std::memcpy(&out[i], &f, 4);
            }
            break;
        }

        // --------------------------------------------------
        // Q8_0 — 32-element blocks: 2-byte f16 scale + 32 int8
        // --------------------------------------------------
        case GGUFType::Q8_0: {

            const uint8_t* p   = raw.data();
            uint64_t num_blocks = n / 32;

            for (uint64_t b = 0; b < num_blocks; b++) {

                uint16_t scale_bits;
                std::memcpy(&scale_bits, p, 2);
                p += 2;

                // Decode f16 scale
                uint32_t sign     = (scale_bits >> 15) & 0x1;
                uint32_t exponent = (scale_bits >> 10) & 0x1F;
                uint32_t mantissa =  scale_bits        & 0x3FF;
                uint32_t f;
                if (exponent == 0)
                    f = 0;
                else if (exponent == 31)
                    f = (sign << 31) | (0xFF << 23) | (mantissa << 13);
                else
                    f = (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13);

                float scale;
                std::memcpy(&scale, &f, 4);

                for (int i = 0; i < 32; i++) {
                    out[b * 32 + i] =
                        static_cast<float>(
                            static_cast<int8_t>(p[i])
                        ) * scale;
                }

                p += 32;
            }
            break;
        }

        // --------------------------------------------------
        // Q4_0 — 32-element blocks: 2-byte f16 scale + 16 bytes (32 x 4-bit)
        // --------------------------------------------------
        case GGUFType::Q4_0: {

            const uint8_t* p    = raw.data();
            uint64_t num_blocks = n / 32;

            for (uint64_t b = 0; b < num_blocks; b++) {

                uint16_t scale_bits;
                std::memcpy(&scale_bits, p, 2);
                p += 2;

                uint32_t sign     = (scale_bits >> 15) & 0x1;
                uint32_t exponent = (scale_bits >> 10) & 0x1F;
                uint32_t mantissa =  scale_bits        & 0x3FF;
                uint32_t f;
                if (exponent == 0)
                    f = 0;
                else if (exponent == 31)
                    f = (sign << 31) | (0xFF << 23) | (mantissa << 13);
                else
                    f = (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13);

                float scale;
                std::memcpy(&scale, &f, 4);

                for (int i = 0; i < 16; i++) {
                    uint8_t byte = p[i];
                    // low nibble = element 2i, high nibble = element 2i+1
                    // values are shifted by -8 (unsigned 0..15 -> signed -8..7)
                    out[b * 32 + 2 * i + 0] =
                        (static_cast<float>(byte & 0x0F) - 8.0f) * scale;
                    out[b * 32 + 2 * i + 1] =
                        (static_cast<float>(byte >> 4)   - 8.0f) * scale;
                }

                p += 16;
            }
            break;
        }

        // --------------------------------------------------
        // Q4_K — 256-element super-blocks
        // Layout: d(f16) dmin(f16) scales(12 bytes) data(128 bytes)
        // --------------------------------------------------
        case GGUFType::Q4_K: {

            const uint8_t* p    = raw.data();
            uint64_t num_blocks = n / 256;

            for (uint64_t b = 0; b < num_blocks; b++) {

                // d and dmin are f16
                uint16_t d_bits, dmin_bits;
                std::memcpy(&d_bits,    p,     2);
                std::memcpy(&dmin_bits, p + 2, 2);
                p += 4;

                auto f16_to_f32 = [](uint16_t h) -> float {
                    uint32_t sign     = (h >> 15) & 0x1;
                    uint32_t exponent = (h >> 10) & 0x1F;
                    uint32_t mantissa =  h        & 0x3FF;
                    uint32_t f;
                    if (exponent == 0)
                        f = 0;
                    else if (exponent == 31)
                        f = (sign << 31) | (0xFF << 23) | (mantissa << 13);
                    else
                        f = (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13);
                    float result;
                    std::memcpy(&result, &f, 4);
                    return result;
                };

                float d    = f16_to_f32(d_bits);
                float dmin = f16_to_f32(dmin_bits);

                // 12 bytes encode 8 sub-block scales + 8 sub-block mins (6-bit each)
                const uint8_t* sc = p;
                p += 12;

                uint8_t scales[8], mins[8];
                scales[0] =  sc[0] & 63;
                scales[1] =  sc[1] & 63;
                scales[2] =  sc[2] & 63;
                scales[3] =  sc[3] & 63;
                scales[4] = (sc[8]  & 0xF) | ((sc[4] >> 6) << 4);
                scales[5] = (sc[9]  & 0xF) | ((sc[5] >> 6) << 4);
                scales[6] = (sc[10] & 0xF) | ((sc[6] >> 6) << 4);
                scales[7] = (sc[11] & 0xF) | ((sc[7] >> 6) << 4);
                mins[0] =  sc[4] & 63;
                mins[1] =  sc[5] & 63;
                mins[2] =  sc[6] & 63;
                mins[3] =  sc[7] & 63;
                mins[4] = (sc[8]  >> 4) | ((sc[4] >> 6) << 4);
                mins[5] = (sc[9]  >> 4) | ((sc[5] >> 6) << 4);
                mins[6] = (sc[10] >> 4) | ((sc[6] >> 6) << 4);
                mins[7] = (sc[11] >> 4) | ((sc[7] >> 6) << 4);

                // 128 bytes = 256 x 4-bit values
                for (int sub = 0; sub < 8; sub++) {

                    float sc_f  = d    * scales[sub];
                    float min_f = dmin * mins[sub];

                    for (int i = 0; i < 16; i++) {
                        uint8_t byte = p[sub * 16 + i];
                        out[b * 256 + sub * 32 + 2 * i + 0] =
                            sc_f * (byte & 0x0F) - min_f;
                        out[b * 256 + sub * 32 + 2 * i + 1] =
                            sc_f * (byte >> 4)   - min_f;
                    }
                }

                p += 128;
            }
            break;
        }

        // --------------------------------------------------
        // Q6_K — 256-element super-blocks
        // Layout: ql(128) qh(64) scales(16) d(f16)
        // --------------------------------------------------
        case GGUFType::Q6_K: {

            const uint8_t* p    = raw.data();
            uint64_t num_blocks = n / 256;

            for (uint64_t b = 0; b < num_blocks; b++) {

                const uint8_t* ql     = p;         // 128 bytes: low 4-bits
                const uint8_t* qh     = p + 128;   //  64 bytes: high 2-bits
                const int8_t*  sc     =
                    reinterpret_cast<const int8_t*>(p + 192); // 16 bytes: scales
                uint16_t d_bits;
                std::memcpy(&d_bits, p + 208, 2);

                auto f16_to_f32 = [](uint16_t h) -> float {
                    uint32_t sign     = (h >> 15) & 0x1;
                    uint32_t exponent = (h >> 10) & 0x1F;
                    uint32_t mantissa =  h        & 0x3FF;
                    uint32_t f;
                    if (exponent == 0)
                        f = 0;
                    else if (exponent == 31)
                        f = (sign << 31) | (0xFF << 23) | (mantissa << 13);
                    else
                        f = (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13);
                    float result;
                    std::memcpy(&result, &f, 4);
                    return result;
                };

                float d = f16_to_f32(d_bits);

                for (int i = 0; i < 128; i++) {

                    // Each ql byte holds low nibbles of two values
                    // Each qh byte holds high 2-bits for four values

                    int i_lo  = i;          // index into ql
                    int i_hi  = i / 2;      // index into qh byte
                    int shift = (i % 2) * 4;// which nibble in ql

                    uint8_t lo_lo = (ql[i_lo] & 0x0F);           // bits [3:0] of value 2i
                    uint8_t lo_hi = (ql[i_lo] >>    4);           // bits [3:0] of value 2i+1
                    uint8_t hi_lo = (qh[i_hi] >> (shift    )) & 3;
                    uint8_t hi_hi = (qh[i_hi] >> (shift + 2)) & 3;

                    int q0 = static_cast<int>((lo_lo | (hi_lo << 4))) - 32;
                    int q1 = static_cast<int>((lo_hi | (hi_hi << 4))) - 32;

                    int sub0 = (2 * i)     / 16;
                    int sub1 = (2 * i + 1) / 16;

                    out[b * 256 + 2 * i + 0] =
                        d * sc[sub0] * static_cast<float>(q0);
                    out[b * 256 + 2 * i + 1] =
                        d * sc[sub1] * static_cast<float>(q1);
                }

                p += (128 + 64 + 16 + 2);
            }
            break;
        }

        // --------------------------------------------------
        // Remaining types: read raw, best-effort cast
        // Add proper dequant here when needed
        // --------------------------------------------------
        default: {

            std::cout
                << "WARNING: no dequant for type "
                << static_cast<int>(info.type)
                << ", doing raw float reinterpret"
                << std::endl;

            uint64_t floats_to_copy =
                std::min(n, info.byte_size / 4);

            std::memcpy(out, raw.data(), floats_to_copy * 4);
            break;
        }
    }

    return t;
}

std::vector<std::string>
GGUFLoader::tensor_names() const {

    std::vector<std::string> names;

    for (auto& kv : tensors) {

        names.push_back(
            kv.first
        );
    }

    return names;
}

GGUFLoader::~GGUFLoader() {

    if (file.is_open()) {

        file.close();
    }
}

}