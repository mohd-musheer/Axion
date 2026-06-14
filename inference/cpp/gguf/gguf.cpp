#include "gguf.hpp"
#include "../core/tensor_factory.hpp"
#include "../core/fp16.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
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

uint64_t gguf_tensor_byte_size(
    GGUFType type,
    uint64_t n
) {

    switch (type) {

        case GGUFType::F32:  return n * 4;
        case GGUFType::F16:  return n * 2;

        // 32-element blocks
        case GGUFType::Q4_0: return (n / 32) * 18;  // f16 d + 16B   (was 16: overread)
        case GGUFType::Q4_1: return (n / 32) * 20;  // f16 d,m + 16B
        case GGUFType::Q5_0: return (n / 32) * 22;  // f16 d + 4B qh + 16B
        case GGUFType::Q5_1: return (n / 32) * 24;  // f16 d,m + 4B qh + 16B
        case GGUFType::Q8_0: return (n / 32) * 34;  // f16 d + 32B   (was 32: overread)
        case GGUFType::Q8_1: return (n / 32) * 36;  // f16 d,s + 32B (was 40)

        // 256-element super-blocks
        case GGUFType::Q2_K: return (n / 256) * 84;   // 16 sc + 64 qs + 2x f16 (was 132)
        case GGUFType::Q3_K: return (n / 256) * 110;  // 32 hm + 64 qs + 12 sc + f16
        case GGUFType::Q4_K: return (n / 256) * 144;  // f16 d,dmin + 12 sc + 128 qs
        case GGUFType::Q5_K: return (n / 256) * 176;  // + 32B qh
        case GGUFType::Q6_K: return (n / 256) * 210;  // 128 ql + 64 qh + 16 sc + f16 d (was 209)
        case GGUFType::Q8_K: return (n / 256) * 292;  // f32 d + 256 qs + 16x i16 bsums (was 290)

        default:
            throw std::runtime_error(
                "Unsupported GGUF tensor type"
            );
    }
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

        GGUFMetadata md;
        md.key  = key;
        md.type = value_type;

        // --------------------------------
        // DECODE AND RETAIN SCALAR VALUES
        // (arrays are still skipped: not needed for scalar hparams)
        // --------------------------------

        switch (value_type) {

            // uint8
            case 0: {

                uint8_t v;

                file.read(
                    reinterpret_cast<char*>(&v),
                    sizeof(v)
                );

                md.i_val = static_cast<int64_t>(v);
                md.d_val = static_cast<double>(v);
                break;
            }

            // int8
            case 1: {

                int8_t v;

                file.read(
                    reinterpret_cast<char*>(&v),
                    sizeof(v)
                );

                md.i_val = static_cast<int64_t>(v);
                md.d_val = static_cast<double>(v);
                break;
            }

            // uint16
            case 2: {

                uint16_t v;

                file.read(
                    reinterpret_cast<char*>(&v),
                    sizeof(v)
                );

                md.i_val = static_cast<int64_t>(v);
                md.d_val = static_cast<double>(v);
                break;
            }

            // int16
            case 3: {

                int16_t v;

                file.read(
                    reinterpret_cast<char*>(&v),
                    sizeof(v)
                );

                md.i_val = static_cast<int64_t>(v);
                md.d_val = static_cast<double>(v);
                break;
            }

            // uint32
            case 4: {

                uint32_t v;

                file.read(
                    reinterpret_cast<char*>(&v),
                    sizeof(v)
                );

                md.i_val = static_cast<int64_t>(v);
                md.d_val = static_cast<double>(v);
                break;
            }

            // int32
            case 5: {

                int32_t v;

                file.read(
                    reinterpret_cast<char*>(&v),
                    sizeof(v)
                );

                md.i_val = static_cast<int64_t>(v);
                md.d_val = static_cast<double>(v);
                break;
            }

            // float32
            case 6: {

                float v;

                file.read(
                    reinterpret_cast<char*>(&v),
                    sizeof(v)
                );

                md.d_val    = static_cast<double>(v);
                md.i_val    = static_cast<int64_t>(v);
                md.is_float = true;
                break;
            }

            // bool
            case 7: {

                bool v;

                file.read(
                    reinterpret_cast<char*>(&v),
                    sizeof(v)
                );

                md.i_val = v ? 1 : 0;
                md.d_val = v ? 1.0 : 0.0;
                break;
            }

            // string
            case 8: {

                md.s_val =
                    read_gguf_string(file);

                md.is_string = true;
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

                md.i_val = static_cast<int64_t>(v);
                md.d_val = static_cast<double>(v);
                break;
            }

            // int64
            case 11: {

                int64_t v;

                file.read(
                    reinterpret_cast<char*>(&v),
                    sizeof(v)
                );

                md.i_val = v;
                md.d_val = static_cast<double>(v);
                break;
            }

            // float64
            case 12: {

                double v;

                file.read(
                    reinterpret_cast<char*>(&v),
                    sizeof(v)
                );

                md.d_val    = v;
                md.i_val    = static_cast<int64_t>(v);
                md.is_float = true;
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

        metadata[key] = md;
    }
}

// --------------------------------------------------------------------
// Metadata accessors
// --------------------------------------------------------------------

bool GGUFLoader::has_metadata(
    const std::string& key
) const {

    return metadata.count(key) != 0;
}

uint32_t GGUFLoader::get_u32(
    const std::string& key,
    uint32_t fallback
) const {

    auto it = metadata.find(key);
    if (it == metadata.end()) return fallback;
    return static_cast<uint32_t>(it->second.i_val);
}

int32_t GGUFLoader::get_i32(
    const std::string& key,
    int32_t fallback
) const {

    auto it = metadata.find(key);
    if (it == metadata.end()) return fallback;
    return static_cast<int32_t>(it->second.i_val);
}

uint64_t GGUFLoader::get_u64(
    const std::string& key,
    uint64_t fallback
) const {

    auto it = metadata.find(key);
    if (it == metadata.end()) return fallback;
    return static_cast<uint64_t>(it->second.i_val);
}

float GGUFLoader::get_f32(
    const std::string& key,
    float fallback
) const {

    auto it = metadata.find(key);
    if (it == metadata.end()) return fallback;
    return static_cast<float>(it->second.d_val);
}

std::string GGUFLoader::get_str(
    const std::string& key,
    const std::string& fallback
) const {

    auto it = metadata.find(key);
    if (it == metadata.end()) return fallback;
    return it->second.s_val;
}

std::string GGUFLoader::architecture() const {

    return get_str("general.architecture", "");
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
        // (centralized + unit-tested; the inline switch this
        //  replaces had wrong sizes for Q4_0/Q8_0/Q8_1/Q2_K/
        //  Q6_K/Q8_K, causing undersized reads and heap overread
        //  during dequantization)
        // -------------------------

        try {

            info.byte_size =
                gguf_tensor_byte_size(
                    info.type,
                    info.element_count
                );
        }
        catch (const std::exception&) {

            std::cout
                << "UNSUPPORTED GGML TENSOR TYPE: "
                << ggml_type
                << std::endl;

            std::cout
                << "TENSOR NAME: "
                << info.name
                << std::endl;

            throw;
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
// K-quant sub-block scale layout shared by Q4_K and Q5_K: 12 bytes
// encode eight 6-bit scales and eight 6-bit mins.
static void decode_k_scales_mins(
    const uint8_t* sc,
    uint8_t scales[8],
    uint8_t mins[8]
) {
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
    //
    // All f16 decoding routes through core/fp16.cpp's fp16_to_fp32 so
    // the loader cannot diverge from the rest of the runtime.
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
        // F16 — centralized decode (core/fp16.cpp)
        // --------------------------------------------------
        case GGUFType::F16: {
            const uint16_t* src =
                reinterpret_cast<const uint16_t*>(raw.data());
            for (uint64_t i = 0; i < n; i++) {
                out[i] = fp16_to_fp32(src[i]);
            }
            break;
        }

        // --------------------------------------------------
        // Q8_0 — 32-element blocks: f16 d + 32 int8
        // --------------------------------------------------
        case GGUFType::Q8_0: {
            const uint8_t* p = raw.data();
            uint64_t nb = n / 32;
            for (uint64_t b = 0; b < nb; b++) {
                uint16_t dbits; std::memcpy(&dbits, p, 2); p += 2;
                float d = fp16_to_fp32(dbits);
                for (int i = 0; i < 32; i++) {
                    out[b * 32 + i] = (float)(int8_t)p[i] * d;
                }
                p += 32;
            }
            break;
        }

        // --------------------------------------------------
        // Q8_1 — 32-element blocks: f16 d + f16 m + 32 int8
        // value = q*d + m
        // --------------------------------------------------
        case GGUFType::Q8_1: {
            const uint8_t* p = raw.data();
            uint64_t nb = n / 32;
            for (uint64_t b = 0; b < nb; b++) {
                uint16_t dbits, mbits;
                std::memcpy(&dbits, p,     2);
                std::memcpy(&mbits, p + 2, 2);
                p += 4;
                float d = fp16_to_fp32(dbits);
                float m = fp16_to_fp32(mbits);
                for (int i = 0; i < 32; i++) {
                    out[b * 32 + i] = (float)(int8_t)p[i] * d + m;
                }
                p += 32;
            }
            break;
        }

        // --------------------------------------------------
        // Q4_0 — 32-element blocks: f16 d + 16B (nibble - 8) * d
        // --------------------------------------------------
        case GGUFType::Q4_0: {
            const uint8_t* p = raw.data();
            uint64_t nb = n / 32;
            for (uint64_t b = 0; b < nb; b++) {
                uint16_t dbits; std::memcpy(&dbits, p, 2); p += 2;
                float d = fp16_to_fp32(dbits);
                for (int i = 0; i < 16; i++) {
                    uint8_t byte = p[i];
                    out[b * 32 + 2 * i + 0] = ((float)(byte & 0x0F) - 8.0f) * d;
                    out[b * 32 + 2 * i + 1] = ((float)(byte >> 4)   - 8.0f) * d;
                }
                p += 16;
            }
            break;
        }

        // --------------------------------------------------
        // Q4_1 — f16 d + f16 m + 16B; value = nibble*d + m
        // --------------------------------------------------
        case GGUFType::Q4_1: {
            const uint8_t* p = raw.data();
            uint64_t nb = n / 32;
            for (uint64_t b = 0; b < nb; b++) {
                uint16_t dbits, mbits;
                std::memcpy(&dbits, p,     2);
                std::memcpy(&mbits, p + 2, 2);
                p += 4;
                float d = fp16_to_fp32(dbits);
                float m = fp16_to_fp32(mbits);
                for (int i = 0; i < 16; i++) {
                    uint8_t byte = p[i];
                    out[b * 32 + 2 * i + 0] = (float)(byte & 0x0F) * d + m;
                    out[b * 32 + 2 * i + 1] = (float)(byte >> 4)   * d + m;
                }
                p += 16;
            }
            break;
        }

        // --------------------------------------------------
        // Q5_0 — f16 d + 4B qh + 16B; value = ((nib|(hbit<<4))-16)*d
        // --------------------------------------------------
        case GGUFType::Q5_0: {
            const uint8_t* p = raw.data();
            uint64_t nb = n / 32;
            for (uint64_t b = 0; b < nb; b++) {
                uint16_t dbits; std::memcpy(&dbits, p, 2); p += 2;
                float d = fp16_to_fp32(dbits);
                uint32_t qh; std::memcpy(&qh, p, 4); p += 4;
                for (int i = 0; i < 16; i++) {
                    uint8_t byte = p[i];
                    uint8_t h0 = (qh >> (i))      & 1;
                    uint8_t h1 = (qh >> (i + 16)) & 1;
                    int q0 = (int)((byte & 0x0F) | (h0 << 4)) - 16;
                    int q1 = (int)((byte >> 4)   | (h1 << 4)) - 16;
                    out[b * 32 + 2 * i + 0] = (float)q0 * d;
                    out[b * 32 + 2 * i + 1] = (float)q1 * d;
                }
                p += 16;
            }
            break;
        }

        // --------------------------------------------------
        // Q5_1 — f16 d + f16 m + 4B qh + 16B; value=(nib|(hbit<<4))*d+m
        // --------------------------------------------------
        case GGUFType::Q5_1: {
            const uint8_t* p = raw.data();
            uint64_t nb = n / 32;
            for (uint64_t b = 0; b < nb; b++) {
                uint16_t dbits, mbits;
                std::memcpy(&dbits, p,     2);
                std::memcpy(&mbits, p + 2, 2);
                p += 4;
                float d = fp16_to_fp32(dbits);
                float m = fp16_to_fp32(mbits);
                uint32_t qh; std::memcpy(&qh, p, 4); p += 4;
                for (int i = 0; i < 16; i++) {
                    uint8_t byte = p[i];
                    uint8_t h0 = (qh >> (i))      & 1;
                    uint8_t h1 = (qh >> (i + 16)) & 1;
                    int q0 = (int)((byte & 0x0F) | (h0 << 4));
                    int q1 = (int)((byte >> 4)   | (h1 << 4));
                    out[b * 32 + 2 * i + 0] = (float)q0 * d + m;
                    out[b * 32 + 2 * i + 1] = (float)q1 * d + m;
                }
                p += 16;
            }
            break;
        }

        // --------------------------------------------------
        // Q2_K — 256-element super-blocks
        // Layout: scales(16) qs(64) d(f16) dmin(f16)
        // --------------------------------------------------
        case GGUFType::Q2_K: {
            const uint8_t* p = raw.data();
            uint64_t nb = n / 256;
            for (uint64_t b = 0; b < nb; b++) {
                const uint8_t* scales = p;          // 16 bytes
                const uint8_t* qs     = p + 16;     // 64 bytes
                uint16_t dbits, dmbits;
                std::memcpy(&dbits,  p + 80, 2);
                std::memcpy(&dmbits, p + 82, 2);
                float d    = fp16_to_fp32(dbits);
                float dmin = fp16_to_fp32(dmbits);
                for (int g = 0; g < 16; g++) {
                    float sc = d    * (scales[g] & 0x0F);
                    float mn = dmin * (scales[g] >> 4);
                    for (int i = 0; i < 16; i++) {
                        int elem = g * 16 + i;
                        int byte_idx = elem / 4;
                        int shift    = (elem % 4) * 2;
                        uint8_t q = (qs[byte_idx] >> shift) & 0x3;
                        out[b * 256 + elem] = sc * (float)q - mn;
                    }
                }
                p += 84;
            }
            break;
        }

        // --------------------------------------------------
        // Q3_K — 256-element super-blocks
        // Layout: hmask(32) qs(64) scales(12) d(f16)
        // --------------------------------------------------
        case GGUFType::Q3_K: {
            const uint8_t* p = raw.data();
            uint64_t nb = n / 256;
            for (uint64_t b = 0; b < nb; b++) {
                const uint8_t* hmask  = p;          // 32 bytes
                const uint8_t* qs     = p + 32;     // 64 bytes
                const uint8_t* scbits = p + 96;     // 12 bytes
                uint16_t dbits; std::memcpy(&dbits, p + 108, 2);
                float d = fp16_to_fp32(dbits);
                int8_t scales[16];
                for (int i = 0; i < 16; i++) {
                    uint8_t s;
                    if (i < 8) {
                        s = (scbits[i] & 0xF) |
                            (((scbits[8 + (i / 2)] >> (4 * (i % 2))) & 0x3) << 4);
                    } else {
                        int j = i - 8;
                        s = (scbits[j] >> 4) |
                            (((scbits[8 + (j / 2)] >> (2 + 4 * (j % 2))) & 0x3) << 4);
                    }
                    scales[i] = (int8_t)s - 32;
                }
                for (int elem = 0; elem < 256; elem++) {
                    int sub = elem / 16;
                    int byte_idx = elem / 4;
                    int shift    = (elem % 4) * 2;
                    int lo = (qs[byte_idx] >> shift) & 0x3;
                    int hi = (hmask[elem / 8] >> (elem % 8)) & 1;
                    int q = lo | (hi << 2);
                    out[b * 256 + elem] =
                        d * (float)scales[sub] * (float)(q - 4);
                }
                p += 110;
            }
            break;
        }

        // --------------------------------------------------
        // Q4_K — 256-element super-blocks
        // Layout: d(f16) dmin(f16) scales(12) qs(128)
        // --------------------------------------------------
        case GGUFType::Q4_K: {
            const uint8_t* p = raw.data();
            uint64_t nb = n / 256;
            for (uint64_t b = 0; b < nb; b++) {
                uint16_t d_bits, dmin_bits;
                std::memcpy(&d_bits,    p,     2);
                std::memcpy(&dmin_bits, p + 2, 2);
                p += 4;
                float d    = fp16_to_fp32(d_bits);
                float dmin = fp16_to_fp32(dmin_bits);
                const uint8_t* sc = p;
                p += 12;
                uint8_t scales[8], mins[8];
                decode_k_scales_mins(sc, scales, mins);
                for (int sub = 0; sub < 8; sub++) {
                    float sc_f  = d    * scales[sub];
                    float min_f = dmin * mins[sub];
                    for (int i = 0; i < 16; i++) {
                        uint8_t byte = p[sub * 16 + i];
                        out[b * 256 + sub * 32 + 2 * i + 0] = sc_f * (byte & 0x0F) - min_f;
                        out[b * 256 + sub * 32 + 2 * i + 1] = sc_f * (byte >> 4)   - min_f;
                    }
                }
                p += 128;
            }
            break;
        }

        // --------------------------------------------------
        // Q5_K — 256-element super-blocks
        // Layout: d(f16) dmin(f16) scales(12) qh(32) qs(128)
        // --------------------------------------------------
        case GGUFType::Q5_K: {
            const uint8_t* p = raw.data();
            uint64_t nb = n / 256;
            for (uint64_t b = 0; b < nb; b++) {
                uint16_t d_bits, dmin_bits;
                std::memcpy(&d_bits,    p,     2);
                std::memcpy(&dmin_bits, p + 2, 2);
                p += 4;
                float d    = fp16_to_fp32(d_bits);
                float dmin = fp16_to_fp32(dmin_bits);
                const uint8_t* sc = p; p += 12;
                const uint8_t* qh = p; p += 32;
                const uint8_t* qs = p; p += 128;
                uint8_t scales[8], mins[8];
                decode_k_scales_mins(sc, scales, mins);
                for (int sub = 0; sub < 8; sub++) {
                    float sc_f  = d    * scales[sub];
                    float min_f = dmin * mins[sub];
                    for (int i = 0; i < 16; i++) {
                        int e0 = sub * 32 + 2 * i + 0;
                        int e1 = sub * 32 + 2 * i + 1;
                        uint8_t byte = qs[sub * 16 + i];
                        uint8_t h0 = (qh[e0 >> 3] >> (e0 & 7)) & 1;
                        uint8_t h1 = (qh[e1 >> 3] >> (e1 & 7)) & 1;
                        float q0 = (float)((byte & 0x0F) | (h0 << 4));
                        float q1 = (float)((byte >> 4)   | (h1 << 4));
                        out[b * 256 + e0] = sc_f * q0 - min_f;
                        out[b * 256 + e1] = sc_f * q1 - min_f;
                    }
                }
            }
            break;
        }

        // --------------------------------------------------
        // Q6_K — 256-element super-blocks
        // Layout: ql(128) qh(64) scales(16, int8) d(f16)
        // --------------------------------------------------
        case GGUFType::Q6_K: {
            const uint8_t* p = raw.data();
            uint64_t nb = n / 256;
            for (uint64_t b = 0; b < nb; b++) {
                const uint8_t* ql = p;
                const uint8_t* qh = p + 128;
                const int8_t*  sc =
                    reinterpret_cast<const int8_t*>(p + 192);
                uint16_t d_bits; std::memcpy(&d_bits, p + 208, 2);
                float d = fp16_to_fp32(d_bits);
                for (int i = 0; i < 128; i++) {
                    int i_hi  = i / 2;
                    int shift = (i % 2) * 4;
                    uint8_t lo_lo = (ql[i] & 0x0F);
                    uint8_t lo_hi = (ql[i] >> 4);
                    uint8_t hi_lo = (qh[i_hi] >> (shift    )) & 3;
                    uint8_t hi_hi = (qh[i_hi] >> (shift + 2)) & 3;
                    int q0 = (int)(lo_lo | (hi_lo << 4)) - 32;
                    int q1 = (int)(lo_hi | (hi_hi << 4)) - 32;
                    int sub0 = (2 * i)     / 16;
                    int sub1 = (2 * i + 1) / 16;
                    out[b * 256 + 2 * i + 0] = d * sc[sub0] * (float)q0;
                    out[b * 256 + 2 * i + 1] = d * sc[sub1] * (float)q1;
                }
                p += 210;
            }
            break;
        }

        // --------------------------------------------------
        // Any type without a decode path must FAIL LOUDLY.
        // Silently reinterpreting quantized bytes as float (the old
        // behavior) produced a model that "loaded" but generated
        // garbage. Real-model validation requires a hard error here.
        // --------------------------------------------------
        default: {
            throw std::runtime_error(
                "GGUF dequant not implemented for tensor type " +
                std::to_string(static_cast<int>(info.type)) +
                " (tensor '" + info.name + "')");
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