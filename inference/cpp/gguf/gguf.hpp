#pragma once

#include "../core/tensor.hpp"

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <cstdint>

namespace axion {

enum class GGUFType {
    F32  = 0,
    F16  = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15
};

// Exact serialized byte size for a GGML tensor type.
// Centralized and unit-tested: Q4_0/Q8_0/Q8_1/Q2_K/Q6_K/Q8_K sizes
// were previously wrong, undersizing the read buffer while dequant
// loops consumed the correct stride (heap overread / corruption).
uint64_t gguf_tensor_byte_size(
    GGUFType type,
    uint64_t element_count
);

struct GGUFMetadata {

    std::string key;

    uint32_t type = 0;

    // Decoded scalar value. Exactly one representation is meaningful
    // depending on `type`; numeric types fill both i_val and d_val so
    // callers can request either an integer or a float width.
    int64_t     i_val = 0;
    double      d_val = 0.0;
    std::string s_val;

    // For array values (type 9): number of elements. The array contents
    // are not retained for numeric scalar hparams, but the length is:
    // the LLaMA-family vocab size is the length of tokenizer.ggml.tokens,
    // since llama.vocab_size is often absent.
    uint64_t arr_len = 0;

    // Retained array contents. The tokenizer needs these: string arrays
    // (tokenizer.ggml.tokens, .merges) and int arrays (.token_type).
    // Float arrays (.scores) are retained as doubles in arr_f. Only the
    // representation matching the element type is populated.
    std::vector<std::string> arr_s;
    std::vector<int64_t>     arr_i;
    std::vector<double>      arr_f;
    uint32_t                 arr_elem_type = 0;

    bool is_string = false;
    bool is_float  = false;
    bool is_array  = false;
};

struct GGUFTensorInfo {

    std::string name;

    GGUFType type;

    std::vector<int64_t> shape;

    uint64_t offset = 0;

    uint64_t element_count = 0;

    uint64_t byte_size = 0;
};

class GGUFLoader {

public:

    GGUFLoader() = default;

    ~GGUFLoader();

    bool load_file(
        const std::string& path
    );

    Tensor load_tensor(
        const std::string& name
    );

    std::vector<std::string>
    tensor_names() const;

    // ----------------------------------------------------------------
    // Metadata accessors (Phase 13 / MR6)
    //
    // Hyperparameters live under architecture-prefixed keys, e.g.
    // "llama.block_count", "llama.attention.head_count". Callers may
    // pass the fully-qualified key, or use architecture() to build it.
    // ----------------------------------------------------------------

    bool has_metadata(
        const std::string& key
    ) const;

    uint32_t get_u32(
        const std::string& key,
        uint32_t fallback = 0
    ) const;

    int32_t get_i32(
        const std::string& key,
        int32_t fallback = 0
    ) const;

    uint64_t get_u64(
        const std::string& key,
        uint64_t fallback = 0
    ) const;

    float get_f32(
        const std::string& key,
        float fallback = 0.0f
    ) const;

    std::string get_str(
        const std::string& key,
        const std::string& fallback = ""
    ) const;

    // Length of a GGUF array-valued key (e.g. tokenizer.ggml.tokens).
    // Returns fallback if the key is missing or is not an array.
    uint64_t get_arr_len(
        const std::string& key,
        uint64_t fallback = 0
    ) const;

    // Retained string array (e.g. tokenizer.ggml.tokens, .merges).
    // Empty if the key is absent or not a string array.
    const std::vector<std::string>& get_str_array(
        const std::string& key
    ) const;

    // Retained integer array (e.g. tokenizer.ggml.token_type).
    // Empty if the key is absent or not an integer array.
    const std::vector<int64_t>& get_i32_array(
        const std::string& key
    ) const;

    // Retained float array (e.g. tokenizer.ggml.scores).
    // Empty if the key is absent or not a float array.
    const std::vector<double>& get_f32_array(
        const std::string& key
    ) const;

    // Shape of a tensor as parsed (row-major [outer, ..., inner]).
    // Empty if the tensor is absent. Used to derive vocab from
    // token_embd.weight when metadata does not carry it.
    std::vector<int64_t> tensor_shape(
        const std::string& name
    ) const;

    // "general.architecture" (e.g. "llama", "qwen2"); empty if absent.
    std::string architecture() const;

private:

    void parse_metadata();
    
    void parse_tensor_directory();

    uint64_t aligned_offset(
        uint64_t offset,
        uint64_t alignment
    );

private:

    std::ifstream file;

    uint32_t version = 0;

    uint64_t tensor_count = 0;

    uint64_t metadata_count = 0;

    uint64_t tensor_data_offset = 0;

    std::unordered_map<
        std::string,
        GGUFMetadata
    > metadata;

    std::unordered_map<
        std::string,
        GGUFTensorInfo
    > tensors;
};

}