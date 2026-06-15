#pragma once

// In-process tokenizer built entirely from GGUF metadata.
//
// LLaMA-family GGUFs embed their tokenizer under tokenizer.ggml.*:
//   tokenizer.ggml.tokens      (string array) - id -> piece
//   tokenizer.ggml.scores      (float array)  - id -> merge score
//   tokenizer.ggml.token_type  (int array)    - id -> type (1 normal,
//                                                2 unknown, 3 control,
//                                                6 byte)
//   tokenizer.ggml.bos_token_id / .eos_token_id (scalars)
//
// This class implements the SentencePiece "unigram-as-BPE" encode used
// by llama.cpp's SPM path: replace spaces with U+2581, seed one symbol
// per UTF-8 character, then greedily merge the adjacent pair with the
// best (highest-score) vocabulary entry until no merge applies. Unknown
// characters fall back to <0xXX> byte tokens. Decode reverses the space
// marker and byte tokens.
//
// No external dependency: the GGUF file is the single source of truth.

#include "gguf.hpp"

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace axion {

class GGUFTokenizer {

public:

    // Build from an already-loaded GGUF. Throws if the model carries no
    // tokenizer.ggml.tokens array.
    explicit GGUFTokenizer(const GGUFLoader& loader);

    // Text -> token ids. When add_bos is true and the model defines a
    // BOS id, it is prepended.
    std::vector<int> encode(
        const std::string& text,
        bool add_bos = true
    ) const;

    // Token ids -> text. Renders the U+2581 space marker as ' ' and
    // decodes <0xXX> byte tokens back to raw bytes. Control tokens
    // (BOS/EOS/etc.) are skipped.
    std::string decode(
        const std::vector<int>& ids
    ) const;

    int bos_id() const { return bos_token_id_; }
    int eos_id() const { return eos_token_id_; }
    int vocab_size() const { return (int)id_to_piece_.size(); }

    // True if id is a control/special token (BOS, EOS, etc.).
    bool is_control(int id) const;

private:

    std::vector<std::string> id_to_piece_;
    std::vector<float>       id_to_score_;
    std::vector<int>         id_to_type_;
    std::unordered_map<std::string, int> piece_to_id_;

    int bos_token_id_ = -1;
    int eos_token_id_ = -1;
    int unk_token_id_ = -1;

    // Look up a piece; -1 if absent.
    int piece_id(const std::string& piece) const;

    // Byte-fallback token id for raw byte b ("<0xXX>"); -1 if the model
    // has no byte tokens.
    int byte_id(uint8_t b) const;
};

}
