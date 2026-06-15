#include "tokenizer.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace axion {

// Token type constants used by the GGUF SPM vocabulary.
static constexpr int TYPE_NORMAL  = 1;
static constexpr int TYPE_UNKNOWN = 2;
static constexpr int TYPE_CONTROL = 3;
static constexpr int TYPE_BYTE    = 6;

// SentencePiece marks a leading word boundary with U+2581 (\xE2\x96\x81).
static const std::string SPM_SPACE = "\xE2\x96\x81";

GGUFTokenizer::GGUFTokenizer(const GGUFLoader& loader) {

    const std::vector<std::string>& toks =
        loader.get_str_array("tokenizer.ggml.tokens");

    if (toks.empty()) {
        throw std::runtime_error(
            "GGUFTokenizer: model has no tokenizer.ggml.tokens array");
    }

    id_to_piece_ = toks;

    const std::vector<double>& scores =
        loader.get_f32_array("tokenizer.ggml.scores");
    const std::vector<int64_t>& types =
        loader.get_i32_array("tokenizer.ggml.token_type");

    id_to_score_.resize(id_to_piece_.size(), 0.0f);
    id_to_type_.resize(id_to_piece_.size(), TYPE_NORMAL);

    for (size_t i = 0; i < id_to_piece_.size(); i++) {
        if (i < scores.size()) id_to_score_[i] = (float)scores[i];
        if (i < types.size())  id_to_type_[i]  = (int)types[i];
        piece_to_id_[id_to_piece_[i]] = (int)i;
    }

    // Special token ids. Defaults match the LLaMA convention used by
    // TinyLlama (BOS=1, EOS=2) when the keys are absent.
    bos_token_id_ = (int)loader.get_u32("tokenizer.ggml.bos_token_id", 1);
    eos_token_id_ = (int)loader.get_u32("tokenizer.ggml.eos_token_id", 2);
    unk_token_id_ = (int)loader.get_u32("tokenizer.ggml.unknown_token_id", 0);
}

int GGUFTokenizer::piece_id(const std::string& piece) const {
    auto it = piece_to_id_.find(piece);
    return it == piece_to_id_.end() ? -1 : it->second;
}

int GGUFTokenizer::byte_id(uint8_t b) const {
    char buf[8];
    std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
    return piece_id(std::string(buf));
}

bool GGUFTokenizer::is_control(int id) const {
    if (id < 0 || id >= (int)id_to_type_.size()) return false;
    return id_to_type_[id] == TYPE_CONTROL ||
           id_to_type_[id] == TYPE_UNKNOWN;
}

// Length in bytes of the UTF-8 sequence beginning with lead byte c.
static int utf8_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c >> 5) == 0x6) return 2;
    if ((c >> 4) == 0xE) return 3;
    if ((c >> 3) == 0x1E) return 4;
    return 1;   // invalid lead byte: treat as single byte
}

std::vector<int> GGUFTokenizer::encode(
    const std::string& text,
    bool add_bos
) const {

    // 1. Normalize: SPM prefixes the string with a space (so the first
    //    word also gets a boundary marker) and replaces every space with
    //    U+2581.
    std::string norm = SPM_SPACE;
    for (char ch : text) {
        if (ch == ' ') norm += SPM_SPACE;
        else           norm.push_back(ch);
    }

    // 2. Seed one symbol per UTF-8 character.
    struct Symbol {
        std::string text;
        int prev;
        int next;
    };
    std::vector<Symbol> syms;
    for (size_t i = 0; i < norm.size();) {
        int len = utf8_len((unsigned char)norm[i]);
        if (i + len > norm.size()) len = 1;
        Symbol s;
        s.text = norm.substr(i, len);
        s.prev = (int)syms.size() - 1;
        s.next = (int)syms.size() + 1;
        syms.push_back(s);
        i += len;
    }
    if (!syms.empty()) syms.back().next = -1;

    // 3. Greedy merge: repeatedly fuse the adjacent symbol pair whose
    //    concatenation is the highest-scoring vocabulary entry.
    while (true) {
        int    best_left  = -1;
        float  best_score = -1e30f;
        for (int i = 0; i >= 0 && i < (int)syms.size(); i = syms[i].next) {
            int j = syms[i].next;
            if (j < 0) break;
            std::string merged = syms[i].text + syms[j].text;
            int id = piece_id(merged);
            if (id >= 0 && id_to_score_[id] > best_score) {
                best_score = id_to_score_[id];
                best_left  = i;
            }
        }
        if (best_left < 0) break;

        int l = best_left;
        int r = syms[l].next;
        syms[l].text += syms[r].text;
        syms[l].next  = syms[r].next;
        if (syms[r].next >= 0) syms[syms[r].next].prev = l;
    }

    // 4. Emit ids. A surviving symbol that is not a known piece is
    //    expanded to byte-fallback tokens (or the unk token if the
    //    model has no byte tokens).
    std::vector<int> out;
    if (add_bos && bos_token_id_ >= 0) {
        out.push_back(bos_token_id_);
    }

    for (int i = 0; i >= 0 && i < (int)syms.size(); i = syms[i].next) {
        int id = piece_id(syms[i].text);
        if (id >= 0) {
            out.push_back(id);
            continue;
        }
        for (unsigned char b : syms[i].text) {
            int bid = byte_id(b);
            if (bid >= 0) out.push_back(bid);
            else if (unk_token_id_ >= 0) out.push_back(unk_token_id_);
        }
    }

    return out;
}

std::string GGUFTokenizer::decode(
    const std::vector<int>& ids
) const {

    std::string out;

    for (int id : ids) {
        if (id < 0 || id >= (int)id_to_piece_.size()) continue;
        if (is_control(id)) continue;   // skip BOS/EOS/etc.

        const std::string& piece = id_to_piece_[id];

        // Byte-fallback token "<0xXX>" -> raw byte.
        if (id_to_type_[id] == TYPE_BYTE &&
            piece.size() == 6 &&
            piece[0] == '<' && piece[1] == '0' && piece[2] == 'x') {
            int byte = (int)std::strtol(piece.c_str() + 3, nullptr, 16);
            out.push_back((char)byte);
            continue;
        }

        // Replace each U+2581 marker with a space.
        for (size_t k = 0; k < piece.size();) {
            if (k + 2 < piece.size() &&
                (unsigned char)piece[k]   == 0xE2 &&
                (unsigned char)piece[k+1] == 0x96 &&
                (unsigned char)piece[k+2] == 0x81) {
                out.push_back(' ');
                k += 3;
            } else {
                out.push_back(piece[k]);
                k += 1;
            }
        }
    }

    return out;
}

}
