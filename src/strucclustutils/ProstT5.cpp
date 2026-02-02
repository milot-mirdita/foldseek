#include "ProstT5.h"

#include "llama.h"

#include <array>
#include <cmath>
#include <cctype>
#include <cstring>
#include <limits>
#include <vector>

static const std::array<float, 20> PROFILE_BACKPROBS = {
    0.0489372f, 0.0306991f, 0.1010490f, 0.0329671f, 0.0276149f,
    0.0416262f, 0.0452521f, 0.0308760f, 0.0297251f, 0.0607036f,
    0.0150238f, 0.0215826f, 0.0783843f, 0.0512926f, 0.0264886f,
    0.0610702f, 0.0201311f, 0.2159980f, 0.0310265f, 0.0295417f,
};
static const float PROFILE_BIT_FACTOR = 8.0f;

static char number_to_char(unsigned int n) {
    switch(n) {
        case 0:  return 'A';
        case 1:  return 'C';
        case 2:  return 'D';
        case 3:  return 'E';
        case 4:  return 'F';
        case 5:  return 'G';
        case 6:  return 'H';
        case 7:  return 'I';
        case 8:  return 'K';
        case 9:  return 'L';
        case 10: return 'M';
        case 11: return 'N';
        case 12: return 'P';
        case 13: return 'Q';
        case 14: return 'R';
        case 15: return 'S';
        case 16: return 'T';
        case 17: return 'V';
        case 18: return 'W';
        case 19: return 'Y';
        default: return 'X'; // Default case for numbers not in the list
    }
}

static llama_token token_from_piece(
    const llama_vocab * vocab,
    const std::string & piece,
    bool parse_special) {
    llama_token buf[8];
    const int32_t n_tokens_max = static_cast<int32_t>(sizeof(buf) / sizeof(buf[0]));
    int32_t n = llama_tokenize(vocab, piece.c_str(), static_cast<int32_t>(piece.size()),
        buf, n_tokens_max, false, parse_special);
    if (n == 1) {
        return buf[0];
    }
    if (n < 0) {
        std::vector<llama_token> tmp(static_cast<size_t>(-n));
        n = llama_tokenize(vocab, piece.c_str(), static_cast<int32_t>(piece.size()),
            tmp.data(), static_cast<int32_t>(tmp.size()), false, parse_special);
        if (n == 1) {
            return tmp[0];
        }
    }
    return LLAMA_TOKEN_NULL;
}

static int encode(
    llama_context * ctx,
    std::vector<llama_token> & enc_input,
    size_t pred_len,
    size_t output_len,
    std::string & result) {
    const struct llama_model * model = llama_get_model(ctx);
    const struct llama_vocab * vocab = llama_model_get_vocab(model);

    if (llama_encode(ctx, llama_batch_get_one(enc_input.data(), enc_input.size())) < 0) {
        // LOG_ERR("%s : failed to encode\n", __func__);
        return 1;
    }
    llama_synchronize(ctx);

    // LOG_INF("%s: n_tokens = %zu, n_seq = %d\n", __func__, enc_input.size(), 1);
    float* embeddings = llama_get_embeddings(ctx);
    if (embeddings == nullptr) {
        return 1;
    }
    if (pred_len == 0 || output_len == 0) {
        return 0;
    }
    if (output_len > pred_len) {
        output_len = pred_len;
    }
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const uint32_t n_cls_out = llama_model_n_cls_out(model);
    uint32_t n_classes = n_cls_out > 0 ? n_cls_out : 20;
    if (n_classes == 1 && n_vocab == 150) {
        n_classes = 20;
    }
    const bool token_major = (n_vocab == 28);
    std::vector<int> arg_max_idx(pred_len);
    std::vector<float> arg_max(pred_len, std::numeric_limits<float>::lowest());
    for (uint32_t i = 0; i < n_classes; ++i) {
        for (size_t j = 0; j < pred_len; ++j) {
            const size_t idx = token_major ? (j*n_classes + i) : (i*pred_len + j);
            if (embeddings[idx] > arg_max[j]) {
                arg_max_idx[j] = i;
                arg_max[j] = embeddings[idx];
            }
        }
    }
    for (size_t i = 0; i < output_len; ++i) {
        result.push_back(number_to_char(arg_max_idx[i]));
    }
    return 0;
}

static inline int8_t pssm_from_prob(float prob, float backprob) {
    float odds = backprob > 0.0f ? (prob / backprob) : 0.0f;
    float log_prob = odds > 0.0f ? std::log2f(odds) : -128.0f;
    float pssm = log_prob * PROFILE_BIT_FACTOR;
    pssm = (pssm < 0.0f) ? (pssm - 0.5f) : (pssm + 0.5f);
    if (pssm < -128.0f) {
        pssm = -128.0f;
    } else if (pssm > 127.0f) {
        pssm = 127.0f;
    }
    return static_cast<int8_t>(pssm);
}

static int encode_profile(
    llama_context * ctx,
    std::vector<llama_token> & enc_input,
    size_t pred_len,
    size_t output_len,
    std::string & result) {
    const struct llama_model * model = llama_get_model(ctx);
    const struct llama_vocab * vocab = llama_model_get_vocab(model);

    if (llama_encode(ctx, llama_batch_get_one(enc_input.data(), enc_input.size())) < 0) {
        return 1;
    }
    llama_synchronize(ctx);

    float * embeddings = llama_get_embeddings(ctx);
    if (embeddings == nullptr) {
        return 1;
    }

    if (pred_len == 0 || output_len == 0) {
        return 0;
    }
    if (output_len > pred_len) {
        output_len = pred_len;
    }

    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const uint32_t n_cls_out = llama_model_n_cls_out(model);
    uint32_t n_classes = n_cls_out > 0 ? n_cls_out : 20;
    if (n_classes == 1 && n_vocab == 150) {
        n_classes = 20;
    }
    if (n_classes != PROFILE_BACKPROBS.size()) {
        return 1;
    }

    const bool token_major = (n_vocab == 28);
    result.clear();
    result.reserve(output_len * 25);

    std::array<float, 20> logits{};
    std::array<float, 20> exps{};
    std::array<char, 20> pssm_row{};

    for (size_t j = 0; j < output_len; ++j) {
        float max_logit = -std::numeric_limits<float>::infinity();
        int8_t consensus = 0;

        for (uint32_t i = 0; i < n_classes; ++i) {
            const size_t idx = token_major ? (j * n_classes + i) : (i * pred_len + j);
            const float v = embeddings[idx];
            logits[i] = v;
            if (v > max_logit) {
                max_logit = v;
                consensus = static_cast<int8_t>(i);
            }
        }

        float sum = 0.0f;
        for (uint32_t i = 0; i < n_classes; ++i) {
            const float e = std::expf(logits[i] - max_logit);
            exps[i] = e;
            sum += e;
        }
        const float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;

        for (uint32_t i = 0; i < n_classes; ++i) {
            const float prob = exps[i] * inv_sum;
            pssm_row[i] = static_cast<char>(pssm_from_prob(prob, PROFILE_BACKPROBS[i]));
        }
        result.append(pssm_row.data(), pssm_row.size());
        result.push_back(static_cast<char>(consensus));
        result.push_back(static_cast<char>(consensus));
        result.push_back(0);
        result.push_back(0);
        result.push_back(0);
    }

    return 0;
}

static std::vector<std::string> string_split(const std::string & input, char separator) {
    std::vector<std::string> parts;
    size_t begin_pos = 0;
    size_t separator_pos = input.find(separator);
    while (separator_pos != std::string::npos) {
        std::string part = input.substr(begin_pos, separator_pos - begin_pos);
        parts.emplace_back(part);
        begin_pos = separator_pos + 1;
        separator_pos = input.find(separator, begin_pos);
    }
    parts.emplace_back(input.substr(begin_pos, separator_pos - begin_pos));
    return parts;
}

static std::vector<ggml_backend_dev_t> parse_device_list(const std::string & value) {
    std::vector<ggml_backend_dev_t> devices;
    auto dev_names = string_split(value, ',');
    if (dev_names.empty()) {
        devices.push_back(nullptr);
        return devices;
    }
    if (dev_names.size() == 1 && dev_names[0] == "none") {
        devices.push_back(nullptr);
    } else {
        for (const auto & device : dev_names) {
            auto * dev = ggml_backend_dev_by_name(device.c_str());
            if (!dev || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
                continue;
            }
            devices.push_back(dev);
        }
        devices.push_back(nullptr);
    }
    return devices;
}

// struct lora_adapter_info {
//     std::string path;
//     float scale;
// };

// struct lora_adapter_container : lora_adapter_info {
//     struct llama_lora_adapter* adapter;
// };

// struct init_result {
//     std::vector<lora_adapter_container> lora_adapters;
// };

LlamaInitGuard::LlamaInitGuard(bool verbose) {
    if (!verbose) {
        llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    }
    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);
}

LlamaInitGuard::~LlamaInitGuard() {
    llama_backend_free();
}

ProstT5Model::ProstT5Model(const std::string& model_file, std::string& device) {
    auto mparams = llama_model_default_params();
    std::vector<ggml_backend_dev_t> devices = parse_device_list(device);
    if (!devices.empty()) {
        mparams.devices = devices.data();
    }

    int gpus = 0;
    for (const auto& dev : devices) {
        if (!dev) {
            continue;
        }
        gpus += ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU;
    }
    if (gpus > 0) {
        mparams.n_gpu_layers = 24;
    } else {
        mparams.n_gpu_layers = 0;
    }
    mparams.use_mmap        = true;
    model = llama_model_load_from_file(model_file.c_str(), mparams);

    // for (auto & la : params.lora_adapters) {
    //     lora_adapter_container loaded_la;
    //     loaded_la.path = la.path;
    //     loaded_la.scale = la.scale;
    //     loaded_la.adapter = llama_lora_adapter_init(model, la.path.c_str());
    //     if (loaded_la.adapter == nullptr) {
    //         llama_free_model(model);
    //         return;
    //     }
    //     lora_adapters.push_back(loaded_la); // copy to list of loaded adapters
    // }
}

ProstT5Model::~ProstT5Model() {
    llama_model_free(model);
}

ProstT5::ProstT5(ProstT5Model& model, int threads) : model(model) {
    auto cparams = llama_context_default_params();
    cparams.n_threads = threads;
    cparams.n_threads_batch = threads;
    // const int32_t model_ctx = llama_model_n_ctx_train(model.model);
    const int32_t model_ctx = 2048;
    cparams.n_ctx = model_ctx > 0 ? static_cast<uint32_t>(model_ctx) : 2048;
    cparams.n_batch = cparams.n_ctx;
    cparams.n_ubatch = cparams.n_ctx;
    cparams.type_k = GGML_TYPE_F32;
    cparams.type_v = GGML_TYPE_F32;
    cparams.embeddings = true;
    cparams.attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

    ctx = llama_init_from_model(model.model, cparams);
    profile_output = false;
    char meta_val[16];
    if (llama_model_meta_val_str(model.model, "modernprost.profiles", meta_val, sizeof(meta_val)) > 0) {
        for (size_t i = 0; meta_val[i] != '\0'; ++i) {
            meta_val[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(meta_val[i])));
        }
        if (strcmp(meta_val, "1") == 0 || strcmp(meta_val, "true") == 0 || strcmp(meta_val, "yes") == 0) {
            profile_output = true;
        }
    }
    // batch = llama_batch_init(4096, 0, 1);
    // if (!params.lora_init_without_apply) {
    //     llama_lora_adapter_clear(lctx);
    //     for (auto & la : iparams.lora_adapters) {
    //         if (la.scale != 0.0f) {
    //             llama_lora_adapter_set(lctx, la.adapter, la.scale);
    //         }
    //     }
    // }
};

ProstT5::~ProstT5() {
    llama_free(ctx);
}

bool ProstT5::outputsProfile() const {
    return profile_output;
}

std::string ProstT5::predict(const std::string& aa) {
    std::string result;
    const llama_vocab * vocab = llama_model_get_vocab(model.model);
    std::vector<llama_token> embd_inp;
    embd_inp.reserve(aa.length() + 2);

    auto token_from_aa = [&](char aa_char) -> llama_token {
        const char upper = static_cast<char>(toupper(aa_char));
        std::string piece(1, upper);
        llama_token token = token_from_piece(vocab, piece, false);
        if (token != LLAMA_TOKEN_NULL) {
            return token;
        }
        std::string sp_piece("▁");
        sp_piece.append(1, upper);
        return token_from_piece(vocab, sp_piece, false);
    };

    llama_token start_token = token_from_piece(vocab, "<AA2fold>", true);
    const bool add_start_end = start_token != LLAMA_TOKEN_NULL;
    if (add_start_end) {
        embd_inp.emplace_back(start_token);
    }

    llama_token unk_aa = token_from_aa('X');
    if (unk_aa == LLAMA_TOKEN_NULL) {
        unk_aa = token_from_piece(vocab, "<unk>", true);
    }
    if (unk_aa == LLAMA_TOKEN_NULL) {
        return result;
    }
    for (size_t i = 0; i < aa.length(); ++i) {
        llama_token token = token_from_aa(aa[i]);
        if (token == LLAMA_TOKEN_NULL) {
            embd_inp.emplace_back(unk_aa);
        } else {
            embd_inp.emplace_back(token);
        }
    }
    if (add_start_end) {
        llama_token end_token = token_from_piece(vocab, "</s>", true);
        if (end_token == LLAMA_TOKEN_NULL) {
            end_token = unk_aa;
        }
        embd_inp.emplace_back(end_token);
    }
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const uint32_t n_cls_out = llama_model_n_cls_out(model.model);
    const bool is_modernprost = (n_cls_out == 20 && n_vocab == 28);
    size_t pred_len = aa.length();
    if (!is_modernprost && embd_inp.size() > 0) {
        pred_len = embd_inp.size() - 1;
    }
    encode(ctx, embd_inp, pred_len, aa.length(), result);
    return result;
}

std::string ProstT5::predictProfile(const std::string& aa) {
    std::string result;
    const llama_vocab * vocab = llama_model_get_vocab(model.model);
    std::vector<llama_token> embd_inp;
    embd_inp.reserve(aa.length() + 2);

    auto token_from_aa = [&](char aa_char) -> llama_token {
        const char upper = static_cast<char>(toupper(aa_char));
        std::string piece(1, upper);
        llama_token token = token_from_piece(vocab, piece, false);
        if (token != LLAMA_TOKEN_NULL) {
            return token;
        }
        std::string sp_piece("▁");
        sp_piece.append(1, upper);
        return token_from_piece(vocab, sp_piece, false);
    };

    llama_token start_token = token_from_piece(vocab, "<AA2fold>", true);
    const bool add_start_end = start_token != LLAMA_TOKEN_NULL;
    if (add_start_end) {
        embd_inp.emplace_back(start_token);
    }

    llama_token unk_aa = token_from_aa('X');
    if (unk_aa == LLAMA_TOKEN_NULL) {
        unk_aa = token_from_piece(vocab, "<unk>", true);
    }
    if (unk_aa == LLAMA_TOKEN_NULL) {
        return result;
    }
    for (size_t i = 0; i < aa.length(); ++i) {
        llama_token token = token_from_aa(aa[i]);
        if (token == LLAMA_TOKEN_NULL) {
            embd_inp.emplace_back(unk_aa);
        } else {
            embd_inp.emplace_back(token);
        }
    }
    if (add_start_end) {
        llama_token end_token = token_from_piece(vocab, "</s>", true);
        if (end_token == LLAMA_TOKEN_NULL) {
            end_token = unk_aa;
        }
        embd_inp.emplace_back(end_token);
    }
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const uint32_t n_cls_out = llama_model_n_cls_out(model.model);
    const bool is_modernprost = (n_cls_out == 20 && n_vocab == 28);
    size_t pred_len = aa.length();
    if (!is_modernprost && embd_inp.size() > 0) {
        pred_len = embd_inp.size() - 1;
    }
    encode_profile(ctx, embd_inp, pred_len, aa.length(), result);
    return result;
}

std::vector<std::string> ProstT5::getDevices() {
    std::vector<std::string> devices;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        std::string name = ggml_backend_dev_name(dev);
        std::string description = ggml_backend_dev_description(dev);
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        // ignore Metal in CI or when the backend reports no usable memory
        if (name == "Metal") {
            const bool bad_desc = description.empty() || description.find("Paravirtual") != std::string::npos;
            const bool no_mem = props.memory_free == 0 && props.memory_total == 0;
            if (bad_desc || no_mem) {
                continue;
            }
        }
        devices.push_back(name);
    }
    return devices;
}

void ProstT5::perf() {
    llama_perf_context_print(ctx);
}
