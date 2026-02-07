#include "ggml.h"
#include "ggml-backend.h"
#include "common.h"
#include "common-ggml.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <regex>

extern "C" {

// Callback for progress reporting
typedef void (*whisper_quantize_progress_callback)(float progress);

// Quantization result codes
enum whisper_quantize_result {
    WHISPER_QUANTIZE_OK = 0,
    WHISPER_QUANTIZE_ERROR_INVALID_MODEL = -1,
    WHISPER_QUANTIZE_ERROR_FILE_OPEN = -2,
    WHISPER_QUANTIZE_ERROR_FILE_WRITE = -3,
    WHISPER_QUANTIZE_ERROR_INVALID_FTYPE = -4,
    WHISPER_QUANTIZE_ERROR_QUANTIZATION_FAILED = -5,
};

// Whisper model header structure
struct whisper_model_hparams {
    int32_t n_vocab;
    int32_t n_audio_ctx;
    int32_t n_audio_state;
    int32_t n_audio_head;
    int32_t n_audio_layer;
    int32_t n_text_ctx;
    int32_t n_text_state;
    int32_t n_text_head;
    int32_t n_text_layer;
    int32_t n_mels;
    int32_t ftype;
};

struct whisper_filters {
    int32_t n_mel;
    int32_t n_fft;
    std::vector<float> data;
};

// Internal quantization function based on whisper.cpp's quantize.cpp
static int whisper_model_quantize_internal(
    const std::string & fname_inp,
    const std::string & fname_out,
    ggml_ftype ftype,
    whisper_quantize_progress_callback progress_callback) {

    printf("%s: loading model from '%s'\n", __func__, fname_inp.c_str());

    auto finp = std::ifstream(fname_inp, std::ios::binary);
    if (!finp) {
        fprintf(stderr, "%s: failed to open '%s' for reading\n", __func__, fname_inp.c_str());
        return WHISPER_QUANTIZE_ERROR_FILE_OPEN;
    }

    auto fout = std::ofstream(fname_out, std::ios::binary);
    if (!fout) {
        fprintf(stderr, "%s: failed to open '%s' for writing\n", __func__, fname_out.c_str());
        return WHISPER_QUANTIZE_ERROR_FILE_WRITE;
    }

    // Verify magic
    {
        uint32_t magic;
        finp.read((char *) &magic, sizeof(magic));
        if (magic != GGML_FILE_MAGIC) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname_inp.c_str());
            return WHISPER_QUANTIZE_ERROR_INVALID_MODEL;
        }

        fout.write((char *) &magic, sizeof(magic));
    }

    whisper_model_hparams hparams = {};

    // Load hparams
    {
        finp.read((char *) &hparams.n_vocab,       sizeof(hparams.n_vocab));
        finp.read((char *) &hparams.n_audio_ctx,   sizeof(hparams.n_audio_ctx));
        finp.read((char *) &hparams.n_audio_state, sizeof(hparams.n_audio_state));
        finp.read((char *) &hparams.n_audio_head,  sizeof(hparams.n_audio_head));
        finp.read((char *) &hparams.n_audio_layer, sizeof(hparams.n_audio_layer));
        finp.read((char *) &hparams.n_text_ctx,    sizeof(hparams.n_text_ctx));
        finp.read((char *) &hparams.n_text_state,  sizeof(hparams.n_text_state));
        finp.read((char *) &hparams.n_text_head,   sizeof(hparams.n_text_head));
        finp.read((char *) &hparams.n_text_layer,  sizeof(hparams.n_text_layer));
        finp.read((char *) &hparams.n_mels,        sizeof(hparams.n_mels));
        finp.read((char *) &hparams.ftype,         sizeof(hparams.ftype));

        const int32_t qntvr_src = hparams.ftype / GGML_QNT_VERSION_FACTOR;
        const int32_t ftype_dst = GGML_QNT_VERSION * GGML_QNT_VERSION_FACTOR + ftype;

        fprintf(stderr, "%s: n_vocab       = %d\n", __func__, hparams.n_vocab);
        fprintf(stderr, "%s: n_audio_ctx   = %d\n", __func__, hparams.n_audio_ctx);
        fprintf(stderr, "%s: n_audio_state = %d\n", __func__, hparams.n_audio_state);
        fprintf(stderr, "%s: n_audio_head  = %d\n", __func__, hparams.n_audio_head);
        fprintf(stderr, "%s: n_audio_layer = %d\n", __func__, hparams.n_audio_layer);
        fprintf(stderr, "%s: n_text_ctx    = %d\n", __func__, hparams.n_text_ctx);
        fprintf(stderr, "%s: n_text_state  = %d\n", __func__, hparams.n_text_state);
        fprintf(stderr, "%s: n_text_head   = %d\n", __func__, hparams.n_text_head);
        fprintf(stderr, "%s: n_text_layer  = %d\n", __func__, hparams.n_text_layer);
        fprintf(stderr, "%s: n_mels        = %d\n", __func__, hparams.n_mels);
        fprintf(stderr, "%s: ftype (src)   = %d\n", __func__, hparams.ftype);
        fprintf(stderr, "%s: qntvr (src)   = %d\n", __func__, qntvr_src);
        fprintf(stderr, "%s: ftype (dst)   = %d\n", __func__, ftype_dst);
        fprintf(stderr, "%s: qntvr (dst)   = %d\n", __func__, GGML_QNT_VERSION);

        fout.write((const char *) &hparams.n_vocab,       sizeof(hparams.n_vocab));
        fout.write((const char *) &hparams.n_audio_ctx,   sizeof(hparams.n_audio_ctx));
        fout.write((const char *) &hparams.n_audio_state, sizeof(hparams.n_audio_state));
        fout.write((const char *) &hparams.n_audio_head,  sizeof(hparams.n_audio_head));
        fout.write((const char *) &hparams.n_audio_layer, sizeof(hparams.n_audio_layer));
        fout.write((const char *) &hparams.n_text_ctx,    sizeof(hparams.n_text_ctx));
        fout.write((const char *) &hparams.n_text_state,  sizeof(hparams.n_text_state));
        fout.write((const char *) &hparams.n_text_head,   sizeof(hparams.n_text_head));
        fout.write((const char *) &hparams.n_text_layer,  sizeof(hparams.n_text_layer));
        fout.write((const char *) &hparams.n_mels,        sizeof(hparams.n_mels));
        fout.write((const char *) &ftype_dst,             sizeof(hparams.ftype));
    }

    // Load mel filters
    {
        whisper_filters filters;

        finp.read ((char *) &filters.n_mel, sizeof(filters.n_mel));
        fout.write((char *) &filters.n_mel, sizeof(filters.n_mel));
        finp.read ((char *) &filters.n_fft, sizeof(filters.n_fft));
        fout.write((char *) &filters.n_fft, sizeof(filters.n_fft));

        filters.data.resize(filters.n_mel * filters.n_fft);
        finp.read ((char *) filters.data.data(), filters.data.size() * sizeof(float));
        fout.write((char *) filters.data.data(), filters.data.size() * sizeof(float));
    }

    // Load vocab - just copy it without parsing
    {
        int32_t n_vocab = 0;
        finp.read ((char *) &n_vocab, sizeof(n_vocab));
        fout.write((char *) &n_vocab, sizeof(n_vocab));

        // Create temporary vocab for ggml_common_quantize_0
        std::map<std::string, int32_t> token_to_id;
        std::map<int32_t, std::string> id_to_token;

        char word[129];

        for (int i = 0; i < n_vocab; i++) {
            uint32_t len;
            finp.read ((char *) &len, sizeof(len));
            fout.write((char *) &len, sizeof(len));

            word[len] = '\0';

            finp.read ((char *) word, len);
            fout.write((char *) word, len);

            // Store in temporary maps (not used by quantization)
            token_to_id[word] = i;
            id_to_token[i] = word;
        }
    }

    // Report progress if callback provided
    if (progress_callback) {
        progress_callback(0.1f); // 10% after loading headers
    }

    // Regexes of tensor names to not be quantized
    const std::vector<std::string> to_skip = {
        //"encoder.*",
        "encoder.conv1.bias",
        "encoder.conv2.bias",
        "encoder.positional_embedding",
        "decoder.positional_embedding",
    };

    // Perform quantization
    if (!ggml_common_quantize_0(finp, fout, ftype, { ".*" }, to_skip)) {
        fprintf(stderr, "%s: failed to quantize model '%s'\n", __func__, fname_inp.c_str());
        return WHISPER_QUANTIZE_ERROR_QUANTIZATION_FAILED;
    }

    if (progress_callback) {
        progress_callback(1.0f); // 100% complete
    }

    finp.close();
    fout.close();

    return WHISPER_QUANTIZE_OK;
}

// Main quantization function exposed to Rust
int whisper_model_quantize(
    const char * fname_inp,
    const char * fname_out,
    int ftype,
    whisper_quantize_progress_callback progress_callback) {

    // Validate ftype
    ggml_ftype ggml_ftype_value = (ggml_ftype)ftype;

    // Check if it's a valid quantization type
    switch (ggml_ftype_value) {
        case GGML_FTYPE_MOSTLY_Q4_0:
        case GGML_FTYPE_MOSTLY_Q4_1:
        case GGML_FTYPE_MOSTLY_Q5_0:
        case GGML_FTYPE_MOSTLY_Q5_1:
        case GGML_FTYPE_MOSTLY_Q8_0:
        case GGML_FTYPE_MOSTLY_Q2_K:
        case GGML_FTYPE_MOSTLY_Q3_K:
        case GGML_FTYPE_MOSTLY_Q4_K:
        case GGML_FTYPE_MOSTLY_Q5_K:
        case GGML_FTYPE_MOSTLY_Q6_K:
            break;
        default:
            fprintf(stderr, "%s: invalid quantization type %d\n", __func__, ftype);
            return WHISPER_QUANTIZE_ERROR_INVALID_FTYPE;
    }

    // Initialize GGML (needed for f16 tables)
    static bool ggml_initialized = false;
    if (!ggml_initialized) {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
        ggml_initialized = true;
    }

    return whisper_model_quantize_internal(
        fname_inp,
        fname_out,
        ggml_ftype_value,
        progress_callback
    );
}

// Get the quantization type of a model file
int whisper_model_get_ftype(const char * fname) {
    std::ifstream fin(fname, std::ios::binary);
    if (!fin) {
        return -1;
    }

    // Check magic
    uint32_t magic;
    fin.read((char *) &magic, sizeof(magic));
    if (magic != GGML_FILE_MAGIC) {
        return -1;
    }

    // Skip to ftype field
    whisper_model_hparams hparams = {};
    fin.read((char *) &hparams.n_vocab,       sizeof(hparams.n_vocab));
    fin.read((char *) &hparams.n_audio_ctx,   sizeof(hparams.n_audio_ctx));
    fin.read((char *) &hparams.n_audio_state, sizeof(hparams.n_audio_state));
    fin.read((char *) &hparams.n_audio_head,  sizeof(hparams.n_audio_head));
    fin.read((char *) &hparams.n_audio_layer, sizeof(hparams.n_audio_layer));
    fin.read((char *) &hparams.n_text_ctx,    sizeof(hparams.n_text_ctx));
    fin.read((char *) &hparams.n_text_state,  sizeof(hparams.n_text_state));
    fin.read((char *) &hparams.n_text_head,   sizeof(hparams.n_text_head));
    fin.read((char *) &hparams.n_text_layer,  sizeof(hparams.n_text_layer));
    fin.read((char *) &hparams.n_mels,        sizeof(hparams.n_mels));
    fin.read((char *) &hparams.ftype,         sizeof(hparams.ftype));

    fin.close();

    // Extract the actual ftype (without version)
    return hparams.ftype % GGML_QNT_VERSION_FACTOR;
}

} // extern "C"