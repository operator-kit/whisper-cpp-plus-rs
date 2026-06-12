# Technical Reference

## whisper.cpp constants

- Sample rate: **16000 Hz** (mono, f32)
- Max segment length: **30 seconds**
- Token context: **1500 tokens** (30s × 50 tokens/sec)
- Model format: GGML `.bin` files
- Memory alignment: 32-byte for SIMD — use `#[repr(C)]` on FFI structs

## Model sizes

| Model | Size | Notes |
|-------|------|-------|
| tiny / tiny.en | 39 MB | Fastest, lowest quality |
| base / base.en | 142 MB | Good balance for dev/test |
| small / small.en | 466 MB | |
| medium / medium.en | 1.5 GB | |
| large-v3 | 3.1 GB | Best quality, multilingual only |

## Error codes

whisper.cpp return codes (not in whisper.h, discovered empirically):

| Code | Meaning |
|------|---------|
| 0 | Success |
| -1 | Invalid model |
| -2 | Out of memory |
| -3 | Failed to process |
| -4 | Invalid context |

## Temperature fallback thresholds

Derived from faster-whisper / OpenAI Whisper defaults:

| Threshold | Default | Purpose |
|-----------|---------|---------|
| `compression_ratio_threshold` | 2.4 | Detect repetitive/degenerate output (zlib ratio) |
| `log_prob_threshold` | -1.0 | Minimum average log probability |
| `no_speech_threshold` | 0.6 | Silence detection — if exceeded AND log_prob fails, treat as silence rather than bad transcription |
| Temperature sequence | 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 | Retry with increasing randomness |
| `prompt_reset_on_temperature` | 0.5 | Reset initial prompt above this temperature |

## Platform linking requirements

### Windows (MSVC)
```
ws2_32, bcrypt, advapi32, userenv
```
- Use `/MT` (static CRT) or set `target-feature=+crt-static` in `.cargo/config.toml`
- Need Visual Studio Build Tools 2022 with C++ workload
- Use x64 Native Tools Command Prompt or ensure x64 MSVC toolchain

### macOS
```
-framework Accelerate, -lc++
```

### Linux
```
-lstdc++, -lm, -lpthread
```

## Windows linking troubleshooting

**Exit code 1120** = unresolved external symbols. Common causes:

1. **Missing system libs** — ensure `build.rs` emits `cargo:rustc-link-lib` for all platform libs above
2. **Missing source files** — all `ggml-*.c` backend files must be compiled
3. **CRT mismatch** — mixing `/MT` and `/MD` across compilation units
4. **Wrong toolchain** — `rustup default stable-x86_64-pc-windows-msvc`

Diagnostic: `cargo test --verbose 2>&1 | grep "unresolved external"` to see exact missing symbols.

## Build environment variables

| Variable | Purpose |
|----------|---------|
| `WHISPER_PREBUILT_PATH` | Path to prebuilt static libs (skips CMake build entirely) |
| `WHISPER_NO_AVX` | Disable AVX/AVX2 for old CPUs |
| `WHISPER_TEST_MODEL_DIR` | Override test model search path |
| `WHISPER_TEST_AUDIO_DIR` | Override test audio search path |
| `CUDA_PATH` | CUDA toolkit root directory (checked first) |
| `CUDA_HOME` | CUDA toolkit root directory (fallback) |
| `CMAKE_*` | Passed through to CMake (e.g. `CMAKE_CUDA_ARCHITECTURES`, `CMAKE_BUILD_TYPE`) |

## Build system

The `cmake` crate invokes CMake from `build.rs` to compile whisper.cpp. The vendor source is copied to `OUT_DIR` before building to avoid polluting the working tree.

Feature flags map to CMake defines:
- `cuda` → `-DGGML_CUDA=ON`
- `metal` → `-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON`
- `openblas` → `-DGGML_BLAS=ON`

The quantization wrapper (`src/quantize_wrapper.cpp`) is compiled separately via the `cc` crate since it's our own C++ code, not part of whisper.cpp's CMake build.

## GPU Linking

### CUDA (`--features cuda`)

CMake handles CUDA compilation automatically. The build script:

1. Invokes CMake with `-DGGML_CUDA=ON` (or uses prebuilt libs from `WHISPER_PREBUILT_PATH`)
2. Links static libs: `whisper`, `ggml`, `ggml-base`, `ggml-cpu`, `ggml-cuda`
3. Locates CUDA toolkit (`CUDA_PATH` → `CUDA_HOME` → standard paths)
4. Links CUDA runtime: `cudart_static`, `cublas`, `cublasLt`, `cuda`

### Metal (`--features metal`)

macOS only. Source builds compile ggml with `GGML_METAL=ON` and link `Metal`, `MetalKit`, and `MetalPerformanceShaders` frameworks. Prebuilt Metal builds require the prebuilt directory to contain `libggml-metal.a`; the default `cargo xtask prebuild` path disables Metal and produces a CPU/BLAS cache.

### OpenBLAS (`--features openblas`)

Links `openblas`. Works with both CMake build and prebuilt.

## Default parameter values

These defaults cover ~90% of use cases:

```rust
n_threads: num_cpus::get() / 2,  // don't saturate all cores
temperature: 0.0,                 // deterministic
language: "en",
suppress_blank: true,
suppress_non_speech_tokens: true,
no_timestamps: false,
single_segment: false,
```

## Performance baselines

- Model loading: < 500ms (base model, modern CPU)
- Transcription: ~1-2x realtime on modern CPU (no GPU)
- Memory overhead: < 10MB above model size
- VAD preprocessing: reduces processing time 50-70% on audio with silence

## VAD integration

Silero VAD requires a separate model file (`ggml-silero-vad.bin` or `ggml-silero-v5.1.2.bin`). The VAD is a preprocessing step — detect speech segments first, then only transcribe those segments. See `enhanced::vad` for segment aggregation that merges small segments into optimal transcription chunks.

## Streaming state reuse

`WhisperStream::reset()` reuses the existing `WhisperState` — it clears buffers but skips the 500MB+ reallocation. This is safe because `whisper_full_with_state()` clears results internally at the start of each transcription.

Use `WhisperStream::recreate_state()` only after errors that may have corrupted state, or when switching between very different audio sources.
