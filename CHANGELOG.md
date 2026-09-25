# Changelog

All notable changes to whisper-cpp-plus will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Updated the pinned whisper.cpp fork to `rmorse/whisper.cpp` `stream-pcm` at `de8fb5fd` (tag `v1.9.4-dev-stream-pcm`), based on upstream `ggml-org/whisper.cpp` `master` after the `v1.9.4` release (`v1.9.4` plus 181 later upstream commits). This picks up upstream releases `v1.8.7` through `v1.9.4` and ggml `0.25.1`.
- Upstream now re-seeds the decoder between calls (ggml-org/whisper.cpp#4025), so temperature-fallback output is deterministic across repeated transcriptions on the same state.
- Upstream now rejects Silero VAD models whose encoder does not have exactly 4 layers (ggml-org/whisper.cpp#4064); loading such a model with `WhisperVadProcessor` now fails at load time.
- On Apple Silicon, upstream's optional ANEForge encoder backend (ggml-org/whisper.cpp#3905) is activated by the `ANEFORGE_ENCODER` and `ANEFORGE_DYLIB` environment variables, which load a dynamic library from the given path when a state is created. It is inactive unless those variables are set.
- NVIDIA Parakeet support is available in the bundled C library but is not yet exposed through the Rust API.
- The new upstream VAD segment and VAD-mapped token timestamp accessors are not exposed: they are only populated by upstream's built-in `whisper_full` VAD, which does not run for per-state transcription (`whisper_full_with_state`, used by this crate; see ggml-org/whisper.cpp#3423). Use the crate's own VAD pipeline instead.
- The `whisper-cpp-plus-sys` package now includes whisper.cpp's public headers (`include/*.h`, `ggml/include/*.h`) and its `LICENSE`. docs.rs builds generate bindings from these headers instead of using hand-written stubs. Regular builds are unchanged: they still download the full pinned whisper.cpp source.
- `WhisperContext` no longer allocates whisper.cpp's default state, which the crate never used (all transcription runs on explicit `WhisperState`s). This saves that state's KV caches and compute buffers for every loaded context: about 146 MB with `ggml-tiny.en.bin` as reported by whisper.cpp, and considerably more for larger models.

### Removed

- **Breaking:** `WhisperState::full_parallel()`. It never returned correct results: whisper.cpp writes parallel results to the context's default state, which the method never read. Use `WhisperContext::full_parallel()`, which returns the merged `TranscriptionResult`.
- **Breaking:** `WhisperContext::n_len()`. It reported the mel length of the context's default state, which the crate never transcribes on. Use `WhisperState::n_len()` for the state you transcribed with.

### Added

- `WhisperState::full_get_segment_no_speech_prob()`, previously only used internally by the temperature-fallback transcriber.
- `WhisperLog` (wrapping `whisper_log_set`) to control whisper.cpp's log output, which covers whisper.cpp, its VAD and the ggml backends: `WhisperLog::set()` routes messages to a Rust callback with a `LogLevel`, `WhisperLog::disable()` silences them, and `WhisperLog::reset()` restores the default stderr output.
- `log` feature: `WhisperLog::use_log_crate()` forwards whisper.cpp log output to the `log` crate with target `whisper_cpp`.
- `WhisperVadProcessor::detect_speech_no_reset()` and `reset_state()` (wrapping `whisper_vad_detect_speech_no_reset` / `whisper_vad_reset_state`) for streaming Silero VAD that keeps its state across chunks, plus `WhisperVadProcessor::WINDOW_SAMPLES` (512 samples per probability).
- `WhisperContext::full_parallel(params, audio, n_processors)`: splits audio into equal chunks, transcribes them concurrently on separate states, and returns the merged `TranscriptionResult` with times on the original timeline. Chunking follows whisper.cpp's `whisper_full_parallel`; in addition, segment times are clamped to their chunk, so whisper reporting a segment end past its audio can no longer push the next chunk's segments later. Words that straddle a chunk boundary may still be cut or misrecognised.
- `WhisperState::n_len()`: mel length of the last transcription on the state (`whisper_n_len_from_state`).

### Fixed

- **Breaking:** segment timestamps are now real milliseconds. whisper.cpp reports segment times in centiseconds, and the crate previously passed them through unconverted, so `Segment::start_ms`/`end_ms`, `start_seconds()`/`end_seconds()`, `WhisperState::full_get_segment_timestamps()`, and the `start`/`end` values passed to `WhisperStreamPcm::run` callbacks were 10x too small. This affects `transcribe*`, `WhisperStream`, `WhisperStreamPcm`, and the temperature-fallback transcriber. The raw `whisper_token_data` returned by `full_get_token_data()` is unchanged and documented as centiseconds.
- Fixed a use-after-free in `FullParams::suppress_regex()`: the regex string was freed immediately after being set, so whisper.cpp read freed memory during transcription.
- Fixed `FullParams::prompt_tokens()` storing a borrowed pointer that dangled once the caller's slice was dropped or the params were moved or cloned. The tokens are now copied into the params.
- **Breaking:** `WhisperState` result getters now validate segment and token indices instead of passing them to whisper.cpp, which does not bounds-check (out-of-range indices were undefined behaviour). `full_get_segment_text()` and `full_get_token_text()` return `WhisperError::InvalidParameter`, `full_get_token_data()` returns `None`, and the plain-value getters (`full_get_segment_timestamps()`, `full_get_segment_speaker_turn_next()`, `full_n_tokens()`, `full_get_token_id()`, `full_get_token_prob()`) panic, like slice indexing.
- Improved Silero VAD accuracy in `WhisperStreamPcm`. Each 200 ms probe was evaluated from a freshly reset model with a zero-padded partial window, so speech onsets and short words were often misclassified: on `jfk.wav`, 16 of 55 probe decisions differed from a full-file Silero pass, cutting "Ask not" short (transcribed as "Ask, knock!") and splitting a sentence. The model state is now carried across probes (reset only when the stream starts) and only whole 32 ms windows are evaluated, which matches the full-file pass.
- Fixed the `whisper-cpp-plus-sys` documentation on docs.rs, which was generated from out-of-date hand-written stubs: it was missing functions, listed functions that no longer exist, and showed some wrong signatures and types. It now matches the real bindings.

### Documentation

- Clarified that `WhisperVadProcessor::detect_speech()` returns whether the computation succeeded, not whether speech was found; speech probabilities come from `get_probs()`.
- Documented that `PcmReaderConfig::buffer_len_ms` drops the oldest samples on overflow, so sources faster than real time (files, in-memory buffers) need a buffer that holds the whole input.

## [0.1.5] - 2026-06-12

### Added

- Added `PcmReader::dropped_samples()` tracking so callers can detect PCM ring-buffer overflow.
- Added macOS GitHub Actions coverage for formatting, model setup, workspace tests, and Metal feature tests.
- Added `cargo xtask prebuild` reporting for optional `ggml-blas` cache artifacts.

### Changed

- Updated the pinned whisper.cpp fork to `rmorse/whisper.cpp` `v1.8.6-stream-pcm` (`ddfe1196`), based on upstream `ggml-org/whisper.cpp` `v1.8.6`.
- Improved macOS build handling by passing opt-in `MACOSX_DEPLOYMENT_TARGET` through to CMake as `CMAKE_OSX_DEPLOYMENT_TARGET`.
- Set best-effort macOS QoS on the PCM capture thread to reduce scheduling-related audio drops.
- Updated `cargo xtask prebuild --force` to remove the existing target/profile cache before rebuilding, preventing stale satellite libraries from surviving.
- Updated macOS default prebuild behavior to produce a CPU/BLAS cache with `GGML_METAL=OFF`.
- Updated test and benchmark model lookup to prefer real `ggml-tiny.en.bin` and `ggml-silero-v6.2.0.bin` models downloaded by `cargo xtask test-setup`.

### Fixed

- Fixed macOS/prebuilt linking by copying and linking `ggml-blas` when whisper.cpp produces it.
- Fixed incomplete Metal prebuilt usage by failing early when `features = ["metal"]` is used with a prebuilt cache missing `libggml-metal.a`.
- Fixed xtask CMake invocation outside Cargo build scripts by setting explicit host, target, and xtask-specific CMake output directories.
- Suppressed the known whisper.cpp `quantize_wrapper` switch warning with a scoped compiler flag.

### Documentation

- Clarified that default macOS xtask prebuilds are CPU/BLAS only and Metal prebuilt use requires a complete custom cache containing `libggml-metal.a`.

## [0.1.0] - Previous Release

Initial release of whisper-cpp-plus with full Rust bindings to whisper.cpp.
