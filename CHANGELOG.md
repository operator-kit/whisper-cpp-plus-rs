# Changelog

All notable changes to whisper-cpp-plus will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Updated the pinned whisper.cpp fork to `rmorse/whisper.cpp` `stream-pcm` at `de8fb5fd` (tag `v1.9.4-dev-stream-pcm`, whisper.cpp `1.9.4-dev`), based on upstream `ggml-org/whisper.cpp` `master` after `v1.9.3`. This picks up upstream releases `v1.8.7` through `v1.9.3` and ggml `0.25.1`.
- Upstream now re-seeds the decoder between calls (ggml-org/whisper.cpp#4025), so temperature-fallback output is deterministic across repeated transcriptions on the same state.
- Upstream now rejects Silero VAD models whose encoder does not have exactly 4 layers (ggml-org/whisper.cpp#4064); loading such a model with `WhisperVadProcessor` now fails at load time.
- On Apple Silicon, upstream's optional ANEForge encoder backend (ggml-org/whisper.cpp#3905) is activated by the `ANEFORGE_ENCODER` and `ANEFORGE_DYLIB` environment variables, which load a dynamic library from the given path when a state is created. It is inactive unless those variables are set.
- NVIDIA Parakeet support is available in the bundled C library but is not yet exposed through the Rust API.
- The new upstream VAD segment and VAD-mapped token timestamp accessors are not exposed: they are only populated by upstream's built-in `whisper_full` VAD, which does not run for per-state transcription (`whisper_full_with_state`, used by this crate; see ggml-org/whisper.cpp#3423). Use the crate's own VAD pipeline instead.

### Added

- `WhisperState::full_get_segment_no_speech_prob()`, previously only used internally by the temperature-fallback transcriber.

### Fixed

- **Breaking:** segment timestamps are now real milliseconds. whisper.cpp reports segment times in centiseconds, and the crate previously passed them through unconverted, so `Segment::start_ms`/`end_ms`, `start_seconds()`/`end_seconds()`, `WhisperState::full_get_segment_timestamps()`, and the `start`/`end` values passed to `WhisperStreamPcm::run` callbacks were 10x too small. This affects `transcribe*`, `WhisperStream`, `WhisperStreamPcm`, and the temperature-fallback transcriber. The raw `whisper_token_data` returned by `full_get_token_data()` is unchanged and documented as centiseconds.
- Fixed a use-after-free in `FullParams::suppress_regex()`: the regex string was freed immediately after being set, so whisper.cpp read freed memory during transcription.
- Fixed `FullParams::prompt_tokens()` storing a borrowed pointer that dangled once the caller's slice was dropped or the params were moved or cloned. The tokens are now copied into the params.
- **Breaking:** `WhisperState` result getters now validate segment and token indices instead of passing them to whisper.cpp, which does not bounds-check (out-of-range indices were undefined behaviour). `full_get_segment_text()` and `full_get_token_text()` return `WhisperError::InvalidParameter`, `full_get_token_data()` returns `None`, and the plain-value getters (`full_get_segment_timestamps()`, `full_get_segment_speaker_turn_next()`, `full_n_tokens()`, `full_get_token_id()`, `full_get_token_prob()`) panic, like slice indexing.
- Fixed an out-of-bounds read in whisper.cpp reachable from safe code: a negative `FullParams::offset_ms` became a negative mel offset in the encoder, which then read before the start of the mel buffer. Transcription (`WhisperState::full`, and through it `transcribe*`, `WhisperStream`, `WhisperStreamPcm` and the temperature-fallback transcriber) now returns `WhisperError::InvalidParameter` for a negative `offset_ms` or `duration_ms`. `WhisperState::full_parallel` also rejects an `offset_ms` at or past the end of the audio, which whisper.cpp turned into negative chunk sizes.
- Audio with more than `i32::MAX` samples is now rejected instead of having its length silently truncated when passed to whisper.cpp (`WhisperState::full`, `full_parallel`, `WhisperVadProcessor::detect_speech` / `segments_from_samples`).

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
