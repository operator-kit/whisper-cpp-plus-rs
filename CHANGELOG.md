# Changelog

All notable changes to whisper-cpp-plus will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

No unreleased changes.

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
