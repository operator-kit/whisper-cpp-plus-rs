# Session Handoff - whisper.cpp Rust Wrapper

## Quick Context
Building a production Rust wrapper for whisper.cpp v1.7.6. **Our ethos: Support the entire API surface, no shortcuts or disabling features.**

## Current Blocker
Linking errors when compiling the full backend system. Getting ~47 unresolved symbols for CPU compute operations after including:
- ggml-backend-reg.cpp (C++17)
- ggml-cpu/*.cpp files
- All core ggml files

## Immediate Need
Find and include ALL required source files for the CPU backend. The missing symbols suggest we're not compiling all necessary implementation files.

## Key Files
- Build config: `whisper-sys/build.rs`
- Test: Run `cargo run --example minimal` to test
- Model: `tests/models/ggml-tiny.en.bin` (exists and loads metadata successfully)

## What NOT to do
- Don't create stub implementations
- Don't disable features
- Don't downgrade whisper.cpp version
- Support everything properly

## Next Action
Investigate whisper.cpp's CMakeLists.txt to identify ALL files that need compilation for CPU backend.