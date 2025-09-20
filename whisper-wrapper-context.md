# Important Context for whisper.cpp Rust Wrapper Implementation

## Essential Technical References

**whisper.cpp v1.7.6 Core Files:**
- Main header: `include/whisper.h` - This is THE reference for the C API
- Implementation: `src/whisper.cpp` and `ggml/src/ggml.c`
- CMakeLists.txt for build flags and options
- The repo moved from `ggerganov/whisper.cpp` to `ggml-org/whisper.cpp`

**Critical whisper.cpp Constants to Preserve:**
- Audio must be 16kHz mono f32 samples
- Maximum segment length is typically 30 seconds
- Context size is 1500 tokens (30 seconds × 50 tokens/sec)
- The model files use `.bin` extension and GGML format

## Gotchas Your Agent Must Know

**Memory Alignment Issues:**
- whisper.cpp uses 32-byte alignment for SIMD operations
- Use `#[repr(C)]` on all structs that cross FFI boundary
- Audio buffers should be aligned - consider using `aligned_vec` crate

**Thread Safety Model:**
- `whisper_context` is thread-safe for reading (multiple states can share it)
- `whisper_state` is NOT thread-safe (one per thread)
- Model loading is NOT thread-safe - wrap in a Mutex

**Platform-Specific Quirks:**
- Windows: Need to link against `ws2_32.lib` for some builds
- macOS: Must link Accelerate framework (`-framework Accelerate`)
- Linux: May need to explicitly link math library (`-lm`)

## The Bindgen Configuration That Actually Works

```rust
// Critical bindgen settings that work with whisper.cpp
bindgen::Builder::default()
    .header("vendor/whisper.cpp/include/whisper.h")
    .clang_arg("-x").clang_arg("c++")
    .clang_arg("-std=c++11")
    .allowlist_function("whisper_.*")
    .allowlist_type("whisper_.*")
    .opaque_type("std::.*")  // Critical: don't try to bind C++ STL
    .generate()
```

## Model File Handling

**Download URLs for Testing:**
```
https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin
https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin
https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin
```

**Model Sizes (Critical for memory planning):**
- tiny: 39 MB
- base: 142 MB  
- small: 466 MB
- medium: 1.5 GB
- large: 2.9 GB

## The VAD Integration Pattern

VAD (Voice Activity Detection) in v1.7.6 requires:
1. Loading the Silero VAD model separately: `ggml-silero-vad.bin`
2. Pre-processing audio through VAD to get speech segments
3. Only transcribing the speech segments

This can reduce processing time by 50-70% on audio with lots of silence.

## Error Codes Not Documented Anywhere

```rust
// These aren't in whisper.h but show up in practice
const WHISPER_ERR_INVALID_MODEL: i32 = -1;
const WHISPER_ERR_NOT_ENOUGH_MEMORY: i32 = -2;  
const WHISPER_ERR_FAILED_TO_PROCESS: i32 = -3;
const WHISPER_ERR_INVALID_CONTEXT: i32 = -4;
```

## The Sampling Strategy Defaults That Work

```rust
// These defaults work well for 90% of use cases
FullParams {
    n_threads: num_cpus::get() / 2,  // Don't use all cores
    n_max_text_ctx: 16384,
    offset_ms: 0,
    duration_ms: 0,
    translate: false,
    no_context: false,
    no_timestamps: false,
    single_segment: false,
    print_special: false,
    print_progress: false,
    print_realtime: false,
    print_timestamps: false,
    language: "en",
    suppress_blank: true,
    suppress_non_speech_tokens: true,
    temperature: 0.0,  // Deterministic by default
    max_initial_ts: 1.0,
    length_penalty: -1.0,
}
```

## Build Script Environment Variables

```rust
// Users will expect these to work
// In build.rs:
if let Ok(path) = env::var("WHISPER_CPP_PATH") {
    // Use user-provided whisper.cpp
}
if env::var("WHISPER_NO_AVX").is_ok() {
    // Disable AVX for old CPUs
}
if env::var("WHISPER_CUDA").is_ok() {
    // Enable CUDA support
}
```

## Testing Strategy Code Structure

```rust
// Essential test structure
#[cfg(test)]
mod tests {
    // Test with a 1-second silence to ensure no crash
    #[test]
    fn test_silence_handling() {
        let silence = vec![0.0f32; 16000];
        // Should return empty transcription, not crash
    }
    
    // Test with generated sine wave
    #[test] 
    fn test_pure_tone() {
        // Generate 440Hz tone - should return empty/noise
    }
    
    // Keep a small test file in repo
    // tests/samples/test_audio.wav (5 seconds, known transcription)
}
```

## The MSRV (Minimum Supported Rust Version)

Set MSRV to 1.70.0 minimum - this gives you:
- `OnceLock` for lazy initialization
- Improved const generics
- Better async trait support
But stay compatible with stable Rust - no nightly features!

## Performance Benchmarking Baseline

Your wrapper should achieve:
- Model loading: < 500ms for base model on modern CPU
- Transcription: ~1-2x realtime on modern CPU without acceleration
- Memory overhead: < 10MB on top of model size
- First transcription: May be slower due to CPU feature detection

## Documentation Examples That Must Work

```rust
//! # Quick Start
//! ```no_run
//! use whisper::{WhisperContext, FullParams, SamplingStrategy};
//! 
//! let ctx = WhisperContext::new("model.bin")?;
//! let audio = vec![0.0f32; 16000]; // 1 second of audio
//! let text = ctx.transcribe(&audio)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
```

## Cargo.toml Categories and Keywords

```toml
# These help with discoverability
categories = ["multimedia::audio", "api-bindings", "science::robotics"]
keywords = ["whisper", "speech", "transcription", "asr", "speech-to-text"]
```

## Critical Success Metrics

The wrapper is successful if:
1. Can transcribe 1 hour of audio without memory leaks
2. Handles dropping contexts/states in any order without segfaults
3. Works on GitHub Actions CI for all tier-1 platforms
4. Can be used in both sync and async contexts
5. Produces identical transcriptions to whisper.cpp CLI

## Additional Implementation Notes

### Crate Structure
```
whisper/
├── whisper-sys/           # Low-level FFI bindings
│   ├── src/
│   │   ├── lib.rs        # Generated bindings
│   │   └── bindings.rs   # Include generated code
│   ├── build.rs          # Build script with cc/bindgen
│   └── Cargo.toml
├── src/                   # High-level safe wrapper
│   ├── lib.rs
│   ├── context.rs        # WhisperContext implementation
│   ├── state.rs          # WhisperState implementation  
│   ├── params.rs         # Parameter builders
│   ├── error.rs          # Error types
│   └── stream.rs         # Streaming support
├── vendor/
│   └── whisper.cpp/      # Git submodule pinned to v1.7.6
├── tests/
│   └── integration.rs
├── examples/
│   ├── basic.rs
│   ├── streaming.rs
│   └── async.rs
└── Cargo.toml
```

### Key Dependencies
```toml
[dependencies]
thiserror = "1.0"
num_cpus = "1.16"

[build-dependencies]
cc = "1.0"
bindgen = "0.69"

[dev-dependencies]
tokio = { version = "1.35", features = ["full"] }
hound = "3.5"  # For WAV file reading in tests
```

### Git Submodule Setup
```bash
git submodule add https://github.com/ggml-org/whisper.cpp vendor/whisper.cpp
cd vendor/whisper.cpp
git checkout v1.7.6
cd ../..
git add .gitmodules vendor/whisper.cpp
git commit -m "Add whisper.cpp v1.7.6 as submodule"
```

This document contains all the critical implementation details needed to build a production-ready whisper.cpp Rust wrapper. Use this alongside the architecture guide for a complete implementation reference.