# whisper-cpp-plus

> **Pinned to whisper.cpp v1.8.3** (fork: [`rmorse/whisper.cpp`](https://github.com/rmorse/whisper.cpp), branch: `stream-pcm`)

Safe, idiomatic Rust bindings for [whisper.cpp](https://github.com/ggerganov/whisper.cpp) — OpenAI's Whisper speech recognition model.

[![Crates.io](https://img.shields.io/crates/v/whisper-cpp-plus.svg)](https://crates.io/crates/whisper-cpp-plus)
[![Documentation](https://docs.rs/whisper-cpp-plus/badge.svg)](https://docs.rs/whisper-cpp-plus)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Quick Start

```rust
use whisper_cpp_plus::WhisperContext;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = WhisperContext::new("models/ggml-base.en.bin")?;

    // Audio must be 16kHz mono f32
    let audio: Vec<f32> = load_audio("audio.wav");
    let text = ctx.transcribe(&audio)?;
    println!("{}", text);

    Ok(())
}
```

## Features

- **Thread-safe** — `WhisperContext` is `Send + Sync`, share via `Arc`
- **Streaming** — real-time transcription via `WhisperStream` and `WhisperStreamPcm`
- **VAD** — Silero Voice Activity Detection integration
- **Enhanced VAD** — segment aggregation for optimal transcription chunks
- **Temperature fallback** — quality-based retry with multiple temperatures
- **Async** — `tokio::spawn_blocking` wrappers (feature = `async`)
- **Cross-platform** — Windows (MSVC), Linux, macOS (Intel & Apple Silicon)
- **Quantization** — model compression via `WhisperQuantize` (feature = `quantization`)
- **Hardware acceleration** — SIMD auto-detected, GPU via feature flags

## Installation

```toml
[dependencies]
whisper-cpp-plus = "0.1.0"

# Optional
hound = "3.5"  # WAV file loading
```

### System Requirements

- Rust 1.70.0+
- CMake 3.14+
- C++ compiler (MSVC on Windows, GCC/Clang on Linux/macOS)

### Feature Flags

```toml
whisper-cpp-plus = { version = "0.1.0", features = ["quantization"] }  # Model quantization
whisper-cpp-plus = { version = "0.1.0", features = ["async"] }         # Async API
whisper-cpp-plus = { version = "0.1.0", features = ["cuda"] }          # NVIDIA GPU
whisper-cpp-plus = { version = "0.1.0", features = ["metal"] }         # macOS GPU
```

### CUDA GPU Acceleration

Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) and build:

```bash
cargo build --features cuda
```

The build script uses CMake to compile whisper.cpp with CUDA support automatically. The CUDA toolkit is located via `CUDA_PATH` → `CUDA_HOME` → standard install paths.

**Advanced: prebuilt libraries** — for CI or to skip recompilation, set `WHISPER_PREBUILT_PATH` to a directory containing pre-compiled static libs. See [docs/CACHING_GUIDE.md](docs/CACHING_GUIDE.md).

## Crate Structure

| Crate | Description |
|-------|-------------|
| `whisper-cpp-plus` | High-level safe Rust bindings |
| `whisper-cpp-plus-sys` | Low-level FFI bindings |

Model quantization available via `features = ["quantization"]`.

## API Overview

### Core Types

| Type | Description | whisper.cpp equivalent |
|------|-------------|------------------------|
| `WhisperContext` | Model context (`Send + Sync`) | `whisper_context*` |
| `WhisperState` | Transcription state (`Send` only) | `whisper_state*` |
| `FullParams` | Transcription parameters | `whisper_full_params` |
| `TranscriptionResult` | Text + timestamped segments | — |
| `WhisperStream` | Chunked real-time streaming | — |
| `WhisperStreamPcm` | Streaming from raw PCM input | `stream-pcm.cpp` |
| `WhisperVadProcessor` | Silero voice activity detection | `whisper_vad_*` |
| `EnhancedWhisperVadProcessor` | VAD + segment aggregation | — |
| `EnhancedWhisperState` | Transcription with temperature fallback | — |
| `WhisperQuantize` | Model quantization (feature) | `quantize.cpp` |

### Examples

**Transcription with parameters:**

```rust
use whisper_cpp_plus::{WhisperContext, TranscriptionParams};

let ctx = WhisperContext::new("model.bin")?;
let params = TranscriptionParams::builder()
    .language("en")
    .temperature(0.0)
    .enable_timestamps()
    .n_threads(4)
    .build();

let result = ctx.transcribe_with_params(&audio, params)?;
for segment in &result.segments {
    println!("[{:.2}s - {:.2}s] {}",
        segment.start_seconds(), segment.end_seconds(), segment.text);
}
```

**Concurrent transcription:**

```rust
use std::sync::Arc;
let ctx = Arc::new(WhisperContext::new("model.bin")?);

// Each thread gets its own WhisperState internally
let handles: Vec<_> = files.iter().map(|file| {
    let ctx = Arc::clone(&ctx);
    std::thread::spawn(move || ctx.transcribe(&load_audio(file)))
}).collect();
```

**Streaming:**

```rust
use whisper_cpp_plus::{WhisperStream, FullParams, SamplingStrategy};

let ctx = WhisperContext::new("model.bin")?;
let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
let mut stream = WhisperStream::new(&ctx, params)?;

loop {
    let chunk = get_audio_chunk(); // your audio source
    stream.feed_audio(&chunk);
    let segments = stream.process_pending()?;
    for seg in &segments {
        println!("{}", seg.text);
    }
}
```

**VAD preprocessing:**

```rust
use whisper_cpp_plus::{WhisperVadProcessor, VadParams};

let mut vad = WhisperVadProcessor::new("models/ggml-silero-vad.bin")?;
let params = VadParams::default();
let segments = vad.segments_from_samples(&audio, &params)?;

for (start, end) in segments.get_all_segments() {
    let start_sample = (start * 16000.0) as usize;
    let end_sample = (end * 16000.0) as usize;
    let text = ctx.transcribe(&audio[start_sample..end_sample])?;
    println!("[{:.1}s-{:.1}s] {}", start, end, text);
}
```

**Enhanced VAD with segment aggregation:**

```rust
use whisper_cpp_plus::enhanced::{EnhancedWhisperVadProcessor, EnhancedVadParams};

let mut vad = EnhancedWhisperVadProcessor::new("models/ggml-silero-vad.bin")?;
let params = EnhancedVadParams::default();
let chunks = vad.process_with_aggregation(&audio, &params)?;

for chunk in &chunks {
    let text = ctx.transcribe(&chunk.audio)?;
    println!("[{:.1}s, {:.1}s long] {}", chunk.offset_seconds, chunk.duration_seconds, text);
}
```

**Temperature fallback for difficult audio:**

```rust
let params = TranscriptionParams::builder()
    .language("en")
    .build();
let result = ctx.transcribe_with_params_enhanced(&audio, params)?;
// Automatically retries with higher temperatures if quality thresholds aren't met
```

More examples in [`whisper-cpp-plus/examples/`](./whisper-cpp-plus/examples/).

## Enhanced Features

Beyond standard whisper.cpp bindings, this crate provides optimizations inspired by [faster-whisper](https://github.com/SYSTRAN/faster-whisper):

### Intelligent VAD Preprocessing

`EnhancedWhisperVadProcessor` aggregates Silero VAD speech segments into optimal-sized chunks for transcription. Instead of transcribing hundreds of tiny segments, it merges adjacent speech into configurable windows — **2-3x faster** on audio with significant silence.

### Temperature Fallback

`EnhancedWhisperState` automatically retries transcription at higher temperatures when quality thresholds aren't met (compression ratio, log probability, no-speech probability). Handles noisy/difficult audio without manual intervention.

Both features are orthogonal — use one, both, or neither. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for design details.

## Design Principles

- **Safety** — all unsafe FFI encapsulated with null checks, lifetime enforcement, RAII cleanup
- **Zero-copy** — audio slices passed directly to C++ via pointer, no intermediate copies
- **Progressive enhancement** — `Enhanced*` types opt-in; base API stays clean
- **Idiomatic Rust** — builder patterns, `thiserror`, correct `Send`/`Sync` bounds
- **Cross-platform** — Windows (MSVC), Linux, macOS; SIMD auto-detected, GPU via feature flags

## Models

### Downloading

The easiest way to get test models:

```bash
cargo xtask test-setup
```

This downloads `ggml-tiny.en.bin` and the Silero VAD model into `whisper-cpp-plus-sys/whisper.cpp/models/` using whisper.cpp's own download scripts.

For production models, download from Hugging Face:

```bash
curl -L -o models/ggml-base.en.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin
```

### Available Models

| Model | Size | English-only | Multilingual |
|-------|------|-------------|--------------|
| tiny | 39 MB | tiny.en | tiny |
| base | 142 MB | base.en | base |
| small | 466 MB | small.en | small |
| medium | 1.5 GB | medium.en | medium |
| large-v3 | 3.1 GB | — | large-v3 |

## Testing

### Setup

```bash
# Download test models (tiny.en + Silero VAD)
cargo xtask test-setup

# Run all tests
cargo test -p whisper-cpp-plus

# With async tests
cargo test -p whisper-cpp-plus --features async
```

Tests that require models skip gracefully if not downloaded.

### Test Suites

```bash
cargo test --lib                          # Unit tests (32 tests)
cargo test --test integration             # Core integration (10 tests)
cargo test --test type_safety             # Send/Sync verification (11 tests)
cargo test --test real_audio              # JFK audio transcription
cargo test --test enhanced_integration    # Enhanced VAD + fallback
cargo test --test stream_pcm_integration  # WhisperStreamPcm modes
cargo test --test vad_integration         # Silero VAD
cargo test --test quantization --features quantization  # Model quantization
```

### Benchmarks

```bash
cargo bench --bench transcription           # Core transcription
cargo bench --bench enhanced_vad_bench      # VAD segment aggregation
cargo bench --bench enhanced_fallback_bench # Quality threshold checks
```

## Build Optimization

C++ compilation takes several minutes. Use `xtask` to cache the compiled library:

```bash
# Build and cache once
cargo xtask prebuild

# Subsequent builds use cache (< 1 second)
cargo build
```

### xtask Commands

```bash
cargo xtask prebuild          # Build precompiled library
cargo xtask prebuild --force  # Force rebuild
cargo xtask info              # Show available prebuilt libraries
cargo xtask clean             # Remove cached libraries
cargo xtask test-setup        # Download test models
```

See [docs/CACHING_GUIDE.md](docs/CACHING_GUIDE.md) for details.

## Safety

### Thread Safety

- `WhisperContext`: `Send + Sync` — share via `Arc`
- `WhisperState`: `Send` only — one per thread
- `FullParams`: not `Send`/`Sync` — create per transcription

### Memory Safety

All unsafe FFI operations encapsulated with null pointer checks, lifetime enforcement, and RAII cleanup.

## Troubleshooting

**"Failed to load model"** — check file path, permissions, available memory

**"Invalid audio format"** — must be 16kHz mono f32, normalized to [-1, 1]

**Linking errors on Windows** — install Visual Studio Build Tools 2022, ensure x64 MSVC toolchain. See [docs/TECHNICAL_REFERENCE.md](docs/TECHNICAL_REFERENCE.md).

## Development

```bash
git clone --recursive https://github.com/getpinch/whisper-cpp-plus-rs
cd whisper-cpp-plus-rs
cargo xtask test-setup
cargo test
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for design decisions and module layout.

## License

[MIT](LICENSE)

### Attribution

- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) by Georgi Gerganov (MIT)
- [OpenAI Whisper](https://github.com/openai/whisper) by OpenAI (MIT)

## Support

- [API Documentation](https://docs.rs/whisper-cpp-plus)
- [Issue Tracker](https://github.com/getpinch/whisper-cpp-plus-rs/issues)
