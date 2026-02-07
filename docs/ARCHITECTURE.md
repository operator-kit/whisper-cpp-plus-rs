# Architecture

## Crate structure

Dual-crate architecture: `whisper-sys` (raw FFI bindings) + `whisper-cpp-rs` (safe idiomatic API). Sys crate stays stable while the high-level API evolves.

## Context / state separation

Maps to whisper.cpp's thread safety model:
- `WhisperContext` wraps `whisper_context*` — immutable, `Send + Sync`, shareable via `Arc`
- `WhisperState` wraps `whisper_state*` — mutable, `Send` only, one per thread
- `FullParams` — create per transcription, not `Send`/`Sync`

`WhisperState` holds `Arc<ContextPtr>` to keep the context alive independently.

## Enhanced features module

All enhancements live under `src/enhanced/` and follow these rules:

### Naming
- Types: `Enhanced` prefix (`EnhancedWhisperVadProcessor`, `EnhancedWhisperState`)
- Methods: `_enhanced` suffix (`transcribe_with_params_enhanced`)
- Module: `whisper_cpp_rs::enhanced::{vad, fallback}`

### Separation of concerns
- **VAD** = preprocessing (before transcription) — `enhanced::vad`
- **Temperature fallback** = transcription quality enhancement — `enhanced::fallback`
- Features are orthogonal: use one, both, or neither independently

### Internal access pattern
Enhanced modules access FFI via `pub(crate)` fields (e.g., `WhisperState.ptr`). Raw pointers never appear in the public API.

### API design
- Enhanced methods mirror base API shape — opt-in, never modifies defaults
- Base types stay clean of enhancement-specific concerns

## Module layout

```
whisper-cpp-rs/src/
├── lib.rs              # Public API, convenience methods on WhisperContext
├── context.rs          # WhisperContext (Arc<ContextPtr>)
├── state.rs            # WhisperState (ptr + Arc<ContextPtr>)
├── params.rs           # FullParams, TranscriptionParams builder
├── error.rs            # WhisperError enum (thiserror)
├── buffer.rs           # AudioBuffer (circular, for streaming)
├── stream.rs           # WhisperStream (chunk-based streaming)
├── stream_pcm.rs       # WhisperStreamPcm (port of stream-pcm.cpp)
├── vad.rs              # WhisperVadProcessor (Silero VAD via whisper.cpp)
├── async_api.rs        # spawn_blocking wrappers (feature = "async")
└── enhanced/
    ├── mod.rs
    ├── vad.rs           # EnhancedWhisperVadProcessor (segment aggregation)
    └── fallback.rs      # Temperature fallback with quality thresholds
```

## Async strategy

whisper.cpp operations are CPU-bound. Async wrappers use `tokio::task::spawn_blocking` — no async whisper.cpp calls.

## Audio data

Zero-copy where possible: `&[f32]` slices passed directly to C++ via `.as_ptr()` + `.len()`. Audio must be 16kHz mono f32.

## Build system

`cc` crate compiles whisper.cpp sources directly (no CMake dependency). whisper.cpp vendored as git submodule at `vendor/whisper.cpp`. Prebuilt library caching available via `cargo xtask prebuild`.
