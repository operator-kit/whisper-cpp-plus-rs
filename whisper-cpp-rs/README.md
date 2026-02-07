# whisper-cpp-rs

Safe, idiomatic Rust bindings for [whisper.cpp](https://github.com/ggerganov/whisper.cpp) — OpenAI's Whisper speech recognition model.

## Quick Start

```rust
use whisper_cpp_rs::{WhisperContext, TranscriptionParams};

let ctx = WhisperContext::new("path/to/ggml-base.bin")?;
let audio: Vec<f32> = load_audio(); // 16kHz mono f32

// Simple transcription
let text = ctx.transcribe(&audio)?;

// With parameters
let params = TranscriptionParams::builder()
    .language("en")
    .temperature(0.8)
    .enable_timestamps()
    .build();
let result = ctx.transcribe_with_params(&audio, params)?;
for seg in &result.segments {
    println!("[{:.1}s-{:.1}s] {}", seg.start_seconds(), seg.end_seconds(), seg.text);
}
```

## Features

| Feature | Description |
|---------|-------------|
| `cuda` | NVIDIA GPU acceleration via CUDA |
| `metal` | Apple Metal acceleration (macOS) |
| `openblas` | OpenBLAS acceleration (Linux) |
| `async` | Async transcription API via tokio |

Enable in `Cargo.toml`:
```toml
[dependencies]
whisper-cpp-rs = { version = "0.1.0", features = ["cuda"] }
```

## Modules

- **Transcription** — `WhisperContext`, `WhisperState`, `FullParams`, `TranscriptionParams` builder
- **Streaming** — `WhisperStream` for chunked real-time transcription
- **StreamPCM** — `WhisperStreamPcm` for raw PCM input with VAD-driven processing
- **VAD** — `WhisperVadProcessor` for Silero-based voice activity detection
- **Enhanced** — Temperature fallback + enhanced VAD aggregation for improved quality
- **Quantization** — See the `whisper-quantize` crate

## Examples

```sh
cargo run --example basic
cargo run --example streaming
cargo run --example streaming_reuse_demo
cargo run --example compare_vad
cargo run --example enhanced_vad
cargo run --example temperature_fallback
cargo run --example async_transcribe --features async
```

## License

MIT
