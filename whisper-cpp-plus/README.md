# whisper-cpp-plus

> **Pinned to whisper.cpp v1.8.3** (fork: [`rmorse/whisper.cpp`](https://github.com/rmorse/whisper.cpp), branch: `stream-pcm`)

Safe Rust bindings for [whisper.cpp](https://github.com/ggerganov/whisper.cpp) with **real-time PCM streaming** and VAD support.

## Highlights

- **Real-time PCM streaming** — feed raw audio chunks, get transcription as you go
- **VAD integration** — Silero-based voice activity detection for intelligent chunking
- **Full whisper.cpp API** — batch transcription, timestamps, language detection
- **GPU acceleration** — CUDA, Metal, OpenBLAS support

## Real-time Streaming

Feed PCM audio chunks directly from a microphone or audio stream:

```rust
use whisper_cpp_plus::{WhisperStreamPcm, StreamPcmParams};

let params = StreamPcmParams::builder()
    .model_path("ggml-tiny.en.bin")
    .vad_model_path("ggml-silero-v6.2.0.bin")
    .build()?;

let mut stream = WhisperStreamPcm::new(params)?;

// Feed PCM chunks as they arrive (16kHz mono f32)
for chunk in audio_source {
    if let Some(text) = stream.process_audio(&chunk)? {
        println!("Transcribed: {}", text);
    }
}
```

## Batch Transcription

For pre-recorded audio files:

```rust
use whisper_cpp_plus::{WhisperContext, TranscriptionParams};

let ctx = WhisperContext::new("path/to/ggml-base.bin")?;
let audio: Vec<f32> = load_audio(); // 16kHz mono f32

// Simple
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
whisper-cpp-plus = { version = "0.1.0", features = ["cuda"] }
```

## Modules

- **Transcription** — `WhisperContext`, `WhisperState`, `FullParams`, `TranscriptionParams` builder
- **Streaming** — `WhisperStream` for chunked real-time transcription
- **StreamPCM** — `WhisperStreamPcm` for raw PCM input with VAD-driven processing
- **VAD** — `WhisperVadProcessor` for Silero-based voice activity detection
- **Enhanced** — Temperature fallback + enhanced VAD aggregation for improved quality
- **Quantization** — `WhisperQuantize` for model compression (feature = `quantization`)

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
