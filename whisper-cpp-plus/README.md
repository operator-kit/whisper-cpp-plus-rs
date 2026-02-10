# whisper-cpp-plus

> **Pinned to whisper.cpp v1.8.3** (fork: [`rmorse/whisper.cpp`](https://github.com/rmorse/whisper.cpp), branch: `stream-pcm`)

Safe Rust bindings for [whisper.cpp](https://github.com/ggerganov/whisper.cpp) with **real-time PCM streaming** and VAD support.

## Highlights

- **Real-time PCM streaming** — feed raw audio chunks, get transcription as you go
- **VAD integration** — Silero-based voice activity detection for intelligent chunking
- **Full whisper.cpp API** — batch transcription, timestamps, language detection
- **GPU acceleration** — CUDA, Metal, OpenBLAS support

## Real-time Streaming

Two streaming APIs for different use cases:

### WhisperStream — Sliding Window (port of stream.cpp)

Feed audio chunks and process with configurable step/overlap:

```rust
use whisper_cpp_plus::{WhisperContext, WhisperStream, WhisperStreamConfig, FullParams};

let ctx = WhisperContext::new("ggml-tiny.en.bin")?;
let params = FullParams::default().language("en");
let config = WhisperStreamConfig { step_ms: 3000, ..Default::default() };

let mut stream = WhisperStream::with_config(&ctx, params, config)?;

// Feed audio chunks as they arrive
stream.feed_audio(&audio_chunk);

// Process when ready
while let Some(segments) = stream.process_step()? {
    for seg in &segments {
        println!("[{:.2}s] {}", seg.start_seconds(), seg.text);
    }
}
```

### WhisperStreamPcm — VAD-driven (port of stream-pcm.cpp)

Process raw PCM from any `Read` source with automatic VAD segmentation:

```rust
use whisper_cpp_plus::{WhisperContext, WhisperStreamPcm, WhisperStreamPcmConfig,
                       PcmReader, PcmReaderConfig, FullParams};
use std::fs::File;

let ctx = WhisperContext::new("ggml-tiny.en.bin")?;
let params = FullParams::default().language("en");
let config = WhisperStreamPcmConfig { use_vad: true, ..Default::default() };

// Create PCM reader from any Read source (file, stdin, socket, etc.)
let file = File::open("audio.pcm")?;
let reader = PcmReader::new(Box::new(file), PcmReaderConfig::default());

let mut stream = WhisperStreamPcm::new(&ctx, params, config, reader)?;

// Process until EOF
stream.run(|segments, start_ms, end_ms| {
    for seg in segments {
        println!("[{:.2}s] {}", seg.start_seconds(), seg.text);
    }
})?;
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
whisper-cpp-plus = { version = "0.1.3", features = ["cuda"] }
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
cargo run --example streaming           # WhisperStream (sliding window + reuse demo)
cargo run --example stream_pcm          # WhisperStreamPcm (VAD-driven, threaded reader)
cargo run --example compare_vad
cargo run --example enhanced_vad
cargo run --example temperature_fallback
cargo run --example async_transcribe --features async
```

## License

MIT
