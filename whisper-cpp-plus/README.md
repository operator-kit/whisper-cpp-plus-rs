# whisper-cpp-plus

> **Pinned to whisper.cpp v1.8.3** (fork: [`rmorse/whisper.cpp`](https://github.com/rmorse/whisper.cpp), branch: `stream-pcm`)

Safe Rust bindings for [whisper.cpp](https://github.com/ggerganov/whisper.cpp) with **real-time PCM streaming** and VAD support.

## Highlights

- **Real-time PCM streaming** — feed raw audio chunks, get transcription as you go
- **VAD integration** — Silero-based voice activity detection for intelligent chunking
- **Full whisper.cpp API** — batch transcription, timestamps, language detection
- **GPU acceleration** — CUDA, Metal, OpenBLAS support

## PCM Streaming (VAD-driven)

Process raw PCM from any `Read` source (file, stdin, socket, microphone) with automatic VAD segmentation. Port of `stream-pcm.cpp`.

```rust
use whisper_cpp_plus::{WhisperContext, WhisperStreamPcm, WhisperStreamPcmConfig,
                       PcmReader, PcmReaderConfig, FullParams, SamplingStrategy};

let ctx = WhisperContext::new("ggml-tiny.en.bin")?;
let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 }).language("en");
let config = WhisperStreamPcmConfig { use_vad: true, ..Default::default() };

// Any Read source — here a file, but could be stdin, socket, etc.
let source = std::fs::File::open("audio.pcm")?;
let reader = PcmReader::new(Box::new(source), PcmReaderConfig::default());

let mut stream = WhisperStreamPcm::new(&ctx, params, config, reader)?;

stream.run(|segments, _start_ms, _end_ms| {
    for seg in segments {
        println!("[{:.2}s - {:.2}s] {}", seg.start_seconds(), seg.end_seconds(), seg.text);
    }
})?;
```

## File Transcription

For pre-recorded audio files — load a WAV (16kHz mono), transcribe in one shot:

```rust
use whisper_cpp_plus::{WhisperContext, FullParams, SamplingStrategy};

let ctx = WhisperContext::new("ggml-base.en.bin")?;

// Load WAV file as 16kHz mono f32 samples (using hound crate)
let mut reader = hound::WavReader::open("audio.wav")?;
let audio: Vec<f32> = reader.samples::<i16>()
    .map(|s| s.unwrap() as f32 / 32768.0)
    .collect();

// Simple — just get the text
let text = ctx.transcribe(&audio)?;
println!("{}", text);

// With parameters — get timestamped segments
let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 })
    .language("en")
    .no_timestamps(false);

let result = ctx.transcribe_with_full_params(&audio, params)?;
for seg in &result.segments {
    println!("[{:.2}s - {:.2}s] {}", seg.start_seconds(), seg.end_seconds(), seg.text);
}
```

## Sliding Window Streaming

Feed audio chunks and process with configurable step/overlap. Port of `stream.cpp`.

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
whisper-cpp-plus = { version = "0.1.4", features = ["cuda"] }
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
