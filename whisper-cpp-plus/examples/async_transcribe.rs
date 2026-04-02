//! Async transcription example
//!
//! Shows async batch transcription using tokio. For real-time streaming,
//! use WhisperStreamPcm which already has its own reader thread.
//!
//! Requires: `cargo run --example async_transcribe --features async`

#[cfg(feature = "async")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use whisper_cpp_plus::WhisperContext;

    let model_path =
        find_model("ggml-tiny.en.bin").ok_or("Model not found. Run: cargo xtask test-setup")?;

    println!("Loading model from {:?}...", model_path);
    let ctx = WhisperContext::new(&model_path)?;

    // Load audio
    let audio = load_audio()?;
    println!(
        "Loaded {} samples ({:.1}s)",
        audio.len(),
        audio.len() as f64 / 16000.0
    );

    // Async transcription — runs in blocking threadpool
    println!("\nTranscribing asynchronously...");
    let text = ctx.transcribe_async(audio).await?;
    println!("Result: {}", text);

    Ok(())
}

#[cfg(not(feature = "async"))]
fn main() {
    eprintln!("This example requires the `async` feature:");
    eprintln!("  cargo run --example async_transcribe --features async");
}

#[cfg(feature = "async")]
fn find_model(name: &str) -> Option<std::path::PathBuf> {
    use std::path::Path;
    for env_var in ["WHISPER_TEST_MODEL_DIR", "WHISPER_MODEL_PATH"] {
        if let Ok(dir) = std::env::var(env_var) {
            let path = std::path::Path::new(&dir).join(name);
            if path.exists() {
                return Some(path);
            }
        }
    }
    let paths = [
        format!("tests/models/{}", name),
        format!("whisper-cpp-plus/tests/models/{}", name),
        format!("../whisper-cpp-plus-sys/whisper.cpp/models/{}", name),
        format!("whisper-cpp-plus-sys/whisper.cpp/models/{}", name),
    ];
    paths
        .iter()
        .find(|p| Path::new(p).exists())
        .map(std::path::PathBuf::from)
}

#[cfg(feature = "async")]
fn load_audio() -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    use std::path::Path;

    let path = std::env::var("WHISPER_TEST_AUDIO_DIR")
        .ok()
        .map(|d| format!("{}/jfk.wav", d))
        .filter(|p| Path::new(p).exists())
        .or_else(|| {
            let paths = [
                "../whisper-cpp-plus-sys/whisper.cpp/samples/jfk.wav",
                "whisper-cpp-plus-sys/whisper.cpp/samples/jfk.wav",
            ];
            paths
                .iter()
                .find(|p| Path::new(*p).exists())
                .map(|s| s.to_string())
        })
        .ok_or("No audio file found")?;

    println!("Loading audio from {}...", path);
    let mut reader = hound::WavReader::open(&path)?;
    let spec = reader.spec();

    let samples: Vec<f32> = reader
        .samples::<i16>()
        .step_by(spec.channels as usize)
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();

    Ok(samples)
}
