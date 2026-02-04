use std::path::{Path, PathBuf};
use whisper_cpp_rs::{WhisperContext, FullParams, SamplingStrategy};
use hound;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Find model using flexible path resolution
    let model_path = find_model("ggml-tiny.en.bin")
        .ok_or("Model file not found. Please download a model or set WHISPER_MODEL_PATH.\n\
                You can download the tiny.en model from:\n\
                https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin")?;

    println!("Loading Whisper model from {:?}...", model_path);
    let ctx = WhisperContext::new(&model_path)?;

    println!("Model loaded successfully!");
    println!("Model info:");
    println!("  - Vocabulary size: {}", ctx.n_vocab());
    println!("  - Audio context: {}", ctx.n_audio_ctx());
    println!("  - Text context: {}", ctx.n_text_ctx());
    println!("  - Multilingual: {}", ctx.is_multilingual());

    // Load real audio for testing
    println!("\nLoading test audio...");
    let audio = find_and_load_audio()?;

    // Transcribe with default parameters
    println!("Transcribing with default parameters...");
    let text = ctx.transcribe(&audio)?;
    println!("Transcription result: '{}'", text);

    // Transcribe with custom parameters
    println!("\nTranscribing with custom parameters...");
    let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 })
        .language("en")
        .translate(false)
        .no_timestamps(false)
        .temperature(0.0)
        .n_threads(2);

    let result = ctx.transcribe_with_full_params(&audio, params)?;
    println!("Full transcription result:");
    println!("  Text: '{}'", result.text);
    println!("  Segments: {}", result.segments.len());

    for (i, segment) in result.segments.iter().enumerate() {
        println!("    Segment {}: [{:.2}s - {:.2}s] '{}'",
            i + 1,
            segment.start_seconds(),
            segment.end_seconds(),
            segment.text
        );
    }

    println!("\nSuccess! The whisper.cpp Rust wrapper is working correctly.");

    Ok(())
}

/// Find model file in common locations
fn find_model(name: &str) -> Option<PathBuf> {
    // Check env var first
    if let Ok(dir) = std::env::var("WHISPER_MODEL_PATH") {
        let path = Path::new(&dir).join(name);
        if path.exists() {
            return Some(path);
        }
        // Also try if env var points directly to a model file
        let path = PathBuf::from(&dir);
        if path.exists() && path.is_file() {
            return Some(path);
        }
    }

    // Common locations to check
    let search_paths = [
        // Workspace-relative (running from root)
        format!("whisper-cpp-rs/tests/models/{}", name),
        // Crate-relative (running from whisper-cpp-rs/)
        format!("tests/models/{}", name),
        // whisper.cpp models directory
        format!("vendor/whisper.cpp/models/{}", name),
        // Current directory
        name.to_string(),
    ];

    for path_str in &search_paths {
        let path = PathBuf::from(path_str);
        if path.exists() {
            return Some(path);
        }
    }

    None
}

/// Find and load audio file
fn find_and_load_audio() -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let audio_paths = [
        // whisper.cpp samples
        "vendor/whisper.cpp/samples/jfk.wav",
        // Crate test audio
        "tests/audio/jfk.wav",
        "whisper-cpp-rs/tests/audio/jfk.wav",
        // User samples
        "samples/test.wav",
    ];

    for path in &audio_paths {
        if Path::new(path).exists() {
            println!("Loading audio from: {}", path);
            return load_wav_file(path);
        }
    }

    Err("No audio files found. Please provide audio at vendor/whisper.cpp/samples/jfk.wav".into())
}

fn load_wav_file(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    // Check format
    if spec.sample_rate != 16000 {
        eprintln!("Warning: Audio sample rate is {}Hz, expected 16000Hz", spec.sample_rate);
    }

    if spec.channels != 1 {
        eprintln!("Warning: Audio has {} channels, using first channel only", spec.channels);
    }

    let samples: Vec<f32> = reader
        .samples::<i16>()
        .step_by(spec.channels as usize)
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();

    Ok(samples)
}
