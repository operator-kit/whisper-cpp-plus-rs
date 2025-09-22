use std::path::Path;
use whisper_cpp_rs::{WhisperContext, FullParams, SamplingStrategy};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Check if model exists
    let model_path = "tests/models/ggml-tiny.en.bin";
    if !Path::new(model_path).exists() {
        eprintln!("Error: Model file not found at {}", model_path);
        eprintln!("Please download a model file first.");
        eprintln!("You can download the tiny.en model from:");
        eprintln!("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin");
        return Ok(());
    }

    println!("Loading Whisper model from {}...", model_path);
    let ctx = WhisperContext::new(model_path)?;

    println!("Model loaded successfully!");
    println!("Model info:");
    println!("  - Vocabulary size: {}", ctx.n_vocab());
    println!("  - Audio context: {}", ctx.n_audio_ctx());
    println!("  - Text context: {}", ctx.n_text_ctx());
    println!("  - Multilingual: {}", ctx.is_multilingual());

    // Create 3 seconds of silence as test audio
    println!("\nGenerating test audio (3 seconds of silence)...");
    let sample_rate = 16000;
    let duration_seconds = 3;
    let audio = vec![0.0f32; sample_rate * duration_seconds];

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