//! Demonstration of efficient WhisperStream reuse
//!
//! This example shows the performance benefit of reusing WhisperStream
//! across multiple streaming sessions, avoiding expensive state reallocation.

use std::path::Path;
use std::time::Duration;
use whisper_cpp_plus::{
    FullParams, SamplingStrategy, StreamConfigBuilder, WhisperContext, WhisperStream,
};
use hound;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== WhisperStream Reuse Demo ===\n");
    println!("This example demonstrates efficient stream reuse.");
    println!("Without an actual model, we'll show the API pattern.\n");

    // Check if model exists
    let model_path = "tests/models/ggml-tiny.en.bin";
    if !Path::new(model_path).exists() {
        println!("Model not found. Showing API usage pattern:\n");
        demonstrate_api_pattern();
        return Ok(());
    }

    // If model exists, run the actual demo
    run_with_model(model_path)?;
    Ok(())
}

/// Demonstrate the API pattern without needing an actual model
fn demonstrate_api_pattern() {
    println!("```rust");
    println!("// EFFICIENT PATTERN: Create stream once, reuse for multiple sessions");
    println!();
    println!("// 1. Create the stream ONCE");
    println!("let mut stream = WhisperStream::new(&context, params)?;");
    println!();
    println!("// 2. Use for first recording session");
    println!("stream.feed_audio(&audio_session_1);");
    println!("let segments = stream.process_pending()?;");
    println!("stream.flush()?;");
    println!();
    println!("// 3. Reset for next session (reuses 500MB+ WhisperState!)");
    println!("stream.reset()?;  // <-- Efficient: no reallocation");
    println!();
    println!("// 4. Use for second recording session");
    println!("stream.feed_audio(&audio_session_2);");
    println!("let segments = stream.process_pending()?;");
    println!("stream.flush()?;");
    println!();
    println!("// 5. Reset again for more sessions");
    println!("stream.reset()?;  // Still efficient!");
    println!();
    println!("// Only use recreate_state() when absolutely necessary:");
    println!("stream.recreate_state()?;  // Expensive: reallocates state");
    println!("```");
    println!();
    println!("BENEFITS:");
    println!("- Avoids repeated 500MB+ allocations for medium/large models");
    println!("- Matches whisper.cpp's internal behavior");
    println!("- Significantly faster session switching");
}

/// Run the actual demo with a model
fn run_with_model(model_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading Whisper model...");
    let context = WhisperContext::new(model_path)?;

    // Configure streaming
    let stream_config = StreamConfigBuilder::new()
        .chunk_seconds(2.0)
        .overlap_seconds(0.3)
        .build();

    let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 })
        .language("en")
        .print_progress(false);

    // Create stream ONCE
    println!("Creating WhisperStream (expensive initial allocation)...");
    let start = std::time::Instant::now();
    let mut stream = WhisperStream::with_config(&context, params, stream_config)?;
    println!("Stream created in {:?}\n", start.elapsed());

    // Simulate 3 recording sessions
    for session in 1..=3 {
        println!("--- Session {} ---", session);

        if session > 1 {
            // Reset the stream (efficient - reuses state!)
            let start = std::time::Instant::now();
            stream.reset()?;
            println!("Stream reset in {:?} (state reused!)", start.elapsed());
        }

        // Process some audio
        let audio = load_demo_audio(2)?; // Load 2 seconds of real audio
        stream.feed_audio(&audio);

        let _ = stream.process_pending()?;
        let _ = stream.flush()?;

        println!("Session {} complete\n", session);
        std::thread::sleep(Duration::from_millis(500));
    }

    // Show the difference with recreate_state()
    println!("--- Demonstrating recreate_state() ---");
    let start = std::time::Instant::now();
    stream.recreate_state()?;
    println!("State recreated in {:?} (expensive reallocation!)", start.elapsed());

    println!("\nDemo complete!");
    Ok(())
}

fn load_demo_audio(duration_seconds: usize) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // Try to load real audio files
    let jfk_path = "vendor/whisper.cpp/samples/jfk.wav";
    let alt_path = "samples/audio.wav";

    let audio = if Path::new(jfk_path).exists() {
        println!("Loading JFK audio from {}...", jfk_path);
        load_wav_file(jfk_path)?
    } else if Path::new(alt_path).exists() {
        println!("Loading audio from {}...", alt_path);
        load_wav_file(alt_path)?
    } else {
        eprintln!("\nError: No audio files found!");
        eprintln!("Please provide audio at one of these locations:");
        eprintln!("  1. {} (JFK sample from whisper.cpp)", jfk_path);
        eprintln!("  2. {} (custom audio sample at 16kHz)", alt_path);
        eprintln!("\nNote: Synthetic audio generation was removed as it doesn't produce meaningful speech.");
        return Err("No audio files found for streaming reuse demo".into());
    };

    // Truncate to requested duration
    let samples_needed = 16000 * duration_seconds;
    Ok(audio.into_iter().take(samples_needed).collect())
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