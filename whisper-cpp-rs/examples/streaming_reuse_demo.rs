//! Demonstration of efficient WhisperStream reuse
//!
//! This example shows the performance benefit of reusing WhisperStream
//! across multiple streaming sessions, avoiding expensive state reallocation.

use std::path::Path;
use std::time::Duration;
use whisper_cpp_rs::{
    FullParams, SamplingStrategy, StreamConfigBuilder, WhisperContext, WhisperStream,
};

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
        let audio = vec![0.0f32; 16000 * 2]; // 2 seconds of silence
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