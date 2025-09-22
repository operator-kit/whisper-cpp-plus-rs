//! Example of async transcription
//!
//! This example demonstrates how to use the async API for non-blocking
//! transcription in async Rust applications.

#[cfg(feature = "async")]
use std::path::Path;
#[cfg(feature = "async")]
use std::time::Duration;
#[cfg(feature = "async")]
use tokio::time::sleep;
#[cfg(feature = "async")]
use whisper_cpp_rs::{
    AsyncWhisperStream, FullParams, SamplingStrategy, StreamConfig, StreamConfigBuilder,
    WhisperContext,
};

#[cfg(not(feature = "async"))]
fn main() {
    eprintln!("This example requires the 'async' feature to be enabled.");
    eprintln!("Run with: cargo run --example async_transcribe --features async");
}

#[cfg(feature = "async")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Check if model exists
    let model_path = "tests/models/ggml-tiny.en.bin";
    if !Path::new(model_path).exists() {
        eprintln!("Model file not found at: {}", model_path);
        eprintln!("Please download a model first. See README.md for instructions.");
        return Ok(());
    }

    println!("Loading Whisper model...");
    let context = WhisperContext::new(model_path)?;

    // Example 1: Simple async transcription
    println!("\n=== Example 1: Simple Async Transcription ===");
    simple_async_transcription(&context).await?;

    // Example 2: Async streaming
    println!("\n=== Example 2: Async Streaming ===");
    async_streaming(&context).await?;

    // Example 3: Concurrent transcriptions
    println!("\n=== Example 3: Concurrent Transcriptions ===");
    concurrent_transcriptions(&context).await?;

    Ok(())
}

#[cfg(feature = "async")]
async fn simple_async_transcription(
    context: &WhisperContext,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Running simple async transcription...");

    // Generate test audio
    let audio = generate_test_audio(3.0); // 3 seconds

    // Transcribe asynchronously
    let start = std::time::Instant::now();
    let text = context.transcribe_async(audio).await?;
    let elapsed = start.elapsed();

    println!("Transcription completed in {:?}", elapsed);
    println!("Text: {}", if text.is_empty() { "(silence)" } else { &text });

    Ok(())
}

#[cfg(feature = "async")]
async fn async_streaming(context: &WhisperContext) -> Result<(), Box<dyn std::error::Error>> {
    println!("Setting up async streaming...");

    // Configure streaming
    let config = StreamConfigBuilder::new()
        .chunk_seconds(2.0)
        .overlap_seconds(0.5)
        .build();

    let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 })
        .language("en")
        .print_progress(false);

    // Create async stream
    let mut stream = AsyncWhisperStream::with_config(context.clone(), params, config)?;

    // Simulate streaming audio input
    println!("Feeding audio chunks...");
    for i in 0..5 {
        let chunk = generate_test_audio(1.0); // 1 second chunks

        println!("  Feeding chunk {}...", i + 1);
        stream.feed_audio(chunk).await?;

        // Simulate real-time delay
        sleep(Duration::from_millis(100)).await;

        // Check for segments
        while let Some(segments) = stream.try_recv_segments() {
            for segment in segments {
                println!(
                    "  [Segment {:.2}s - {:.2}s]: {}",
                    segment.start_seconds(),
                    segment.end_seconds(),
                    segment.text
                );
            }
        }
    }

    // Flush remaining audio
    println!("Flushing stream...");
    let final_segments = stream.flush().await?;
    for segment in final_segments {
        println!(
            "  [Final {:.2}s - {:.2}s]: {}",
            segment.start_seconds(),
            segment.end_seconds(),
            segment.text
        );
    }

    // Stop the stream
    stream.stop().await?;
    println!("Stream stopped successfully");

    Ok(())
}

#[cfg(feature = "async")]
async fn concurrent_transcriptions(
    context: &WhisperContext,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Running concurrent transcriptions...");

    // Create multiple transcription tasks
    let mut handles = Vec::new();

    for i in 0..3 {
        let ctx = context.clone();
        let handle = tokio::spawn(async move {
            let audio = generate_test_audio(2.0); // 2 seconds each

            let start = std::time::Instant::now();
            let result = ctx.transcribe_async(audio).await;
            let elapsed = start.elapsed();

            (i, result, elapsed)
        });
        handles.push(handle);
    }

    // Wait for all tasks to complete
    for handle in handles {
        let (task_id, result, elapsed) = handle.await?;
        match result {
            Ok(text) => {
                println!(
                    "  Task {} completed in {:?}: {}",
                    task_id,
                    elapsed,
                    if text.is_empty() { "(silence)" } else { &text }
                );
            }
            Err(e) => {
                println!("  Task {} failed: {}", task_id, e);
            }
        }
    }

    println!("All concurrent tasks completed");
    Ok(())
}

#[cfg(feature = "async")]
fn generate_test_audio(seconds: f32) -> Vec<f32> {
    // Generate silence with very slight noise
    let samples = (seconds * 16000.0) as usize;
    let mut audio = vec![0.0f32; samples];

    // Add minimal noise to avoid complete silence
    for sample in audio.iter_mut() {
        *sample = (rand::random::<f32>() - 0.5) * 0.0001;
    }

    audio
}

/// Simple rand implementation for demo
#[cfg(all(feature = "async", not(feature = "rand")))]
mod rand {
    use std::time::{SystemTime, UNIX_EPOCH};

    pub fn random<T>() -> T
    where
        T: From<f32>,
    {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .subsec_nanos();
        let value = ((nanos % 1000) as f32) / 1000.0;
        T::from(value)
    }
}