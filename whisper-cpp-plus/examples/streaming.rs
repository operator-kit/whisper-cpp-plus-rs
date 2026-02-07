//! Example of streaming transcription with stream reuse
//!
//! This example demonstrates how to use WhisperStream for real-time
//! transcription of audio chunks as they arrive, and how to efficiently
//! reuse the stream across multiple sessions.

use std::path::Path;
use std::time::Duration;
use whisper_cpp_plus::{
    FullParams, SamplingStrategy, StreamConfigBuilder, WhisperContext, WhisperStream,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Check if model exists
    let model_path = "tests/models/ggml-tiny.en.bin";
    if !Path::new(model_path).exists() {
        eprintln!("Model file not found at: {}", model_path);
        eprintln!("Please download a model first. See README.md for instructions.");
        return Ok(());
    }

    println!("Loading Whisper model...");
    let context = WhisperContext::new(model_path)?;

    // Configure streaming parameters
    let stream_config = StreamConfigBuilder::new()
        .chunk_seconds(3.0)      // Process 3-second chunks
        .overlap_seconds(0.5)     // 0.5 second overlap between chunks
        .min_chunk_size(16000)    // Minimum 1 second before processing
        .partial_timeout(Duration::from_secs(2))
        .build();

    // Set up transcription parameters
    let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 })
        .language("en")
        .no_timestamps(false)
        .print_progress(false);

    // Create the stream ONCE - we'll reuse it for multiple sessions
    let mut stream = WhisperStream::with_config(&context, params, stream_config)?;

    println!("Streaming transcription initialized!");
    println!("Demonstrating stream reuse across multiple sessions...\n");

    // Simulate multiple streaming sessions (e.g., multiple recordings)
    for session in 1..=3 {
        println!("=== SESSION {} ===", session);

        // For sessions 2 and onwards, reset the stream
        // This efficiently reuses the WhisperState (no model/ram reallocation)
        if session > 1 {
            println!("Resetting stream for new session (reusing state)...");
            stream.reset()?;
            // Note: We're reusing the same WhisperState internally
        }

        // Simulate streaming for this session
        process_audio_session(&mut stream, session)?;

        println!("Session {} complete!\n", session);
        std::thread::sleep(Duration::from_secs(1));
    }

    // Example: If you need to completely recreate the state
    // (e.g., after an error or switching audio sources dramatically)
    println!("Demonstrating state recreation...");
    stream.recreate_state()?;  // This WILL reallocate the state (expensive!)
    println!("State recreated - use this sparingly!");

    println!("\nAll sessions complete!");

    Ok(())
}

/// Process a single audio session
fn process_audio_session(stream: &mut WhisperStream, session_num: usize) -> Result<(), Box<dyn std::error::Error>> {
    // Simulate streaming audio input
    let chunk_size = 16000; // 1 second chunks at 16kHz

    // For demo, we'll use shorter audio for sessions 2 and 3
    let duration_seconds = if session_num == 1 { 5 } else { 3 };
    let sample_audio = generate_demo_audio(duration_seconds)?;

    // Process audio in chunks to simulate streaming
    let mut chunk_count = 0;
    for chunk in sample_audio.chunks(chunk_size) {
        chunk_count += 1;

        // Simulate real-time delay
        std::thread::sleep(Duration::from_millis(300));

        println!("  Session {}, Chunk {}: {} samples",
                 session_num, chunk_count, chunk.len());
        stream.feed_audio(chunk);

        // Process pending audio
        let segments = stream.process_pending()?;

        // Print new segments
        for segment in segments {
            println!(
                "  [{:.2}s - {:.2}s]: {}",
                segment.start_seconds(),
                segment.end_seconds(),
                segment.text
            );
        }
    }

    // Flush any remaining audio
    let final_segments = stream.flush()?;
    for segment in final_segments {
        println!(
            "  [{:.2}s - {:.2}s]: {}",
            segment.start_seconds(),
            segment.end_seconds(),
            segment.text
        );
    }

    println!("  Processed {} samples total", stream.processed_samples());

    Ok(())
}

/// Generate demo audio for a session
fn generate_demo_audio(duration_seconds: usize) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // Try to load real audio samples
    let jfk_path = "vendor/whisper.cpp/samples/jfk.wav";
    let alt_path = "samples/audio.wav";

    if Path::new(jfk_path).exists() {
        println!("  Loading JFK sample audio from {}...", jfk_path);
        let audio = load_wav_16khz_mono(jfk_path)?;
        // Truncate to requested duration
        let samples_needed = 16000 * duration_seconds;
        Ok(audio.into_iter().take(samples_needed).collect())
    } else if Path::new(alt_path).exists() {
        println!("  Loading audio from {}...", alt_path);
        let audio = load_wav_16khz_mono(alt_path)?;
        let samples_needed = 16000 * duration_seconds;
        Ok(audio.into_iter().take(samples_needed).collect())
    } else {
        eprintln!("\nError: No audio files found!");
        eprintln!("Please provide audio at one of these locations:");
        eprintln!("  1. {} (JFK sample from whisper.cpp)", jfk_path);
        eprintln!("  2. {} (custom audio sample at 16kHz)", alt_path);
        eprintln!("\nNote: Synthetic audio generation was removed as it doesn't produce meaningful speech.");
        Err("No audio files found for streaming example".into())
    }
}

/// Load a WAV file and convert to 16kHz mono f32
fn load_wav_16khz_mono(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    use hound;

    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    // Check sample rate
    if spec.sample_rate != 16000 {
        return Err(format!(
            "Audio must be 16kHz, but got {}Hz. Please resample the audio.",
            spec.sample_rate
        ).into());
    }

    // Convert samples to f32
    let samples: Result<Vec<f32>, _> = match spec.bits_per_sample {
        16 => reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
            .collect(),
        32 => reader
            .samples::<i32>()
            .map(|s| s.map(|v| v as f32 / i32::MAX as f32))
            .collect(),
        _ => {
            return Err(format!(
                "Unsupported bits per sample: {}",
                spec.bits_per_sample
            ).into())
        }
    };

    let samples = samples?;

    // Convert to mono if necessary
    let mono_samples = match spec.channels {
        1 => samples,
        2 => {
            // Average stereo channels to mono
            samples
                .chunks(2)
                .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
                .collect()
        }
        _ => {
            return Err(format!(
                "Unsupported number of channels: {}",
                spec.channels
            ).into())
        }
    };

    Ok(mono_samples)
}

