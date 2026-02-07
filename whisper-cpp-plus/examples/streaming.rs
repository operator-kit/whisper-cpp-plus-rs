//! Streaming transcription example — new WhisperStream API (port of stream.cpp)
//!
//! Creates a WhisperStream, feeds audio from a wav file, and calls
//! process_step() in a loop to get segments.

use std::path::Path;
use whisper_cpp_plus::{FullParams, SamplingStrategy, WhisperContext, WhisperStream, WhisperStreamConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "tests/models/ggml-tiny.en.bin";
    if !Path::new(model_path).exists() {
        eprintln!("Model not found at: {}", model_path);
        return Ok(());
    }

    println!("Loading model...");
    let ctx = WhisperContext::new(model_path)?;

    let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 })
        .language("en");

    let config = WhisperStreamConfig {
        step_ms: 3000,
        length_ms: 10000,
        keep_ms: 200,
        no_context: true,
        ..Default::default()
    };

    let mut stream = WhisperStream::with_config(&ctx, params, config)?;
    println!("Stream created.");

    // Load audio
    let audio = load_audio()?;
    println!("Loaded {} samples ({:.1}s)", audio.len(), audio.len() as f64 / 16000.0);

    // Feed in 1-second chunks to simulate real-time
    let chunk_size = 16000;
    for (i, chunk) in audio.chunks(chunk_size).enumerate() {
        stream.feed_audio(chunk);
        println!("Fed chunk {} ({} samples, buf={})", i + 1, chunk.len(), stream.buffer_size());

        // Process any ready steps
        while let Some(segments) = stream.process_step()? {
            for seg in &segments {
                println!(
                    "  [{:.2}s - {:.2}s]: {}",
                    seg.start_seconds(),
                    seg.end_seconds(),
                    seg.text
                );
            }
        }
    }

    // Flush remaining
    let final_segments = stream.flush()?;
    for seg in &final_segments {
        println!(
            "  [flush {:.2}s - {:.2}s]: {}",
            seg.start_seconds(),
            seg.end_seconds(),
            seg.text
        );
    }

    println!("Done. Processed {} samples total.", stream.processed_samples());
    Ok(())
}

fn load_audio() -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let jfk = "vendor/whisper.cpp/samples/jfk.wav";
    let alt = "samples/audio.wav";

    let path = if Path::new(jfk).exists() {
        jfk
    } else if Path::new(alt).exists() {
        alt
    } else {
        return Err("No audio file found (tried jfk.wav and samples/audio.wav)".into());
    };

    println!("Loading audio from {}...", path);
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    if spec.sample_rate != 16000 {
        return Err(format!("Expected 16kHz, got {}Hz", spec.sample_rate).into());
    }

    let samples: Vec<f32> = reader
        .samples::<i16>()
        .step_by(spec.channels as usize)
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();

    Ok(samples)
}
