//! Demonstrates WhisperStream reuse across multiple sessions.

use std::path::Path;
use whisper_cpp_plus::{FullParams, SamplingStrategy, WhisperContext, WhisperStream, WhisperStreamConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "tests/models/ggml-tiny.en.bin";
    if !Path::new(model_path).exists() {
        eprintln!("Model not found at: {}", model_path);
        eprintln!("Showing API pattern instead:\n");
        show_pattern();
        return Ok(());
    }

    println!("Loading model...");
    let ctx = WhisperContext::new(model_path)?;

    let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 })
        .language("en");

    let config = WhisperStreamConfig::default();
    let mut stream = WhisperStream::with_config(&ctx, params, config)?;

    for session in 1..=3 {
        println!("--- Session {} ---", session);

        if session > 1 {
            stream.reset();
            println!("  (reset — state reused)");
        }

        let audio = load_audio(2)?;
        stream.feed_audio(&audio);

        while let Some(segments) = stream.process_step()? {
            for seg in &segments {
                println!("  [{:.2}s-{:.2}s]: {}", seg.start_seconds(), seg.end_seconds(), seg.text);
            }
        }

        let flush_segs = stream.flush()?;
        for seg in &flush_segs {
            println!("  [flush {:.2}s-{:.2}s]: {}", seg.start_seconds(), seg.end_seconds(), seg.text);
        }

        println!("  processed: {} samples\n", stream.processed_samples());
    }

    println!("Done.");
    Ok(())
}

fn show_pattern() {
    println!("let mut stream = WhisperStream::new(&ctx, params)?;");
    println!("stream.feed_audio(&audio);");
    println!("while let Some(segs) = stream.process_step()? {{ ... }}");
    println!("stream.flush()?;");
    println!("stream.reset();  // reuse for next session");
}

fn load_audio(duration_secs: usize) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let jfk = "vendor/whisper.cpp/samples/jfk.wav";
    let alt = "samples/audio.wav";

    let path = if Path::new(jfk).exists() {
        jfk
    } else if Path::new(alt).exists() {
        alt
    } else {
        return Err("No audio file found".into());
    };

    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    let samples: Vec<f32> = reader
        .samples::<i16>()
        .step_by(spec.channels as usize)
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();

    let needed = 16000 * duration_secs;
    Ok(samples.into_iter().take(needed).collect())
}
