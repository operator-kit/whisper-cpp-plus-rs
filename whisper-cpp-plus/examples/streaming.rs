//! Streaming transcription example — WhisperStream API (port of stream.cpp)
//!
//! Demonstrates:
//! 1. Basic streaming with sliding window
//! 2. Stream reuse across multiple sessions via .reset()

use std::path::{Path, PathBuf};
use whisper_cpp_plus::{
    FullParams, SamplingStrategy, WhisperContext, WhisperStream, WhisperStreamConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path =
        find_model("ggml-tiny.en.bin").ok_or("Model not found. Run: cargo xtask test-setup")?;

    println!("Loading model from {:?}...", model_path);
    let ctx = WhisperContext::new(&model_path)?;

    let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 }).language("en");

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
    println!(
        "Loaded {} samples ({:.1}s)",
        audio.len(),
        audio.len() as f64 / 16000.0
    );

    // Feed in 1-second chunks to simulate real-time
    let chunk_size = 16000;
    for (i, chunk) in audio.chunks(chunk_size).enumerate() {
        stream.feed_audio(chunk);
        println!(
            "Fed chunk {} ({} samples, buf={})",
            i + 1,
            chunk.len(),
            stream.buffer_size()
        );

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

    println!(
        "Done. Processed {} samples total.",
        stream.processed_samples()
    );

    // === Part 2: Stream reuse across sessions ===
    println!("\n=== Stream Reuse Demo ===\n");
    demo_stream_reuse(&ctx)?;

    Ok(())
}

/// Demonstrates reusing a WhisperStream across multiple sessions
fn demo_stream_reuse(ctx: &WhisperContext) -> Result<(), Box<dyn std::error::Error>> {
    let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 }).language("en");
    let mut stream = WhisperStream::with_config(ctx, params, WhisperStreamConfig::default())?;

    for session in 1..=3 {
        println!("--- Session {} ---", session);

        if session > 1 {
            stream.reset(); // Reuse stream without recreating context
            println!("  (reset — state reused)");
        }

        // Feed 2 seconds of audio
        let audio = load_audio_chunk(2)?;
        stream.feed_audio(&audio);

        while let Some(segments) = stream.process_step()? {
            for seg in &segments {
                println!(
                    "  [{:.2}s-{:.2}s]: {}",
                    seg.start_seconds(),
                    seg.end_seconds(),
                    seg.text
                );
            }
        }

        let flush_segs = stream.flush()?;
        for seg in &flush_segs {
            println!(
                "  [flush {:.2}s-{:.2}s]: {}",
                seg.start_seconds(),
                seg.end_seconds(),
                seg.text
            );
        }

        println!("  processed: {} samples\n", stream.processed_samples());
    }

    Ok(())
}

fn load_audio_chunk(duration_secs: usize) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let audio = load_audio()?;
    let needed = 16000 * duration_secs;
    Ok(audio.into_iter().take(needed).collect())
}

fn load_audio() -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // Check env var first
    let path = if let Ok(dir) = std::env::var("WHISPER_TEST_AUDIO_DIR") {
        let p = format!("{}/jfk.wav", dir);
        if Path::new(&p).exists() {
            Some(p)
        } else {
            None
        }
    } else {
        None
    };

    let path = path.or_else(|| {
        let paths = [
            "../whisper-cpp-plus-sys/whisper.cpp/samples/jfk.wav",
            "whisper-cpp-plus-sys/whisper.cpp/samples/jfk.wav",
            "samples/audio.wav",
        ];
        paths
            .iter()
            .find(|p| Path::new(p).exists())
            .map(|s| s.to_string())
    });

    let path = path.ok_or("No audio file found. Set WHISPER_TEST_AUDIO_DIR.")?;

    println!("Loading audio from {}...", path);
    let mut reader = hound::WavReader::open(&path)?;
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

fn find_model(name: &str) -> Option<PathBuf> {
    for env_var in ["WHISPER_TEST_MODEL_DIR", "WHISPER_MODEL_PATH"] {
        if let Ok(dir) = std::env::var(env_var) {
            let path = Path::new(&dir).join(name);
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
        .map(PathBuf::from)
}
