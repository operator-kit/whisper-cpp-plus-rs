//! Compare standard VAD vs enhanced VAD with aggregation

use std::path::{Path, PathBuf};
use whisper_cpp_plus::enhanced::vad::{EnhancedVadParamsBuilder, EnhancedWhisperVadProcessor};
use whisper_cpp_plus::{VadParams, WhisperVadProcessor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vad_model_path = find_model("ggml-silero-v6.2.0.bin")
        .ok_or("VAD model not found. Run: cargo xtask test-setup")?;

    println!("VAD model: {:?}", vad_model_path);

    println!("=== VAD Comparison: Standard vs Enhanced ===\n");

    // Load audio (you would load real audio here)
    let audio = load_audio_example()?;
    let duration_seconds = audio.len() as f32 / 16000.0;
    println!("Audio duration: {:.2} seconds\n", duration_seconds);

    // Run standard VAD
    println!("1. STANDARD VAD:");
    println!("{}", "-".repeat(50));
    let mut standard_vad = WhisperVadProcessor::new(&vad_model_path)?;
    let standard_params = VadParams::default();

    let start = std::time::Instant::now();
    let segments = standard_vad.segments_from_samples(&audio, &standard_params)?;
    let standard_time = start.elapsed();

    let standard_segments = segments.get_all_segments();
    println!("  Segments found: {}", standard_segments.len());
    println!("  Processing time: {:?}", standard_time);

    let total_speech_duration: f32 = standard_segments
        .iter()
        .map(|(start, end)| end - start)
        .sum();
    println!("  Total speech duration: {:.2}s", total_speech_duration);
    println!(
        "  Silence removed: {:.2}s ({:.1}%)",
        duration_seconds - total_speech_duration,
        ((duration_seconds - total_speech_duration) / duration_seconds) * 100.0
    );

    println!("\n  Segment details:");
    for (i, (start, end)) in standard_segments.iter().enumerate() {
        let duration = end - start;
        println!(
            "    Segment {}: {:.2}s - {:.2}s (duration: {:.2}s)",
            i + 1,
            start,
            end,
            duration
        );
    }

    // Run enhanced VAD with aggregation
    println!("\n2. ENHANCED VAD WITH AGGREGATION:");
    println!("{}", "-".repeat(50));
    let mut enhanced_vad = EnhancedWhisperVadProcessor::new(&vad_model_path)?;
    let enhanced_params = EnhancedVadParamsBuilder::new()
        .max_segment_duration(30.0)
        .merge_segments(true)
        .min_gap_ms(250) // Merge segments with < 250ms gap (will catch the 230ms gap)
        .build();

    let start = std::time::Instant::now();
    let chunks = enhanced_vad.process_with_aggregation(&audio, &enhanced_params)?;
    let enhanced_time = start.elapsed();

    println!("  Aggregated chunks: {}", chunks.len());
    println!("  Processing time: {:?}", enhanced_time);

    let total_chunk_duration: f32 = chunks.iter().map(|c| c.duration_seconds).sum();
    println!("  Total speech duration: {:.2}s", total_chunk_duration);
    println!(
        "  Silence removed: {:.2}s ({:.1}%)",
        duration_seconds - total_chunk_duration,
        ((duration_seconds - total_chunk_duration) / duration_seconds) * 100.0
    );

    println!("\n  Aggregated chunk details:");
    for (i, chunk) in chunks.iter().enumerate() {
        println!(
            "    Chunk {}: {:.2}s - {:.2}s (duration: {:.2}s)",
            i + 1,
            chunk.offset_seconds,
            chunk.offset_seconds + chunk.duration_seconds,
            chunk.duration_seconds
        );
    }

    // Show comparison
    println!("\n3. COMPARISON SUMMARY:");
    println!("{}", "-".repeat(50));
    let reduction =
        ((standard_segments.len() - chunks.len()) as f32 / standard_segments.len() as f32) * 100.0;
    println!(
        "  Segment reduction: {} → {} ({:.1}% fewer chunks)",
        standard_segments.len(),
        chunks.len(),
        reduction
    );

    let speedup = standard_time.as_secs_f32() / enhanced_time.as_secs_f32();
    println!("  Processing speedup: {:.2}x", speedup);

    println!("\n  Benefits:");
    println!(
        "  - {} fewer transcription API calls needed",
        standard_segments.len() - chunks.len()
    );
    println!("  - Larger chunks are more efficient for transcription");
    println!("  - Better context for speech recognition");

    Ok(())
}

fn load_wav_file(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    use hound;

    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    if spec.sample_rate != 16000 {
        eprintln!(
            "Warning: Sample rate is {}Hz, expected 16000Hz",
            spec.sample_rate
        );
    }

    let samples: Vec<f32> = reader
        .samples::<i16>()
        .step_by(spec.channels as usize)
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();

    Ok(samples)
}

fn load_audio_example() -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // Check env var first
    if let Ok(dir) = std::env::var("WHISPER_TEST_AUDIO_DIR") {
        let path = Path::new(&dir).join("jfk.wav");
        if path.exists() {
            println!("Loading audio from: {}", path.display());
            return load_wav_file(path.to_str().unwrap());
        }
    }

    let paths = vec![
        // whisper.cpp submodule samples
        "../whisper-cpp-plus-sys/whisper.cpp/samples/jfk.wav",
        "whisper-cpp-plus-sys/whisper.cpp/samples/jfk.wav",
        "samples/audio.wav",
    ];

    for audio_path in &paths {
        if Path::new(audio_path).exists() {
            println!("Loading audio from: {}", audio_path);
            return load_wav_file(audio_path);
        }
    }

    eprintln!("\nError: No audio files found!");
    eprintln!("Set WHISPER_TEST_AUDIO_DIR env var or provide audio at one of these locations:");
    for path in &paths {
        eprintln!("  - {}", path);
    }
    Err("No audio files found for VAD comparison".into())
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
