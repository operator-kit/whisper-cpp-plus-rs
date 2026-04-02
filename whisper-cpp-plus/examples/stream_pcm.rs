//! PCM streaming example — WhisperStreamPcm API (port of stream-pcm.cpp)
//!
//! Reads raw PCM audio from a file and transcribes with VAD-driven segmentation.
//! Demonstrates the threaded reader + ring buffer architecture.

use std::path::{Path, PathBuf};
use whisper_cpp_plus::{
    FullParams, PcmFormat, PcmReader, PcmReaderConfig, SamplingStrategy, WhisperContext,
    WhisperStreamPcm, WhisperStreamPcmConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path =
        find_model("ggml-tiny.en.bin").ok_or("Model not found. Run: cargo xtask test-setup")?;

    println!("Loading model from {:?}...", model_path);
    let ctx = WhisperContext::new(&model_path)?;

    let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 }).language("en");

    // VAD-driven config for automatic speech segmentation
    let config = WhisperStreamPcmConfig {
        use_vad: true,
        vad_thold: 0.6,
        vad_silence_ms: 800,
        vad_pre_roll_ms: 300,
        length_ms: 10000,
        ..Default::default()
    };

    // Create PCM source — here we convert wav to raw PCM
    let audio = load_audio_as_pcm()?;
    println!(
        "Loaded {} samples ({:.1}s)",
        audio.len(),
        audio.len() as f64 / 16000.0
    );

    // Wrap in a cursor to simulate a Read source (could be stdin, socket, etc.)
    let cursor = std::io::Cursor::new(audio);

    let reader_config = PcmReaderConfig {
        buffer_len_ms: 10000,
        sample_rate: 16000,
        format: PcmFormat::F32,
    };
    let reader = PcmReader::new(Box::new(cursor), reader_config);

    println!("Creating stream with VAD...");
    let mut stream = WhisperStreamPcm::new(&ctx, params, config, reader)?;

    println!("Processing...\n");

    // Run until EOF, callback for each transcribed segment
    stream.run(|segments, _start_ms, _end_ms| {
        for seg in segments {
            println!(
                "[{:.2}s - {:.2}s]: {}",
                seg.start_seconds(),
                seg.end_seconds(),
                seg.text.trim()
            );
        }
    })?;

    println!("\nDone. Total iterations: {}", stream.n_iter());
    Ok(())
}

/// Load wav file and convert to raw f32 PCM bytes
fn load_audio_as_pcm() -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let path = find_audio_file()?;
    println!("Loading audio from {}...", path);

    let mut reader = hound::WavReader::open(&path)?;
    let spec = reader.spec();

    if spec.sample_rate != 16000 {
        return Err(format!("Expected 16kHz, got {}Hz", spec.sample_rate).into());
    }

    // Convert to f32 samples
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .step_by(spec.channels as usize)
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();

    // Convert to raw bytes (f32 little-endian)
    let mut bytes = Vec::with_capacity(samples.len() * 4);
    for s in &samples {
        bytes.extend_from_slice(&s.to_le_bytes());
    }

    Ok(bytes)
}

fn find_audio_file() -> Result<String, Box<dyn std::error::Error>> {
    // Check env var first
    if let Ok(dir) = std::env::var("WHISPER_TEST_AUDIO_DIR") {
        let p = format!("{}/jfk.wav", dir);
        if Path::new(&p).exists() {
            return Ok(p);
        }
    }

    let paths = [
        "../whisper-cpp-plus-sys/whisper.cpp/samples/jfk.wav",
        "whisper-cpp-plus-sys/whisper.cpp/samples/jfk.wav",
        "samples/audio.wav",
    ];

    for p in &paths {
        if Path::new(p).exists() {
            return Ok(p.to_string());
        }
    }

    Err("No audio file found. Set WHISPER_TEST_AUDIO_DIR.".into())
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
