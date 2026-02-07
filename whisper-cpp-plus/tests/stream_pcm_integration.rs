//! Integration tests for WhisperStreamPcm — pipe real WAV audio through PcmReader → WhisperStreamPcm

mod common;

use common::TestModels;
use std::path::Path;
use whisper_cpp_plus::{
    FullParams, PcmFormat, PcmReader, PcmReaderConfig, SamplingStrategy, WhisperStreamPcm,
    WhisperStreamPcmConfig, WhisperContext,
};

/// Load a WAV file, return raw PCM bytes in the given format + the f32 sample count.
fn wav_to_raw_pcm(path: &Path, format: PcmFormat) -> (Vec<u8>, usize) {
    let mut reader = hound::WavReader::open(path).expect("Failed to open WAV");
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 16000, "WAV must be 16kHz");

    let samples: Vec<f32> = match spec.bits_per_sample {
        16 => reader
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / i16::MAX as f32)
            .collect(),
        32 => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        b => panic!("Unsupported bit depth: {}", b),
    };

    let mono = if spec.channels == 2 {
        samples
            .chunks(2)
            .map(|c| (c[0] + c[1]) / 2.0)
            .collect()
    } else {
        samples
    };

    let n_samples = mono.len();

    let bytes = match format {
        PcmFormat::F32 => mono.iter().flat_map(|s| s.to_le_bytes()).collect(),
        PcmFormat::S16 => mono
            .iter()
            .flat_map(|s| {
                let v = (*s * 32767.0).clamp(-32768.0, 32767.0) as i16;
                v.to_le_bytes().to_vec()
            })
            .collect(),
    };

    (bytes, n_samples)
}

fn check_jfk_keywords(text: &str) {
    let lower = text.to_lowercase();
    let keywords = ["ask", "not", "what", "country", "you"];
    let found: Vec<&&str> = keywords.iter().filter(|k| lower.contains(**k)).collect();
    println!("Transcript: {}", text);
    println!(
        "Found {}/{} keywords: {:?}",
        found.len(),
        keywords.len(),
        found
    );
    assert!(
        found.len() >= 3,
        "Expected >= 3 JFK keywords, found {}: {:?}",
        found.len(),
        found
    );
}

// ---- Fixed-step mode (F32) ----

#[test]
fn test_stream_pcm_fixed_step_f32() {
    let Some(model_path) = TestModels::tiny_en() else {
        eprintln!("Skipping: whisper model not found. Run `cargo xtask test-setup`");
        return;
    };
    let Some(jfk_path) = TestModels::jfk_audio() else {
        eprintln!("Skipping: jfk.wav not found. Run `cargo xtask test-setup`");
        return;
    };

    let (raw_bytes, _) = wav_to_raw_pcm(&jfk_path, PcmFormat::F32);

    let ctx = WhisperContext::new(&model_path).unwrap();
    let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 })
        .language("en")
        .no_timestamps(true)
        .single_segment(true);

    let reader = PcmReader::new(
        Box::new(std::io::Cursor::new(raw_bytes)),
        PcmReaderConfig {
            buffer_len_ms: 10000,
            sample_rate: 16000,
            format: PcmFormat::F32,
        },
    );

    let config = WhisperStreamPcmConfig {
        step_ms: 3000,
        length_ms: 10000,
        keep_ms: 200,
        use_vad: false,
        ..Default::default()
    };

    let mut stream = WhisperStreamPcm::new(&ctx, params, config, reader).unwrap();

    let mut all_text = String::new();
    stream
        .run(|segments, _start, _end| {
            for seg in segments {
                all_text.push_str(&seg.text);
            }
        })
        .expect("WhisperStreamPcm::run failed");

    assert!(!all_text.is_empty(), "Should produce non-empty transcription");
    check_jfk_keywords(&all_text);
}

// ---- Fixed-step mode (S16) ----

#[test]
fn test_stream_pcm_fixed_step_s16() {
    let Some(model_path) = TestModels::tiny_en() else {
        eprintln!("Skipping: whisper model not found. Run `cargo xtask test-setup`");
        return;
    };
    let Some(jfk_path) = TestModels::jfk_audio() else {
        eprintln!("Skipping: jfk.wav not found. Run `cargo xtask test-setup`");
        return;
    };

    let (raw_bytes, _) = wav_to_raw_pcm(&jfk_path, PcmFormat::S16);

    let ctx = WhisperContext::new(&model_path).unwrap();
    let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 })
        .language("en")
        .no_timestamps(true)
        .single_segment(true);

    let reader = PcmReader::new(
        Box::new(std::io::Cursor::new(raw_bytes)),
        PcmReaderConfig {
            buffer_len_ms: 10000,
            sample_rate: 16000,
            format: PcmFormat::S16,
        },
    );

    let config = WhisperStreamPcmConfig {
        step_ms: 3000,
        length_ms: 10000,
        keep_ms: 200,
        use_vad: false,
        ..Default::default()
    };

    let mut stream = WhisperStreamPcm::new(&ctx, params, config, reader).unwrap();

    let mut all_text = String::new();
    stream
        .run(|segments, _start, _end| {
            for seg in segments {
                all_text.push_str(&seg.text);
            }
        })
        .expect("WhisperStreamPcm::run failed");

    assert!(!all_text.is_empty(), "Should produce non-empty transcription");
    check_jfk_keywords(&all_text);
}

// ---- VAD mode (simple energy VAD) ----

#[test]
fn test_stream_pcm_vad_simple() {
    let Some(model_path) = TestModels::tiny_en() else {
        eprintln!("Skipping: whisper model not found. Run `cargo xtask test-setup`");
        return;
    };
    let Some(jfk_path) = TestModels::jfk_audio() else {
        eprintln!("Skipping: jfk.wav not found. Run `cargo xtask test-setup`");
        return;
    };

    let (raw_bytes, _) = wav_to_raw_pcm(&jfk_path, PcmFormat::F32);

    let ctx = WhisperContext::new(&model_path).unwrap();
    let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 })
        .language("en");

    let reader = PcmReader::new(
        Box::new(std::io::Cursor::new(raw_bytes)),
        PcmReaderConfig {
            buffer_len_ms: 10000,
            sample_rate: 16000,
            format: PcmFormat::F32,
        },
    );

    let config = WhisperStreamPcmConfig {
        use_vad: true,
        length_ms: 10000,
        vad_thold: 0.6,
        freq_thold: 100.0,
        vad_probe_ms: 200,
        vad_silence_ms: 800,
        vad_pre_roll_ms: 300,
        ..Default::default()
    };

    let mut stream = WhisperStreamPcm::new(&ctx, params, config, reader).unwrap();

    let mut all_text = String::new();
    let mut segment_count = 0;
    stream
        .run(|segments, start_ms, end_ms| {
            segment_count += 1;
            println!(
                "VAD segment {}: {}ms-{}ms",
                segment_count, start_ms, end_ms
            );
            for seg in segments {
                all_text.push_str(&seg.text);
                all_text.push(' ');
            }
        })
        .expect("WhisperStreamPcm::run with VAD failed");

    println!("VAD produced {} transcription segments", segment_count);
    assert!(segment_count > 0, "VAD should produce at least 1 segment");
    assert!(!all_text.is_empty(), "Should produce non-empty transcription");
    check_jfk_keywords(&all_text);
}

// ---- VAD mode with Silero ----

#[test]
fn test_stream_pcm_vad_silero() {
    let Some(model_path) = TestModels::tiny_en() else {
        eprintln!("Skipping: whisper model not found. Run `cargo xtask test-setup`");
        return;
    };
    let Some(vad_model_path) = TestModels::vad() else {
        eprintln!("Skipping: VAD model not found. Run `cargo xtask test-setup`");
        return;
    };
    let Some(jfk_path) = TestModels::jfk_audio() else {
        eprintln!("Skipping: jfk.wav not found. Run `cargo xtask test-setup`");
        return;
    };

    let (raw_bytes, _) = wav_to_raw_pcm(&jfk_path, PcmFormat::F32);

    let ctx = WhisperContext::new(&model_path).unwrap();
    let vad = whisper_cpp_plus::WhisperVadProcessor::new(&vad_model_path).unwrap();
    let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 })
        .language("en");

    let reader = PcmReader::new(
        Box::new(std::io::Cursor::new(raw_bytes)),
        PcmReaderConfig {
            buffer_len_ms: 10000,
            sample_rate: 16000,
            format: PcmFormat::F32,
        },
    );

    let config = WhisperStreamPcmConfig {
        use_vad: true,
        length_ms: 10000,
        vad_thold: 0.5,
        vad_probe_ms: 200,
        vad_silence_ms: 800,
        vad_pre_roll_ms: 300,
        ..Default::default()
    };

    let mut stream = WhisperStreamPcm::with_vad(&ctx, params, config, reader, vad).unwrap();

    let mut all_text = String::new();
    let mut segment_count = 0;
    stream
        .run(|segments, start_ms, end_ms| {
            segment_count += 1;
            println!(
                "Silero VAD segment {}: {}ms-{}ms",
                segment_count, start_ms, end_ms
            );
            for seg in segments {
                all_text.push_str(&seg.text);
                all_text.push(' ');
            }
        })
        .expect("WhisperStreamPcm::run with Silero VAD failed");

    println!("Silero VAD produced {} segments", segment_count);
    assert!(segment_count > 0, "Silero VAD should produce >= 1 segment");
    assert!(!all_text.is_empty(), "Should produce non-empty transcription");
    check_jfk_keywords(&all_text);
}
