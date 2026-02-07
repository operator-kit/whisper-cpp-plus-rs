//! Benchmarks for temperature fallback mechanism

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use whisper_cpp_plus::enhanced::fallback::calculate_compression_ratio;
use std::time::Duration;

fn benchmark_compression_ratio(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression_ratio");

    // Different text sizes and patterns
    let short_text = "Hello world";
    let medium_text = "The quick brown fox jumps over the lazy dog. ".repeat(10);
    let long_text = include_str!("../src/lib.rs"); // Use a real source file
    let repetitive_text = "a".repeat(1000);
    let mixed_text = "aaaaa bbbbb ccccc ddddd eeeee fffff ".repeat(50);

    group.bench_function("short_text", |b| {
        b.iter(|| calculate_compression_ratio(black_box(short_text)))
    });

    group.bench_function("medium_text", |b| {
        b.iter(|| calculate_compression_ratio(black_box(&medium_text)))
    });

    group.bench_function("long_text", |b| {
        b.iter(|| calculate_compression_ratio(black_box(long_text)))
    });

    group.bench_function("repetitive_text", |b| {
        b.iter(|| calculate_compression_ratio(black_box(&repetitive_text)))
    });

    group.bench_function("mixed_text", |b| {
        b.iter(|| calculate_compression_ratio(black_box(&mixed_text)))
    });

    group.finish();
}

fn benchmark_quality_checks(c: &mut Criterion) {
    use whisper_cpp_plus::enhanced::fallback::{TranscriptionAttempt, QualityThresholds};

    let mut group = c.benchmark_group("quality_checks");

    let thresholds = QualityThresholds {
        compression_ratio_threshold: Some(2.4),
        log_prob_threshold: Some(-1.0),
        no_speech_threshold: Some(0.6),
    };

    // Good quality transcription
    let good_attempt = TranscriptionAttempt {
        text: "This is a clear transcription with good quality metrics.".to_string(),
        segments: vec![],
        temperature: 0.0,
        compression_ratio: 1.5,
        avg_logprob: -0.5,
        no_speech_prob: 0.1,
    };

    // Poor quality transcription (repetitive)
    let poor_attempt = TranscriptionAttempt {
        text: "a".repeat(500),
        segments: vec![],
        temperature: 0.0,
        compression_ratio: 15.0,
        avg_logprob: -2.0,
        no_speech_prob: 0.1,
    };

    // Borderline quality
    let borderline_attempt = TranscriptionAttempt {
        text: "This text has borderline quality metrics that might trigger fallback.".to_string(),
        segments: vec![],
        temperature: 0.0,
        compression_ratio: 2.3,
        avg_logprob: -0.95,
        no_speech_prob: 0.55,
    };

    group.bench_function("good_quality_check", |b| {
        b.iter(|| good_attempt.meets_thresholds(black_box(&thresholds)))
    });

    group.bench_function("poor_quality_check", |b| {
        b.iter(|| poor_attempt.meets_thresholds(black_box(&thresholds)))
    });

    group.bench_function("borderline_quality_check", |b| {
        b.iter(|| borderline_attempt.meets_thresholds(black_box(&thresholds)))
    });

    group.finish();
}

fn benchmark_transcription_with_fallback_simulation(c: &mut Criterion) {
    // Skip if model doesn't exist
    let model_path = "tests/models/ggml-base.en.bin";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Model not found at {}. Skipping transcription benchmarks.", model_path);
        return;
    }

    use whisper_cpp_plus::{WhisperContext, TranscriptionParams};
    use std::sync::Arc;

    let ctx = Arc::new(WhisperContext::new(model_path).unwrap());

    let mut group = c.benchmark_group("transcription_fallback");
    group.measurement_time(Duration::from_secs(15));

    // Load real audio for benchmarks
    let audio = load_benchmark_audio().unwrap_or_else(|e| {
        eprintln!("Failed to load audio: {}. Skipping transcription benchmarks.", e);
        vec![0.0f32; 16000] // Fallback to minimal audio
    });

    // Create noisy version for comparison
    let noisy_audio = add_noise_to_audio(&audio);

    let params = TranscriptionParams::builder()
        .language("en")
        .build();

    // Benchmark standard transcription (no fallback)
    group.bench_function("standard_clear", |b| {
        let ctx = Arc::clone(&ctx);
        let audio = audio.clone();
        b.iter(|| {
            ctx.transcribe_with_params(black_box(&audio), params.clone()).unwrap()
        })
    });

    group.bench_function("standard_noisy", |b| {
        let ctx = Arc::clone(&ctx);
        let audio = noisy_audio.clone();
        b.iter(|| {
            ctx.transcribe_with_params(black_box(&audio), params.clone()).unwrap()
        })
    });

    // Benchmark enhanced transcription with fallback
    group.bench_function("enhanced_clear", |b| {
        let ctx = Arc::clone(&ctx);
        let audio = audio.clone();
        b.iter(|| {
            ctx.transcribe_with_params_enhanced(black_box(&audio), params.clone()).unwrap()
        })
    });

    group.bench_function("enhanced_noisy", |b| {
        let ctx = Arc::clone(&ctx);
        let audio = noisy_audio.clone();
        b.iter(|| {
            ctx.transcribe_with_params_enhanced(black_box(&audio), params.clone()).unwrap()
        })
    });

    group.finish();
}

fn load_benchmark_audio() -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let jfk_path = "vendor/whisper.cpp/samples/jfk.wav";
    let alt_path = "samples/benchmark_audio.wav";

    if std::path::Path::new(jfk_path).exists() {
        load_wav_file(jfk_path)
    } else if std::path::Path::new(alt_path).exists() {
        load_wav_file(alt_path)
    } else {
        Err(format!("No audio files found at {} or {}", jfk_path, alt_path).into())
    }
}

fn load_wav_file(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    use hound;

    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    if spec.sample_rate != 16000 {
        eprintln!("Warning: Audio sample rate is {}Hz, expected 16000Hz", spec.sample_rate);
    }

    let samples: Vec<f32> = reader
        .samples::<i16>()
        .step_by(spec.channels as usize)
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();

    // Truncate to 3 seconds for benchmark
    let max_samples = 16000 * 3;
    Ok(samples.into_iter().take(max_samples).collect())
}

fn add_noise_to_audio(audio: &[f32]) -> Vec<f32> {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hash, Hasher};

    let mut rng = RandomState::new().build_hasher();

    audio.iter().enumerate().map(|(i, &sample)| {
        i.hash(&mut rng);
        let noise_val = (rng.finish() as f32 / u64::MAX as f32 - 0.5) * 0.1;
        let noisy = sample + noise_val;
        noisy.max(-1.0).min(1.0)
    }).collect()
}

criterion_group!(
    benches,
    benchmark_compression_ratio,
    benchmark_quality_checks,
    benchmark_transcription_with_fallback_simulation
);
criterion_main!(benches);