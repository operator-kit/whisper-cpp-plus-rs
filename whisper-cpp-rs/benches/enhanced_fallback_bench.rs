//! Benchmarks for temperature fallback mechanism

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use whisper_cpp_rs::enhanced::fallback::calculate_compression_ratio;
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
    use whisper_cpp_rs::enhanced::fallback::{TranscriptionAttempt, QualityThresholds};

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

    use whisper_cpp_rs::{WhisperContext, TranscriptionParams};
    use std::sync::Arc;

    let ctx = Arc::new(WhisperContext::new(model_path).unwrap());

    let mut group = c.benchmark_group("transcription_fallback");
    group.measurement_time(Duration::from_secs(15));

    // Generate test audio
    let clear_audio = vec![0.0f32; 16000 * 3]; // 3 seconds of silence (clear)
    let noisy_audio: Vec<f32> = (0..16000 * 3)
        .map(|_| (rand::random::<f32>() - 0.5) * 0.1)
        .collect(); // 3 seconds of noise

    let params = TranscriptionParams::builder()
        .language("en")
        .build();

    // Benchmark standard transcription (no fallback)
    group.bench_function("standard_clear", |b| {
        let ctx = Arc::clone(&ctx);
        let audio = clear_audio.clone();
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
        let audio = clear_audio.clone();
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

criterion_group!(
    benches,
    benchmark_compression_ratio,
    benchmark_quality_checks,
    benchmark_transcription_with_fallback_simulation
);
criterion_main!(benches);