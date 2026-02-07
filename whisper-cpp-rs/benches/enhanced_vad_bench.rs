//! Benchmarks comparing standard VAD vs enhanced VAD with aggregation

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use whisper_cpp_rs::bench_helpers::{WhisperVadProcessor, VadParams};
use whisper_cpp_rs::enhanced::vad::{EnhancedWhisperVadProcessor, EnhancedVadParamsBuilder};
use std::time::Duration;

fn load_jfk_audio() -> Vec<f32> {
    // Try multiple locations for the audio file
    let paths = vec![
        "vendor/whisper.cpp/samples/jfk.wav",
        "../vendor/whisper.cpp/samples/jfk.wav",
        "samples/benchmark_audio.wav",
    ];

    for audio_path in &paths {
        if std::path::Path::new(audio_path).exists() {
            eprintln!("Loading JFK audio from: {}", audio_path);
            return load_wav_file(audio_path).unwrap_or_else(|e| {
                eprintln!("Failed to load audio file: {}", e);
                panic!("Cannot run benchmarks without audio files");
            });
        }
    }

    eprintln!("\nError: No audio files found for benchmarks!");
    eprintln!("Please provide audio at one of these locations:");
    for path in &paths {
        eprintln!("  - {}", path);
    }
    eprintln!("\nNote: Benchmarks require real audio for meaningful results.");
    eprintln!("Falling back to synthetic audio for demonstration only.");
    eprintln!("Results will not be representative of real performance!\n");

    // Still generate synthetic as last resort for CI/testing
    generate_synthetic_speech(11) // JFK clip is ~11 seconds

}

fn load_wav_file(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    use hound;

    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    // Convert to mono f32 samples at 16kHz
    let samples: Vec<f32> = if spec.channels == 2 {
        // Convert stereo to mono by averaging channels
        reader
            .samples::<i16>()
            .enumerate()
            .filter_map(|(i, s)| {
                if i % 2 == 0 {
                    Some(s.unwrap() as f32 / 32768.0)
                } else {
                    None
                }
            })
            .collect()
    } else {
        // Already mono
        reader
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect()
    };

    // Note: This assumes the audio is already at 16kHz
    // In production, you'd resample if needed
    Ok(samples)
}

fn generate_synthetic_speech(duration_seconds: usize) -> Vec<f32> {
    // Better synthetic speech simulation with actual patterns
    let sample_rate = 16000;
    let mut audio = Vec::with_capacity(sample_rate * duration_seconds);

    // Add some leading silence
    audio.extend(vec![0.0f32; sample_rate / 2]);

    // Generate speech-like patterns
    for i in 0..duration_seconds {
        if i % 3 == 2 {
            // Add silence between "sentences"
            audio.extend(vec![0.0f32; sample_rate / 2]);
        } else {
            // Generate speech-like audio with varying amplitude
            for j in 0..sample_rate {
                let t = j as f32 / sample_rate as f32;
                // Mix of frequencies to simulate speech formants
                let sample = 0.1 * (2.0 * std::f32::consts::PI * 200.0 * t).sin()
                    + 0.05 * (2.0 * std::f32::consts::PI * 500.0 * t).sin()
                    + 0.03 * (2.0 * std::f32::consts::PI * 1000.0 * t).sin();
                // Add envelope to simulate word boundaries
                let envelope = (t * 10.0).sin().abs() * 0.5 + 0.5;
                audio.push(sample * envelope);
            }
        }
    }

    // Add trailing silence
    audio.extend(vec![0.0f32; sample_rate / 2]);

    audio
}

fn benchmark_vad_processing(c: &mut Criterion) {
    // Skip if model doesn't exist
    let vad_model_path = "tests/models/ggml-silero-vad.bin";
    if !std::path::Path::new(vad_model_path).exists() {
        eprintln!("VAD model not found at {}. Skipping VAD benchmarks.", vad_model_path);
        eprintln!("Looking for model in tests/models/ directory");
        return;
    }

    let mut group = c.benchmark_group("vad_processing");
    group.measurement_time(Duration::from_secs(10));

    // Load real JFK audio (about 11 seconds)
    let jfk_audio = load_jfk_audio();

    // Create different audio samples for testing
    let test_audios = vec![
        ("jfk_original", jfk_audio.clone()),
        ("jfk_with_silence", {
            // Add silence padding to simulate longer audio with gaps
            let mut padded = vec![0.0f32; 16000 * 2]; // 2s silence
            padded.extend(jfk_audio.clone());
            padded.extend(vec![0.0f32; 16000 * 3]); // 3s silence
            padded.extend(jfk_audio.clone());
            padded.extend(vec![0.0f32; 16000 * 2]); // 2s silence
            padded
        }),
    ];

    for (name, audio) in test_audios.iter() {
        // Benchmark standard VAD
        group.bench_with_input(
            BenchmarkId::new("standard", name),
            audio,
            |b, audio| {
                let mut vad = WhisperVadProcessor::new(vad_model_path).unwrap();
                let params = VadParams::default();
                b.iter(|| {
                    let segments = vad.segments_from_samples(black_box(audio), &params).unwrap();
                    segments.get_all_segments().len()
                })
            }
        );

        // Benchmark enhanced VAD with aggregation
        group.bench_with_input(
            BenchmarkId::new("enhanced_aggregated", name),
            audio,
            |b, audio| {
                let mut vad = EnhancedWhisperVadProcessor::new(vad_model_path).unwrap();
                let params = EnhancedVadParamsBuilder::new()
                    .max_segment_duration(30.0)
                    .merge_segments(true)
                    .min_gap_ms(100)
                    .build();
                b.iter(|| {
                    let chunks = vad.process_with_aggregation(black_box(audio), &params).unwrap();
                    chunks.len()
                })
            }
        );
    }

    group.finish();
}

fn benchmark_segment_aggregation(c: &mut Criterion) {
    // Skip if model doesn't exist (needed for real processor)
    let vad_model_path = "tests/models/ggml-silero-vad.bin";
    if !std::path::Path::new(vad_model_path).exists() {
        eprintln!("VAD model not found, skipping segment aggregation benchmarks");
        return;
    }

    let processor = EnhancedWhisperVadProcessor::new(vad_model_path).unwrap();
    let mut group = c.benchmark_group("segment_aggregation");

    // Create different segment patterns
    let many_small_segments: Vec<(f32, f32)> = (0..100)
        .map(|i| {
            let start = i as f32 * 0.5;
            let end = start + 0.4;
            (start, end)
        })
        .collect();

    let few_large_segments: Vec<(f32, f32)> = vec![
        (0.0, 10.0),
        (11.0, 21.0),
        (22.0, 32.0),
        (33.0, 43.0),
    ];

    let mixed_segments: Vec<(f32, f32)> = vec![
        (0.0, 2.0),
        (2.1, 4.0),
        (4.5, 6.0),
        (10.0, 20.0),
        (20.5, 22.0),
        (22.1, 23.0),
        (30.0, 40.0),
    ];

    group.bench_function("many_small_segments", |b| {
        b.iter(|| {
            let aggregated = processor.aggregate_segments(
                black_box(many_small_segments.clone()),
                30.0,
                100,
                true
            );
            aggregated.len()
        })
    });

    group.bench_function("few_large_segments", |b| {
        b.iter(|| {
            let aggregated = processor.aggregate_segments(
                black_box(few_large_segments.clone()),
                30.0,
                100,
                true
            );
            aggregated.len()
        })
    });

    group.bench_function("mixed_segments", |b| {
        b.iter(|| {
            let aggregated = processor.aggregate_segments(
                black_box(mixed_segments.clone()),
                30.0,
                100,
                true
            );
            aggregated.len()
        })
    });

    group.finish();
}

fn benchmark_vad_efficiency_metrics(c: &mut Criterion) {
    let vad_model_path = "tests/models/ggml-silero-vad.bin";
    if !std::path::Path::new(vad_model_path).exists() {
        return;
    }

    let mut group = c.benchmark_group("vad_efficiency");
    group.measurement_time(Duration::from_secs(5));

    // Load real JFK audio and create version with silence
    let jfk_audio = load_jfk_audio();

    // Create audio with significant silence sections
    let mut audio = vec![0.0f32; 16000 * 2]; // 2s silence
    audio.extend(jfk_audio.clone());
    audio.extend(vec![0.0f32; 16000 * 3]); // 3s silence
    audio.extend(jfk_audio);
    audio.extend(vec![0.0f32; 16000 * 2]); // 2s silence
    // Total: ~29 seconds (7s silence + 22s speech)

    // Measure VAD processing without sleep simulation
    group.bench_function("standard_vad_processing", |b| {
        let mut vad = WhisperVadProcessor::new(vad_model_path).unwrap();
        let params = VadParams::default();

        b.iter(|| {
            let segments = vad.segments_from_samples(&audio, &params).unwrap();
            let segments = segments.get_all_segments();

            // Calculate total audio duration that would be transcribed
            let total_duration: f32 = segments.iter()
                .map(|(start, end)| end - start)
                .sum();

            (segments.len(), total_duration)
        })
    });

    // Measure enhanced VAD with aggregation
    group.bench_function("enhanced_vad_processing", |b| {
        let mut vad = EnhancedWhisperVadProcessor::new(vad_model_path).unwrap();
        let params = EnhancedVadParamsBuilder::new()
            .max_segment_duration(30.0)
            .merge_segments(true)
            .build();

        b.iter(|| {
            let chunks = vad.process_with_aggregation(&audio, &params).unwrap();

            // Calculate total audio duration that would be transcribed
            let total_duration: f32 = chunks.iter()
                .map(|c| c.duration_seconds)
                .sum();

            (chunks.len(), total_duration)
        })
    });

    group.finish();
}


criterion_group!(
    benches,
    benchmark_vad_processing,
    benchmark_segment_aggregation,
    benchmark_vad_efficiency_metrics
);
criterion_main!(benches);