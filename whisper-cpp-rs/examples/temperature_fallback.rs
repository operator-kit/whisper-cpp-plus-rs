//! Example demonstrating temperature fallback for improved transcription quality

use std::path::Path;
use whisper_cpp_rs::{WhisperContext, TranscriptionParams, FullParams, SamplingStrategy};
use whisper_cpp_rs::enhanced::fallback::{
    EnhancedTranscriptionParams, EnhancedTranscriptionParamsBuilder,
    QualityThresholds, EnhancedWhisperState
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Check for model file
    let model_path = "tests/models/ggml-base.en.bin";

    if !Path::new(model_path).exists() {
        eprintln!("Model file not found at: {}", model_path);
        eprintln!("Please download a Whisper model from https://huggingface.co/ggerganov/whisper.cpp");
        return Ok(());
    }

    println!("Loading model...");
    let ctx = WhisperContext::new(model_path)?;

    // Load audio (you would load real audio here)
    let (clear_audio, noisy_audio) = load_audio_examples()?;

    // Example 1: Standard transcription vs Enhanced with fallback
    println!("\n=== Example 1: Clear Audio ===");
    compare_transcription_methods(&ctx, &clear_audio)?;

    // Example 2: Noisy/difficult audio
    println!("\n=== Example 2: Noisy/Difficult Audio ===");
    compare_transcription_methods(&ctx, &noisy_audio)?;

    // Example 3: Custom quality thresholds
    println!("\n=== Example 3: Custom Quality Thresholds ===");
    demonstrate_custom_thresholds(&ctx, &noisy_audio)?;

    // Example 4: Direct enhanced state usage
    println!("\n=== Example 4: Direct Enhanced State Control ===");
    demonstrate_direct_enhanced_state(&ctx, &noisy_audio)?;

    Ok(())
}

fn compare_transcription_methods(
    ctx: &WhisperContext,
    audio: &[f32]
) -> Result<(), Box<dyn std::error::Error>> {
    // Standard transcription
    println!("1. Standard transcription:");
    let start = std::time::Instant::now();
    let standard_text = ctx.transcribe(audio)?;
    let standard_time = start.elapsed();
    println!("   Text: {}", standard_text);
    println!("   Time: {:?}", standard_time);

    // Enhanced transcription with automatic fallback
    println!("\n2. Enhanced transcription with temperature fallback:");
    let params = TranscriptionParams::builder()
        .language("en")
        .build();

    let start = std::time::Instant::now();
    let enhanced_result = ctx.transcribe_with_params_enhanced(audio, params)?;
    let enhanced_time = start.elapsed();
    println!("   Text: {}", enhanced_result.text);
    println!("   Time: {:?}", enhanced_time);

    if enhanced_result.text != standard_text {
        println!("   Note: Enhanced version produced different (likely better) result!");
    }

    Ok(())
}

fn demonstrate_custom_thresholds(
    ctx: &WhisperContext,
    audio: &[f32]
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Creating enhanced parameters with custom quality thresholds...");

    // Build custom enhanced parameters
    let base_params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 })
        .language("en");

    let enhanced_params = EnhancedTranscriptionParamsBuilder::new()
        .base_params(base_params)
        .temperatures(vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        .compression_ratio_threshold(Some(2.0))  // Stricter than default 2.4
        .log_prob_threshold(Some(-0.5))          // Stricter than default -1.0
        .build();

    println!("Quality thresholds:");
    println!("  - Max compression ratio: {:?}", enhanced_params.thresholds.compression_ratio_threshold);
    println!("  - Min log probability: {:?}", enhanced_params.thresholds.log_prob_threshold);
    println!("  - Temperature sequence: {:?}", enhanced_params.temperatures);

    // Transcribe with custom thresholds
    let mut state = ctx.create_state()?;
    let mut enhanced_state = EnhancedWhisperState::new(&mut state);

    let result = enhanced_state.transcribe_with_fallback(enhanced_params, audio)?;

    println!("\nTranscription result:");
    println!("  Text: {}", result.text);
    println!("  Segments: {}", result.segments.len());

    for (i, segment) in result.segments.iter().enumerate() {
        println!("    Segment {}: [{:.2}s - {:.2}s] {}",
            i + 1,
            segment.start_seconds(),
            segment.end_seconds(),
            segment.text
        );
    }

    Ok(())
}

fn demonstrate_direct_enhanced_state(
    ctx: &WhisperContext,
    audio: &[f32]
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Using enhanced state directly for fine control...");

    // Create state once and reuse
    let mut state = ctx.create_state()?;

    // Configure different quality thresholds for experimentation
    let relaxed_thresholds = QualityThresholds {
        compression_ratio_threshold: Some(3.0),  // More relaxed
        log_prob_threshold: Some(-2.0),          // More relaxed
        no_speech_threshold: Some(0.8),
    };

    let strict_thresholds = QualityThresholds {
        compression_ratio_threshold: Some(1.5),  // Very strict
        log_prob_threshold: Some(-0.3),          // Very strict
        no_speech_threshold: Some(0.4),
    };

    // Try with relaxed thresholds
    println!("\n1. With relaxed thresholds:");
    let params = EnhancedTranscriptionParams {
        base: FullParams::default().language("en"),
        temperatures: vec![0.0, 0.5, 1.0],
        thresholds: relaxed_thresholds,
        prompt_reset_on_temperature: 0.5,
    };

    let mut enhanced_state = EnhancedWhisperState::new(&mut state);
    let result = enhanced_state.transcribe_with_fallback(params, audio)?;
    println!("   Result: {}", result.text);

    // Try with strict thresholds
    println!("\n2. With strict thresholds:");
    let params = EnhancedTranscriptionParams {
        base: FullParams::default().language("en"),
        temperatures: vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        thresholds: strict_thresholds,
        prompt_reset_on_temperature: 0.5,
    };

    let result = enhanced_state.transcribe_with_fallback(params, audio)?;
    println!("   Result: {}", result.text);
    println!("   Note: Stricter thresholds may have triggered more temperature fallbacks");

    Ok(())
}

fn load_audio_examples() -> Result<(Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
    // Check for actual audio files
    let clear_path = "samples/clear_speech.wav";
    let noisy_path = "samples/noisy_speech.wav";

    let clear_audio = if Path::new(clear_path).exists() {
        println!("Loading clear audio from: {}", clear_path);
        load_wav_file(clear_path)?
    } else {
        println!("Generating synthetic clear audio...");
        generate_clear_synthetic_audio()
    };

    let noisy_audio = if Path::new(noisy_path).exists() {
        println!("Loading noisy audio from: {}", noisy_path);
        load_wav_file(noisy_path)?
    } else {
        println!("Generating synthetic noisy audio...");
        generate_noisy_synthetic_audio()
    };

    Ok((clear_audio, noisy_audio))
}

fn generate_clear_synthetic_audio() -> Vec<f32> {
    // Generate clean synthetic audio (would be actual speech in practice)
    let sample_rate = 16000;
    let duration = 3;
    let mut audio = Vec::with_capacity(sample_rate * duration);

    // Simple sine wave modulated to simulate speech patterns
    for i in 0..sample_rate * duration {
        let t = i as f32 / sample_rate as f32;
        let freq = 200.0 + 100.0 * (t * 2.0).sin(); // Varying frequency
        let amplitude = 0.3 * (1.0 + 0.5 * (t * 3.0).cos()); // Varying amplitude
        let sample = amplitude * (2.0 * std::f32::consts::PI * freq * t).sin();
        audio.push(sample);
    }

    audio
}

fn generate_noisy_synthetic_audio() -> Vec<f32> {
    // Generate noisy synthetic audio (would be actual noisy speech in practice)
    let clear = generate_clear_synthetic_audio();

    // Add noise
    clear.iter().map(|&sample| {
        let noise = (rand::random::<f32>() - 0.5) * 0.2; // Random noise
        let clipped = (sample + noise).max(-1.0).min(1.0); // Clip to valid range
        clipped
    }).collect()
}

fn load_wav_file(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    use hound;

    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    // Check format
    if spec.sample_rate != 16000 {
        eprintln!("Warning: Audio sample rate is {}Hz, expected 16000Hz", spec.sample_rate);
    }

    if spec.channels != 1 {
        eprintln!("Warning: Audio has {} channels, using first channel only", spec.channels);
    }

    let samples: Vec<f32> = reader
        .samples::<i16>()
        .step_by(spec.channels as usize)
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();

    Ok(samples)
}

// Fallback for rand if not available
#[cfg(not(feature = "rand"))]
fn rand_random() -> f32 {
    0.5
}