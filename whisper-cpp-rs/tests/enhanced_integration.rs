mod common;

use common::TestModels;
use whisper_cpp_rs::enhanced::vad::{EnhancedVadParams, EnhancedVadProcessor};
use whisper_cpp_rs::enhanced::fallback::{EnhancedTranscriptionParams, EnhancedWhisperState};
use whisper_cpp_rs::{FullParams, SamplingStrategy, WhisperContext};

#[test]
fn test_enhanced_vad_with_real_audio() {
    let Some(vad_model) = TestModels::vad() else {
        eprintln!("Skipping: VAD model not found. Run `cargo xtask test-setup`");
        return;
    };
    let Some(audio_path) = TestModels::jfk_audio() else {
        eprintln!("Skipping: jfk.wav not found");
        return;
    };

    // Load audio
    let reader = hound::WavReader::open(&audio_path).unwrap();
    let samples: Vec<f32> = reader
        .into_samples::<i16>()
        .map(|s| s.unwrap() as f32 / i16::MAX as f32)
        .collect();

    let mut processor = EnhancedVadProcessor::new(&vad_model).unwrap();
    let params = EnhancedVadParams::default();
    let chunks = processor.process_with_aggregation(&samples, &params).unwrap();

    // JFK audio has speech — should produce at least one chunk
    assert!(!chunks.is_empty(), "Expected speech chunks from jfk.wav");

    // Each chunk should have valid audio data
    for chunk in &chunks {
        assert!(!chunk.audio.is_empty());
        assert!(chunk.duration_seconds > 0.0);
        assert!(chunk.offset_seconds >= 0.0);
    }
}

#[test]
fn test_enhanced_vad_aggregation_merges_segments() {
    let Some(vad_model) = TestModels::vad() else {
        eprintln!("Skipping: VAD model not found. Run `cargo xtask test-setup`");
        return;
    };
    let Some(audio_path) = TestModels::jfk_audio() else {
        eprintln!("Skipping: jfk.wav not found");
        return;
    };

    let reader = hound::WavReader::open(&audio_path).unwrap();
    let samples: Vec<f32> = reader
        .into_samples::<i16>()
        .map(|s| s.unwrap() as f32 / i16::MAX as f32)
        .collect();

    let mut processor = EnhancedVadProcessor::new(&vad_model).unwrap();

    // With merging enabled (default)
    let merged_params = EnhancedVadParams {
        merge_segments: true,
        min_gap_ms: 500,
        ..Default::default()
    };
    let merged = processor.process_with_aggregation(&samples, &merged_params).unwrap();

    // Without merging
    let unmerged_params = EnhancedVadParams {
        merge_segments: false,
        ..Default::default()
    };
    let unmerged = processor.process_with_aggregation(&samples, &unmerged_params).unwrap();

    // Merged should have fewer or equal segments
    assert!(
        merged.len() <= unmerged.len(),
        "Merged ({}) should have <= segments than unmerged ({})",
        merged.len(),
        unmerged.len()
    );
}

#[test]
fn test_temperature_fallback_transcription() {
    let Some(model_path) = TestModels::tiny_en() else {
        eprintln!("Skipping: model not found. Run `cargo xtask test-setup`");
        return;
    };
    let Some(audio_path) = TestModels::jfk_audio() else {
        eprintln!("Skipping: jfk.wav not found");
        return;
    };

    let reader = hound::WavReader::open(&audio_path).unwrap();
    let samples: Vec<f32> = reader
        .into_samples::<i16>()
        .map(|s| s.unwrap() as f32 / i16::MAX as f32)
        .collect();

    let ctx = WhisperContext::new(&model_path).unwrap();
    let mut state = ctx.create_state().unwrap();

    let base_params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 })
        .language("en");
    let enhanced_params = EnhancedTranscriptionParams::from_base(base_params);

    let mut enhanced_state = EnhancedWhisperState::new(&mut state);
    let result = enhanced_state
        .transcribe_with_fallback(enhanced_params, &samples)
        .unwrap();

    // Should produce non-empty transcription
    assert!(!result.text.is_empty(), "Expected non-empty transcription");
    assert!(!result.segments.is_empty(), "Expected segments");

    // JFK audio should contain recognizable words
    let lower = result.text.to_lowercase();
    assert!(
        lower.contains("ask") || lower.contains("country") || lower.contains("do"),
        "Expected recognizable JFK speech, got: {}",
        result.text
    );
}

#[test]
fn test_enhanced_transcription_via_context() {
    let Some(model_path) = TestModels::tiny_en() else {
        eprintln!("Skipping: model not found. Run `cargo xtask test-setup`");
        return;
    };
    let Some(audio_path) = TestModels::jfk_audio() else {
        eprintln!("Skipping: jfk.wav not found");
        return;
    };

    let reader = hound::WavReader::open(&audio_path).unwrap();
    let samples: Vec<f32> = reader
        .into_samples::<i16>()
        .map(|s| s.unwrap() as f32 / i16::MAX as f32)
        .collect();

    let ctx = WhisperContext::new(&model_path).unwrap();

    let params = whisper_cpp_rs::TranscriptionParams::builder()
        .language("en")
        .build();

    let result = ctx.transcribe_with_params_enhanced(&samples, params).unwrap();

    assert!(!result.text.is_empty());
    assert!(!result.segments.is_empty());
}
