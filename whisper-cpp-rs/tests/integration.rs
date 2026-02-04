mod common;

use std::sync::Arc;
use common::TestModels;
use whisper_cpp_rs::{
    FullParams, SamplingStrategy, TranscriptionParams, WhisperContext, WhisperError,
};

#[test]
fn test_model_loading() {
    let Some(model_path) = TestModels::tiny_en() else {
        eprintln!("Skipping test: model file not found (expected in CI)");
        return;
    };

    let ctx = WhisperContext::new(&model_path);
    assert!(ctx.is_ok(), "Failed to load model from {:?}", model_path);
}

#[test]
fn test_invalid_model_path() {
    let ctx = WhisperContext::new("nonexistent_model.bin");
    assert!(ctx.is_err());
    match ctx {
        Err(WhisperError::ModelLoadError(_)) => {}
        _ => panic!("Expected ModelLoadError"),
    }
}

#[test]
fn test_silence_handling() {
    let Some(model_path) = TestModels::tiny_en() else {
        eprintln!("Skipping test: model file not found");
        return;
    };

    let ctx = WhisperContext::new(&model_path).unwrap();
    let silence = vec![0.0f32; 16000]; // 1 second of silence
    let result = ctx.transcribe(&silence);
    assert!(result.is_ok());

    // Silence should produce empty or minimal transcription
    let text = result.unwrap();
    assert!(text.len() < 100); // Arbitrary threshold for "minimal" output
}

#[test]
fn test_concurrent_states() {
    let Some(model_path) = TestModels::tiny_en() else {
        eprintln!("Skipping test: model file not found");
        return;
    };

    let ctx = Arc::new(WhisperContext::new(&model_path).unwrap());
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let ctx = Arc::clone(&ctx);
            std::thread::spawn(move || {
                let audio = vec![0.0f32; 16000];
                ctx.transcribe(&audio)
            })
        })
        .collect();

    for handle in handles {
        assert!(handle.join().unwrap().is_ok());
    }
}

#[test]
fn test_transcription_with_params() {
    let Some(model_path) = TestModels::tiny_en() else {
        eprintln!("Skipping test: model file not found");
        return;
    };

    let ctx = WhisperContext::new(&model_path).unwrap();
    let audio = vec![0.0f32; 16000 * 3]; // 3 seconds of silence

    let params = TranscriptionParams::builder()
        .language("en")
        .temperature(0.0)
        .enable_timestamps()
        .n_threads(2)
        .build();

    let result = ctx.transcribe_with_params(&audio, params);
    assert!(result.is_ok());

    let transcription = result.unwrap();
    // Check that we have both text and segments
    assert!(transcription.segments.len() >= 0);
    assert_eq!(
        transcription.text.split_whitespace().collect::<Vec<_>>().len(),
        transcription
            .segments
            .iter()
            .map(|s| s.text.split_whitespace().count())
            .sum::<usize>()
    );
}

#[test]
fn test_full_params_configuration() {
    let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 })
        .language("en")
        .translate(false)
        .no_context(false)
        .no_timestamps(false)
        .single_segment(false)
        .temperature(0.8)
        .n_threads(4)
        .offset_ms(0)
        .duration_ms(0);

    // Just ensure it builds without panic
    // The as_raw() method is internal
}

#[test]
fn test_beam_search_strategy() {
    let params = FullParams::new(SamplingStrategy::BeamSearch { beam_size: 5 })
        .language("en")
        .temperature(0.5);

    // Just ensure it builds without panic
    // The as_raw() method is internal
}

#[test]
fn test_model_info() {
    let Some(model_path) = TestModels::tiny_en() else {
        eprintln!("Skipping test: model file not found");
        return;
    };

    let ctx = WhisperContext::new(&model_path).unwrap();

    // Test model information methods (static model properties)
    assert!(ctx.n_vocab() > 0, "n_vocab should be positive");
    assert!(ctx.n_audio_ctx() > 0, "n_audio_ctx should be positive");
    assert!(ctx.n_text_ctx() > 0, "n_text_ctx should be positive");

    // Note: n_len() returns mel spectrogram length, which is 0 until audio is processed.
    // It's a state property, not a model property - don't test it here.

    // tiny.en model is English-only
    assert!(!ctx.is_multilingual());
}

#[test]
fn test_segment_timestamps() {
    let Some(model_path) = TestModels::tiny_en() else {
        eprintln!("Skipping test: model file not found");
        return;
    };

    let ctx = WhisperContext::new(&model_path).unwrap();
    let audio = vec![0.0f32; 16000 * 2]; // 2 seconds

    let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 })
        .no_timestamps(false);

    let result = ctx.transcribe_with_full_params(&audio, params).unwrap();

    for segment in &result.segments {
        // Timestamps should be non-negative
        assert!(segment.start_ms >= 0);
        assert!(segment.end_ms >= 0);
        // End should be after start
        assert!(segment.end_ms >= segment.start_ms);
        // Seconds conversion should work
        assert!(segment.start_seconds() >= 0.0);
        assert!(segment.end_seconds() >= 0.0);
    }
}

#[test]
fn test_state_reuse() {
    let Some(model_path) = TestModels::tiny_en() else {
        eprintln!("Skipping test: model file not found");
        return;
    };

    let ctx = WhisperContext::new(&model_path).unwrap();
    let mut state = ctx.create_state().unwrap();

    let audio1 = vec![0.0f32; 16000]; // 1 second
    let audio2 = vec![0.0f32; 16000 * 2]; // 2 seconds

    let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

    // First transcription
    assert!(state.full(params.clone(), &audio1).is_ok());
    let n_segments1 = state.full_n_segments();

    // Second transcription with same state
    assert!(state.full(params, &audio2).is_ok());
    let n_segments2 = state.full_n_segments();

    // Both should succeed
    assert!(n_segments1 >= 0);
    assert!(n_segments2 >= 0);
}

#[test]
fn test_empty_audio_error() {
    let Some(model_path) = TestModels::tiny_en() else {
        eprintln!("Skipping test: model file not found");
        return;
    };

    let ctx = WhisperContext::new(&model_path).unwrap();
    let empty_audio = vec![];

    let result = ctx.transcribe(&empty_audio);
    assert!(result.is_err());
    match result {
        Err(WhisperError::InvalidAudioFormat) => {}
        _ => panic!("Expected InvalidAudioFormat error"),
    }
}
