//! `WhisperContext` loads the model without whisper.cpp's default state; states are only
//! allocated by `WhisperState::new`. Observed through whisper.cpp's own allocation log messages,
//! which needs the process-global log hook, so this lives in its own test binary.

mod common;

use common::TestModels;
use std::sync::{Arc, Mutex};
use whisper_cpp_plus::{FullParams, SamplingStrategy, WhisperContext, WhisperLog, WhisperState};

#[test]
fn test_context_allocates_no_default_state() {
    let Some(model_path) = TestModels::tiny_en() else {
        eprintln!("Skipping: model not found. Run `cargo xtask test-setup`");
        return;
    };

    let messages: Arc<Mutex<Vec<String>>> = Arc::default();
    let sink = Arc::clone(&messages);
    WhisperLog::set(move |_, message| sink.lock().unwrap().push(message.to_owned()));

    let state_allocations = |messages: &Mutex<Vec<String>>| {
        messages
            .lock()
            .unwrap()
            .iter()
            .filter(|m| m.starts_with("whisper_init_state:"))
            .count()
    };

    let ctx = WhisperContext::new(&model_path).expect("Failed to load model");
    assert!(
        messages
            .lock()
            .unwrap()
            .iter()
            .any(|m| m.contains("loading model")),
        "expected model loading messages"
    );
    assert_eq!(
        state_allocations(&messages),
        0,
        "loading a context should not allocate a whisper state"
    );

    let mut state = WhisperState::new(&ctx).expect("Failed to create state");
    assert!(
        state_allocations(&messages) > 0,
        "creating a WhisperState should allocate one"
    );

    // The context-less path still works end to end, and n_len is per-state.
    assert_eq!(state.n_len(), 0);
    let audio = vec![0.0f32; 16000];
    state
        .full(
            FullParams::new(SamplingStrategy::Greedy { best_of: 1 }),
            &audio,
        )
        .expect("transcription failed");
    assert!(state.n_len() > 0, "n_len should be set after transcribing");

    WhisperLog::reset();
}
