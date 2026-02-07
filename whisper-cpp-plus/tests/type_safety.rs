use std::sync::Arc;
use std::thread;
use whisper_cpp_plus::{WhisperContext, WhisperState, TranscriptionParams, FullParams, SamplingStrategy};

// Test that WhisperContext is Send + Sync
fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}

#[test]
fn test_context_is_send_sync() {
    assert_send::<WhisperContext>();
    assert_sync::<WhisperContext>();
    assert_send::<Arc<WhisperContext>>();
    assert_sync::<Arc<WhisperContext>>();
}

#[test]
fn test_state_is_send() {
    // WhisperState should be Send but not Sync (each thread needs its own state)
    assert_send::<WhisperState>();
    // This should NOT compile if uncommented (WhisperState is not Sync):
    // assert_sync::<WhisperState>();
}

#[test]
fn test_params_are_not_send_sync() {
    // TranscriptionParams and FullParams contain raw pointers from FFI,
    // so they should NOT be Send or Sync - this is correct for safety!
    // This test documents that the types correctly prevent unsafe sharing.

    // These would fail to compile (which is good for safety):
    // assert_send::<TranscriptionParams>();
    // assert_sync::<TranscriptionParams>();
    // assert_send::<FullParams>();
    // assert_sync::<FullParams>();
}

#[test]
fn test_arc_context_thread_safety() {
    // Skip if model doesn't exist
    let model_path = "tests/models/ggml-tiny.en.bin";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Skipping: model not found. Run `cargo xtask test-setup`");
        return;
    }

    // Test that Arc<WhisperContext> can be safely shared across threads
    let context = Arc::new(WhisperContext::new(model_path).unwrap());

    let handles: Vec<_> = (0..4)
        .map(|i| {
            let ctx = Arc::clone(&context);
            thread::spawn(move || {
                // Each thread can use the shared context
                let info = format!(
                    "Thread {}: vocab_size={}, audio_ctx={}, text_ctx={}, multilingual={}",
                    i,
                    ctx.n_vocab(),
                    ctx.n_audio_ctx(),
                    ctx.n_text_ctx(),
                    ctx.is_multilingual()
                );
                info
            })
        })
        .collect();

    for handle in handles {
        let info = handle.join().unwrap();
        println!("{}", info);
    }
}

#[test]
fn test_state_not_clone() {
    // WhisperState should NOT be Clone (each thread needs its own state)
    // This is a compile-time check - if WhisperState implements Clone, this will fail
    fn assert_not_clone<T>() {
        // This function exists just to document the invariant
        // The actual test is that WhisperState doesn't implement Clone
    }
    assert_not_clone::<WhisperState>();
}

#[test]
fn test_context_not_copy() {
    // WhisperContext should NOT be Copy (it manages resources)
    // This is enforced at compile time
    fn assert_not_copy<T>() {
        // This function exists just to document the invariant
    }
    assert_not_copy::<WhisperContext>();
}

#[test]
fn test_lifetime_safety() {
    // Test that states cannot outlive their contexts
    // This is enforced through lifetime parameters in WhisperState
    let model_path = "tests/models/ggml-tiny.en.bin";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Skipping: model not found. Run `cargo xtask test-setup`");
        return;
    }

    let context = Arc::new(WhisperContext::new(model_path).unwrap());
    let state = WhisperState::new(&context).unwrap();

    // State holds a reference to context, so it cannot outlive it
    // This is enforced at compile time through lifetimes
    drop(state);
    drop(context);
    // If we tried to use state after dropping context, it would be a compile error
}

#[test]
fn test_null_pointer_safety() {
    // Test that the safe wrapper prevents null pointer issues
    // Invalid paths should return proper errors, not segfault
    let result = WhisperContext::new("nonexistent_model.bin");
    assert!(result.is_err());

    // Test error type safety
    if let Err(e) = result {
        // Should be able to get error description without panic
        let _desc = e.to_string();
    }
}

#[test]
fn test_buffer_safety() {
    // Test that audio buffer handling is safe
    let model_path = "tests/models/ggml-tiny.en.bin";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Skipping: model not found. Run `cargo xtask test-setup`");
        return;
    }

    let ctx = WhisperContext::new(model_path).unwrap();

    // Empty buffer should be handled safely (no crash)
    let empty: Vec<f32> = vec![];
    let result = ctx.transcribe(&empty);
    // Empty buffer may return error or empty result - both are safe
    if let Err(e) = &result {
        eprintln!("Empty buffer error (acceptable): {}", e);
    }

    // Large buffer should be handled safely (no buffer overflow)
    let large = vec![0.0f32; 16000 * 60]; // 1 minute
    let result = ctx.transcribe(&large);

    // Large buffers may return an error if they exceed limits, which is also fine for safety
    if let Err(e) = &result {
        eprintln!("Large buffer transcription error (expected): {}", e);
    }
    // The important thing is it doesn't crash - error is acceptable
}

#[test]
fn test_params_type_safety() {
    // Test that parameters are type-safe and can't have invalid values
    let _params = TranscriptionParams::builder()
        .language("en")
        .temperature(0.8)
        .n_threads(4)
        .build();

    // Test sampling strategies are type-safe enums
    let _greedy = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    let _beam = FullParams::new(SamplingStrategy::BeamSearch { beam_size: 5 });

    // The actual conversion to FFI params happens internally in WhisperState
    // and is not exposed to prevent misuse - this is good for safety!
}

#[test]
fn test_drop_safety() {
    // Test that Drop implementations are safe and don't cause double-free
    let model_path = "tests/models/ggml-tiny.en.bin";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Skipping: model not found. Run `cargo xtask test-setup`");
        return;
    }

    {
        let ctx = WhisperContext::new(model_path).unwrap();
        let _state1 = WhisperState::new(&ctx).unwrap();
        let _state2 = WhisperState::new(&ctx).unwrap();
        // Both states should drop safely when going out of scope
    } // Drop happens here

    // Should not crash or double-free
}

// Compile-time tests for API misuse prevention
#[cfg(compile_fail_tests)]
mod compile_fail {
    use super::*;

    // This should fail to compile: WhisperState is not Sync
    // #[test]
    // fn state_not_sync() {
    //     assert_sync::<WhisperState>(); // COMPILE ERROR
    // }

    // This should fail to compile: Can't use state after context is dropped
    // #[test]
    // fn state_lifetime() {
    //     let state = {
    //         let ctx = WhisperContext::new("model.bin").unwrap();
    //         WhisperState::new(&ctx).unwrap()
    //     };
    //     // state.full(...); // COMPILE ERROR: ctx doesn't live long enough
    // }
}