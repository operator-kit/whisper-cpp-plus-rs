//! Log routing is process-global, so these tests live in their own test binary and in a single
//! test function.

mod common;

use common::TestModels;
use std::sync::{Arc, Mutex};
use whisper_cpp_plus::{LogLevel, WhisperContext, WhisperLog};

type Captured = Arc<Mutex<Vec<(LogLevel, String)>>>;

fn capture_into(captured: &Captured) {
    let sink = Arc::clone(captured);
    WhisperLog::set(move |level, message| {
        sink.lock().unwrap().push((level, message.to_owned()));
    });
}

#[test]
fn test_log_routing() {
    let Some(model_path) = TestModels::tiny_en() else {
        eprintln!("Skipping: model not found. Run `cargo xtask test-setup`");
        return;
    };

    let captured: Captured = Arc::default();

    // A callback receives whisper.cpp's messages from model loading.
    capture_into(&captured);
    drop(WhisperContext::new(&model_path).expect("Failed to load model"));
    {
        let messages = captured.lock().unwrap();
        assert!(
            messages
                .iter()
                .any(|(level, message)| *level == LogLevel::Info
                    && message.contains("loading model")),
            "expected an info 'loading model' message, got {:?}",
            *messages
        );
        assert!(
            messages.iter().all(|(_, message)| !message.ends_with('\n')),
            "messages should have their trailing newline trimmed"
        );
    }

    // disable() silences everything, and the previous callback is no longer called.
    let before = captured.lock().unwrap().len();
    WhisperLog::disable();
    drop(WhisperContext::new(&model_path).expect("Failed to load model"));
    assert_eq!(captured.lock().unwrap().len(), before);

    // A callback can be set again after disabling.
    capture_into(&captured);
    drop(WhisperContext::new(&model_path).expect("Failed to load model"));
    assert!(captured.lock().unwrap().len() > before);

    // reset() restores whisper.cpp's default stderr output and detaches the callback.
    let before = captured.lock().unwrap().len();
    WhisperLog::reset();
    drop(WhisperContext::new(&model_path).expect("Failed to load model"));
    assert_eq!(captured.lock().unwrap().len(), before);

    // Changing the routing while another thread is inside whisper.cpp is allowed: it only
    // touches Rust-side state, never whisper.cpp's global log hook. (A data race here would
    // need a thread sanitizer to detect; this checks the calls are accepted and routing still
    // works afterwards.)
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let toggler = {
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || {
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                WhisperLog::set(|_, _| {});
                WhisperLog::disable();
                WhisperLog::reset();
            }
        })
    };
    for _ in 0..3 {
        drop(WhisperContext::new(&model_path).expect("Failed to load model"));
    }
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    toggler.join().expect("toggler thread panicked");

    capture_into(&captured);
    let before = captured.lock().unwrap().len();
    drop(WhisperContext::new(&model_path).expect("Failed to load model"));
    assert!(captured.lock().unwrap().len() > before);
    WhisperLog::reset();
}
