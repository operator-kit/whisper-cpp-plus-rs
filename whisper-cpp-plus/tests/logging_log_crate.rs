//! `WhisperLog::use_log_crate()` (feature `log`). Installs a global `log` logger, so it lives in
//! its own test binary.
#![cfg(feature = "log")]

mod common;

use common::TestModels;
use std::sync::Mutex;
use whisper_cpp_plus::{WhisperContext, WhisperLog};

struct CapturingLogger {
    records: Mutex<Vec<(log::Level, String, String)>>,
}

impl log::Log for CapturingLogger {
    fn enabled(&self, _metadata: &log::Metadata) -> bool {
        true
    }

    fn log(&self, record: &log::Record) {
        self.records.lock().unwrap().push((
            record.level(),
            record.target().to_owned(),
            record.args().to_string(),
        ));
    }

    fn flush(&self) {}
}

static LOGGER: CapturingLogger = CapturingLogger {
    records: Mutex::new(Vec::new()),
};

#[test]
fn test_use_log_crate() {
    let Some(model_path) = TestModels::tiny_en() else {
        eprintln!("Skipping: model not found. Run `cargo xtask test-setup`");
        return;
    };

    log::set_logger(&LOGGER).expect("logger already set");
    log::set_max_level(log::LevelFilter::Trace);

    WhisperLog::use_log_crate();
    drop(WhisperContext::new(&model_path).expect("Failed to load model"));
    WhisperLog::reset();

    let records = LOGGER.records.lock().unwrap();
    assert!(
        records.iter().any(|(level, target, message)| {
            *level == log::Level::Info
                && target == "whisper_cpp"
                && message.contains("loading model")
        }),
        "expected an info 'loading model' record with target whisper_cpp, got {:?}",
        *records
    );
}
