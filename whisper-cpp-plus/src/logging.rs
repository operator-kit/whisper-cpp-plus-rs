//! Control over whisper.cpp's log output.
//!
//! whisper.cpp (including its VAD and the ggml backends) writes log messages to stderr by
//! default. [`WhisperLog`] redirects that output to a Rust callback, silences it, or forwards it
//! to the [`log`](https://docs.rs/log) crate (feature `log`).
//!
//! whisper.cpp's log hook (`whisper_log_set`) is process-global state that whisper.cpp writes
//! and reads without synchronisation, so changing it while another thread is inside whisper.cpp
//! is a data race. The crate therefore installs its own hook exactly once, before its first call
//! into whisper.cpp, and never changes it again. [`WhisperLog`] only changes where that hook
//! sends messages, which is synchronised on the Rust side.

use std::ffi::{c_char, c_void, CStr};
use std::io::Write;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Once, RwLock};
use whisper_cpp_plus_sys as ffi;

/// Severity of a whisper.cpp log message (`ggml_log_level`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
}

type LogCallback = Arc<dyn Fn(LogLevel, &str) + Send + Sync>;

// Where `log_trampoline` sends messages.
enum Sink {
    // What whisper.cpp does when no log callback has been set: every message goes to stderr.
    Stderr,
    Discard,
    Callback(LogCallback),
}

// whisper.cpp only ever holds a pointer to `log_trampoline`, which looks the sink up here, so no
// Rust data is ever handed to C.
static SINK: RwLock<Sink> = RwLock::new(Sink::Stderr);

static INSTALL: Once = Once::new();

// Level of the last message, used for `GGML_LOG_LEVEL_CONT` ("continue previous message").
static LAST_LEVEL: AtomicU8 = AtomicU8::new(LogLevel::Info as u8);

/// Configures where whisper.cpp's log output goes.
///
/// This covers whisper.cpp, its VAD, and the ggml backends it initialises. Output goes to
/// stderr until one of these functions is called. They are safe to call at any time, including
/// while other threads are transcribing.
///
/// The crate installs its own hook with `whisper_log_set` before its first call into
/// whisper.cpp; a hook set earlier by calling `whisper_log_set` directly through
/// `whisper-cpp-plus-sys` is replaced at that point.
pub struct WhisperLog;

impl WhisperLog {
    /// Sends whisper.cpp log messages to `callback` instead of stderr.
    ///
    /// Messages are passed without their trailing newline; empty messages are skipped. The
    /// callback may be called from any thread, including whisper.cpp's worker threads. A panic
    /// inside the callback is caught and the message dropped, since unwinding into C is not
    /// allowed.
    pub fn set<F>(callback: F)
    where
        F: Fn(LogLevel, &str) + Send + Sync + 'static,
    {
        set_sink(Sink::Callback(Arc::new(callback)));
    }

    /// Discards all whisper.cpp log output.
    pub fn disable() {
        set_sink(Sink::Discard);
    }

    /// Restores the default: every message goes to stderr unchanged, as it does when no log
    /// callback has been set in whisper.cpp.
    pub fn reset() {
        set_sink(Sink::Stderr);
    }

    /// Forwards whisper.cpp log messages to the [`log`](https://docs.rs/log) crate with target
    /// `whisper_cpp`, mapping [`LogLevel`] to the matching `log::Level`.
    #[cfg(feature = "log")]
    pub fn use_log_crate() {
        Self::set(|level, message| {
            let level = match level {
                LogLevel::Debug => log::Level::Debug,
                LogLevel::Info => log::Level::Info,
                LogLevel::Warn => log::Level::Warn,
                LogLevel::Error => log::Level::Error,
            };
            log::log!(target: "whisper_cpp", level, "{}", message);
        });
    }
}

/// Installs the crate's log hook in whisper.cpp, once per process.
///
/// Every safe entry point that can call into whisper.cpp without going through an existing
/// context or VAD context (constructors, quantization) must call this first. `Once` orders the
/// `whisper_log_set` write before every later crate call into whisper.cpp, so the hook is never
/// written while the crate is inside whisper.cpp.
pub(crate) fn ensure_installed() {
    INSTALL
        .call_once(|| unsafe { ffi::whisper_log_set(Some(log_trampoline), std::ptr::null_mut()) });
}

fn set_sink(sink: Sink) {
    ensure_installed();
    *SINK
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = sink;
}

fn map_level(level: ffi::ggml_log_level) -> Option<LogLevel> {
    let mapped = match level {
        ffi::ggml_log_level_GGML_LOG_LEVEL_DEBUG => LogLevel::Debug,
        ffi::ggml_log_level_GGML_LOG_LEVEL_INFO => LogLevel::Info,
        ffi::ggml_log_level_GGML_LOG_LEVEL_WARN => LogLevel::Warn,
        ffi::ggml_log_level_GGML_LOG_LEVEL_ERROR => LogLevel::Error,
        ffi::ggml_log_level_GGML_LOG_LEVEL_CONT => {
            return Some(level_from_u8(LAST_LEVEL.load(Ordering::Relaxed)))
        }
        _ => return None,
    };
    LAST_LEVEL.store(mapped as u8, Ordering::Relaxed);
    Some(mapped)
}

fn level_from_u8(value: u8) -> LogLevel {
    match value {
        v if v == LogLevel::Debug as u8 => LogLevel::Debug,
        v if v == LogLevel::Warn as u8 => LogLevel::Warn,
        v if v == LogLevel::Error as u8 => LogLevel::Error,
        _ => LogLevel::Info,
    }
}

unsafe extern "C" fn log_trampoline(
    level: ffi::ggml_log_level,
    text: *const c_char,
    _user_data: *mut c_void,
) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if text.is_null() {
            return;
        }
        let callback = match &*SINK.read().unwrap_or_else(|poisoned| poisoned.into_inner()) {
            Sink::Stderr => {
                let mut stderr = std::io::stderr().lock();
                let _ = stderr.write_all(CStr::from_ptr(text).to_bytes());
                let _ = stderr.flush();
                return;
            }
            Sink::Discard => return,
            Sink::Callback(callback) => Arc::clone(callback),
        };
        let Some(level) = map_level(level) else {
            return;
        };

        let text = CStr::from_ptr(text).to_string_lossy();
        let message = text.trim_end_matches(&['\n', '\r'][..]);
        if !message.is_empty() {
            callback(level, message);
        }
    }));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use std::sync::Mutex;

    fn emit(level: ffi::ggml_log_level, text: &str) {
        let text = CString::new(text).unwrap();
        unsafe { log_trampoline(level, text.as_ptr(), std::ptr::null_mut()) };
    }

    // Sets the sink without installing the trampoline in whisper.cpp, so other tests' log output
    // doesn't reach it.
    fn set_sink_for_test(sink: Sink) {
        *SINK
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = sink;
    }

    // One test so the shared SINK isn't raced by other tests in this module.
    #[test]
    fn trampoline_maps_levels_trims_and_contains_panics() {
        let received: Arc<Mutex<Vec<(LogLevel, String)>>> = Arc::default();
        let sink = Arc::clone(&received);
        set_sink_for_test(Sink::Callback(Arc::new(move |level, message: &str| {
            sink.lock().unwrap().push((level, message.to_owned()));
        })));

        emit(ffi::ggml_log_level_GGML_LOG_LEVEL_DEBUG, "debug line\n");
        emit(ffi::ggml_log_level_GGML_LOG_LEVEL_INFO, "info line\n");
        emit(ffi::ggml_log_level_GGML_LOG_LEVEL_WARN, "warn line\r\n");
        emit(ffi::ggml_log_level_GGML_LOG_LEVEL_CONT, "continued");
        emit(ffi::ggml_log_level_GGML_LOG_LEVEL_ERROR, "error line");
        emit(ffi::ggml_log_level_GGML_LOG_LEVEL_INFO, "\n");
        emit(ffi::ggml_log_level_GGML_LOG_LEVEL_NONE, "ignored\n");
        unsafe {
            log_trampoline(
                ffi::ggml_log_level_GGML_LOG_LEVEL_INFO,
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };

        // A panicking callback must not unwind into C.
        set_sink_for_test(Sink::Callback(Arc::new(|_, _: &str| {
            panic!("callback panicked")
        })));
        emit(ffi::ggml_log_level_GGML_LOG_LEVEL_INFO, "boom\n");

        // Discard: messages are dropped.
        set_sink_for_test(Sink::Discard);
        emit(ffi::ggml_log_level_GGML_LOG_LEVEL_INFO, "dropped\n");

        // Stderr: written unchanged, nothing reaches the old callback.
        set_sink_for_test(Sink::Stderr);
        emit(ffi::ggml_log_level_GGML_LOG_LEVEL_INFO, "stderr line\n");

        assert_eq!(
            *received.lock().unwrap(),
            vec![
                (LogLevel::Debug, "debug line".to_owned()),
                (LogLevel::Info, "info line".to_owned()),
                (LogLevel::Warn, "warn line".to_owned()),
                (LogLevel::Warn, "continued".to_owned()),
                (LogLevel::Error, "error line".to_owned()),
            ]
        );
    }

    #[test]
    fn log_levels_are_ordered_by_severity() {
        assert!(LogLevel::Debug < LogLevel::Info);
        assert!(LogLevel::Info < LogLevel::Warn);
        assert!(LogLevel::Warn < LogLevel::Error);
    }
}
