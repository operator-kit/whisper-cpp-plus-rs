//! Control over whisper.cpp's log output.
//!
//! whisper.cpp (including its VAD and the ggml backends) writes log messages to stderr by
//! default. [`WhisperLog`] wraps `whisper_log_set` so the output can be redirected to a Rust
//! callback, silenced, or forwarded to the [`log`](https://docs.rs/log) crate (feature `log`).
//!
//! The log hook is process-global state in whisper.cpp, so configure it once at startup,
//! before loading models or starting transcriptions.

use std::ffi::{c_char, c_void, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex, RwLock};
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

// The active Rust callback. whisper.cpp only ever holds a pointer to `log_trampoline`, which
// looks the callback up here, so no Rust data is ever handed to C.
static CALLBACK: RwLock<Option<LogCallback>> = RwLock::new(None);

// Whether `log_trampoline` is installed in whisper.cpp. The lock also serialises our calls to
// `whisper_log_set`, which writes whisper.cpp's global state without synchronisation.
static INSTALLED: Mutex<bool> = Mutex::new(false);

// Level of the last message, used for `GGML_LOG_LEVEL_CONT` ("continue previous message").
static LAST_LEVEL: AtomicU8 = AtomicU8::new(LogLevel::Info as u8);

/// Configures where whisper.cpp's log output goes (`whisper_log_set`).
///
/// This covers whisper.cpp, its VAD, and the ggml backends it initialises. Output goes to
/// stderr until one of these functions is called.
pub struct WhisperLog;

impl WhisperLog {
    /// Sends whisper.cpp log messages to `callback` instead of stderr.
    ///
    /// Messages are passed without their trailing newline; empty messages are skipped. Debug
    /// messages are included (whisper.cpp's default stderr output hides them). The callback may
    /// be called from any thread, including whisper.cpp's worker threads. A panic inside the
    /// callback is caught and the message dropped, since unwinding into C is not allowed.
    ///
    /// Replacing the callback later is cheap and safe at any time.
    pub fn set<F>(callback: F)
    where
        F: Fn(LogLevel, &str) + Send + Sync + 'static,
    {
        *write_callback() = Some(Arc::new(callback));
        install_trampoline();
    }

    /// Discards all whisper.cpp log output.
    pub fn disable() {
        *write_callback() = None;
        install_trampoline();
    }

    /// Restores whisper.cpp's default behaviour: messages above debug level go to stderr.
    pub fn reset() {
        let mut installed = lock_installed();
        unsafe { ffi::whisper_log_set(None, std::ptr::null_mut()) };
        *installed = false;
        *write_callback() = None;
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

fn write_callback() -> std::sync::RwLockWriteGuard<'static, Option<LogCallback>> {
    CALLBACK
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn lock_installed() -> std::sync::MutexGuard<'static, bool> {
    INSTALLED
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn install_trampoline() {
    let mut installed = lock_installed();
    if !*installed {
        unsafe { ffi::whisper_log_set(Some(log_trampoline), std::ptr::null_mut()) };
        *installed = true;
    }
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
        let Some(level) = map_level(level) else {
            return;
        };
        let callback = CALLBACK
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        let Some(callback) = callback else {
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

    fn emit(level: ffi::ggml_log_level, text: &str) {
        let text = CString::new(text).unwrap();
        unsafe { log_trampoline(level, text.as_ptr(), std::ptr::null_mut()) };
    }

    // One test so the shared CALLBACK isn't raced by other tests in this module. The
    // trampoline is called directly and never installed in whisper.cpp, so other tests' log
    // output doesn't reach the callback.
    #[test]
    fn trampoline_maps_levels_trims_and_contains_panics() {
        let received: Arc<Mutex<Vec<(LogLevel, String)>>> = Arc::default();
        let sink = Arc::clone(&received);
        *write_callback() = Some(Arc::new(move |level, message: &str| {
            sink.lock().unwrap().push((level, message.to_owned()));
        }));

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

        // A panicking callback must not unwind into C.
        *write_callback() = Some(Arc::new(|_, _: &str| panic!("callback panicked")));
        emit(ffi::ggml_log_level_GGML_LOG_LEVEL_INFO, "boom\n");

        // No callback: messages are dropped.
        *write_callback() = None;
        emit(ffi::ggml_log_level_GGML_LOG_LEVEL_INFO, "dropped\n");
    }

    #[test]
    fn log_levels_are_ordered_by_severity() {
        assert!(LogLevel::Debug < LogLevel::Info);
        assert!(LogLevel::Info < LogLevel::Warn);
        assert!(LogLevel::Warn < LogLevel::Error);
    }
}
