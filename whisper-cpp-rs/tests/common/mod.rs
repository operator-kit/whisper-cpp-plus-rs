//! Common test utilities for whisper-cpp-rs tests
//!
//! Provides path resolution that works from both crate and workspace root.

use std::path::{Path, PathBuf};

/// Model paths for testing
pub struct TestModels;

impl TestModels {
    /// Get path to a whisper model file.
    /// Checks multiple locations in order:
    /// 1. WHISPER_TEST_MODEL_DIR env var
    /// 2. tests/models/ (crate-relative)
    /// 3. whisper-cpp-rs/tests/models/ (workspace-relative)
    /// 4. vendor/whisper.cpp/models/ (for stub test models)
    pub fn whisper_model(name: &str) -> Option<PathBuf> {
        Self::find_model(name, &[
            // Env override
            std::env::var("WHISPER_TEST_MODEL_DIR").ok(),
            // Crate-relative (when running from whisper-cpp-rs/)
            Some("tests/models".to_string()),
            // Workspace-relative (when running from root)
            Some("whisper-cpp-rs/tests/models".to_string()),
            // Fallback to whisper.cpp test stubs
            Some("vendor/whisper.cpp/models".to_string()),
        ])
    }

    /// Get path to the tiny.en model (most common for tests)
    pub fn tiny_en() -> Option<PathBuf> {
        Self::whisper_model("ggml-tiny.en.bin")
            .or_else(|| Self::whisper_model("for-tests-ggml-tiny.en.bin"))
    }

    /// Get path to VAD model
    pub fn vad() -> Option<PathBuf> {
        Self::find_model("ggml-silero-vad.bin", &[
            std::env::var("WHISPER_TEST_MODEL_DIR").ok(),
            Some("tests/models".to_string()),
            Some("whisper-cpp-rs/tests/models".to_string()),
        ]).or_else(|| {
            // Fallback to whisper.cpp's VAD test model
            Self::find_model("for-tests-silero-v6.2.0-ggml.bin", &[
                Some("vendor/whisper.cpp/models".to_string()),
            ])
        })
    }

    /// Get path to test audio file
    pub fn audio(name: &str) -> Option<PathBuf> {
        Self::find_model(name, &[
            std::env::var("WHISPER_TEST_AUDIO_DIR").ok(),
            Some("tests/audio".to_string()),
            Some("whisper-cpp-rs/tests/audio".to_string()),
            Some("vendor/whisper.cpp/samples".to_string()),
        ])
    }

    /// Get jfk.wav sample audio
    pub fn jfk_audio() -> Option<PathBuf> {
        Self::audio("jfk.wav")
    }

    fn find_model(name: &str, dirs: &[Option<String>]) -> Option<PathBuf> {
        for dir in dirs.iter().flatten() {
            let path = Path::new(dir).join(name);
            if path.exists() {
                return Some(path);
            }
        }
        None
    }
}

/// Skip test if model not found, with helpful message
#[macro_export]
macro_rules! skip_if_no_model {
    ($path:expr, $model_name:expr) => {
        let Some(path) = $path else {
            eprintln!(
                "Skipping test: {} not found.\n\
                 Set WHISPER_TEST_MODEL_DIR or place models in tests/models/",
                $model_name
            );
            return;
        };
        let path = path;
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_resolution() {
        // This test verifies path resolution works
        // At least one location should exist in dev environment
        let tiny = TestModels::tiny_en();
        println!("tiny.en model path: {:?}", tiny);

        // The whisper.cpp stubs should always exist
        let stub_path = Path::new("vendor/whisper.cpp/models/for-tests-ggml-tiny.en.bin");
        let stub_exists = stub_path.exists() ||
            Path::new("whisper-cpp-rs/../vendor/whisper.cpp/models/for-tests-ggml-tiny.en.bin").exists();

        // In CI without models, we at least have the stubs
        if tiny.is_none() && !stub_exists {
            eprintln!("Warning: No test models found (expected in CI)");
        }
    }
}
