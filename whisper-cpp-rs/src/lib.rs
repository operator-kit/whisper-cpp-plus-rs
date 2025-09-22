//! Safe, idiomatic Rust bindings for whisper.cpp
//!
//! This crate provides high-level, safe Rust bindings for whisper.cpp,
//! OpenAI's Whisper automatic speech recognition (ASR) model implementation in C++.
//!
//! # Quick Start
//!
//! ```no_run
//! use whisper_cpp_rs::{WhisperContext, FullParams, SamplingStrategy};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Load a Whisper model
//! let ctx = WhisperContext::new("path/to/model.bin")?;
//!
//! // Transcribe audio (must be 16kHz mono f32 samples)
//! let audio = vec![0.0f32; 16000]; // 1 second of silence
//! let text = ctx.transcribe(&audio)?;
//! println!("Transcription: {}", text);
//! # Ok(())
//! # }
//! ```
//!
//! # Advanced Usage
//!
//! ```no_run
//! use whisper_cpp_rs::{WhisperContext, FullParams, SamplingStrategy, TranscriptionParams};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let ctx = WhisperContext::new("path/to/model.bin")?;
//!
//! // Configure parameters using builder pattern
//! let params = TranscriptionParams::builder()
//!     .language("en")
//!     .temperature(0.8)
//!     .enable_timestamps()
//!     .build();
//!
//! // Transcribe with custom parameters
//! let result = ctx.transcribe_with_params(&audio, params)?;
//!
//! // Access segments with timestamps
//! for segment in result.segments {
//!     println!("[{}-{}]: {}", segment.start_seconds(), segment.end_seconds(), segment.text);
//! }
//! # Ok(())
//! # }
//! ```

mod buffer;
mod context;
mod error;
mod params;
mod state;
mod stream;
mod vad;

pub mod enhanced;

#[cfg(feature = "async")]
mod async_api;

pub use context::WhisperContext;
pub use error::{Result, WhisperError};
pub use params::{
    FullParams, SamplingStrategy, TranscriptionParams, TranscriptionParamsBuilder,
};
pub use state::{Segment, TranscriptionResult, WhisperState};
pub use stream::{StreamConfig, StreamConfigBuilder, WhisperStream};
pub use vad::{
    VadContextParams, VadParams, VadParamsBuilder, VadProcessor, VadSegments,
};

// Re-export for benchmarks
#[doc(hidden)]
pub mod bench_helpers {
    pub use crate::vad::{VadProcessor, VadParams};
}

#[cfg(feature = "async")]
pub use async_api::{AsyncWhisperStream, SharedAsyncStream};

// Re-export the sys crate for advanced users who need lower-level access
pub use whisper_sys;

impl WhisperContext {
    /// Transcribe audio using default parameters
    ///
    /// # Arguments
    /// * `audio` - Audio samples (must be 16kHz mono f32)
    ///
    /// # Returns
    /// The transcribed text as a string
    ///
    /// # Example
    /// ```no_run
    /// # use whisper_cpp_rs::WhisperContext;
    /// # fn main() -> whisper_cpp_rs::Result<()> {
    /// let ctx = WhisperContext::new("model.bin")?;
    /// let audio = vec![0.0f32; 16000]; // 1 second
    /// let text = ctx.transcribe(&audio)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn transcribe(&self, audio: &[f32]) -> Result<String> {
        let mut state = WhisperState::new(self)?;
        let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

        state.full(params, audio)?;

        let n_segments = state.full_n_segments();
        let mut text = String::new();

        for i in 0..n_segments {
            if i > 0 {
                text.push(' ');
            }
            text.push_str(&state.full_get_segment_text(i)?);
        }

        Ok(text)
    }

    /// Transcribe audio with custom parameters
    ///
    /// # Arguments
    /// * `audio` - Audio samples (must be 16kHz mono f32)
    /// * `params` - Custom transcription parameters
    ///
    /// # Returns
    /// A `TranscriptionResult` containing the full text and individual segments
    pub fn transcribe_with_params(
        &self,
        audio: &[f32],
        params: TranscriptionParams,
    ) -> Result<TranscriptionResult> {
        self.transcribe_with_full_params(audio, params.into_full_params())
    }

    /// Transcribe audio with full control over parameters
    ///
    /// # Arguments
    /// * `audio` - Audio samples (must be 16kHz mono f32)
    /// * `params` - Full parameter configuration
    ///
    /// # Returns
    /// A `TranscriptionResult` containing the full text and individual segments
    pub fn transcribe_with_full_params(
        &self,
        audio: &[f32],
        params: FullParams,
    ) -> Result<TranscriptionResult> {
        let mut state = WhisperState::new(self)?;
        state.full(params, audio)?;

        let n_segments = state.full_n_segments();
        let mut segments = Vec::with_capacity(n_segments as usize);
        let mut full_text = String::new();

        for i in 0..n_segments {
            let text = state.full_get_segment_text(i)?;
            let (start_ms, end_ms) = state.full_get_segment_timestamps(i);
            let speaker_turn_next = state.full_get_segment_speaker_turn_next(i);

            if i > 0 {
                full_text.push(' ');
            }
            full_text.push_str(&text);

            segments.push(Segment {
                start_ms,
                end_ms,
                text,
                speaker_turn_next,
            });
        }

        Ok(TranscriptionResult {
            text: full_text,
            segments,
        })
    }

    /// Create a new state for manual transcription control
    ///
    /// This allows you to reuse a state for multiple transcriptions,
    /// which can be more efficient than creating a new state each time.
    pub fn create_state(&self) -> Result<WhisperState> {
        WhisperState::new(self)
    }

    /// Enhanced transcription with custom parameters and temperature fallback
    ///
    /// This method provides quality-based retry with multiple temperatures
    /// if the initial transcription doesn't meet quality thresholds.
    ///
    /// # Arguments
    /// * `audio` - Audio samples (must be 16kHz mono f32)
    /// * `params` - Custom transcription parameters
    ///
    /// # Returns
    /// A `TranscriptionResult` containing the full text and individual segments
    ///
    /// # Example
    /// ```no_run
    /// # use whisper_cpp_rs::{WhisperContext, TranscriptionParams};
    /// # fn main() -> whisper_cpp_rs::Result<()> {
    /// let ctx = WhisperContext::new("model.bin")?;
    /// let params = TranscriptionParams::builder()
    ///     .language("en")
    ///     .build();
    /// let audio = vec![0.0f32; 16000];
    /// let result = ctx.transcribe_with_params_enhanced(&audio, params)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn transcribe_with_params_enhanced(
        &self,
        audio: &[f32],
        params: TranscriptionParams,
    ) -> Result<TranscriptionResult> {
        self.transcribe_with_full_params_enhanced(audio, params.into_full_params())
    }

    /// Enhanced transcription with full parameters and temperature fallback
    ///
    /// This method provides quality-based retry with multiple temperatures
    /// if the initial transcription doesn't meet quality thresholds.
    ///
    /// # Arguments
    /// * `audio` - Audio samples (must be 16kHz mono f32)
    /// * `params` - Full parameter configuration
    ///
    /// # Returns
    /// A `TranscriptionResult` containing the full text and individual segments
    pub fn transcribe_with_full_params_enhanced(
        &self,
        audio: &[f32],
        params: FullParams,
    ) -> Result<TranscriptionResult> {
        use crate::enhanced::fallback::{EnhancedTranscriptionParams, EnhancedWhisperState};

        // Convert to enhanced params with default fallback settings
        let enhanced_params = EnhancedTranscriptionParams::from_base(params);

        // Use enhanced state with temperature fallback logic
        let mut state = self.create_state()?;
        let mut enhanced_state = EnhancedWhisperState::new(&mut state);
        enhanced_state.transcribe_with_fallback(enhanced_params, audio)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::sync::Arc;

    #[test]
    fn test_error_on_invalid_model() {
        let result = WhisperContext::new("nonexistent_model.bin");
        assert!(result.is_err());
    }

    #[test]
    fn test_model_loading() {
        let model_path = "tests/models/ggml-tiny.en.bin";
        if Path::new(model_path).exists() {
            let ctx = WhisperContext::new(model_path);
            assert!(ctx.is_ok());
        } else {
            eprintln!("Skipping test_model_loading: model file not found");
        }
    }

    #[test]
    fn test_silence_handling() {
        let model_path = "tests/models/ggml-tiny.en.bin";
        if Path::new(model_path).exists() {
            let ctx = WhisperContext::new(model_path).unwrap();
            let silence = vec![0.0f32; 16000]; // 1 second of silence
            let result = ctx.transcribe(&silence);
            assert!(result.is_ok());
        } else {
            eprintln!("Skipping test_silence_handling: model file not found");
        }
    }

    #[test]
    fn test_concurrent_states() {
        let model_path = "tests/models/ggml-tiny.en.bin";
        if Path::new(model_path).exists() {
            let ctx = Arc::new(WhisperContext::new(model_path).unwrap());
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
        } else {
            eprintln!("Skipping test_concurrent_states: model file not found");
        }
    }

    #[test]
    fn test_params_builder() {
        let params = TranscriptionParams::builder()
            .language("en")
            .temperature(0.8)
            .enable_timestamps()
            .n_threads(4)
            .build();

        // Just ensure it builds without panic
        let _ = params.into_full_params();
    }
}