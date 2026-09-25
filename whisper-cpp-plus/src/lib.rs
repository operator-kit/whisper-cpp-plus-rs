//! Safe, idiomatic Rust bindings for whisper.cpp
//!
//! This crate provides high-level, safe Rust bindings for whisper.cpp,
//! OpenAI's Whisper automatic speech recognition (ASR) model implementation in C++.
//!
//! # Quick Start
//!
//! ```no_run
//! use whisper_cpp_plus::{WhisperContext, FullParams, SamplingStrategy};
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
//! use whisper_cpp_plus::{WhisperContext, FullParams, SamplingStrategy, TranscriptionParams};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let ctx = WhisperContext::new("path/to/model.bin")?;
//! let audio = vec![0.0f32; 16000]; // 1 second of audio
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

mod context;
mod error;
mod logging;
mod params;
mod state;
mod stream;
mod stream_pcm;
#[cfg(test)]
mod test_support;
mod vad;

pub mod enhanced;

#[cfg(feature = "quantization")]
mod quantize;

#[cfg(feature = "async")]
mod async_api;

pub use context::WhisperContext;
pub use error::{Result, WhisperError};
pub use logging::{LogLevel, WhisperLog};
pub use params::{FullParams, SamplingStrategy, TranscriptionParams, TranscriptionParamsBuilder};
#[cfg(feature = "quantization")]
pub use quantize::{QuantizationType, QuantizeError, WhisperQuantize};
pub use state::{Segment, TranscriptionResult, WhisperState};
pub use stream::{WhisperStream, WhisperStreamConfig};
pub use stream_pcm::{
    vad_simple, PcmFormat, PcmReader, PcmReaderConfig, WhisperStreamPcm, WhisperStreamPcmConfig,
};
pub use vad::{VadContextParams, VadParams, VadParamsBuilder, VadSegments, WhisperVadProcessor};

// Re-export for benchmarks
#[doc(hidden)]
pub mod bench_helpers {
    pub use crate::vad::{VadParams, WhisperVadProcessor};
}

#[cfg(feature = "async")]
pub use async_api::{AsyncWhisperStream, SharedAsyncStream};

// Re-export the sys crate for advanced users who need lower-level access
pub use whisper_cpp_plus_sys;

fn result_from_segments(segments: Vec<Segment>) -> TranscriptionResult {
    let text = segments
        .iter()
        .map(|segment| segment.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    TranscriptionResult { text, segments }
}

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
    /// # use whisper_cpp_plus::WhisperContext;
    /// # fn main() -> whisper_cpp_plus::Result<()> {
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
        Ok(result_from_segments(state.collect_segments()?))
    }

    /// Transcribe audio by splitting it into `n_processors` chunks that are transcribed
    /// concurrently (`whisper_full_parallel`).
    ///
    /// The audio after `params.offset_ms` is split into `n_processors` equal chunks. Each chunk
    /// is transcribed on its own [`WhisperState`] in a separate thread, and the segments are
    /// merged in order with their times shifted onto the original timeline (a segment never
    /// starts before the previous one ends). This follows whisper.cpp's `whisper_full_parallel`.
    ///
    /// Chunks are not overlapped, so a word that straddles a chunk boundary may be cut or
    /// misrecognised. Each chunk uses `params.n_threads` threads, so up to
    /// `n_processors * n_threads` threads run at once. With `n_processors == 1`, or audio too
    /// short to split, this is a single transcription.
    ///
    /// # Arguments
    /// * `params` - Full parameter configuration, applied to every chunk
    /// * `audio` - Audio samples (must be 16kHz mono f32)
    /// * `n_processors` - Number of chunks to transcribe concurrently (at least 1)
    pub fn full_parallel(
        &self,
        params: FullParams,
        audio: &[f32],
        n_processors: usize,
    ) -> Result<TranscriptionResult> {
        if audio.is_empty() {
            return Err(WhisperError::InvalidAudioFormat);
        }
        if n_processors == 0 {
            return Err(WhisperError::InvalidParameter(
                "n_processors must be at least 1".into(),
            ));
        }

        let sample_rate = whisper_cpp_plus_sys::WHISPER_SAMPLE_RATE as usize;
        let offset_ms = params.inner.offset_ms.max(0) as usize;
        let offset_samples = (sample_rate * offset_ms / 1000).min(audio.len());
        let n_samples_per_processor = (audio.len() - offset_samples) / n_processors;

        if n_processors == 1 || n_samples_per_processor == 0 {
            return self.transcribe_with_full_params(audio, params);
        }

        let transcribe_chunk = |params: FullParams, chunk: &[f32]| -> Result<Vec<Segment>> {
            let mut state = WhisperState::new(self)?;
            state.full(params, chunk)?;
            state.collect_segments()
        };

        // As in whisper.cpp: the first chunk also covers the `offset_ms` lead-in (so whisper
        // skips it itself and reports times from the start of `audio`); later chunks start
        // after it and are transcribed without an offset.
        let chunk_results: Vec<Result<Vec<Segment>>> = std::thread::scope(|scope| {
            let workers: Vec<_> = (1..n_processors)
                .map(|i| {
                    let start = offset_samples + i * n_samples_per_processor;
                    let end = if i == n_processors - 1 {
                        audio.len()
                    } else {
                        start + n_samples_per_processor
                    };
                    let chunk_params = params
                        .clone()
                        .offset_ms(0)
                        .print_progress(false)
                        .print_realtime(false);
                    let transcribe_chunk = &transcribe_chunk;
                    scope.spawn(move || transcribe_chunk(chunk_params, &audio[start..end]))
                })
                .collect();

            let first = transcribe_chunk(
                params.clone().print_realtime(false),
                &audio[..offset_samples + n_samples_per_processor],
            );

            std::iter::once(first)
                .chain(workers.into_iter().map(|worker| {
                    worker
                        .join()
                        .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
                }))
                .collect()
        });

        let samples_to_ms = |samples: usize| samples as i64 * 1000 / sample_rate as i64;

        let mut segments: Vec<Segment> = Vec::new();
        for (i, chunk_segments) in chunk_results.into_iter().enumerate() {
            let chunk_offset_ms = if i == 0 {
                0
            } else {
                samples_to_ms(offset_samples + i * n_samples_per_processor)
            };
            let chunk_end_ms = if i == n_processors - 1 {
                samples_to_ms(audio.len())
            } else {
                samples_to_ms(offset_samples + (i + 1) * n_samples_per_processor)
            };

            for mut segment in chunk_segments? {
                // whisper can report segment ends past the audio it was given; clamp to the
                // chunk so an overrun can't push the next chunk's segments later (whisper.cpp's
                // own merge doesn't do this).
                segment.start_ms = (segment.start_ms + chunk_offset_ms).min(chunk_end_ms);
                segment.end_ms = (segment.end_ms + chunk_offset_ms).min(chunk_end_ms);
                // Keep segments from overlapping across chunk boundaries.
                if let Some(previous) = segments.last() {
                    segment.start_ms = segment.start_ms.max(previous.end_ms);
                }
                segment.end_ms = segment.end_ms.max(segment.start_ms);
                segments.push(segment);
            }
        }

        Ok(result_from_segments(segments))
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
    /// # use whisper_cpp_plus::{WhisperContext, TranscriptionParams};
    /// # fn main() -> whisper_cpp_plus::Result<()> {
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
    use std::sync::Arc;

    #[test]
    fn test_error_on_invalid_model() {
        let result = WhisperContext::new("nonexistent_model.bin");
        assert!(result.is_err());
    }

    #[test]
    fn test_model_loading() {
        let Some(model_path) = crate::test_support::tiny_en() else {
            crate::test_support::note_missing_fixture("tiny.en model");
            return;
        };

        let ctx = WhisperContext::new(&model_path);
        assert!(ctx.is_ok());
    }

    #[test]
    fn test_silence_handling() {
        let Some(model_path) = crate::test_support::tiny_en() else {
            crate::test_support::note_missing_fixture("tiny.en model");
            return;
        };

        let ctx = WhisperContext::new(&model_path).unwrap();
        let silence = vec![0.0f32; 16000]; // 1 second of silence
        let result = ctx.transcribe(&silence);
        assert!(result.is_ok());
    }

    #[test]
    fn test_concurrent_states() {
        let Some(model_path) = crate::test_support::tiny_en() else {
            crate::test_support::note_missing_fixture("tiny.en model");
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
