//! Temperature fallback mechanism for improved transcription quality
//!
//! This module implements quality-based retry logic inspired by faster-whisper

use crate::{WhisperState, FullParams, Result, WhisperError, Segment, TranscriptionResult};
use std::io::Write;
use flate2::Compression;
use flate2::write::ZlibEncoder;
use whisper_sys as ffi;

/// Quality thresholds for transcription validation
#[derive(Debug, Clone)]
pub struct QualityThresholds {
    /// Maximum compression ratio (default: 2.4)
    pub compression_ratio_threshold: Option<f32>,
    /// Minimum average log probability (default: -1.0)
    pub log_prob_threshold: Option<f32>,
    /// Maximum no-speech probability (default: 0.6)
    pub no_speech_threshold: Option<f32>,
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            compression_ratio_threshold: Some(2.4),
            log_prob_threshold: Some(-1.0),
            no_speech_threshold: Some(0.6),
        }
    }
}

/// Enhanced transcription parameters with fallback support
#[derive(Clone)]
pub struct EnhancedTranscriptionParams {
    /// Base parameters
    pub base: FullParams,
    /// Temperature sequence for fallback
    pub temperatures: Vec<f32>,
    /// Quality thresholds
    pub thresholds: QualityThresholds,
    /// Whether to reset prompt on temperature increase
    pub prompt_reset_on_temperature: f32,
}

impl EnhancedTranscriptionParams {
    /// Create from base params with default enhancement settings
    pub fn from_base(base: FullParams) -> Self {
        Self {
            base,
            temperatures: vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            thresholds: QualityThresholds::default(),
            prompt_reset_on_temperature: 0.5,
        }
    }

    pub fn builder() -> EnhancedTranscriptionParamsBuilder {
        EnhancedTranscriptionParamsBuilder::new()
    }
}

pub struct EnhancedTranscriptionParamsBuilder {
    params: EnhancedTranscriptionParams,
}

impl EnhancedTranscriptionParamsBuilder {
    pub fn new() -> Self {
        Self {
            params: EnhancedTranscriptionParams::from_base(FullParams::default()),
        }
    }

    pub fn base_params(mut self, params: FullParams) -> Self {
        self.params.base = params;
        self
    }

    pub fn language(mut self, lang: &str) -> Self {
        self.params.base = self.params.base.language(lang);
        self
    }

    pub fn temperatures(mut self, temps: Vec<f32>) -> Self {
        self.params.temperatures = temps;
        self
    }

    pub fn compression_ratio_threshold(mut self, threshold: Option<f32>) -> Self {
        self.params.thresholds.compression_ratio_threshold = threshold;
        self
    }

    pub fn log_prob_threshold(mut self, threshold: Option<f32>) -> Self {
        self.params.thresholds.log_prob_threshold = threshold;
        self
    }

    pub fn build(self) -> EnhancedTranscriptionParams {
        self.params
    }
}

impl Default for EnhancedTranscriptionParamsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate compression ratio for text using zlib
pub fn calculate_compression_ratio(text: &str) -> f32 {
    let text_bytes = text.as_bytes();
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(text_bytes).unwrap();
    let compressed = encoder.finish().unwrap();

    text_bytes.len() as f32 / compressed.len() as f32
}

/// Result of a single transcription attempt
#[derive(Debug)]
pub struct TranscriptionAttempt {
    pub text: String,
    pub segments: Vec<Segment>,
    pub temperature: f32,
    pub compression_ratio: f32,
    pub avg_logprob: f32,
    pub no_speech_prob: f32,
}

impl TranscriptionAttempt {
    /// Check if this attempt meets quality thresholds
    pub fn meets_thresholds(&self, thresholds: &QualityThresholds) -> bool {
        let mut meets = true;

        if let Some(cr_threshold) = thresholds.compression_ratio_threshold {
            if self.compression_ratio > cr_threshold {
                meets = false;
            }
        }

        if let Some(lp_threshold) = thresholds.log_prob_threshold {
            if self.avg_logprob < lp_threshold {
                // Check for silence exception
                if let Some(ns_threshold) = thresholds.no_speech_threshold {
                    if self.no_speech_prob <= ns_threshold {
                        meets = false;
                    }
                } else {
                    meets = false;
                }
            }
        }

        meets
    }
}

/// Enhanced state with fallback support
pub struct EnhancedWhisperState<'a> {
    state: &'a mut WhisperState,
}

impl<'a> EnhancedWhisperState<'a> {
    pub fn new(state: &'a mut WhisperState) -> Self {
        Self { state }
    }

    /// Get no-speech probability for a segment (enhanced feature)
    fn get_no_speech_prob(&self, segment_idx: i32) -> f32 {
        unsafe {
            // Direct FFI call using the exposed ptr
            ffi::whisper_full_get_segment_no_speech_prob_from_state(
                self.state.ptr,
                segment_idx
            )
        }
    }

    /// Calculate average log probability from token probabilities
    fn calculate_avg_logprob(&self, segment_idx: i32) -> f32 {
        let n_tokens = self.state.full_n_tokens(segment_idx);
        if n_tokens == 0 {
            return 0.0;
        }

        let mut sum_logprob = 0.0;
        for i in 0..n_tokens {
            let prob = self.state.full_get_token_prob(segment_idx, i);
            if prob > 0.0 {
                sum_logprob += prob.ln();
            }
        }

        sum_logprob / n_tokens as f32
    }

    /// Transcribe with temperature fallback
    pub fn transcribe_with_fallback(
        &mut self,
        params: EnhancedTranscriptionParams,
        audio: &[f32],
    ) -> Result<TranscriptionResult> {
        let mut all_attempts = Vec::new();
        let mut below_cr_attempts = Vec::new();

        for temperature in &params.temperatures {
            // Update temperature in params
            let mut current_params = params.base.clone();
            current_params = current_params.temperature(*temperature);

            // Reset prompt if temperature is high
            if *temperature > params.prompt_reset_on_temperature {
                current_params = current_params.initial_prompt("");
            }

            // Attempt transcription
            self.state.full(current_params, audio)?;

            // Extract results
            let n_segments = self.state.full_n_segments();
            let mut segments = Vec::new();
            let mut text = String::new();
            let mut total_logprob = 0.0;
            let mut total_tokens = 0;

            for i in 0..n_segments {
                let segment_text = self.state.full_get_segment_text(i)?;
                let (start_ms, end_ms) = self.state.full_get_segment_timestamps(i);
                let speaker_turn_next = self.state.full_get_segment_speaker_turn_next(i);

                if i > 0 {
                    text.push(' ');
                }
                text.push_str(&segment_text);

                segments.push(Segment {
                    start_ms,
                    end_ms,
                    text: segment_text,
                    speaker_turn_next,
                });

                // Calculate average log probability
                let avg_lp = self.calculate_avg_logprob(i);
                let n_tokens = self.state.full_n_tokens(i);
                total_logprob += avg_lp * n_tokens as f32;
                total_tokens += n_tokens;
            }

            let avg_logprob = if total_tokens > 0 {
                total_logprob / total_tokens as f32
            } else {
                0.0
            };

            // Calculate quality metrics
            let compression_ratio = calculate_compression_ratio(&text);
            let no_speech_prob = if n_segments > 0 {
                self.get_no_speech_prob(0)
            } else {
                0.0
            };

            let attempt = TranscriptionAttempt {
                text: text.clone(),
                segments: segments.clone(),
                temperature: *temperature,
                compression_ratio,
                avg_logprob,
                no_speech_prob,
            };

            // Check if attempt meets thresholds
            if attempt.meets_thresholds(&params.thresholds) {
                return Ok(TranscriptionResult {
                    text: attempt.text,
                    segments: attempt.segments,
                });
            }

            // Store attempt for potential fallback selection
            if let Some(cr_threshold) = params.thresholds.compression_ratio_threshold {
                if attempt.compression_ratio <= cr_threshold {
                    below_cr_attempts.push(attempt);
                } else {
                    all_attempts.push(attempt);
                }
            } else {
                all_attempts.push(attempt);
            }
        }

        // All temperatures failed, select best attempt
        let best_attempt = if !below_cr_attempts.is_empty() {
            below_cr_attempts.into_iter()
                .max_by(|a, b| a.avg_logprob.partial_cmp(&b.avg_logprob).unwrap())
        } else {
            all_attempts.into_iter()
                .max_by(|a, b| a.avg_logprob.partial_cmp(&b.avg_logprob).unwrap())
        };

        best_attempt
            .map(|a| TranscriptionResult {
                text: a.text,
                segments: a.segments,
            })
            .ok_or_else(|| WhisperError::TranscriptionError(
                "Failed to produce acceptable transcription with any temperature".into()
            ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_ratio_calculation() {
        // Short text might not compress well due to compression overhead
        let text = "The quick brown fox jumps over the lazy dog";
        let ratio = calculate_compression_ratio(text);
        assert!(ratio > 0.0); // Ratio should be positive

        // Longer text should compress better
        let longer_text = "The quick brown fox jumps over the lazy dog. ".repeat(10);
        let longer_ratio = calculate_compression_ratio(&longer_text);
        assert!(longer_ratio > 1.0); // Should achieve compression

        // Highly repetitive text should compress very well
        let repetitive = "a".repeat(1000);
        let repetitive_ratio = calculate_compression_ratio(&repetitive);
        assert!(repetitive_ratio > 5.0); // Highly compressible
    }

    #[test]
    fn test_quality_threshold_checking() {
        let thresholds = QualityThresholds {
            compression_ratio_threshold: Some(2.4),
            log_prob_threshold: Some(-1.0),
            no_speech_threshold: Some(0.6),
        };

        let good_attempt = TranscriptionAttempt {
            text: "Hello world".to_string(),
            segments: vec![],
            temperature: 0.0,
            compression_ratio: 1.5,
            avg_logprob: -0.5,
            no_speech_prob: 0.1,
        };

        assert!(good_attempt.meets_thresholds(&thresholds));

        let bad_attempt = TranscriptionAttempt {
            text: "a".repeat(100),
            segments: vec![],
            temperature: 0.0,
            compression_ratio: 10.0, // Too repetitive
            avg_logprob: -0.5,
            no_speech_prob: 0.1,
        };

        assert!(!bad_attempt.meets_thresholds(&thresholds));
    }

    #[test]
    fn test_enhanced_params_from_base() {
        let base = FullParams::default()
            .language("en");

        let enhanced = EnhancedTranscriptionParams::from_base(base);

        assert_eq!(enhanced.temperatures.len(), 6);
        assert_eq!(enhanced.temperatures[0], 0.0);
        assert_eq!(enhanced.prompt_reset_on_temperature, 0.5);
        assert!(enhanced.thresholds.compression_ratio_threshold.is_some());
    }

    #[test]
    fn test_enhanced_transcription_params_builder() {
        let params = EnhancedTranscriptionParamsBuilder::new()
            .language("en")
            .temperatures(vec![0.0, 0.5, 1.0])
            .compression_ratio_threshold(Some(3.0))
            .build();

        assert_eq!(params.temperatures.len(), 3);
        assert_eq!(params.thresholds.compression_ratio_threshold, Some(3.0));
    }
}