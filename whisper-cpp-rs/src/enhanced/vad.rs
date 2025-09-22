//! Enhanced VAD functionality with segment aggregation
//!
//! This module provides advanced VAD features beyond the basic whisper.cpp implementation,
//! inspired by faster-whisper's optimizations. VAD is a preprocessing step that happens
//! BEFORE transcription, not part of the transcription API itself.

use crate::vad::{VadProcessor, VadParams};
use crate::error::Result;
use std::path::Path;

/// Enhanced VAD parameters with aggregation settings
#[derive(Debug, Clone)]
pub struct EnhancedVadParams {
    /// Base VAD parameters from whisper.cpp
    pub base: VadParams,
    /// Maximum duration for aggregated segments (seconds)
    pub max_segment_duration_s: f32,
    /// Whether to merge adjacent segments
    pub merge_segments: bool,
    /// Minimum gap between segments to keep them separate (ms)
    pub min_gap_ms: i32,
}

impl Default for EnhancedVadParams {
    fn default() -> Self {
        Self {
            base: VadParams::default(),
            max_segment_duration_s: 30.0,
            merge_segments: true,
            min_gap_ms: 100,
        }
    }
}

/// Enhanced VAD processor with segment aggregation
pub struct EnhancedVadProcessor {
    inner: VadProcessor,
}

impl EnhancedVadProcessor {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        Ok(Self {
            inner: VadProcessor::new(model_path)?,
        })
    }

    /// Process audio with segment aggregation
    /// Returns aggregated speech chunks optimized for transcription
    pub fn process_with_aggregation(
        &mut self,
        audio: &[f32],
        params: &EnhancedVadParams,
    ) -> Result<Vec<AudioChunk>> {
        // Get raw segments from base VAD
        let segments = self.inner.segments_from_samples(audio, &params.base)?;
        let raw_segments = segments.get_all_segments();

        // Apply aggregation
        let aggregated = self.aggregate_segments(
            raw_segments,
            params.max_segment_duration_s,
            params.min_gap_ms,
            params.merge_segments,
        );

        // Extract audio chunks with metadata
        let chunks = self.extract_audio_chunks(audio, aggregated, 16000.0);
        Ok(chunks)
    }

    /// Aggregate segments to optimize for transcription
    #[doc(hidden)]
    pub fn aggregate_segments(
        &self,
        segments: Vec<(f32, f32)>,
        max_duration: f32,
        min_gap_ms: i32,
        merge: bool,
    ) -> Vec<(f32, f32)> {
        if segments.is_empty() {
            return Vec::new();
        }

        let mut aggregated = Vec::new();
        let min_gap = min_gap_ms as f32 / 1000.0;

        let mut current_start = segments[0].0;
        let mut current_end = segments[0].1;

        for (start, end) in segments.iter().skip(1) {
            let gap = start - current_end;
            let combined_duration = end - current_start;

            // Check if we should merge with current segment
            if merge && gap < min_gap && combined_duration <= max_duration {
                // Extend current segment
                current_end = *end;
            } else {
                // Save current segment and start new one
                aggregated.push((current_start, current_end));
                current_start = *start;
                current_end = *end;
            }
        }

        // Don't forget the last segment
        aggregated.push((current_start, current_end));

        aggregated
    }

    /// Extract audio chunks with metadata
    fn extract_audio_chunks(
        &self,
        audio: &[f32],
        segments: Vec<(f32, f32)>,
        sample_rate: f32,
    ) -> Vec<AudioChunk> {
        segments
            .into_iter()
            .map(|(start, end)| {
                let start_sample = (start * sample_rate) as usize;
                let end_sample = ((end * sample_rate) as usize).min(audio.len());

                AudioChunk {
                    audio: audio[start_sample..end_sample].to_vec(),
                    offset_seconds: start,
                    duration_seconds: end - start,
                    metadata: ChunkMetadata {
                        original_start: start,
                        original_end: end,
                        sample_offset: start_sample,
                    },
                }
            })
            .collect()
    }
}

/// Audio chunk with metadata for transcription
#[derive(Debug, Clone)]
pub struct AudioChunk {
    /// Audio samples
    pub audio: Vec<f32>,
    /// Offset from original audio start (seconds)
    pub offset_seconds: f32,
    /// Duration of this chunk (seconds)
    pub duration_seconds: f32,
    /// Additional metadata
    pub metadata: ChunkMetadata,
}

#[derive(Debug, Clone)]
pub struct ChunkMetadata {
    /// Original segment start time
    pub original_start: f32,
    /// Original segment end time
    pub original_end: f32,
    /// Sample offset in original audio
    pub sample_offset: usize,
}

/// Builder for enhanced VAD parameters
pub struct EnhancedVadParamsBuilder {
    params: EnhancedVadParams,
}

impl EnhancedVadParamsBuilder {
    pub fn new() -> Self {
        Self {
            params: EnhancedVadParams::default(),
        }
    }

    pub fn threshold(mut self, threshold: f32) -> Self {
        self.params.base.threshold = threshold;
        self
    }

    pub fn max_segment_duration(mut self, seconds: f32) -> Self {
        self.params.max_segment_duration_s = seconds;
        self
    }

    pub fn merge_segments(mut self, merge: bool) -> Self {
        self.params.merge_segments = merge;
        self
    }

    pub fn min_gap_ms(mut self, ms: i32) -> Self {
        self.params.min_gap_ms = ms;
        self
    }

    pub fn speech_pad_ms(mut self, ms: i32) -> Self {
        self.params.base.speech_pad_ms = ms;
        self
    }

    pub fn build(self) -> EnhancedVadParams {
        self.params
    }
}

impl Default for EnhancedVadParamsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_aggregation() {
        let processor = EnhancedVadProcessor {
            inner: unsafe { std::mem::zeroed() }, // Mock for testing aggregation logic
        };

        let segments = vec![
            (0.0, 2.0),
            (2.1, 4.0),  // Small gap - should merge
            (4.5, 6.0),  // Larger gap
            (10.0, 12.0), // Large gap - separate segment
        ];

        let aggregated = processor.aggregate_segments(segments, 30.0, 100, true);

        assert_eq!(aggregated.len(), 3);
        assert_eq!(aggregated[0], (0.0, 4.0)); // First two merged
        assert_eq!(aggregated[1], (4.5, 6.0));
        assert_eq!(aggregated[2], (10.0, 12.0));
    }

    #[test]
    fn test_max_duration_split() {
        let processor = EnhancedVadProcessor {
            inner: unsafe { std::mem::zeroed() },
        };

        let segments = vec![
            (0.0, 20.0),
            (20.1, 40.0), // Would exceed 30s if merged
        ];

        let aggregated = processor.aggregate_segments(segments, 30.0, 100, true);

        assert_eq!(aggregated.len(), 2); // Should not merge due to max duration
    }

    #[test]
    fn test_enhanced_vad_params_builder() {
        let params = EnhancedVadParamsBuilder::new()
            .threshold(0.6)
            .max_segment_duration(25.0)
            .merge_segments(false)
            .min_gap_ms(200)
            .build();

        assert_eq!(params.base.threshold, 0.6);
        assert_eq!(params.max_segment_duration_s, 25.0);
        assert!(!params.merge_segments);
        assert_eq!(params.min_gap_ms, 200);
    }
}