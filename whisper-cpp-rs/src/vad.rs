//! Voice Activity Detection (VAD) support
//!
//! This module provides VAD capabilities for detecting speech segments
//! in audio before transcription, improving performance and accuracy.

use crate::error::{Result, WhisperError};
use std::path::Path;
use whisper_sys as ffi;

/// VAD parameters for speech detection
#[derive(Debug, Clone)]
pub struct VadParams {
    /// Probability threshold to consider as speech (0.0 - 1.0)
    pub threshold: f32,
    /// Minimum duration for a valid speech segment (in milliseconds)
    pub min_speech_duration_ms: i32,
    /// Minimum duration of silence to split segments (in milliseconds)
    pub min_silence_duration_ms: i32,
    /// Maximum speech duration before forcing a segment break (in seconds)
    pub max_speech_duration_s: f32,
    /// Padding added before and after speech segments (in milliseconds)
    pub speech_pad_ms: i32,
    /// Overlap in seconds when copying audio samples from speech segment
    pub samples_overlap: f32,
}

impl Default for VadParams {
    fn default() -> Self {
        // Use whisper.cpp's default VAD parameters
        let default_params = unsafe { ffi::whisper_vad_default_params() };

        Self {
            threshold: default_params.threshold,
            min_speech_duration_ms: default_params.min_speech_duration_ms,
            min_silence_duration_ms: default_params.min_silence_duration_ms,
            max_speech_duration_s: default_params.max_speech_duration_s,
            speech_pad_ms: default_params.speech_pad_ms,
            samples_overlap: default_params.samples_overlap,
        }
    }
}

impl VadParams {
    /// Convert to FFI params
    fn to_ffi(&self) -> ffi::whisper_vad_params {
        ffi::whisper_vad_params {
            threshold: self.threshold,
            min_speech_duration_ms: self.min_speech_duration_ms,
            min_silence_duration_ms: self.min_silence_duration_ms,
            max_speech_duration_s: self.max_speech_duration_s,
            speech_pad_ms: self.speech_pad_ms,
            samples_overlap: self.samples_overlap,
        }
    }
}

/// VAD context parameters
#[derive(Debug, Clone)]
pub struct VadContextParams {
    /// Number of threads to use for processing
    pub n_threads: i32,
    /// Whether to use GPU acceleration
    pub use_gpu: bool,
    /// GPU device ID to use
    pub gpu_device: i32,
}

impl Default for VadContextParams {
    fn default() -> Self {
        let default_params = unsafe { ffi::whisper_vad_default_context_params() };

        Self {
            n_threads: default_params.n_threads,
            use_gpu: default_params.use_gpu,
            gpu_device: default_params.gpu_device,
        }
    }
}

impl VadContextParams {
    /// Convert to FFI params
    fn to_ffi(&self) -> ffi::whisper_vad_context_params {
        ffi::whisper_vad_context_params {
            n_threads: self.n_threads,
            use_gpu: self.use_gpu,
            gpu_device: self.gpu_device,
        }
    }
}

/// Voice Activity Detector
pub struct VadProcessor {
    ctx: *mut ffi::whisper_vad_context,
}

unsafe impl Send for VadProcessor {}
unsafe impl Sync for VadProcessor {}

impl Drop for VadProcessor {
    fn drop(&mut self) {
        unsafe {
            if !self.ctx.is_null() {
                ffi::whisper_vad_free(self.ctx);
            }
        }
    }
}

impl VadProcessor {
    /// Create a new VAD processor from a model file
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        Self::new_with_params(model_path, VadContextParams::default())
    }

    /// Create a new VAD processor with custom parameters
    pub fn new_with_params<P: AsRef<Path>>(
        model_path: P,
        params: VadContextParams,
    ) -> Result<Self> {
        let path_str = model_path
            .as_ref()
            .to_str()
            .ok_or_else(|| WhisperError::ModelLoadError("Invalid path".into()))?;

        let c_path = std::ffi::CString::new(path_str)?;

        let ctx = unsafe {
            ffi::whisper_vad_init_from_file_with_params(c_path.as_ptr(), params.to_ffi())
        };

        if ctx.is_null() {
            return Err(WhisperError::ModelLoadError(
                "Failed to load VAD model".into(),
            ));
        }

        Ok(Self { ctx })
    }

    /// Detect speech in audio samples
    pub fn detect_speech(&mut self, samples: &[f32]) -> bool {
        if samples.is_empty() {
            return false;
        }

        unsafe {
            ffi::whisper_vad_detect_speech(
                self.ctx,
                samples.as_ptr(),
                samples.len() as i32,
            )
        }
    }

    /// Get the number of probability values
    pub fn n_probs(&self) -> i32 {
        unsafe { ffi::whisper_vad_n_probs(self.ctx) }
    }

    /// Get probability values
    pub fn get_probs(&self) -> Vec<f32> {
        let n = self.n_probs();
        if n == 0 {
            return Vec::new();
        }

        let probs_ptr = unsafe { ffi::whisper_vad_probs(self.ctx) };
        if probs_ptr.is_null() {
            return Vec::new();
        }

        let slice = unsafe { std::slice::from_raw_parts(probs_ptr, n as usize) };
        slice.to_vec()
    }

    /// Get speech segments from probability values
    pub fn segments_from_probs(&mut self, params: &VadParams) -> Result<VadSegments> {
        let segments_ptr = unsafe {
            ffi::whisper_vad_segments_from_probs(self.ctx, params.to_ffi())
        };

        if segments_ptr.is_null() {
            return Err(WhisperError::InvalidContext);
        }

        Ok(VadSegments {
            ptr: segments_ptr,
        })
    }

    /// Get speech segments directly from audio samples
    pub fn segments_from_samples(
        &mut self,
        samples: &[f32],
        params: &VadParams,
    ) -> Result<VadSegments> {
        if samples.is_empty() {
            return Err(WhisperError::InvalidAudioFormat);
        }

        let segments_ptr = unsafe {
            ffi::whisper_vad_segments_from_samples(
                self.ctx,
                params.to_ffi(),
                samples.as_ptr(),
                samples.len() as i32,
            )
        };

        if segments_ptr.is_null() {
            return Err(WhisperError::InvalidContext);
        }

        Ok(VadSegments {
            ptr: segments_ptr,
        })
    }
}

/// Speech segments detected by VAD
pub struct VadSegments {
    ptr: *mut ffi::whisper_vad_segments,
}

impl Drop for VadSegments {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.is_null() {
                ffi::whisper_vad_free_segments(self.ptr);
            }
        }
    }
}

impl VadSegments {
    /// Get the number of segments
    pub fn n_segments(&self) -> i32 {
        unsafe { ffi::whisper_vad_segments_n_segments(self.ptr) }
    }

    /// Get segment start time in seconds
    pub fn get_segment_t0(&self, i_segment: i32) -> f32 {
        // The FFI returns time in centiseconds, convert to seconds
        unsafe { ffi::whisper_vad_segments_get_segment_t0(self.ptr, i_segment) / 100.0 }
    }

    /// Get segment end time in seconds
    pub fn get_segment_t1(&self, i_segment: i32) -> f32 {
        // The FFI returns time in centiseconds, convert to seconds
        unsafe { ffi::whisper_vad_segments_get_segment_t1(self.ptr, i_segment) / 100.0 }
    }

    /// Get all segments as tuples of (start, end) times in seconds
    pub fn get_all_segments(&self) -> Vec<(f32, f32)> {
        let n = self.n_segments();
        let mut segments = Vec::with_capacity(n as usize);

        for i in 0..n {
            segments.push((self.get_segment_t0(i), self.get_segment_t1(i)));
        }

        segments
    }

    /// Extract audio segments from the original audio based on VAD segments
    pub fn extract_audio_segments(&self, audio: &[f32], sample_rate: f32) -> Vec<Vec<f32>> {
        let segments = self.get_all_segments();
        let mut audio_segments = Vec::with_capacity(segments.len());

        for (start, end) in segments {
            let start_sample = (start * sample_rate) as usize;
            let end_sample = (end * sample_rate) as usize;

            if start_sample < audio.len() && end_sample <= audio.len() {
                audio_segments.push(audio[start_sample..end_sample].to_vec());
            }
        }

        audio_segments
    }
}

/// Builder for VadParams
pub struct VadParamsBuilder {
    params: VadParams,
}

impl VadParamsBuilder {
    /// Create a new builder with default values
    pub fn new() -> Self {
        Self {
            params: VadParams::default(),
        }
    }

    /// Set the probability threshold (0.0 - 1.0)
    pub fn threshold(mut self, threshold: f32) -> Self {
        self.params.threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set minimum speech duration in milliseconds
    pub fn min_speech_duration_ms(mut self, ms: i32) -> Self {
        self.params.min_speech_duration_ms = ms.max(0);
        self
    }

    /// Set minimum silence duration in milliseconds
    pub fn min_silence_duration_ms(mut self, ms: i32) -> Self {
        self.params.min_silence_duration_ms = ms.max(0);
        self
    }

    /// Set maximum speech duration in seconds
    pub fn max_speech_duration_s(mut self, seconds: f32) -> Self {
        self.params.max_speech_duration_s = seconds.max(0.0);
        self
    }

    /// Set speech padding in milliseconds
    pub fn speech_pad_ms(mut self, ms: i32) -> Self {
        self.params.speech_pad_ms = ms.max(0);
        self
    }

    /// Set samples overlap
    pub fn samples_overlap(mut self, overlap: f32) -> Self {
        self.params.samples_overlap = overlap.max(0.0);
        self
    }

    /// Build the parameters
    pub fn build(self) -> VadParams {
        self.params
    }
}

impl Default for VadParamsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vad_params_default() {
        let params = VadParams::default();
        assert!(params.threshold > 0.0 && params.threshold < 1.0);
        assert!(params.min_speech_duration_ms >= 0);
        assert!(params.max_speech_duration_s > 0.0);
    }

    #[test]
    fn test_vad_params_builder() {
        let params = VadParamsBuilder::new()
            .threshold(0.6)
            .min_speech_duration_ms(250)
            .min_silence_duration_ms(100)
            .max_speech_duration_s(30.0)
            .speech_pad_ms(100)
            .build();

        assert_eq!(params.threshold, 0.6);
        assert_eq!(params.min_speech_duration_ms, 250);
        assert_eq!(params.min_silence_duration_ms, 100);
        assert_eq!(params.max_speech_duration_s, 30.0);
        assert_eq!(params.speech_pad_ms, 100);
    }

    #[test]
    fn test_vad_params_builder_clamps() {
        let params = VadParamsBuilder::new()
            .threshold(1.5) // Should be clamped to 1.0
            .min_speech_duration_ms(-100) // Should be clamped to 0
            .build();

        assert_eq!(params.threshold, 1.0);
        assert_eq!(params.min_speech_duration_ms, 0);
    }

    #[test]
    fn test_vad_processor_creation() {
        // This test will only run if a VAD model is available
        let model_path = "tests/models/ggml-silero-vad.bin";
        if Path::new(model_path).exists() {
            let processor = VadProcessor::new(model_path);
            assert!(processor.is_ok());
        }
    }

    #[test]
    fn test_vad_context_params() {
        let params = VadContextParams::default();
        assert!(params.n_threads > 0);

        let custom_params = VadContextParams {
            n_threads: 4,
            use_gpu: true,
            gpu_device: 0,
        };
        assert_eq!(custom_params.n_threads, 4);
        assert!(custom_params.use_gpu);
        assert_eq!(custom_params.gpu_device, 0);
    }
}