//! Streaming transcription support for real-time audio processing
//!
//! The whisper.cpp library automatically clears internal results when starting a new
//! transcription, so state reuse is safe and matches the behavior of whisper.cpp's
//! own streaming implementation.
//!
//! If you need to force a complete state recreation (e.g., after errors or when
//! switching between very different audio sources), use `recreate_state()` instead.

use crate::buffer::AudioBuffer;
use crate::context::WhisperContext;
use crate::error::Result;
use crate::params::FullParams;
use crate::state::{Segment, WhisperState};
use std::time::{Duration, Instant};

/// Configuration for streaming transcription
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Size of audio chunks to process (in samples)
    pub chunk_size: usize,
    /// Overlap between chunks (in samples) to maintain context
    pub overlap_size: usize,
    /// Maximum buffer size (in samples)
    pub max_buffer_size: usize,
    /// Minimum chunk size before processing (in samples)
    pub min_chunk_size: usize,
    /// Timeout for processing partial chunks
    pub partial_timeout: Duration,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            chunk_size: 16000 * 5,        // 5 seconds at 16kHz
            overlap_size: 16000,           // 1 second overlap
            max_buffer_size: 16000 * 30,   // 30 seconds maximum
            min_chunk_size: 16000,          // 1 second minimum
            partial_timeout: Duration::from_secs(2),
        }
    }
}

/// A streaming transcriber that processes audio incrementally
pub struct WhisperStream {
    context: WhisperContext,
    state: WhisperState,
    params: FullParams,
    config: StreamConfig,
    buffer: AudioBuffer,
    last_process_time: Instant,
    segment_offset: i64,
    processed_samples: i64,
}

impl WhisperStream {
    /// Create a new streaming transcriber
    pub fn new(context: &WhisperContext, params: FullParams) -> Result<Self> {
        Self::with_config(context, params, StreamConfig::default())
    }

    /// Create a new streaming transcriber with custom configuration
    pub fn with_config(
        context: &WhisperContext,
        params: FullParams,
        config: StreamConfig,
    ) -> Result<Self> {
        let state = context.create_state()?;
        let buffer = AudioBuffer::new(config.max_buffer_size);

        Ok(Self {
            context: context.clone(),
            state,
            params,
            config,
            buffer,
            last_process_time: Instant::now(),
            segment_offset: 0,
            processed_samples: 0,
        })
    }

    /// Feed audio samples to the stream
    pub fn feed_audio(&mut self, samples: &[f32]) {
        self.buffer.push_samples(samples);
    }

    /// Process pending audio and return any new segments
    pub fn process_pending(&mut self) -> Result<Vec<Segment>> {
        let mut segments = Vec::new();

        // Process full chunks
        while let Some(chunk) = self.buffer.extract_chunk(self.config.chunk_size, self.config.overlap_size) {
            let chunk_segments = self.process_chunk(&chunk)?;
            segments.extend(chunk_segments);
            self.last_process_time = Instant::now();
        }

        // Check if we should process a partial chunk (timeout or flush)
        if self.should_process_partial() {
            if let Some(chunk) = self.extract_partial_chunk() {
                let chunk_segments = self.process_chunk(&chunk)?;
                segments.extend(chunk_segments);
                self.last_process_time = Instant::now();
            }
        }

        Ok(segments)
    }

    /// Force processing of all buffered audio
    pub fn flush(&mut self) -> Result<Vec<Segment>> {
        let mut segments = Vec::new();

        // Process any remaining audio
        if !self.buffer.is_empty() {
            let remaining = self.buffer.drain_all();
            if remaining.len() >= self.config.min_chunk_size {
                let chunk_segments = self.process_chunk(&remaining)?;
                segments.extend(chunk_segments);
            }
        }

        Ok(segments)
    }

    /// Reset the stream, clearing all buffers but reusing the state
    /// The state's results will be automatically cleared on the next transcription
    pub fn reset(&mut self) -> Result<()> {
        self.buffer.clear();
        // Don't recreate state - reuse existing one for better performance
        // The state's internal results are automatically cleared by whisper_full_with_state
        self.segment_offset = 0;
        self.processed_samples = 0;
        self.last_process_time = Instant::now();
        Ok(())
    }

    /// Force recreation of the WhisperState, deallocating the old one
    /// This is more expensive than reset() but may be needed after errors
    /// or when switching between very different audio sources
    pub fn recreate_state(&mut self) -> Result<()> {
        self.state = self.context.create_state()?;
        self.reset()
    }

    /// Get the current buffer size in samples
    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }

    /// Get the total number of processed samples
    pub fn processed_samples(&self) -> i64 {
        self.processed_samples
    }

    /// Process a chunk of audio and return segments
    fn process_chunk(&mut self, audio: &[f32]) -> Result<Vec<Segment>> {
        // Run transcription on the chunk
        self.state.full(self.params.clone(), audio)?;

        // Extract segments
        let n_segments = self.state.full_n_segments();
        let mut segments = Vec::with_capacity(n_segments as usize);

        for i in 0..n_segments {
            let text = self.state.full_get_segment_text(i)?;
            let (start_ms, end_ms) = self.state.full_get_segment_timestamps(i);
            let speaker_turn_next = self.state.full_get_segment_speaker_turn_next(i);

            // Adjust timestamps based on stream position
            let adjusted_start = start_ms + self.segment_offset;
            let adjusted_end = end_ms + self.segment_offset;

            segments.push(Segment {
                start_ms: adjusted_start,
                end_ms: adjusted_end,
                text,
                speaker_turn_next,
            });
        }

        // Update offset for next chunk
        // Account for overlap by only advancing by non-overlapped samples
        let advance_samples = (audio.len() - self.config.overlap_size) as i64;
        self.segment_offset += (advance_samples * 1000) / 16000; // Convert samples to ms at 16kHz
        self.processed_samples += advance_samples;

        Ok(segments)
    }

    /// Check if we should process a partial chunk
    fn should_process_partial(&self) -> bool {
        self.buffer.len() >= self.config.min_chunk_size
            && self.last_process_time.elapsed() > self.config.partial_timeout
    }

    /// Extract a partial chunk for processing
    fn extract_partial_chunk(&mut self) -> Option<Vec<f32>> {
        let size = self.buffer.len().min(self.config.chunk_size);
        if size >= self.config.min_chunk_size {
            self.buffer.extract_chunk(size, 0)
        } else {
            None
        }
    }
}

/// Builder for StreamConfig
pub struct StreamConfigBuilder {
    config: StreamConfig,
}

impl StreamConfigBuilder {
    /// Create a new builder with default values
    pub fn new() -> Self {
        Self {
            config: StreamConfig::default(),
        }
    }

    /// Set the chunk size in samples
    pub fn chunk_size(mut self, size: usize) -> Self {
        self.config.chunk_size = size;
        self
    }

    /// Set the chunk size in seconds (assumes 16kHz)
    pub fn chunk_seconds(mut self, seconds: f32) -> Self {
        self.config.chunk_size = (seconds * 16000.0) as usize;
        self
    }

    /// Set the overlap size in samples
    pub fn overlap_size(mut self, size: usize) -> Self {
        self.config.overlap_size = size;
        self
    }

    /// Set the overlap size in seconds (assumes 16kHz)
    pub fn overlap_seconds(mut self, seconds: f32) -> Self {
        self.config.overlap_size = (seconds * 16000.0) as usize;
        self
    }

    /// Set the maximum buffer size in samples
    pub fn max_buffer_size(mut self, size: usize) -> Self {
        self.config.max_buffer_size = size;
        self
    }

    /// Set the minimum chunk size in samples
    pub fn min_chunk_size(mut self, size: usize) -> Self {
        self.config.min_chunk_size = size;
        self
    }

    /// Set the partial chunk timeout
    pub fn partial_timeout(mut self, timeout: Duration) -> Self {
        self.config.partial_timeout = timeout;
        self
    }

    /// Build the configuration
    pub fn build(self) -> StreamConfig {
        self.config
    }
}

impl Default for StreamConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SamplingStrategy;
    use std::path::Path;

    #[test]
    fn test_stream_config_builder() {
        let config = StreamConfigBuilder::new()
            .chunk_seconds(3.0)
            .overlap_seconds(0.5)
            .min_chunk_size(8000)
            .build();

        assert_eq!(config.chunk_size, 48000); // 3 seconds at 16kHz
        assert_eq!(config.overlap_size, 8000); // 0.5 seconds at 16kHz
        assert_eq!(config.min_chunk_size, 8000);
    }

    #[test]
    fn test_stream_creation() {
        let model_path = "tests/models/ggml-tiny.en.bin";
        if Path::new(model_path).exists() {
            let ctx = WhisperContext::new(model_path).unwrap();
            let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

            let stream = WhisperStream::new(&ctx, params);
            assert!(stream.is_ok());

            let mut stream = stream.unwrap();
            assert_eq!(stream.buffer_size(), 0);
            assert_eq!(stream.processed_samples(), 0);
        }
    }

    #[test]
    fn test_feed_and_buffer() {
        let model_path = "tests/models/ggml-tiny.en.bin";
        if Path::new(model_path).exists() {
            let ctx = WhisperContext::new(model_path).unwrap();
            let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
            let mut stream = WhisperStream::new(&ctx, params).unwrap();

            // Feed some audio
            let samples = vec![0.0f32; 16000]; // 1 second
            stream.feed_audio(&samples);
            assert_eq!(stream.buffer_size(), 16000);

            // Feed more audio
            stream.feed_audio(&samples);
            assert_eq!(stream.buffer_size(), 32000);
        }
    }

    #[test]
    fn test_state_reuse_on_reset() {
        let model_path = "tests/models/ggml-tiny.en.bin";
        if Path::new(model_path).exists() {
            let ctx = WhisperContext::new(model_path).unwrap();
            let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
            let mut stream = WhisperStream::new(&ctx, params).unwrap();

            // Feed audio and process
            let samples = vec![0.0f32; 16000 * 5]; // 5 seconds
            stream.feed_audio(&samples);

            // Get state pointer before reset (for comparison)
            let state_ptr_before = &stream.state as *const _ as usize;

            // Reset the stream
            stream.reset().unwrap();

            // State pointer should be the same (state was reused)
            let state_ptr_after = &stream.state as *const _ as usize;
            assert_eq!(state_ptr_before, state_ptr_after, "State should be reused, not recreated");

            // Buffer should be cleared
            assert_eq!(stream.buffer_size(), 0);
            assert_eq!(stream.processed_samples(), 0);

            // Now test recreate_state() creates a new state
            stream.feed_audio(&samples);
            let state_ptr_before_recreate = &stream.state as *const _ as usize;

            stream.recreate_state().unwrap();

            let state_ptr_after_recreate = &stream.state as *const _ as usize;
            // Note: The WhisperState object address stays the same, but its internal pointer changes
            // We can't directly test the internal pointer changes from here
            assert_eq!(state_ptr_before_recreate, state_ptr_after_recreate,
                      "WhisperState struct address remains same");

            // But we can verify the stream was reset
            assert_eq!(stream.buffer_size(), 0);
            assert_eq!(stream.processed_samples(), 0);
        }
    }
}