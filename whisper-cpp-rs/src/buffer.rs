//! Audio buffer utilities for streaming support

use std::collections::VecDeque;

/// A circular audio buffer for streaming transcription
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    buffer: VecDeque<f32>,
    max_capacity: usize,
}

impl AudioBuffer {
    /// Create a new audio buffer with specified maximum capacity
    pub fn new(max_capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(max_capacity),
            max_capacity,
        }
    }

    /// Add audio samples to the buffer
    pub fn push_samples(&mut self, samples: &[f32]) {
        // If adding these samples would exceed capacity, remove old samples
        let space_needed = samples.len().saturating_sub(self.available_space());
        if space_needed > 0 {
            self.buffer.drain(..space_needed.min(self.buffer.len()));
        }

        self.buffer.extend(samples);
    }

    /// Get the number of samples currently in the buffer
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Get available space in the buffer
    pub fn available_space(&self) -> usize {
        self.max_capacity.saturating_sub(self.buffer.len())
    }

    /// Extract a chunk of audio from the buffer
    /// Returns None if not enough samples are available
    pub fn extract_chunk(&mut self, size: usize, keep_overlap: usize) -> Option<Vec<f32>> {
        if self.buffer.len() < size {
            return None;
        }

        // Collect the chunk
        let chunk: Vec<f32> = self.buffer.iter().take(size).copied().collect();

        // Remove processed samples, keeping overlap
        let to_remove = size.saturating_sub(keep_overlap);
        self.buffer.drain(..to_remove);

        Some(chunk)
    }

    /// Clear all samples from the buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Drain all samples from the buffer
    pub fn drain_all(&mut self) -> Vec<f32> {
        self.buffer.drain(..).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_buffer_push_and_extract() {
        let mut buffer = AudioBuffer::new(100);

        // Push samples
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        buffer.push_samples(&samples);
        assert_eq!(buffer.len(), 5);

        // Extract chunk with overlap
        let chunk = buffer.extract_chunk(4, 2);
        assert!(chunk.is_some());
        assert_eq!(chunk.unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(buffer.len(), 3); // 2 samples kept as overlap + 1 remaining
    }

    #[test]
    fn test_audio_buffer_overflow() {
        let mut buffer = AudioBuffer::new(5);

        // Fill buffer to capacity
        buffer.push_samples(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(buffer.len(), 5);

        // Push more samples, should remove oldest
        buffer.push_samples(&[6.0, 7.0]);
        assert_eq!(buffer.len(), 5);
        // Verify newest samples are at the end via extract
        let chunk = buffer.extract_chunk(5, 0).unwrap();
        assert_eq!(chunk, vec![3.0, 4.0, 5.0, 6.0, 7.0]);
    }
}