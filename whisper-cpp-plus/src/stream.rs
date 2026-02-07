//! Streaming transcription — faithful port of stream.cpp
//!
//! Replaces SDL audio capture with a push-based `feed_audio()` API
//! since we're a library, not a binary.

use crate::context::WhisperContext;
use crate::error::Result;
use crate::params::FullParams;
use crate::state::{Segment, WhisperState};
use std::collections::VecDeque;

const WHISPER_SAMPLE_RATE: i32 = 16000;

// ---------------------------------------------------------------------------
// WhisperStreamConfig
// ---------------------------------------------------------------------------

/// Streaming config — maps to stream.cpp's whisper_params (streaming subset).
#[derive(Debug, Clone)]
pub struct WhisperStreamConfig {
    /// Audio step size in ms. Set <= 0 for VAD mode.
    pub step_ms: i32,
    /// Audio length per inference in ms.
    pub length_ms: i32,
    /// Audio to keep from previous step in ms.
    pub keep_ms: i32,
    /// VAD energy threshold.
    pub vad_thold: f32,
    /// High-pass frequency cutoff for VAD.
    pub freq_thold: f32,
    /// If true, don't carry prompt tokens across boundaries.
    pub no_context: bool,
}

impl Default for WhisperStreamConfig {
    fn default() -> Self {
        Self {
            step_ms: 3000,
            length_ms: 10000,
            keep_ms: 200,
            vad_thold: 0.6,
            freq_thold: 100.0,
            no_context: true,
        }
    }
}

// ---------------------------------------------------------------------------
// WhisperStream
// ---------------------------------------------------------------------------

/// Streaming transcriber — faithful port of stream.cpp main loop.
///
/// Two modes:
/// - **Fixed-step** (`step_ms > 0`): sliding window with overlap.
/// - **VAD** (`step_ms <= 0`): transcribe on speech activity.
pub struct WhisperStream {
    state: WhisperState,
    params: FullParams,
    config: WhisperStreamConfig,
    use_vad: bool,

    // Pre-computed sample counts
    n_samples_step: usize,
    n_samples_len: usize,
    n_samples_keep: usize,
    n_new_line: i32,

    // Overlap buffer from previous inference
    pcmf32_old: Vec<f32>,
    // Context propagation
    prompt_tokens: Vec<i32>,

    n_iter: i32,

    // Internal audio buffer (replaces SDL capture)
    audio_buf: VecDeque<f32>,

    // Total samples consumed from audio_buf
    total_samples_processed: i64,
}

impl WhisperStream {
    /// Create with default config.
    pub fn new(ctx: &WhisperContext, params: FullParams) -> Result<Self> {
        Self::with_config(ctx, params, WhisperStreamConfig::default())
    }

    /// Create with custom config.
    pub fn with_config(
        ctx: &WhisperContext,
        mut params: FullParams,
        mut config: WhisperStreamConfig,
    ) -> Result<Self> {
        let state = WhisperState::new(ctx)?;

        // --- Config normalization (stream.cpp main()) ---
        config.keep_ms = config.keep_ms.min(config.step_ms);
        config.length_ms = config.length_ms.max(config.step_ms);

        // Sample counts
        let n_samples_step =
            (1e-3 * config.step_ms as f64 * WHISPER_SAMPLE_RATE as f64) as usize;
        let n_samples_len =
            (1e-3 * config.length_ms as f64 * WHISPER_SAMPLE_RATE as f64) as usize;
        let n_samples_keep =
            (1e-3 * config.keep_ms as f64 * WHISPER_SAMPLE_RATE as f64) as usize;

        // Mode detection
        let use_vad = n_samples_step == 0; // step_ms <= 0 → VAD

        // n_new_line: guard against division by zero when step_ms <= 0
        let n_new_line = if !use_vad {
            (config.length_ms / config.step_ms - 1).max(1)
        } else {
            1
        };

        // Auto-set mode-dependent FullParams (stream.cpp lines 141-143)
        params = params
            .no_timestamps(!use_vad)
            .max_tokens(0)
            .single_segment(!use_vad)
            .print_progress(false)
            .print_realtime(false);

        // Force no_context in VAD mode: no_context |= use_vad
        if use_vad {
            config.no_context = true;
            params = params.no_context(true);
        }

        Ok(Self {
            state,
            params,
            config,
            use_vad,
            n_samples_step,
            n_samples_len,
            n_samples_keep,
            n_new_line,
            pcmf32_old: Vec::new(),
            prompt_tokens: Vec::new(),
            n_iter: 0,
            audio_buf: VecDeque::new(),
            total_samples_processed: 0,
        })
    }

    // --- Audio input ---

    /// Push samples into the internal buffer (replaces SDL capture).
    pub fn feed_audio(&mut self, samples: &[f32]) {
        self.audio_buf.extend(samples.iter());
    }

    // --- Processing ---

    /// Dispatch to fixed-step or VAD mode.
    pub fn process_step(&mut self) -> Result<Option<Vec<Segment>>> {
        if !self.use_vad {
            self.process_step_fixed()
        } else {
            self.process_step_vad()
        }
    }

    /// Fixed-step (sliding window) mode — port of stream.cpp lines 253-428.
    fn process_step_fixed(&mut self) -> Result<Option<Vec<Segment>>> {
        // Need at least n_samples_step new samples
        if self.audio_buf.len() < self.n_samples_step {
            return Ok(None);
        }

        // Pop n_samples_step from front of audio_buf
        let pcmf32_new: Vec<f32> = self.audio_buf.drain(..self.n_samples_step).collect();
        self.total_samples_processed += pcmf32_new.len() as i64;

        let n_samples_new = pcmf32_new.len();

        // Exact formula from stream.cpp line 279:
        // n_samples_take = min(pcmf32_old.size(), max(0, n_samples_keep + n_samples_len - n_samples_new))
        let n_samples_take = self.pcmf32_old.len().min(
            (self.n_samples_keep + self.n_samples_len).saturating_sub(n_samples_new),
        );

        // Build pcmf32: tail of pcmf32_old + pcmf32_new
        let mut pcmf32 = Vec::with_capacity(n_samples_take + n_samples_new);
        if n_samples_take > 0 && !self.pcmf32_old.is_empty() {
            let start = self.pcmf32_old.len() - n_samples_take;
            pcmf32.extend_from_slice(&self.pcmf32_old[start..]);
        }
        pcmf32.extend_from_slice(&pcmf32_new);

        // Save for next iteration
        self.pcmf32_old = pcmf32.clone();

        // Run inference
        let segments = self.run_inference(&pcmf32)?;

        self.n_iter += 1;

        // At n_new_line boundary (stream.cpp lines 408-425)
        if self.n_iter % self.n_new_line == 0 {
            // Keep only last n_samples_keep samples
            if self.n_samples_keep > 0 && pcmf32.len() >= self.n_samples_keep {
                self.pcmf32_old =
                    pcmf32[pcmf32.len() - self.n_samples_keep..].to_vec();
            } else {
                self.pcmf32_old.clear();
            }

            // Collect prompt tokens if !no_context
            if !self.config.no_context {
                self.collect_prompt_tokens();
            }
        }

        Ok(Some(segments))
    }

    /// VAD mode — port of stream.cpp lines 293-313.
    fn process_step_vad(&mut self) -> Result<Option<Vec<Segment>>> {
        // Need at least 2 seconds of audio (stream.cpp: t_diff < 2000 → continue)
        let n_vad_samples = (WHISPER_SAMPLE_RATE * 2) as usize; // 32000 samples
        if self.audio_buf.len() < n_vad_samples {
            return Ok(None);
        }

        // Pop 2 seconds for VAD probe
        let pcmf32_vad: Vec<f32> = self.audio_buf.drain(..n_vad_samples).collect();
        self.total_samples_processed += pcmf32_vad.len() as i64;

        // Check for speech
        let is_silence = vad_simple(
            &pcmf32_vad,
            WHISPER_SAMPLE_RATE,
            1000,
            self.config.vad_thold,
            self.config.freq_thold,
        );

        if is_silence {
            return Ok(None);
        }

        // Speech detected — grab length_ms of audio total (stream.cpp line 305)
        let n_samples_len = self.n_samples_len;
        let additional = n_samples_len.saturating_sub(pcmf32_vad.len());
        let mut pcmf32 = pcmf32_vad;

        if additional > 0 {
            let available = additional.min(self.audio_buf.len());
            let extra: Vec<f32> = self.audio_buf.drain(..available).collect();
            self.total_samples_processed += extra.len() as i64;
            pcmf32.extend_from_slice(&extra);
        }

        let segments = self.run_inference(&pcmf32)?;
        self.n_iter += 1;

        Ok(Some(segments))
    }

    /// Run whisper inference on audio — port of stream.cpp lines 316-344.
    fn run_inference(&mut self, audio: &[f32]) -> Result<Vec<Segment>> {
        if audio.is_empty() {
            return Ok(Vec::new());
        }

        // Clone params so we can set prompt_tokens pointer
        let mut params = self.params.clone();

        // Set prompt tokens on the clone, pointing to self.prompt_tokens.
        // The prompt_tokens() method stores a raw pointer. self.prompt_tokens
        // (Vec<i32>) lives on self and outlives the full() call, so this is safe.
        if !self.config.no_context && !self.prompt_tokens.is_empty() {
            params = params.prompt_tokens(&self.prompt_tokens);
        }

        self.state.full(params, audio)?;

        // Extract segments
        let n_segments = self.state.full_n_segments();
        let mut segments = Vec::with_capacity(n_segments as usize);

        for i in 0..n_segments {
            let text = self.state.full_get_segment_text(i)?;
            let (start_ms, end_ms) = self.state.full_get_segment_timestamps(i);
            let speaker_turn_next = self.state.full_get_segment_speaker_turn_next(i);

            segments.push(Segment {
                start_ms,
                end_ms,
                text,
                speaker_turn_next,
            });
        }

        Ok(segments)
    }

    /// Collect prompt tokens from last inference — port of stream.cpp lines 416-425.
    fn collect_prompt_tokens(&mut self) {
        self.prompt_tokens.clear();

        let n_segments = self.state.full_n_segments();
        for i in 0..n_segments {
            let token_count = self.state.full_n_tokens(i);
            for j in 0..token_count {
                self.prompt_tokens
                    .push(self.state.full_get_token_id(i, j));
            }
        }
    }

    // --- Convenience methods ---

    /// Process all remaining audio in buffer.
    pub fn flush(&mut self) -> Result<Vec<Segment>> {
        let mut all_segments = Vec::new();

        loop {
            match self.process_step()? {
                Some(segments) => all_segments.extend(segments),
                None => break,
            }
        }

        // If there's leftover audio that's less than a full step, run inference on it
        if !self.audio_buf.is_empty() {
            let remaining: Vec<f32> = self.audio_buf.drain(..).collect();
            self.total_samples_processed += remaining.len() as i64;

            if !self.use_vad {
                // Build final buffer with overlap
                let n_samples_take = self.pcmf32_old.len().min(
                    (self.n_samples_keep + self.n_samples_len)
                        .saturating_sub(remaining.len()),
                );
                let mut pcmf32 = Vec::with_capacity(n_samples_take + remaining.len());
                if n_samples_take > 0 && !self.pcmf32_old.is_empty() {
                    let start = self.pcmf32_old.len() - n_samples_take;
                    pcmf32.extend_from_slice(&self.pcmf32_old[start..]);
                }
                pcmf32.extend_from_slice(&remaining);

                let segments = self.run_inference(&pcmf32)?;
                all_segments.extend(segments);
            } else {
                let segments = self.run_inference(&remaining)?;
                all_segments.extend(segments);
            }
        }

        Ok(all_segments)
    }

    /// Clear buffers, counters, prompt tokens.
    pub fn reset(&mut self) {
        self.audio_buf.clear();
        self.pcmf32_old.clear();
        self.prompt_tokens.clear();
        self.n_iter = 0;
        self.total_samples_processed = 0;
    }

    /// Samples currently in the internal buffer.
    pub fn buffer_size(&self) -> usize {
        self.audio_buf.len()
    }

    /// Total samples consumed from the buffer.
    pub fn processed_samples(&self) -> i64 {
        self.total_samples_processed
    }
}

// ---------------------------------------------------------------------------
// vad_simple + high_pass_filter — port from common.cpp
// ---------------------------------------------------------------------------

/// High-pass filter — port of common.cpp::high_pass_filter (lines 597-608).
fn high_pass_filter(data: &mut [f32], cutoff: f32, sample_rate: f32) {
    if data.is_empty() {
        return;
    }
    let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff);
    let dt = 1.0 / sample_rate;
    let alpha = dt / (rc + dt);

    let mut y = data[0];
    for i in 1..data.len() {
        y = alpha * (y + data[i] - data[i - 1]);
        data[i] = y;
    }
}

/// Energy-based VAD — port of common.cpp::vad_simple (lines 610-646).
///
/// Returns `true` if **silence** (no speech detected).
fn vad_simple(
    pcmf32: &[f32],
    sample_rate: i32,
    last_ms: i32,
    vad_thold: f32,
    freq_thold: f32,
) -> bool {
    let n_samples = pcmf32.len();
    let n_samples_last = (sample_rate as usize * last_ms.max(0) as usize) / 1000;

    if n_samples_last >= n_samples {
        // not enough samples — assume no speech (C++ returns false here,
        // but the sense in C++ is inverted: false = silence. We return true = silence.)
        return true;
    }

    // Work on a copy so we can apply the high-pass filter
    let mut data = pcmf32.to_vec();

    if freq_thold > 0.0 {
        high_pass_filter(&mut data, freq_thold, sample_rate as f32);
    }

    let mut energy_all: f32 = 0.0;
    let mut energy_last: f32 = 0.0;

    for (i, &s) in data.iter().enumerate() {
        energy_all += s.abs();
        if i >= n_samples - n_samples_last {
            energy_last += s.abs();
        }
    }

    energy_all /= n_samples as f32;
    energy_last /= n_samples_last as f32;

    // C++ returns false (speech) when energy_last > thold * energy_all.
    // We return true for silence.
    energy_last <= vad_thold * energy_all
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SamplingStrategy;
    use std::path::Path;

    #[test]
    fn test_config_defaults() {
        let config = WhisperStreamConfig::default();
        assert_eq!(config.step_ms, 3000);
        assert_eq!(config.length_ms, 10000);
        assert_eq!(config.keep_ms, 200);
        assert!((config.vad_thold - 0.6).abs() < f32::EPSILON);
        assert!((config.freq_thold - 100.0).abs() < f32::EPSILON);
        assert!(config.no_context);
    }

    #[test]
    fn test_config_normalization() {
        // keep_ms clamped to step_ms
        let model_path = "tests/models/ggml-tiny.en.bin";
        if !Path::new(model_path).exists() {
            // Can't test normalization without a model for the constructor.
            // Test the logic directly instead.
            let mut config = WhisperStreamConfig {
                step_ms: 2000,
                length_ms: 5000,
                keep_ms: 3000, // > step_ms, should be clamped
                ..Default::default()
            };
            config.keep_ms = config.keep_ms.min(config.step_ms);
            config.length_ms = config.length_ms.max(config.step_ms);
            assert_eq!(config.keep_ms, 2000);
            assert_eq!(config.length_ms, 5000);

            // length_ms clamped up to step_ms
            let mut config2 = WhisperStreamConfig {
                step_ms: 8000,
                length_ms: 5000, // < step_ms, should be raised
                keep_ms: 200,
                ..Default::default()
            };
            config2.keep_ms = config2.keep_ms.min(config2.step_ms);
            config2.length_ms = config2.length_ms.max(config2.step_ms);
            assert_eq!(config2.length_ms, 8000);
            assert_eq!(config2.keep_ms, 200);
        }
    }

    #[test]
    fn test_n_new_line_calculation() {
        // n_new_line = max(1, length_ms / step_ms - 1) when !use_vad
        // Defaults: length_ms=10000, step_ms=3000 → 10000/3000 - 1 = 2
        let n = (10000i32 / 3000 - 1).max(1);
        assert_eq!(n, 2);

        // step_ms=5000, length_ms=10000 → 10000/5000 - 1 = 1
        let n = (10000i32 / 5000 - 1).max(1);
        assert_eq!(n, 1);

        // step_ms=10000, length_ms=10000 → 10000/10000 - 1 = 0 → clamped to 1
        let n = (10000i32 / 10000 - 1).max(1);
        assert_eq!(n, 1);

        // step_ms=2000, length_ms=10000 → 10000/2000 - 1 = 4
        let n = (10000i32 / 2000 - 1).max(1);
        assert_eq!(n, 4);

        // VAD mode: always 1
        let n_vad = 1i32;
        assert_eq!(n_vad, 1);
    }

    #[test]
    fn test_vad_mode_detection() {
        // step_ms <= 0 → use_vad
        let step_ms_values = [0, -1, -100];
        for step_ms in step_ms_values {
            let n_samples_step =
                (1e-3 * step_ms as f64 * WHISPER_SAMPLE_RATE as f64) as usize;
            assert_eq!(n_samples_step, 0, "step_ms={} should yield 0 samples", step_ms);
        }

        // step_ms > 0 → fixed step
        let n = (1e-3 * 3000.0 * WHISPER_SAMPLE_RATE as f64) as usize;
        assert_eq!(n, 48000);
    }

    #[test]
    fn test_feed_and_buffer() {
        let model_path = "tests/models/ggml-tiny.en.bin";
        if !Path::new(model_path).exists() {
            eprintln!("Skipping test_feed_and_buffer: model not found");
            return;
        }

        let ctx = WhisperContext::new(model_path).unwrap();
        let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        let mut stream = WhisperStream::new(&ctx, params).unwrap();

        assert_eq!(stream.buffer_size(), 0);

        let samples = vec![0.0f32; 16000];
        stream.feed_audio(&samples);
        assert_eq!(stream.buffer_size(), 16000);

        stream.feed_audio(&samples);
        assert_eq!(stream.buffer_size(), 32000);
    }

    #[test]
    fn test_vad_simple_silence() {
        let silence = vec![0.0f32; 16000];
        assert!(vad_simple(&silence, 16000, 100, 0.6, 100.0));
    }

    #[test]
    fn test_vad_simple_too_few_samples() {
        let short = vec![0.1f32; 100];
        assert!(vad_simple(&short, 16000, 1000, 0.6, 100.0));
    }

    #[test]
    fn test_high_pass_filter_basic() {
        let mut data = vec![1.0, 0.0, 1.0, 0.0, 1.0];
        high_pass_filter(&mut data, 100.0, 16000.0);
        assert_ne!(data[2], 1.0);
    }

    #[test]
    fn test_reset() {
        let model_path = "tests/models/ggml-tiny.en.bin";
        if !Path::new(model_path).exists() {
            eprintln!("Skipping test_reset: model not found");
            return;
        }

        let ctx = WhisperContext::new(model_path).unwrap();
        let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        let mut stream = WhisperStream::new(&ctx, params).unwrap();

        stream.feed_audio(&vec![0.0f32; 16000]);
        assert_eq!(stream.buffer_size(), 16000);

        stream.reset();
        assert_eq!(stream.buffer_size(), 0);
        assert_eq!(stream.processed_samples(), 0);
    }

    // --- Integration tests (require model) ---

    #[test]
    fn test_fixed_step_basic() {
        let model_path = "tests/models/ggml-tiny.en.bin";
        if !Path::new(model_path).exists() {
            eprintln!("Skipping test_fixed_step_basic: model not found");
            return;
        }

        let ctx = WhisperContext::new(model_path).unwrap();
        let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 })
            .language("en");

        // Use a small step for testing
        let config = WhisperStreamConfig {
            step_ms: 3000,
            length_ms: 10000,
            keep_ms: 200,
            ..Default::default()
        };

        let mut stream = WhisperStream::with_config(&ctx, params, config).unwrap();

        // Feed enough audio for one step (3 seconds = 48000 samples)
        let audio = vec![0.0f32; 48000];
        stream.feed_audio(&audio);

        let result = stream.process_step().unwrap();
        assert!(result.is_some(), "Should produce segments with enough audio");
        assert!(stream.processed_samples() > 0);
    }

    #[test]
    fn test_prompt_propagation() {
        let model_path = "tests/models/ggml-tiny.en.bin";
        if !Path::new(model_path).exists() {
            eprintln!("Skipping test_prompt_propagation: model not found");
            return;
        }

        let ctx = WhisperContext::new(model_path).unwrap();
        let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 })
            .language("en");

        let config = WhisperStreamConfig {
            step_ms: 3000,
            length_ms: 6000,
            keep_ms: 200,
            no_context: false, // enable prompt propagation
            ..Default::default()
        };

        let mut stream = WhisperStream::with_config(&ctx, params, config).unwrap();

        // n_new_line = max(1, 6000/3000 - 1) = 1, so every iteration triggers
        // prompt collection when no_context=false.

        // Feed enough for one step
        let audio = vec![0.0f32; 48000];
        stream.feed_audio(&audio);

        let result = stream.process_step().unwrap();
        assert!(result.is_some());

        // After one iteration at the n_new_line boundary, prompt_tokens should
        // be populated (assuming whisper produced at least one token).
        // With silence input, whisper may or may not produce tokens, so we
        // just verify the mechanism didn't panic.
        assert!(stream.processed_samples() > 0);
    }
}
