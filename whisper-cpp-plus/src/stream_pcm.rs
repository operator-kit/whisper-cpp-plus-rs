//! Direct port of stream-pcm.cpp — streaming transcription from raw PCM input.
//!
//! Architecture: async PCM reader → ring buffer → fixed-step or VAD-driven processing.

use crate::context::WhisperContext;
use crate::error::{Result, WhisperError};
use crate::params::FullParams;
use crate::state::{Segment, WhisperState};
use crate::vad::WhisperVadProcessor;

use std::io::Read;
use std::sync::{Arc, Mutex};
use std::thread;

const WHISPER_SAMPLE_RATE: i32 = 16000;

// ---------------------------------------------------------------------------
// PcmFormat
// ---------------------------------------------------------------------------

/// Input PCM sample format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcmFormat {
    F32,
    S16,
}

// ---------------------------------------------------------------------------
// PcmReader — direct port of pcm_async
// ---------------------------------------------------------------------------

/// Configuration for [`PcmReader`].
#[derive(Debug, Clone)]
pub struct PcmReaderConfig {
    /// Ring buffer length in milliseconds (maps to `m_len_ms`).
    pub buffer_len_ms: i32,
    /// Sample rate (must be 16000).
    pub sample_rate: i32,
    /// Input PCM format.
    pub format: PcmFormat,
}

impl Default for PcmReaderConfig {
    fn default() -> Self {
        Self {
            buffer_len_ms: 10000,
            sample_rate: WHISPER_SAMPLE_RATE,
            format: PcmFormat::F32,
        }
    }
}

/// Shared ring-buffer state (behind Mutex, matching C++ `m_mutex`).
struct RingBuffer {
    audio: Vec<f32>,
    /// Write position in ring (next sample to write).
    audio_pos: usize,
    /// Total unread samples available.
    audio_len: usize,
    /// Read position in ring (next sample to pop).
    audio_read: usize,
    eof: bool,
}

/// Threaded PCM reader — direct port of `pcm_async`.
///
/// Reads raw PCM from any `Read` source on a background thread,
/// converts S16→f32 if needed, and fills a ring buffer.
pub struct PcmReader {
    shared: Arc<Mutex<RingBuffer>>,
    handle: Option<thread::JoinHandle<()>>,
    stop: Arc<std::sync::atomic::AtomicBool>,
}

impl PcmReader {
    /// Create and immediately start the reader thread.
    pub fn new(source: Box<dyn Read + Send>, config: PcmReaderConfig) -> Self {
        let ring_samples = (config.sample_rate as usize * config.buffer_len_ms as usize) / 1000;

        let shared = Arc::new(Mutex::new(RingBuffer {
            audio: vec![0.0; ring_samples],
            audio_pos: 0,
            audio_len: 0,
            audio_read: 0,
            eof: false,
        }));

        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));

        let shared_clone = Arc::clone(&shared);
        let stop_clone = Arc::clone(&stop);
        let format = config.format;

        let handle = thread::spawn(move || {
            reader_loop(source, shared_clone, stop_clone, format);
        });

        Self {
            shared,
            handle: Some(handle),
            stop,
        }
    }

    /// Pop up to `ms` milliseconds of audio from the ring buffer.
    /// Returns fewer samples if not enough are available.
    pub fn pop_ms(&self, ms: i32) -> Vec<f32> {
        let mut ring = self.shared.lock().unwrap();
        let n_samples = ((WHISPER_SAMPLE_RATE as usize) * ms.max(0) as usize) / 1000;
        let n = n_samples.min(ring.audio_len);

        if n == 0 {
            return Vec::new();
        }

        let mut result = vec![0.0f32; n];
        let cap = ring.audio.len();
        let s0 = ring.audio_read;

        if s0 + n > cap {
            let n0 = cap - s0;
            result[..n0].copy_from_slice(&ring.audio[s0..]);
            result[n0..].copy_from_slice(&ring.audio[..n - n0]);
        } else {
            result.copy_from_slice(&ring.audio[s0..s0 + n]);
        }

        ring.audio_read = (ring.audio_read + n) % cap;
        ring.audio_len -= n;
        result
    }

    /// Number of unread samples currently in the ring buffer.
    pub fn available_samples(&self) -> usize {
        self.shared.lock().unwrap().audio_len
    }

    /// Whether the source has reached EOF.
    pub fn is_eof(&self) -> bool {
        self.shared.lock().unwrap().eof
    }

    /// Signal the reader thread to stop.
    pub fn stop(&mut self) {
        self.stop
            .store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for PcmReader {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Background reader loop — direct port of `pcm_async::reader_loop`.
fn reader_loop(
    mut source: Box<dyn Read + Send>,
    shared: Arc<Mutex<RingBuffer>>,
    stop: Arc<std::sync::atomic::AtomicBool>,
    format: PcmFormat,
) {
    let bytes_per_sample: usize = match format {
        PcmFormat::F32 => 4,
        PcmFormat::S16 => 2,
    };

    let mut buffer = vec![0u8; 4096];
    let mut carry: Vec<u8> = Vec::new();

    loop {
        if stop.load(std::sync::atomic::Ordering::Relaxed) {
            break;
        }

        let n_read = match source.read(&mut buffer) {
            Ok(0) => {
                shared.lock().unwrap().eof = true;
                break;
            }
            Ok(n) => n,
            Err(_) => {
                shared.lock().unwrap().eof = true;
                break;
            }
        };

        // Combine carry bytes with freshly read bytes
        let mut data = Vec::with_capacity(carry.len() + n_read);
        data.extend_from_slice(&carry);
        data.extend_from_slice(&buffer[..n_read]);
        carry.clear();

        let total_bytes = data.len();
        let n_samples = total_bytes / bytes_per_sample;
        let rem = total_bytes % bytes_per_sample;

        if rem > 0 {
            carry.extend_from_slice(&data[total_bytes - rem..]);
        }

        if n_samples == 0 {
            continue;
        }

        // Convert to f32
        let samples: Vec<f32> = match format {
            PcmFormat::F32 => {
                (0..n_samples)
                    .map(|i| {
                        let o = i * 4;
                        f32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]])
                    })
                    .collect()
            }
            PcmFormat::S16 => {
                (0..n_samples)
                    .map(|i| {
                        let o = i * 2;
                        i16::from_le_bytes([data[o], data[o + 1]]) as f32 / 32768.0
                    })
                    .collect()
            }
        };

        // Push into ring buffer
        push_samples(&shared, &samples);
    }
}

/// Push samples into the ring buffer — direct port of `pcm_async::push_samples`.
fn push_samples(shared: &Arc<Mutex<RingBuffer>>, data: &[f32]) {
    if data.is_empty() {
        return;
    }

    let mut ring = shared.lock().unwrap();
    let cap = ring.audio.len();
    let mut src = data;
    let mut n = data.len();

    // If more samples than ring capacity, skip the oldest
    if n > cap {
        src = &data[n - cap..];
        n = cap;
    }

    // Drop oldest unread samples if we'd overflow
    if n > cap - ring.audio_len {
        let drop = n - (cap - ring.audio_len);
        ring.audio_read = (ring.audio_read + drop) % cap;
        ring.audio_len -= drop;
    }

    // Write into ring
    let pos = ring.audio_pos;
    if pos + n > cap {
        let n0 = cap - pos;
        ring.audio[pos..].copy_from_slice(&src[..n0]);
        ring.audio[..n - n0].copy_from_slice(&src[n0..]);
    } else {
        ring.audio[pos..pos + n].copy_from_slice(src);
    }

    ring.audio_pos = (ring.audio_pos + n) % cap;
    ring.audio_len = (ring.audio_len + n).min(cap);
}

// ---------------------------------------------------------------------------
// vad_simple — port of common.cpp::vad_simple + high_pass_filter
// ---------------------------------------------------------------------------

/// High-pass filter — port of `common.cpp::high_pass_filter`.
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

/// Energy-based VAD — port of `common.cpp::vad_simple`.
///
/// Returns `true` if the audio chunk is **silence** (no speech detected).
pub fn vad_simple(
    pcmf32: &[f32],
    sample_rate: i32,
    last_ms: i32,
    vad_thold: f32,
    freq_thold: f32,
) -> bool {
    let n_samples = pcmf32.len();
    let n_samples_last = (sample_rate as usize * last_ms.max(0) as usize) / 1000;

    if n_samples_last >= n_samples {
        // not enough samples — assume no speech
        return true; // silence
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

    // C++ returns false (= NOT silence) when energy_last > thold*energy_all
    // We return true for silence, matching the C++ sense where true = silence.
    energy_last <= vad_thold * energy_all
}

// ---------------------------------------------------------------------------
// WhisperStreamPcmConfig
// ---------------------------------------------------------------------------

/// Configuration for [`WhisperStreamPcm`] — maps to `whisper_params` streaming subset.
#[derive(Debug, Clone)]
pub struct WhisperStreamPcmConfig {
    /// Fixed-step chunk size in ms (non-VAD mode).
    pub step_ms: i32,
    /// Max audio length per inference in ms.
    pub length_ms: i32,
    /// Overlap to retain from previous step in ms.
    pub keep_ms: i32,
    /// Enable VAD-driven segmentation.
    pub use_vad: bool,
    /// VAD threshold (both simple & Silero).
    pub vad_thold: f32,
    /// High-pass frequency cutoff for simple VAD.
    pub freq_thold: f32,
    /// VAD probe chunk size in ms.
    pub vad_probe_ms: i32,
    /// Silence duration to end a segment in ms.
    pub vad_silence_ms: i32,
    /// Audio prepended before VAD trigger in ms.
    pub vad_pre_roll_ms: i32,
}

impl Default for WhisperStreamPcmConfig {
    fn default() -> Self {
        Self {
            step_ms: 3000,
            length_ms: 10000,
            keep_ms: 200,
            use_vad: false,
            vad_thold: 0.6,
            freq_thold: 100.0,
            vad_probe_ms: 200,
            vad_silence_ms: 800,
            vad_pre_roll_ms: 300,
        }
    }
}

// ---------------------------------------------------------------------------
// WhisperStreamPcm — main processor
// ---------------------------------------------------------------------------

/// Streaming PCM transcriber — direct port of `stream-pcm.cpp` main loop.
///
/// Two modes:
/// - **Fixed-step** (`use_vad = false`): process `step_ms` chunks with overlap.
/// - **VAD-driven** (`use_vad = true`): accumulate speech, transcribe on silence.
pub struct WhisperStreamPcm {
    state: WhisperState,
    params: FullParams,
    config: WhisperStreamPcmConfig,
    reader: PcmReader,
    vad: Option<WhisperVadProcessor>,

    // Pre-computed sample counts
    n_samples_step: usize,
    n_samples_len: usize,
    n_samples_keep: usize,

    // Fixed-step state
    pcmf32_old: Vec<f32>,

    // VAD state machine
    in_speech: bool,
    speech_buf: Vec<f32>,
    pre_roll: Vec<f32>,
    silence_samples: usize,

    total_samples: i64,
    n_iter: i32,

    // VAD pre-computed
    vad_last_ms: i32,
    vad_pre_roll_samples: usize,
    vad_silence_samples: usize,
    vad_max_segment_samples: usize,
}

impl WhisperStreamPcm {
    /// Create a new WhisperStreamPcm processor (simple VAD or no VAD).
    pub fn new(
        ctx: &WhisperContext,
        params: FullParams,
        mut config: WhisperStreamPcmConfig,
        reader: PcmReader,
    ) -> Result<Self> {
        Self::build(ctx, params, &mut config, reader, None)
    }

    /// Create a new WhisperStreamPcm processor with Silero VAD.
    pub fn with_vad(
        ctx: &WhisperContext,
        params: FullParams,
        mut config: WhisperStreamPcmConfig,
        reader: PcmReader,
        vad: WhisperVadProcessor,
    ) -> Result<Self> {
        Self::build(ctx, params, &mut config, reader, Some(vad))
    }

    fn build(
        ctx: &WhisperContext,
        params: FullParams,
        config: &mut WhisperStreamPcmConfig,
        reader: PcmReader,
        vad: Option<WhisperVadProcessor>,
    ) -> Result<Self> {
        let state = WhisperState::new(ctx)?;

        // Normalize config (matches C++ main)
        if !config.use_vad {
            if config.step_ms <= 0 {
                return Err(WhisperError::InvalidParameter(
                    "step_ms must be > 0 unless use_vad is true".into(),
                ));
            }
            config.keep_ms = config.keep_ms.min(config.step_ms);
            config.length_ms = config.length_ms.max(config.step_ms);
        } else {
            if config.length_ms <= 0 {
                config.length_ms = 5000;
            }
            config.keep_ms = 0;
        }

        let n_samples_step = if config.use_vad {
            0
        } else {
            (config.step_ms as f64 * 0.001 * WHISPER_SAMPLE_RATE as f64) as usize
        };
        let n_samples_len =
            (config.length_ms as f64 * 0.001 * WHISPER_SAMPLE_RATE as f64) as usize;
        let n_samples_keep =
            (config.keep_ms as f64 * 0.001 * WHISPER_SAMPLE_RATE as f64) as usize;

        let vad_probe_ms = config.vad_probe_ms.max(1);
        let vad_last_ms = (vad_probe_ms / 2).clamp(1, 1000);
        let vad_pre_roll_samples =
            (WHISPER_SAMPLE_RATE as usize * config.vad_pre_roll_ms.max(0) as usize) / 1000;
        let vad_silence_samples =
            (WHISPER_SAMPLE_RATE as usize * config.vad_silence_ms.max(0) as usize) / 1000;

        Ok(Self {
            state,
            params,
            config: config.clone(),
            reader,
            vad,
            n_samples_step,
            n_samples_len,
            n_samples_keep,
            pcmf32_old: Vec::new(),
            in_speech: false,
            speech_buf: Vec::new(),
            pre_roll: Vec::new(),
            silence_samples: 0,
            total_samples: 0,
            n_iter: 0,
            vad_last_ms,
            vad_pre_roll_samples,
            vad_silence_samples,
            vad_max_segment_samples: n_samples_len,
        })
    }

    /// Run one iteration of the main loop.
    ///
    /// Returns `Ok(Some(segments))` if transcription occurred,
    /// `Ok(None)` if waiting for more audio or sleeping,
    /// `Err` on fatal error.
    ///
    /// Returns `Ok(None)` with no more audio when EOF + drained.
    pub fn process_step(&mut self) -> Result<Option<Vec<Segment>>> {
        if !self.config.use_vad {
            self.process_step_fixed()
        } else {
            self.process_step_vad()
        }
    }

    /// Run until EOF or error. Calls `callback` for each transcription.
    pub fn run<F>(&mut self, mut callback: F) -> Result<()>
    where
        F: FnMut(&[Segment], i64, i64),
    {
        loop {
            match self.process_step()? {
                Some(segments) if !segments.is_empty() => {
                    let start = segments.first().map(|s| s.start_ms).unwrap_or(0);
                    let end = segments.last().map(|s| s.end_ms).unwrap_or(0);
                    callback(&segments, start, end);
                }
                Some(_) => {} // empty segments, keep going
                None => {
                    // Check if truly done (EOF + no audio left)
                    if self.reader.is_eof() && self.reader.available_samples() == 0 {
                        // Flush any remaining VAD speech
                        if self.config.use_vad && self.in_speech && !self.speech_buf.is_empty() {
                            let segments =
                                self.run_inference(&self.speech_buf.clone())?;
                            if !segments.is_empty() {
                                let start =
                                    segments.first().map(|s| s.start_ms).unwrap_or(0);
                                let end =
                                    segments.last().map(|s| s.end_ms).unwrap_or(0);
                                callback(&segments, start, end);
                            }
                            self.speech_buf.clear();
                            self.in_speech = false;
                        }
                        break;
                    }
                    // Still waiting for audio
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
            }
        }
        Ok(())
    }

    /// Fixed-step processing — port of the non-VAD branch of the C++ main loop.
    fn process_step_fixed(&mut self) -> Result<Option<Vec<Segment>>> {
        let available = self.reader.available_samples();

        if available < self.n_samples_step {
            if self.reader.is_eof() {
                if available == 0 {
                    return Ok(None); // done
                }
                // Fall through to process remaining
            } else {
                return Ok(None); // wait for more audio
            }
        }

        let pcmf32_new = self.reader.pop_ms(self.config.step_ms);
        if pcmf32_new.is_empty() {
            return Ok(None);
        }

        self.total_samples += pcmf32_new.len() as i64;

        let n_samples_new = pcmf32_new.len();
        let n_samples_take = self
            .pcmf32_old
            .len()
            .min((self.n_samples_keep + self.n_samples_len).saturating_sub(n_samples_new));

        let mut pcmf32 = Vec::with_capacity(n_samples_new + n_samples_take);

        // Prepend overlap from previous step
        if n_samples_take > 0 && !self.pcmf32_old.is_empty() {
            let start = self.pcmf32_old.len() - n_samples_take;
            pcmf32.extend_from_slice(&self.pcmf32_old[start..]);
        }
        pcmf32.extend_from_slice(&pcmf32_new);

        self.pcmf32_old = pcmf32.clone();

        let segments = self.run_inference(&pcmf32)?;
        self.n_iter += 1;

        // Keep overlap for next iteration
        let n_new_line = (self.config.length_ms / self.config.step_ms - 1).max(1);
        if self.n_iter % n_new_line == 0 && self.n_samples_keep > 0 && pcmf32.len() >= self.n_samples_keep {
            self.pcmf32_old = pcmf32[pcmf32.len() - self.n_samples_keep..].to_vec();
        }

        Ok(Some(segments))
    }

    /// VAD-driven processing — port of the VAD branch of the C++ main loop.
    fn process_step_vad(&mut self) -> Result<Option<Vec<Segment>>> {
        let available = self.reader.available_samples();

        if available == 0 {
            if self.reader.is_eof() {
                // Flush remaining speech
                if self.in_speech && !self.speech_buf.is_empty() {
                    let segments = self.run_inference(
                        &self.speech_buf.clone(),
                    )?;
                    self.speech_buf.clear();
                    self.in_speech = false;
                    self.n_iter += 1;
                    return Ok(Some(segments));
                }
                return Ok(None);
            }
            return Ok(None); // wait
        }

        let pcmf32_new = self.reader.pop_ms(self.config.vad_probe_ms);
        if pcmf32_new.is_empty() {
            return Ok(None);
        }

        self.total_samples += pcmf32_new.len() as i64;

        // Determine silence via Silero or simple VAD
        let silence = if let Some(ref mut vad) = self.vad {
            if vad.detect_speech(&pcmf32_new) {
                let probs = vad.get_probs();
                let avg = if probs.is_empty() {
                    0.0
                } else {
                    probs.iter().sum::<f32>() / probs.len() as f32
                };
                avg < self.config.vad_thold
            } else {
                true // detect failed → treat as silence
            }
        } else {
            vad_simple(
                &pcmf32_new,
                WHISPER_SAMPLE_RATE,
                self.vad_last_ms,
                self.config.vad_thold,
                self.config.freq_thold,
            )
        };

        let mut result_segments: Option<Vec<Segment>> = None;

        if !self.in_speech {
            if !silence {
                self.in_speech = true;
                self.silence_samples = 0;
                self.speech_buf.clear();
                if !self.pre_roll.is_empty() {
                    self.speech_buf.extend_from_slice(&self.pre_roll);
                }
                self.speech_buf.extend_from_slice(&pcmf32_new);
            }
        } else {
            self.speech_buf.extend_from_slice(&pcmf32_new);
            if !silence {
                self.silence_samples = 0;
            } else {
                self.silence_samples += pcmf32_new.len();
            }

            if self.speech_buf.len() >= self.vad_max_segment_samples
                || self.silence_samples >= self.vad_silence_samples
            {
                let segments = self.run_inference(
                    &self.speech_buf.clone(),
                )?;
                self.speech_buf.clear();
                self.in_speech = false;
                self.silence_samples = 0;
                self.n_iter += 1;
                result_segments = Some(segments);
            }
        }

        // Maintain pre-roll buffer
        if self.vad_pre_roll_samples > 0 {
            self.pre_roll.extend_from_slice(&pcmf32_new);
            if self.pre_roll.len() > self.vad_pre_roll_samples {
                let excess = self.pre_roll.len() - self.vad_pre_roll_samples;
                self.pre_roll.drain(..excess);
            }
        }

        Ok(result_segments)
    }

    /// Run whisper inference on an audio buffer — port of `run_inference` lambda.
    fn run_inference(&mut self, audio: &[f32]) -> Result<Vec<Segment>> {
        if audio.is_empty() {
            return Ok(Vec::new());
        }

        self.state.full(self.params.clone(), audio)?;

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

    /// Get the total number of processed samples.
    pub fn total_samples(&self) -> i64 {
        self.total_samples
    }

    /// Get the iteration count.
    pub fn n_iter(&self) -> i32 {
        self.n_iter
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pcm_format_eq() {
        assert_eq!(PcmFormat::F32, PcmFormat::F32);
        assert_ne!(PcmFormat::F32, PcmFormat::S16);
    }

    #[test]
    fn test_vad_simple_silence() {
        // All zeros = silence
        let silence = vec![0.0f32; 16000];
        assert!(vad_simple(&silence, 16000, 100, 0.6, 100.0));
    }

    #[test]
    fn test_vad_simple_too_few_samples() {
        let short = vec![0.1f32; 100];
        // last_ms=1000 → needs 16000 samples, only have 100 → silence
        assert!(vad_simple(&short, 16000, 1000, 0.6, 100.0));
    }

    #[test]
    fn test_high_pass_filter_basic() {
        let mut data = vec![1.0, 0.0, 1.0, 0.0, 1.0];
        high_pass_filter(&mut data, 100.0, 16000.0);
        // After filter, values should be modified
        assert_ne!(data[2], 1.0);
    }

    #[test]
    fn test_pcm_reader_f32() {
        // Simulate 1 second of f32 PCM data (16000 samples)
        let n = 16000;
        let mut raw = Vec::with_capacity(n * 4);
        for i in 0..n {
            let val = (i as f32 / n as f32) * 2.0 - 1.0; // ramp -1..1
            raw.extend_from_slice(&val.to_le_bytes());
        }

        let cursor = std::io::Cursor::new(raw);
        let config = PcmReaderConfig {
            buffer_len_ms: 2000,
            sample_rate: 16000,
            format: PcmFormat::F32,
        };
        let reader = PcmReader::new(Box::new(cursor), config);

        // Wait for reader thread to consume
        std::thread::sleep(std::time::Duration::from_millis(100));

        assert!(reader.is_eof());
        let samples = reader.pop_ms(1000);
        assert_eq!(samples.len(), 16000);
    }

    #[test]
    fn test_pcm_reader_s16() {
        let n = 16000;
        let mut raw = Vec::with_capacity(n * 2);
        for i in 0..n {
            let val = ((i as f32 / n as f32) * 2.0 - 1.0) * 32767.0;
            raw.extend_from_slice(&(val as i16).to_le_bytes());
        }

        let cursor = std::io::Cursor::new(raw);
        let config = PcmReaderConfig {
            buffer_len_ms: 2000,
            sample_rate: 16000,
            format: PcmFormat::S16,
        };
        let reader = PcmReader::new(Box::new(cursor), config);

        std::thread::sleep(std::time::Duration::from_millis(100));

        assert!(reader.is_eof());
        let samples = reader.pop_ms(1000);
        assert_eq!(samples.len(), 16000);

        // Check conversion — first sample should be near -1.0
        assert!(samples[0] < -0.9);
    }

    #[test]
    fn test_ring_buffer_overflow() {
        // Buffer only holds 500ms = 8000 samples, but we push 16000
        let n = 16000;
        let mut raw = Vec::with_capacity(n * 4);
        for i in 0..n {
            raw.extend_from_slice(&(i as f32).to_le_bytes());
        }

        let cursor = std::io::Cursor::new(raw);
        let config = PcmReaderConfig {
            buffer_len_ms: 500,
            sample_rate: 16000,
            format: PcmFormat::F32,
        };
        let reader = PcmReader::new(Box::new(cursor), config);

        std::thread::sleep(std::time::Duration::from_millis(100));

        // Should only have 8000 samples (most recent)
        let available = reader.available_samples();
        assert!(available <= 8000);

        let samples = reader.pop_ms(500);
        assert_eq!(samples.len(), 8000);
        // Last sample should be 15999.0
        assert!((samples[samples.len() - 1] - 15999.0).abs() < 0.01);
    }

    #[test]
    fn test_stream_pcm_config_defaults() {
        let config = WhisperStreamPcmConfig::default();
        assert_eq!(config.step_ms, 3000);
        assert_eq!(config.length_ms, 10000);
        assert_eq!(config.keep_ms, 200);
        assert!(!config.use_vad);
    }

    #[test]
    fn test_stream_pcm_config_vad_normalization() {
        // When use_vad=true, keep_ms should be forced to 0
        use std::path::Path;
        let model_path = "tests/models/ggml-tiny.en.bin";
        if !Path::new(model_path).exists() {
            eprintln!("Skipping: model not found");
            return;
        }

        let ctx = WhisperContext::new(model_path).unwrap();
        let params = FullParams::default();
        let cursor = std::io::Cursor::new(Vec::<u8>::new());
        let reader = PcmReader::new(Box::new(cursor), PcmReaderConfig::default());
        let config = WhisperStreamPcmConfig {
            use_vad: true,
            keep_ms: 500, // should be forced to 0
            ..Default::default()
        };

        let stream = WhisperStreamPcm::new(&ctx, params, config, reader).unwrap();
        assert_eq!(stream.config.keep_ms, 0);
    }
}
