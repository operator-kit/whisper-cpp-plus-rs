use crate::context::{ContextPtr, WhisperContext};
use crate::error::{Result, WhisperError};
use crate::params::FullParams;
use std::sync::Arc;
use whisper_cpp_plus_sys as ffi;

pub struct WhisperState {
    pub(crate) ptr: *mut ffi::whisper_state,
    pub(crate) _context: Arc<ContextPtr>,
}

impl Drop for WhisperState {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.is_null() {
                ffi::whisper_free_state(self.ptr);
            }
        }
    }
}

impl WhisperState {
    pub fn new(context: &WhisperContext) -> Result<Self> {
        let ptr = unsafe { ffi::whisper_init_state(context.ptr.0) };

        if ptr.is_null() {
            return Err(WhisperError::OutOfMemory);
        }

        Ok(Self {
            ptr,
            _context: Arc::clone(&context.ptr),
        })
    }

    pub fn full(&mut self, params: FullParams, audio: &[f32]) -> Result<()> {
        if audio.is_empty() {
            return Err(WhisperError::InvalidAudioFormat);
        }

        let ret = unsafe {
            ffi::whisper_full_with_state(
                self._context.0,
                self.ptr,
                params.as_raw(),
                audio.as_ptr(),
                audio.len() as i32,
            )
        };

        if ret != 0 {
            return Err(WhisperError::TranscriptionError(format!(
                "whisper_full returned {}",
                ret
            )));
        }

        Ok(())
    }

    /// Collects the segments of the last transcription on this state.
    pub(crate) fn collect_segments(&self) -> Result<Vec<Segment>> {
        (0..self.full_n_segments())
            .map(|i| {
                let text = self.full_get_segment_text(i)?;
                let (start_ms, end_ms) = self.full_get_segment_timestamps(i);
                Ok(Segment {
                    start_ms,
                    end_ms,
                    text,
                    speaker_turn_next: self.full_get_segment_speaker_turn_next(i),
                })
            })
            .collect()
    }

    pub fn full_n_segments(&self) -> i32 {
        unsafe { ffi::whisper_full_n_segments_from_state(self.ptr) }
    }

    pub fn full_lang_id(&self) -> i32 {
        unsafe { ffi::whisper_full_lang_id_from_state(self.ptr) }
    }

    /// Returns the length, in mel frames, of the audio from the last transcription on this state
    /// (`whisper_n_len_from_state`); 0 before any transcription.
    pub fn n_len(&self) -> i32 {
        unsafe { ffi::whisper_n_len_from_state(self.ptr) }
    }

    // The whisper.cpp result getters index their vectors without bounds checks, so every
    // wrapper validates indices before calling into C.

    fn segment_in_range(&self, i_segment: i32) -> bool {
        i_segment >= 0 && i_segment < self.full_n_segments()
    }

    fn token_in_range(&self, i_segment: i32, i_token: i32) -> bool {
        self.segment_in_range(i_segment)
            && i_token >= 0
            && i_token < unsafe { ffi::whisper_full_n_tokens_from_state(self.ptr, i_segment) }
    }

    fn assert_segment_in_range(&self, i_segment: i32) {
        assert!(
            self.segment_in_range(i_segment),
            "segment index {} out of range (n_segments = {})",
            i_segment,
            self.full_n_segments()
        );
    }

    fn assert_token_in_range(&self, i_segment: i32, i_token: i32) {
        assert!(
            self.token_in_range(i_segment, i_token),
            "token index ({}, {}) out of range",
            i_segment,
            i_token
        );
    }

    /// Returns the text of segment `i_segment`.
    ///
    /// Returns [`WhisperError::InvalidParameter`] if the index is out of range.
    pub fn full_get_segment_text(&self, i_segment: i32) -> Result<String> {
        if !self.segment_in_range(i_segment) {
            return Err(WhisperError::InvalidParameter(format!(
                "segment index {} out of range",
                i_segment
            )));
        }

        let text_ptr =
            unsafe { ffi::whisper_full_get_segment_text_from_state(self.ptr, i_segment) };

        if text_ptr.is_null() {
            return Err(WhisperError::InvalidContext);
        }

        let c_str = unsafe { std::ffi::CStr::from_ptr(text_ptr) };
        Ok(c_str.to_string_lossy().into_owned())
    }

    /// Returns the `(start, end)` time of segment `i_segment` in milliseconds.
    ///
    /// # Panics
    ///
    /// Panics if `i_segment` is out of range.
    pub fn full_get_segment_timestamps(&self, i_segment: i32) -> (i64, i64) {
        self.assert_segment_in_range(i_segment);
        // whisper.cpp reports segment times in centiseconds (10 ms units).
        unsafe {
            let t0 = ffi::whisper_full_get_segment_t0_from_state(self.ptr, i_segment);
            let t1 = ffi::whisper_full_get_segment_t1_from_state(self.ptr, i_segment);
            (t0 * 10, t1 * 10)
        }
    }

    /// Returns whether the next segment starts with a speaker turn (tinydiarize).
    ///
    /// # Panics
    ///
    /// Panics if `i_segment` is out of range.
    pub fn full_get_segment_speaker_turn_next(&self, i_segment: i32) -> bool {
        self.assert_segment_in_range(i_segment);
        unsafe { ffi::whisper_full_get_segment_speaker_turn_next_from_state(self.ptr, i_segment) }
    }

    /// Returns the no-speech probability of segment `i_segment`.
    ///
    /// # Panics
    ///
    /// Panics if `i_segment` is out of range.
    pub fn full_get_segment_no_speech_prob(&self, i_segment: i32) -> f32 {
        self.assert_segment_in_range(i_segment);
        unsafe { ffi::whisper_full_get_segment_no_speech_prob_from_state(self.ptr, i_segment) }
    }

    /// Returns the number of tokens in segment `i_segment`.
    ///
    /// # Panics
    ///
    /// Panics if `i_segment` is out of range.
    pub fn full_n_tokens(&self, i_segment: i32) -> i32 {
        self.assert_segment_in_range(i_segment);
        unsafe { ffi::whisper_full_n_tokens_from_state(self.ptr, i_segment) }
    }

    /// Returns the text of token `i_token` in segment `i_segment`.
    ///
    /// Returns [`WhisperError::InvalidParameter`] if either index is out of range.
    pub fn full_get_token_text(&self, i_segment: i32, i_token: i32) -> Result<String> {
        if !self.token_in_range(i_segment, i_token) {
            return Err(WhisperError::InvalidParameter(format!(
                "token index ({}, {}) out of range",
                i_segment, i_token
            )));
        }

        let text_ptr = unsafe {
            ffi::whisper_full_get_token_text_from_state(
                self._context.0,
                self.ptr,
                i_segment,
                i_token,
            )
        };

        if text_ptr.is_null() {
            return Err(WhisperError::InvalidContext);
        }

        let c_str = unsafe { std::ffi::CStr::from_ptr(text_ptr) };
        Ok(c_str.to_string_lossy().into_owned())
    }

    /// Returns the id of token `i_token` in segment `i_segment`.
    ///
    /// # Panics
    ///
    /// Panics if either index is out of range.
    pub fn full_get_token_id(&self, i_segment: i32, i_token: i32) -> i32 {
        self.assert_token_in_range(i_segment, i_token);
        unsafe { ffi::whisper_full_get_token_id_from_state(self.ptr, i_segment, i_token) }
    }

    /// Returns the raw token data for token `i_token` in segment `i_segment`, or `None` if
    /// either index is out of range.
    ///
    /// This is the unmodified whisper.cpp struct: its `t0`, `t1` and `t_dtw` fields are in
    /// centiseconds (10 ms units), unlike [`WhisperState::full_get_segment_timestamps`].
    pub fn full_get_token_data(
        &self,
        i_segment: i32,
        i_token: i32,
    ) -> Option<ffi::whisper_token_data> {
        if !self.token_in_range(i_segment, i_token) {
            return None;
        }

        Some(unsafe { ffi::whisper_full_get_token_data_from_state(self.ptr, i_segment, i_token) })
    }

    /// Returns the probability of token `i_token` in segment `i_segment`.
    ///
    /// # Panics
    ///
    /// Panics if either index is out of range.
    pub fn full_get_token_prob(&self, i_segment: i32, i_token: i32) -> f32 {
        self.assert_token_in_range(i_segment, i_token);
        unsafe { ffi::whisper_full_get_token_p_from_state(self.ptr, i_segment, i_token) }
    }
}

unsafe impl Send for WhisperState {}

#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    pub text: String,
    pub segments: Vec<Segment>,
}

#[derive(Debug, Clone)]
pub struct Segment {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
    pub speaker_turn_next: bool,
}

impl Segment {
    pub fn start_seconds(&self) -> f64 {
        self.start_ms as f64 / 1000.0
    }

    pub fn end_seconds(&self) -> f64 {
        self.end_ms as f64 / 1000.0
    }
}
