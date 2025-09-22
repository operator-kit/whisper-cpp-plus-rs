use crate::context::{ContextPtr, WhisperContext};
use crate::error::{Result, WhisperError};
use crate::params::FullParams;
use std::sync::Arc;
use whisper_sys as ffi;

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

    pub fn full_parallel(
        &mut self,
        params: FullParams,
        audio: &[f32],
        n_processors: i32,
    ) -> Result<()> {
        if audio.is_empty() {
            return Err(WhisperError::InvalidAudioFormat);
        }

        if n_processors < 1 {
            return Err(WhisperError::InvalidParameter(
                "n_processors must be at least 1".into(),
            ));
        }

        let ret = unsafe {
            ffi::whisper_full_parallel(
                self._context.0,
                params.as_raw(),
                audio.as_ptr(),
                audio.len() as i32,
                n_processors,
            )
        };

        if ret != 0 {
            return Err(WhisperError::TranscriptionError(format!(
                "whisper_full_parallel returned {}",
                ret
            )));
        }

        Ok(())
    }

    pub fn full_n_segments(&self) -> i32 {
        unsafe { ffi::whisper_full_n_segments_from_state(self.ptr) }
    }

    pub fn full_lang_id(&self) -> i32 {
        unsafe { ffi::whisper_full_lang_id_from_state(self.ptr) }
    }

    pub fn full_get_segment_text(&self, i_segment: i32) -> Result<String> {
        let text_ptr = unsafe { ffi::whisper_full_get_segment_text_from_state(self.ptr, i_segment) };

        if text_ptr.is_null() {
            return Err(WhisperError::InvalidContext);
        }

        let c_str = unsafe { std::ffi::CStr::from_ptr(text_ptr) };
        Ok(c_str.to_string_lossy().into_owned())
    }

    pub fn full_get_segment_timestamps(&self, i_segment: i32) -> (i64, i64) {
        unsafe {
            let t0 = ffi::whisper_full_get_segment_t0_from_state(self.ptr, i_segment);
            let t1 = ffi::whisper_full_get_segment_t1_from_state(self.ptr, i_segment);
            (t0, t1)
        }
    }

    pub fn full_get_segment_speaker_turn_next(&self, i_segment: i32) -> bool {
        unsafe {
            ffi::whisper_full_get_segment_speaker_turn_next_from_state(self.ptr, i_segment)
        }
    }

    pub fn full_n_tokens(&self, i_segment: i32) -> i32 {
        unsafe { ffi::whisper_full_n_tokens_from_state(self.ptr, i_segment) }
    }

    pub fn full_get_token_text(&self, i_segment: i32, i_token: i32) -> Result<String> {
        let text_ptr = unsafe {
            ffi::whisper_full_get_token_text_from_state(self._context.0, self.ptr, i_segment, i_token)
        };

        if text_ptr.is_null() {
            return Err(WhisperError::InvalidContext);
        }

        let c_str = unsafe { std::ffi::CStr::from_ptr(text_ptr) };
        Ok(c_str.to_string_lossy().into_owned())
    }

    pub fn full_get_token_id(&self, i_segment: i32, i_token: i32) -> i32 {
        unsafe { ffi::whisper_full_get_token_id_from_state(self.ptr, i_segment, i_token) }
    }

    pub fn full_get_token_data(
        &self,
        i_segment: i32,
        i_token: i32,
    ) -> Option<ffi::whisper_token_data> {
        let data = unsafe {
            ffi::whisper_full_get_token_data_from_state(self.ptr, i_segment, i_token)
        };

        if data.id == -1 {
            None
        } else {
            Some(data)
        }
    }

    pub fn full_get_token_prob(&self, i_segment: i32, i_token: i32) -> f32 {
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