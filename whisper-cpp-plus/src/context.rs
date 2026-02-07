use crate::error::{Result, WhisperError};
use std::path::Path;
use std::sync::Arc;
use whisper_cpp_plus_sys as ffi;

pub struct WhisperContext {
    pub(crate) ptr: Arc<ContextPtr>,
}

pub(crate) struct ContextPtr(pub(crate) *mut ffi::whisper_context);

unsafe impl Send for ContextPtr {}
unsafe impl Sync for ContextPtr {}

impl Drop for ContextPtr {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                ffi::whisper_free(self.0);
            }
        }
    }
}

impl WhisperContext {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let path_str = model_path
            .as_ref()
            .to_str()
            .ok_or_else(|| WhisperError::ModelLoadError("Invalid path".into()))?;

        let c_path = std::ffi::CString::new(path_str)?;

        let ptr = unsafe {
            ffi::whisper_init_from_file_with_params(
                c_path.as_ptr(),
                ffi::whisper_context_default_params(),
            )
        };

        if ptr.is_null() {
            return Err(WhisperError::ModelLoadError(
                "Failed to load model".into(),
            ));
        }

        Ok(Self {
            ptr: Arc::new(ContextPtr(ptr)),
        })
    }

    pub fn new_from_buffer(buffer: &[u8]) -> Result<Self> {
        let ptr = unsafe {
            ffi::whisper_init_from_buffer_with_params(
                buffer.as_ptr() as *mut std::os::raw::c_void,
                buffer.len(),
                ffi::whisper_context_default_params(),
            )
        };

        if ptr.is_null() {
            return Err(WhisperError::ModelLoadError(
                "Failed to load model from buffer".into(),
            ));
        }

        Ok(Self {
            ptr: Arc::new(ContextPtr(ptr)),
        })
    }

    pub fn is_multilingual(&self) -> bool {
        unsafe { ffi::whisper_is_multilingual(self.ptr.0) != 0 }
    }

    pub fn n_vocab(&self) -> i32 {
        unsafe { ffi::whisper_n_vocab(self.ptr.0) }
    }

    pub fn n_audio_ctx(&self) -> i32 {
        unsafe { ffi::whisper_n_audio_ctx(self.ptr.0) }
    }

    pub fn n_text_ctx(&self) -> i32 {
        unsafe { ffi::whisper_n_text_ctx(self.ptr.0) }
    }

    pub fn n_len(&self) -> i32 {
        unsafe { ffi::whisper_n_len(self.ptr.0) }
    }

}

impl Clone for WhisperContext {
    fn clone(&self) -> Self {
        Self {
            ptr: Arc::clone(&self.ptr),
        }
    }
}