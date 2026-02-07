use thiserror::Error;

#[derive(Error, Debug)]
pub enum WhisperError {
    #[error("Failed to load model: {0}")]
    ModelLoadError(String),

    #[error("Invalid audio format: expected 16kHz mono f32")]
    InvalidAudioFormat,

    #[error("Transcription failed: {0}")]
    TranscriptionError(String),

    #[error("Invalid context")]
    InvalidContext,

    #[error("Out of memory")]
    OutOfMemory,

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("FFI error: code {code}")]
    CppError { code: i32 },

    #[error("Invalid UTF-8 string from C")]
    InvalidUtf8,

    #[error("Null pointer error")]
    NullPointer,
}

pub type Result<T> = std::result::Result<T, WhisperError>;

impl From<std::ffi::NulError> for WhisperError {
    fn from(_: std::ffi::NulError) -> Self {
        WhisperError::InvalidParameter("String contains null byte".to_string())
    }
}

impl From<std::str::Utf8Error> for WhisperError {
    fn from(_: std::str::Utf8Error) -> Self {
        WhisperError::InvalidUtf8
    }
}