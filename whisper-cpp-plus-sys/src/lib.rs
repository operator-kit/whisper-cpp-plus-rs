#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(improper_ctypes)]

// Include the bindgen-generated bindings
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

// Manual error code constants not documented in whisper.h
// These are based on common error patterns seen in whisper.cpp
pub const WHISPER_ERR_INVALID_MODEL: i32 = -1;
pub const WHISPER_ERR_NOT_ENOUGH_MEMORY: i32 = -2;
pub const WHISPER_ERR_FAILED_TO_PROCESS: i32 = -3;
pub const WHISPER_ERR_INVALID_CONTEXT: i32 = -4;

// Quantization-related constants and functions
#[cfg(feature = "quantization")]
pub const WHISPER_QUANTIZE_OK: i32 = 0;
#[cfg(feature = "quantization")]
pub const WHISPER_QUANTIZE_ERROR_INVALID_MODEL: i32 = -1;
#[cfg(feature = "quantization")]
pub const WHISPER_QUANTIZE_ERROR_FILE_OPEN: i32 = -2;
#[cfg(feature = "quantization")]
pub const WHISPER_QUANTIZE_ERROR_FILE_WRITE: i32 = -3;
#[cfg(feature = "quantization")]
pub const WHISPER_QUANTIZE_ERROR_INVALID_FTYPE: i32 = -4;
#[cfg(feature = "quantization")]
pub const WHISPER_QUANTIZE_ERROR_QUANTIZATION_FAILED: i32 = -5;

// GGML quantization types
pub const GGML_FTYPE_UNKNOWN: i32 = -1;
pub const GGML_FTYPE_ALL_F32: i32 = 0;
pub const GGML_FTYPE_MOSTLY_F16: i32 = 1;
pub const GGML_FTYPE_MOSTLY_Q4_0: i32 = 2;
pub const GGML_FTYPE_MOSTLY_Q4_1: i32 = 3;
pub const GGML_FTYPE_MOSTLY_Q4_1_SOME_F16: i32 = 4;
pub const GGML_FTYPE_MOSTLY_Q8_0: i32 = 7;
pub const GGML_FTYPE_MOSTLY_Q5_0: i32 = 8;
pub const GGML_FTYPE_MOSTLY_Q5_1: i32 = 9;
pub const GGML_FTYPE_MOSTLY_Q2_K: i32 = 10;
pub const GGML_FTYPE_MOSTLY_Q3_K: i32 = 11;
pub const GGML_FTYPE_MOSTLY_Q4_K: i32 = 12;
pub const GGML_FTYPE_MOSTLY_Q5_K: i32 = 13;
pub const GGML_FTYPE_MOSTLY_Q6_K: i32 = 14;

// Progress callback type
#[cfg(feature = "quantization")]
pub type whisper_quantize_progress_callback = Option<extern "C" fn(progress: f32)>;

#[cfg(feature = "quantization")]
extern "C" {
    /// Quantize a Whisper model file
    ///
    /// # Parameters
    /// - `fname_inp`: Path to the input model file
    /// - `fname_out`: Path to the output quantized model file
    /// - `ftype`: Quantization type (one of the GGML_FTYPE_MOSTLY_* constants)
    /// - `progress_callback`: Optional callback function for progress updates
    ///
    /// # Returns
    /// - `WHISPER_QUANTIZE_OK` on success
    /// - One of the `WHISPER_QUANTIZE_ERROR_*` codes on failure
    pub fn whisper_model_quantize(
        fname_inp: *const ::std::os::raw::c_char,
        fname_out: *const ::std::os::raw::c_char,
        ftype: ::std::os::raw::c_int,
        progress_callback: whisper_quantize_progress_callback,
    ) -> ::std::os::raw::c_int;

    /// Get the quantization type of a model file
    ///
    /// # Parameters
    /// - `fname`: Path to the model file
    ///
    /// # Returns
    /// - The GGML_FTYPE_* constant representing the model's quantization
    /// - -1 on error (file not found or invalid model)
    pub fn whisper_model_get_ftype(
        fname: *const ::std::os::raw::c_char,
    ) -> ::std::os::raw::c_int;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants_defined() {
        // Just verify that our custom error constants are accessible
        assert_eq!(WHISPER_ERR_INVALID_MODEL, -1);
        assert_eq!(WHISPER_ERR_NOT_ENOUGH_MEMORY, -2);
        assert_eq!(WHISPER_ERR_FAILED_TO_PROCESS, -3);
        assert_eq!(WHISPER_ERR_INVALID_CONTEXT, -4);
    }
}