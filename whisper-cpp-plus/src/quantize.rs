//! Model quantization for reducing model size and improving inference speed.
//!
//! Provides functionality to quantize Whisper models to various bit depths,
//! reducing model size while maintaining reasonable accuracy. Quantization is
//! particularly useful for deployment on resource-constrained devices.
//!
//! Enable with the `quantization` feature flag:
//! ```toml
//! whisper-cpp-plus = { version = "0.1.1", features = ["quantization"] }
//! ```

use std::ffi::CString;
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;
use thiserror::Error;

use whisper_cpp_plus_sys as ffi;

/// Error type for quantization operations
#[derive(Debug, Error)]
pub enum QuantizeError {
    #[error("Model file not found: {0}")]
    FileNotFound(String),

    #[error("Failed to open file: {0}")]
    FileOpenError(String),

    #[error("Failed to write file: {0}")]
    FileWriteError(String),

    #[error("Invalid model format")]
    InvalidModel,

    #[error("Invalid quantization type")]
    InvalidQuantizationType,

    #[error("Quantization failed: {0}")]
    QuantizationFailed(String),
}

type Result<T> = std::result::Result<T, QuantizeError>;

/// Quantization types supported by whisper.cpp
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
#[allow(non_camel_case_types)]
pub enum QuantizationType {
    /// 4-bit quantization (method 0) - ~3.5 GB for base model
    Q4_0 = ffi::GGML_FTYPE_MOSTLY_Q4_0,

    /// 4-bit quantization (method 1) - ~3.9 GB for base model
    Q4_1 = ffi::GGML_FTYPE_MOSTLY_Q4_1,

    /// 5-bit quantization (method 0) - ~4.3 GB for base model
    Q5_0 = ffi::GGML_FTYPE_MOSTLY_Q5_0,

    /// 5-bit quantization (method 1) - ~4.7 GB for base model
    Q5_1 = ffi::GGML_FTYPE_MOSTLY_Q5_1,

    /// 8-bit quantization - ~7.7 GB for base model
    Q8_0 = ffi::GGML_FTYPE_MOSTLY_Q8_0,

    /// 2-bit k-quantization
    Q2_K = ffi::GGML_FTYPE_MOSTLY_Q2_K,

    /// 3-bit k-quantization
    Q3_K = ffi::GGML_FTYPE_MOSTLY_Q3_K,

    /// 4-bit k-quantization
    Q4_K = ffi::GGML_FTYPE_MOSTLY_Q4_K,

    /// 5-bit k-quantization
    Q5_K = ffi::GGML_FTYPE_MOSTLY_Q5_K,

    /// 6-bit k-quantization
    Q6_K = ffi::GGML_FTYPE_MOSTLY_Q6_K,
}

impl QuantizationType {
    /// Get a human-readable name for the quantization type
    pub fn name(&self) -> &'static str {
        match self {
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q2_K => "Q2_K",
            Self::Q3_K => "Q3_K",
            Self::Q4_K => "Q4_K",
            Self::Q5_K => "Q5_K",
            Self::Q6_K => "Q6_K",
        }
    }

    /// Estimate the size reduction factor for this quantization type.
    /// Returns the approximate size as a fraction of the original F32 model.
    pub fn size_factor(&self) -> f32 {
        match self {
            Self::Q2_K => 0.19,  // ~19% of original
            Self::Q3_K => 0.26,  // ~26% of original
            Self::Q4_0 => 0.31,  // ~31% of original
            Self::Q4_1 => 0.35,  // ~35% of original
            Self::Q4_K => 0.33,  // ~33% of original
            Self::Q5_0 => 0.39,  // ~39% of original
            Self::Q5_1 => 0.43,  // ~43% of original
            Self::Q5_K => 0.41,  // ~41% of original
            Self::Q6_K => 0.49,  // ~49% of original
            Self::Q8_0 => 0.69,  // ~69% of original
        }
    }

    /// Get all available quantization types
    pub fn all() -> &'static [QuantizationType] {
        &[
            Self::Q4_0,
            Self::Q4_1,
            Self::Q5_0,
            Self::Q5_1,
            Self::Q8_0,
            Self::Q2_K,
            Self::Q3_K,
            Self::Q4_K,
            Self::Q5_K,
            Self::Q6_K,
        ]
    }

}

impl std::str::FromStr for QuantizationType {
    type Err = QuantizeError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "Q4_0" | "Q40" => Ok(Self::Q4_0),
            "Q4_1" | "Q41" => Ok(Self::Q4_1),
            "Q5_0" | "Q50" => Ok(Self::Q5_0),
            "Q5_1" | "Q51" => Ok(Self::Q5_1),
            "Q8_0" | "Q80" => Ok(Self::Q8_0),
            "Q2_K" | "Q2K" => Ok(Self::Q2_K),
            "Q3_K" | "Q3K" => Ok(Self::Q3_K),
            "Q4_K" | "Q4K" => Ok(Self::Q4_K),
            "Q5_K" | "Q5K" => Ok(Self::Q5_K),
            "Q6_K" | "Q6K" => Ok(Self::Q6_K),
            _ => Err(QuantizeError::InvalidQuantizationType),
        }
    }
}

impl std::fmt::Display for QuantizationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Progress callback for quantization operations
pub type ProgressCallback = Box<dyn Fn(f32) + Send>;

/// Model quantizer for converting Whisper models to different quantization formats
pub struct WhisperQuantize;

impl WhisperQuantize {
    /// Quantize a model file to a specified quantization type
    ///
    /// # Arguments
    /// * `input_path` - Path to the input model file (must be in GGML format)
    /// * `output_path` - Path where the quantized model will be saved
    /// * `qtype` - The quantization type to use
    ///
    /// # Example
    /// ```no_run
    /// use whisper_cpp_plus::{WhisperQuantize, QuantizationType};
    ///
    /// WhisperQuantize::quantize_model_file(
    ///     "models/ggml-base.bin",
    ///     "models/ggml-base-q5_0.bin",
    ///     QuantizationType::Q5_0
    /// ).expect("Failed to quantize model");
    /// ```
    pub fn quantize_model_file<P: AsRef<Path>>(
        input_path: P,
        output_path: P,
        qtype: QuantizationType,
    ) -> Result<()> {
        Self::quantize_model_file_impl(input_path.as_ref(), output_path.as_ref(), qtype, None)
    }

    /// Quantize a model file with progress callback
    ///
    /// # Arguments
    /// * `input_path` - Path to the input model file
    /// * `output_path` - Path where the quantized model will be saved
    /// * `qtype` - The quantization type to use
    /// * `callback` - Progress callback function (receives values from 0.0 to 1.0)
    ///
    /// # Example
    /// ```no_run
    /// use whisper_cpp_plus::{WhisperQuantize, QuantizationType};
    ///
    /// WhisperQuantize::quantize_model_file_with_progress(
    ///     "models/ggml-base.bin",
    ///     "models/ggml-base-q4_0.bin",
    ///     QuantizationType::Q4_0,
    ///     |progress| {
    ///         println!("Progress: {:.1}%", progress * 100.0);
    ///     }
    /// ).expect("Failed to quantize model");
    /// ```
    pub fn quantize_model_file_with_progress<P, F>(
        input_path: P,
        output_path: P,
        qtype: QuantizationType,
        callback: F,
    ) -> Result<()>
    where
        P: AsRef<Path>,
        F: Fn(f32) + Send + 'static,
    {
        Self::quantize_model_file_impl(
            input_path.as_ref(),
            output_path.as_ref(),
            qtype,
            Some(Box::new(callback)),
        )
    }

    fn quantize_model_file_impl(
        input_path: &Path,
        output_path: &Path,
        qtype: QuantizationType,
        callback: Option<ProgressCallback>,
    ) -> Result<()> {
        // Validate input file exists
        if !input_path.exists() {
            return Err(QuantizeError::FileNotFound(format!(
                "{}",
                input_path.display()
            )));
        }

        // Convert paths to C strings
        let input_cstr = path_to_cstring(input_path)?;
        let output_cstr = path_to_cstring(output_path)?;

        // Set up progress callback if provided
        let callback_data = callback.map(|cb| Arc::new(Mutex::new(cb)));
        let callback_ptr = callback_data.as_ref().map(|data| {
            Arc::clone(data) as Arc<Mutex<dyn Fn(f32) + Send>>
        });

        // Create the FFI callback function
        let ffi_callback: ffi::whisper_quantize_progress_callback = if callback_ptr.is_some() {
            Some(quantize_progress_callback)
        } else {
            None
        };

        // Store callback data in thread-local storage for the callback to access
        if let Some(ptr) = callback_ptr {
            CALLBACK_DATA.with(|data| {
                *data.borrow_mut() = Some(ptr);
            });
        }

        // Perform quantization
        let result = unsafe {
            ffi::whisper_model_quantize(
                input_cstr.as_ptr(),
                output_cstr.as_ptr(),
                qtype as i32,
                ffi_callback,
            )
        };

        // Clear callback data
        CALLBACK_DATA.with(|data| {
            *data.borrow_mut() = None;
        });

        // Check result
        match result {
            ffi::WHISPER_QUANTIZE_OK => Ok(()),
            ffi::WHISPER_QUANTIZE_ERROR_INVALID_MODEL => {
                Err(QuantizeError::QuantizationFailed("Invalid model file".to_string()))
            }
            ffi::WHISPER_QUANTIZE_ERROR_FILE_OPEN => {
                Err(QuantizeError::QuantizationFailed(format!(
                    "Failed to open input file: {}",
                    input_path.display()
                )))
            }
            ffi::WHISPER_QUANTIZE_ERROR_FILE_WRITE => {
                Err(QuantizeError::QuantizationFailed(format!(
                    "Failed to write output file: {}",
                    output_path.display()
                )))
            }
            ffi::WHISPER_QUANTIZE_ERROR_INVALID_FTYPE => {
                Err(QuantizeError::QuantizationFailed(format!(
                    "Invalid quantization type: {}",
                    qtype
                )))
            }
            ffi::WHISPER_QUANTIZE_ERROR_QUANTIZATION_FAILED => {
                Err(QuantizeError::QuantizationFailed("Quantization failed".to_string()))
            }
            _ => Err(QuantizeError::QuantizationFailed(format!(
                "Unknown quantization error: {}",
                result
            ))),
        }
    }

    /// Get the quantization type of an existing model file
    ///
    /// # Returns
    /// * `Ok(Some(qtype))` - The quantization type if the model is quantized
    /// * `Ok(None)` - If the model is in full precision (F32 or F16)
    /// * `Err(_)` - If the file cannot be read or is not a valid model
    ///
    /// # Example
    /// ```no_run
    /// use whisper_cpp_plus::WhisperQuantize;
    ///
    /// match WhisperQuantize::get_model_quantization_type("models/ggml-base-q5_0.bin") {
    ///     Ok(Some(qtype)) => println!("Model is quantized as: {}", qtype),
    ///     Ok(None) => println!("Model is not quantized"),
    ///     Err(e) => println!("Error reading model: {}", e),
    /// }
    /// ```
    pub fn get_model_quantization_type<P: AsRef<Path>>(
        model_path: P,
    ) -> Result<Option<QuantizationType>> {
        let path = model_path.as_ref();

        if !path.exists() {
            return Err(QuantizeError::FileNotFound(format!(
                "{}",
                path.display()
            )));
        }

        let path_cstr = path_to_cstring(path)?;

        let ftype = unsafe {
            ffi::whisper_model_get_ftype(path_cstr.as_ptr())
        };

        if ftype < 0 {
            return Err(QuantizeError::FileOpenError(format!(
                "{}",
                path.display()
            )));
        }

        // Map the ftype to our QuantizationType enum
        let qtype = match ftype {
            x if x == ffi::GGML_FTYPE_ALL_F32 => None,
            x if x == ffi::GGML_FTYPE_MOSTLY_F16 => None,
            x if x == QuantizationType::Q4_0 as i32 => Some(QuantizationType::Q4_0),
            x if x == QuantizationType::Q4_1 as i32 => Some(QuantizationType::Q4_1),
            x if x == QuantizationType::Q5_0 as i32 => Some(QuantizationType::Q5_0),
            x if x == QuantizationType::Q5_1 as i32 => Some(QuantizationType::Q5_1),
            x if x == QuantizationType::Q8_0 as i32 => Some(QuantizationType::Q8_0),
            x if x == QuantizationType::Q2_K as i32 => Some(QuantizationType::Q2_K),
            x if x == QuantizationType::Q3_K as i32 => Some(QuantizationType::Q3_K),
            x if x == QuantizationType::Q4_K as i32 => Some(QuantizationType::Q4_K),
            x if x == QuantizationType::Q5_K as i32 => Some(QuantizationType::Q5_K),
            x if x == QuantizationType::Q6_K as i32 => Some(QuantizationType::Q6_K),
            _ => None,
        };

        Ok(qtype)
    }

    /// Estimate the size of a quantized model given the original model path and target quantization type
    ///
    /// # Returns
    /// Estimated size in bytes of the quantized model
    ///
    /// # Example
    /// ```no_run
    /// use whisper_cpp_plus::{WhisperQuantize, QuantizationType};
    ///
    /// let estimated_size = WhisperQuantize::estimate_quantized_size(
    ///     "models/ggml-base.bin",
    ///     QuantizationType::Q5_0
    /// ).unwrap_or(0);
    ///
    /// println!("Estimated after Q5_0: {} MB", estimated_size / 1024 / 1024);
    /// ```
    pub fn estimate_quantized_size<P: AsRef<Path>>(
        model_path: P,
        qtype: QuantizationType,
    ) -> Result<u64> {
        let path = model_path.as_ref();
        let metadata = std::fs::metadata(path)
            .map_err(|e| QuantizeError::QuantizationFailed(format!("Failed to read model file: {}", e)))?;

        let original_size = metadata.len();
        let estimated_size = (original_size as f64 * qtype.size_factor() as f64) as u64;

        Ok(estimated_size)
    }
}

// Thread-local storage for callback data
thread_local! {
    static CALLBACK_DATA: std::cell::RefCell<Option<Arc<Mutex<dyn Fn(f32) + Send>>>> =
        std::cell::RefCell::new(None);
}

// FFI callback function that forwards to the Rust callback
extern "C" fn quantize_progress_callback(progress: f32) {
    CALLBACK_DATA.with(|data| {
        if let Some(callback) = data.borrow().as_ref() {
            if let Ok(cb) = callback.lock() {
                cb(progress);
            }
        }
    });
}

// Helper function to convert Path to CString
fn path_to_cstring(path: &Path) -> Result<CString> {
    let path_str = path.to_str()
        .ok_or_else(|| QuantizeError::QuantizationFailed("Invalid UTF-8 in path".to_string()))?;

    CString::new(path_str)
        .map_err(|_| QuantizeError::QuantizationFailed("Path contains null byte".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_type_names() {
        assert_eq!(QuantizationType::Q4_0.name(), "Q4_0");
        assert_eq!(QuantizationType::Q5_1.name(), "Q5_1");
        assert_eq!(QuantizationType::Q8_0.name(), "Q8_0");
        assert_eq!(QuantizationType::Q3_K.name(), "Q3_K");
    }

    #[test]
    fn test_quantization_type_from_str() {
        assert_eq!("q4_0".parse::<QuantizationType>().unwrap(), QuantizationType::Q4_0);
        assert_eq!("Q5_1".parse::<QuantizationType>().unwrap(), QuantizationType::Q5_1);
        assert_eq!("q8_0".parse::<QuantizationType>().unwrap(), QuantizationType::Q8_0);
        assert_eq!("Q3K".parse::<QuantizationType>().unwrap(), QuantizationType::Q3_K);
        assert!("invalid".parse::<QuantizationType>().is_err());
    }

    #[test]
    fn test_size_factors() {
        for qtype in QuantizationType::all() {
            let factor = qtype.size_factor();
            assert!(factor > 0.0 && factor < 1.0,
                "{} has invalid size factor: {}", qtype, factor);
        }
    }
}
