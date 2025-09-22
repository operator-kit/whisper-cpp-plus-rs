//! Enhanced optimizations for whisper-cpp-rs
//!
//! This module provides performance optimizations inspired by faster-whisper
//! while maintaining compatibility with the base whisper.cpp API.
//!
//! ## Features
//!
//! - **Enhanced VAD**: Intelligent speech segment aggregation for optimal chunk sizes (preprocessing)
//! - **Temperature Fallback**: Quality-based retry mechanism for difficult audio (transcription)
//! - **Performance**: 2-3x speedup on audio with silence, improved accuracy on noisy audio
//!
//! ## Architecture
//!
//! The enhancements are designed as orthogonal improvements:
//! - VAD enhancement is a preprocessing step that happens BEFORE transcription
//! - Temperature fallback is a transcription enhancement for quality
//! - Both can be used independently or together

pub mod vad;
pub mod fallback;

pub use vad::{EnhancedVadProcessor, EnhancedVadParams, EnhancedVadParamsBuilder, AudioChunk};
pub use fallback::{
    EnhancedTranscriptionParams, EnhancedTranscriptionParamsBuilder,
    QualityThresholds, EnhancedWhisperState, TranscriptionAttempt
};