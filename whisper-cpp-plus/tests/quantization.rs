//! Integration tests for model quantization functionality

#![cfg(feature = "quantization")]

mod common;

use common::TestModels;
use std::fs;
use std::path::Path;
use whisper_cpp_plus::{ModelQuantizer, QuantizationType};

#[test]
fn test_quantization_types() {
    let types = [
        QuantizationType::Q4_0,
        QuantizationType::Q4_1,
        QuantizationType::Q5_0,
        QuantizationType::Q5_1,
        QuantizationType::Q8_0,
        QuantizationType::Q2_K,
        QuantizationType::Q3_K,
        QuantizationType::Q4_K,
        QuantizationType::Q5_K,
        QuantizationType::Q6_K,
    ];

    for qtype in &types {
        assert!(!qtype.name().is_empty());

        let factor = qtype.size_factor();
        assert!(factor > 0.0 && factor < 1.0,
            "{} has invalid size factor: {}", qtype, factor);
    }
}

#[test]
fn test_quantization_type_parsing() {
    assert_eq!(QuantizationType::from_str("Q4_0"), Some(QuantizationType::Q4_0));
    assert_eq!(QuantizationType::from_str("q4_0"), Some(QuantizationType::Q4_0));
    assert_eq!(QuantizationType::from_str("Q40"), Some(QuantizationType::Q4_0));

    assert_eq!(QuantizationType::from_str("Q5_K"), Some(QuantizationType::Q5_K));
    assert_eq!(QuantizationType::from_str("q5k"), Some(QuantizationType::Q5_K));

    assert_eq!(QuantizationType::from_str("invalid"), None);
    assert_eq!(QuantizationType::from_str(""), None);
}

#[test]
fn test_quantization_display() {
    assert_eq!(format!("{}", QuantizationType::Q4_0), "Q4_0");
    assert_eq!(format!("{}", QuantizationType::Q5_K), "Q5_K");
}

#[test]
fn test_quantize_model() {
    let Some(model_path) = TestModels::tiny_en() else {
        eprintln!("Skipping: model not found. Run `cargo xtask test-setup`");
        return;
    };

    let output_path = model_path.with_file_name("ggml-tiny.en-q5_0.bin");
    let _ = fs::remove_file(&output_path);

    let result = ModelQuantizer::quantize_model_file(
        model_path.to_str().unwrap(),
        output_path.to_str().unwrap(),
        QuantizationType::Q5_0,
    );

    assert!(result.is_ok(), "Quantization failed: {:?}", result);
    assert!(output_path.exists(), "Output file was not created");

    let input_size = fs::metadata(&model_path).unwrap().len();
    let output_size = fs::metadata(&output_path).unwrap().len();
    assert!(output_size < input_size,
        "Quantized model should be smaller: {} >= {}", output_size, input_size);

    let _ = fs::remove_file(&output_path);
}

#[test]
fn test_quantize_with_progress() {
    let Some(model_path) = TestModels::tiny_en() else {
        eprintln!("Skipping: model not found. Run `cargo xtask test-setup`");
        return;
    };

    let output_path = model_path.with_file_name("ggml-tiny.en-q4_0.bin");
    let _ = fs::remove_file(&output_path);

    let result = ModelQuantizer::quantize_model_file_with_progress(
        model_path.to_str().unwrap(),
        output_path.to_str().unwrap(),
        QuantizationType::Q4_0,
        |progress| {
            assert!(progress >= 0.0 && progress <= 1.0, "Invalid progress value");
        },
    );

    assert!(result.is_ok(), "Quantization failed: {:?}", result);
    assert!(output_path.exists(), "Output file was not created");

    let _ = fs::remove_file(&output_path);
}

#[test]
fn test_get_model_quantization_type() {
    let Some(model_path) = TestModels::tiny_en() else {
        eprintln!("Skipping: model not found. Run `cargo xtask test-setup`");
        return;
    };

    let result = ModelQuantizer::get_model_quantization_type(model_path.to_str().unwrap());
    assert!(result.is_ok(), "Failed to check model type: {:?}", result);

    match result.unwrap() {
        Some(qtype) => println!("Model is quantized as: {}", qtype),
        None => println!("Model is in full precision"),
    }
}

#[test]
fn test_estimate_quantized_size() {
    let Some(model_path) = TestModels::tiny_en() else {
        eprintln!("Skipping: model not found. Run `cargo xtask test-setup`");
        return;
    };

    let original_size = fs::metadata(&model_path).unwrap().len();

    for qtype in QuantizationType::all() {
        let estimated = ModelQuantizer::estimate_quantized_size(model_path.to_str().unwrap(), *qtype).unwrap();

        assert!(estimated < original_size,
            "{} estimation {} >= original {}", qtype, estimated, original_size);

        let expected = (original_size as f64 * qtype.size_factor() as f64) as u64;
        let diff = if estimated > expected {
            estimated - expected
        } else {
            expected - estimated
        };

        let margin = (expected as f64 * 0.1) as u64;
        assert!(diff < margin,
            "{}: estimated {} differs too much from expected {} (diff: {})",
            qtype, estimated, expected, diff);
    }
}

#[test]
fn test_error_handling() {
    let result = ModelQuantizer::quantize_model_file(
        "non_existent_model.bin",
        "output.bin",
        QuantizationType::Q4_0,
    );
    assert!(result.is_err(), "Should fail with non-existent input");

    let result = ModelQuantizer::get_model_quantization_type("non_existent.bin");
    assert!(result.is_err(), "Should fail with non-existent file");

    let result = ModelQuantizer::estimate_quantized_size(
        "non_existent.bin",
        QuantizationType::Q5_0
    );
    assert!(result.is_err(), "Should fail with non-existent file");
}
