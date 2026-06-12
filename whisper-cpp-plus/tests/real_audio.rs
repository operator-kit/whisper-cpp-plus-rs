use hound;
use std::path::Path;
use whisper_cpp_plus::{FullParams, SamplingStrategy, WhisperContext};

/// Find Whisper model (env var or default paths)
fn find_whisper_model() -> Option<String> {
    if let Ok(dir) = std::env::var("WHISPER_TEST_MODEL_DIR") {
        let p = format!("{}/ggml-tiny.en.bin", dir);
        if Path::new(&p).exists() {
            return Some(p);
        }
    }
    let paths = [
        "tests/models/ggml-tiny.en.bin",
        "../whisper-cpp-plus-sys/whisper.cpp/models/ggml-tiny.en.bin",
        "whisper-cpp-plus-sys/whisper.cpp/models/ggml-tiny.en.bin",
        "../whisper-cpp-plus-sys/whisper.cpp/models/for-tests-ggml-tiny.en.bin",
        "whisper-cpp-plus-sys/whisper.cpp/models/for-tests-ggml-tiny.en.bin",
    ];
    paths
        .iter()
        .find(|p| Path::new(p).exists())
        .map(|s| s.to_string())
}

/// Find JFK audio (env var or default paths)
fn find_jfk_audio() -> Option<String> {
    if let Ok(dir) = std::env::var("WHISPER_TEST_AUDIO_DIR") {
        let p = format!("{}/jfk.wav", dir);
        if Path::new(&p).exists() {
            return Some(p);
        }
    }
    let paths = [
        "../whisper-cpp-plus-sys/whisper.cpp/samples/jfk.wav",
        "whisper-cpp-plus-sys/whisper.cpp/samples/jfk.wav",
    ];
    paths
        .iter()
        .find(|p| Path::new(p).exists())
        .map(|s| s.to_string())
}

/// Load a WAV file and convert to f32 samples
fn load_wav_file<P: AsRef<Path>>(path: P) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    // Validate it's 16kHz (whisper requirement)
    if spec.sample_rate != 16000 {
        return Err(format!("Expected 16kHz sample rate, got {}Hz", spec.sample_rate).into());
    }

    // Convert samples to f32 normalized to [-1, 1]
    let samples: Result<Vec<f32>, _> = match spec.sample_format {
        hound::SampleFormat::Int => match spec.bits_per_sample {
            16 => reader
                .samples::<i16>()
                .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
                .collect(),
            _ => return Err(format!("Unsupported bit depth: {}", spec.bits_per_sample).into()),
        },
        hound::SampleFormat::Float => reader.samples::<f32>().collect(),
    };

    samples.map_err(|e| e.into())
}

#[test]
fn test_jfk_transcription() {
    // Skip if model doesn't exist (env var or default paths)
    let model_path = find_whisper_model();
    if model_path.is_none() {
        eprintln!(
            "Skipping: model not found. Set WHISPER_TEST_MODEL_DIR or run `cargo xtask test-setup`"
        );
        return;
    }
    let model_path = model_path.unwrap();

    // Load the JFK audio sample
    let audio_path = find_jfk_audio();
    if audio_path.is_none() {
        eprintln!("Skipping: JFK audio not found. Set WHISPER_TEST_AUDIO_DIR or run `cargo xtask test-setup`");
        return;
    }
    let audio_path = audio_path.unwrap();

    let audio = load_wav_file(&audio_path).expect("Failed to load JFK audio");

    // Create context and transcribe
    let ctx = WhisperContext::new(&model_path).expect("Failed to load model");
    let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

    let result = ctx
        .transcribe_with_full_params(&audio, params)
        .expect("Failed to transcribe");

    println!("Transcription: {}", result.text);
    println!("Number of segments: {}", result.segments.len());

    // Verify key phrases are present (case-insensitive)
    let text_lower = result.text.to_lowercase();

    // Check for key phrases from JFK's famous quote
    assert!(
        text_lower.contains("fellow americans") || text_lower.contains("fellow american"),
        "Should contain 'fellow Americans'"
    );
    assert!(
        text_lower.contains("ask not") || text_lower.contains("asked not"),
        "Should contain 'ask not'"
    );
    assert!(
        text_lower.contains("country") || text_lower.contains("countries"),
        "Should contain 'country'"
    );

    // Should have at least one segment
    assert!(
        !result.segments.is_empty(),
        "Should have at least one segment"
    );

    // Segments should have valid timestamps
    for segment in &result.segments {
        assert!(
            segment.start_ms >= 0,
            "Segment start time should be non-negative"
        );
        assert!(
            segment.end_ms > segment.start_ms,
            "Segment end should be after start"
        );
        assert!(!segment.text.is_empty(), "Segment text should not be empty");
    }
}

#[test]
fn test_audio_duration_handling() {
    let Some(model_path) = find_whisper_model() else {
        eprintln!(
            "Skipping: model not found. Set WHISPER_TEST_MODEL_DIR or run `cargo xtask test-setup`"
        );
        return;
    };

    let ctx = WhisperContext::new(&model_path).expect("Failed to load model");

    // Test various audio durations
    let test_cases = vec![
        (16000, "1 second"),        // 1 second
        (16000 * 5, "5 seconds"),   // 5 seconds
        (16000 * 30, "30 seconds"), // 30 seconds
    ];

    for (sample_count, description) in test_cases {
        println!("Testing {} of silence", description);

        // Create silence audio
        let audio = vec![0.0f32; sample_count];

        // Should handle without crashing
        let result = ctx.transcribe(&audio);
        assert!(result.is_ok(), "Should handle {} of audio", description);
    }
}

#[test]
fn test_stereo_to_mono_conversion() {
    // This test documents that stereo audio needs to be converted to mono
    // before passing to whisper - this is currently the user's responsibility

    // Simulate stereo audio by interleaving samples
    let mono_samples = vec![0.1, 0.2, 0.3, 0.4];
    let stereo_samples = vec![0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4]; // L, R, L, R...

    // Convert stereo to mono by averaging channels
    let converted_mono: Vec<f32> = stereo_samples
        .chunks(2)
        .map(|lr| (lr[0] + lr[1]) / 2.0)
        .collect();

    assert_eq!(converted_mono.len(), mono_samples.len());
    for (converted, expected) in converted_mono.iter().zip(mono_samples.iter()) {
        assert!((converted - expected).abs() < 0.001);
    }
}
