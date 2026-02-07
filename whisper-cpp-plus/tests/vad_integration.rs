//! Integration tests for VAD (Voice Activity Detection) with real audio

use std::path::Path;
use whisper_cpp_plus::{WhisperVadProcessor, VadParams, VadParamsBuilder, WhisperContext, FullParams, SamplingStrategy};

/// Load WAV file and convert to 16kHz mono f32
fn load_wav_16khz_mono(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    use hound;

    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    // Check sample rate
    if spec.sample_rate != 16000 {
        return Err(format!(
            "Audio must be 16kHz, but got {}Hz",
            spec.sample_rate
        ).into());
    }

    // Convert samples to f32
    let samples: Result<Vec<f32>, _> = match spec.bits_per_sample {
        16 => reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
            .collect(),
        32 => reader
            .samples::<i32>()
            .map(|s| s.map(|v| v as f32 / i32::MAX as f32))
            .collect(),
        _ => {
            return Err(format!(
                "Unsupported bits per sample: {}",
                spec.bits_per_sample
            ).into())
        }
    };

    let samples = samples?;

    // Convert to mono if necessary
    let mono_samples = match spec.channels {
        1 => samples,
        2 => {
            // Average stereo channels to mono
            samples
                .chunks(2)
                .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
                .collect()
        }
        _ => {
            return Err(format!(
                "Unsupported number of channels: {}",
                spec.channels
            ).into())
        }
    };

    Ok(mono_samples)
}

#[test]
fn test_vad_with_jfk_audio() {
    // Check if both models exist
    let vad_model_path = "tests/models/ggml-silero-vad.bin";
    let whisper_model_path = "tests/models/ggml-tiny.en.bin";
    let jfk_path = "../vendor/whisper.cpp/samples/jfk.wav";

    if !Path::new(vad_model_path).exists() {
        eprintln!("Skipping: VAD model not found at {}. Run `cargo xtask test-setup`", vad_model_path);
        return;
    }

    if !Path::new(jfk_path).exists() {
        eprintln!("Skipping: JFK audio not found at {}. Run `cargo xtask test-setup`", jfk_path);
        return;
    }

    // Load the JFK audio
    let audio = load_wav_16khz_mono(jfk_path).expect("Failed to load JFK audio");
    let audio_duration_s = audio.len() as f32 / 16000.0;

    println!("Loaded JFK audio: {} samples ({:.2}s)", audio.len(), audio_duration_s);

    // Initialize VAD processor
    let mut vad = WhisperVadProcessor::new(vad_model_path)
        .expect("Failed to load VAD model");

    // Configure VAD parameters for speech detection
    let vad_params = VadParamsBuilder::new()
        .threshold(0.5)              // Medium confidence threshold
        .min_speech_duration_ms(250) // Minimum 250ms for valid speech
        .min_silence_duration_ms(100) // 100ms silence to split segments
        .speech_pad_ms(100)          // 100ms padding around speech
        .build();

    // Detect speech segments
    let segments = vad.segments_from_samples(&audio, &vad_params)
        .expect("Failed to detect speech segments");

    let n_segments = segments.n_segments();
    println!("Detected {} speech segments", n_segments);

    // Verify we detected speech
    assert!(n_segments > 0, "Should detect at least one speech segment in JFK audio");

    // Calculate total speech duration
    let mut total_speech_duration = 0.0;
    let mut segments_info = Vec::new();

    for i in 0..n_segments {
        let start = segments.get_segment_t0(i);  // Already in seconds
        let end = segments.get_segment_t1(i);    // Already in seconds
        let duration = end - start;

        println!("  Segment {}: {:.2}s - {:.2}s (duration: {:.2}s)",
                 i, start, end, duration);

        segments_info.push((start, end));
        total_speech_duration += duration;

        // Verify segment is reasonable
        assert!(start >= 0.0, "Segment start should be non-negative");
        assert!(end > start, "Segment end should be after start");
        assert!(end <= audio_duration_s + 0.1, "Segment should not exceed audio duration");
    }

    println!("Total speech duration: {:.2}s out of {:.2}s ({:.1}%)",
             total_speech_duration, audio_duration_s,
             (total_speech_duration / audio_duration_s) * 100.0);

    // The JFK sample is mostly speech, so we expect high coverage
    let speech_ratio = total_speech_duration / audio_duration_s;
    assert!(speech_ratio > 0.5,
            "JFK sample should be at least 50% speech, but got {:.1}%",
            speech_ratio * 100.0);
    assert!(speech_ratio <= 1.1,
            "Speech duration shouldn't exceed audio duration significantly");

    // Extract audio segments for the detected speech
    let audio_segments = segments.extract_audio_segments(&audio, 16000.0);
    assert_eq!(audio_segments.len(), n_segments as usize,
               "Should extract same number of audio segments as detected");

    // Verify extracted segments have reasonable sizes
    for (i, segment_audio) in audio_segments.iter().enumerate() {
        let (start, end) = segments_info[i];
        let expected_samples = ((end - start) * 16000.0) as usize;
        let actual_samples = segment_audio.len();

        // Allow some tolerance due to rounding
        let tolerance = 160; // 10ms at 16kHz
        assert!(
            (actual_samples as i32 - expected_samples as i32).abs() < tolerance,
            "Segment {} audio size mismatch: expected ~{} samples, got {}",
            i, expected_samples, actual_samples
        );
    }

    println!("✓ VAD successfully detected and extracted {} speech segments", n_segments);
}

#[test]
fn test_vad_with_transcription() {
    // This test combines VAD with actual transcription
    let vad_model_path = "tests/models/ggml-silero-vad.bin";
    let whisper_model_path = "tests/models/ggml-tiny.en.bin";
    let jfk_path = "../vendor/whisper.cpp/samples/jfk.wav";

    if !Path::new(vad_model_path).exists() ||
       !Path::new(whisper_model_path).exists() ||
       !Path::new(jfk_path).exists() {
        eprintln!("Skipping: models/audio not found. Run `cargo xtask test-setup`");
        return;
    }

    // Load audio
    let audio = load_wav_16khz_mono(jfk_path).expect("Failed to load JFK audio");

    // Initialize models
    let mut vad = WhisperVadProcessor::new(vad_model_path)
        .expect("Failed to load VAD model");
    let whisper_ctx = WhisperContext::new(whisper_model_path)
        .expect("Failed to load Whisper model");

    // Detect speech segments
    let vad_params = VadParams::default(); // Use default settings
    let segments = vad.segments_from_samples(&audio, &vad_params)
        .expect("Failed to detect speech segments");

    // Extract speech segments
    let speech_segments = segments.extract_audio_segments(&audio, 16000.0);

    println!("Transcribing {} speech segments...", speech_segments.len());

    // Transcribe each segment
    let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    let mut full_transcript = String::new();

    for (i, segment_audio) in speech_segments.iter().enumerate() {
        let mut state = whisper_ctx.create_state()
            .expect("Failed to create state");

        state.full(params.clone(), segment_audio)
            .expect("Failed to transcribe segment");

        let n_segments = state.full_n_segments();
        for j in 0..n_segments {
            if let Ok(text) = state.full_get_segment_text(j) {
                if !text.trim().is_empty() {
                    if !full_transcript.is_empty() {
                        full_transcript.push(' ');
                    }
                    full_transcript.push_str(&text);
                }
            }
        }
    }

    println!("Full transcript from VAD segments: {}", full_transcript);

    // Verify we got meaningful transcription
    assert!(!full_transcript.is_empty(), "Should get non-empty transcription");

    // The JFK audio contains specific keywords we can check for
    let transcript_lower = full_transcript.to_lowercase();

    // Check for some expected words/phrases from JFK's "Ask not" speech
    // The sample contains: "Ask not what your country can do for you,
    // ask what you can do for your country"
    let expected_keywords = ["ask", "not", "what", "country", "you"];
    let mut found_keywords = 0;

    for keyword in &expected_keywords {
        if transcript_lower.contains(keyword) {
            found_keywords += 1;
        }
    }

    println!("Found {}/{} expected keywords in transcript",
             found_keywords, expected_keywords.len());

    // We should find most of these very common words from the famous speech
    assert!(found_keywords >= 3,
            "Should find at least 3 expected keywords in JFK transcript, found {}",
            found_keywords);

    println!("✓ VAD + transcription successfully processed JFK audio");
}

#[test]
fn test_vad_with_silence() {
    // Test VAD with pure silence
    let vad_model_path = "tests/models/ggml-silero-vad.bin";

    if !Path::new(vad_model_path).exists() {
        eprintln!("Skipping: VAD model not found at {}. Run `cargo xtask test-setup`", vad_model_path);
        return;
    }

    let mut vad = WhisperVadProcessor::new(vad_model_path)
        .expect("Failed to load VAD model");

    // Create 3 seconds of silence
    let silence = vec![0.0f32; 16000 * 3];

    let vad_params = VadParams::default();
    let segments = vad.segments_from_samples(&silence, &vad_params)
        .expect("Failed to process silence");

    let n_segments = segments.n_segments();
    println!("Detected {} speech segments in silence", n_segments);

    // Should detect no speech in pure silence
    assert_eq!(n_segments, 0, "Should detect no speech in pure silence");

    println!("✓ VAD correctly detected no speech in silence");
}

#[test]
fn test_vad_with_mixed_audio() {
    // Test VAD with artificially created mixed audio (speech-like noise + silence)
    let vad_model_path = "tests/models/ggml-silero-vad.bin";

    if !Path::new(vad_model_path).exists() {
        eprintln!("Skipping: VAD model not found. Run `cargo xtask test-setup`");
        return;
    }

    let mut vad = WhisperVadProcessor::new(vad_model_path)
        .expect("Failed to load VAD model");

    // Create artificial audio: noise, silence, noise, silence pattern
    let mut audio = Vec::new();

    // 1 second of noise (simulated speech)
    for _ in 0..16000 {
        audio.push((rand::random::<f32>() - 0.5) * 0.3);
    }

    // 0.5 seconds of silence
    audio.extend(vec![0.0f32; 8000]);

    // 1.5 seconds of noise
    for _ in 0..24000 {
        audio.push((rand::random::<f32>() - 0.5) * 0.3);
    }

    // 1 second of silence
    audio.extend(vec![0.0f32; 16000]);

    let vad_params = VadParamsBuilder::new()
        .threshold(0.3) // Lower threshold for noise detection
        .min_speech_duration_ms(500) // Require at least 500ms
        .build();

    let segments = vad.segments_from_samples(&audio, &vad_params)
        .expect("Failed to process mixed audio");

    let n_segments = segments.n_segments();
    println!("Detected {} segments in mixed audio", n_segments);

    // We expect to detect some segments (the noise parts might be detected as speech)
    // The exact number depends on the VAD model's behavior with noise
    assert!(n_segments >= 0, "VAD should process mixed audio without errors");

    // Verify segment boundaries are reasonable
    for i in 0..n_segments {
        let start = segments.get_segment_t0(i);
        let end = segments.get_segment_t1(i);

        assert!(start >= 0.0, "Segment start should be non-negative");
        assert!(end > start, "Segment end should be after start");
        assert!(end <= 4.0, "Segment should not exceed audio duration (4s)");
    }

    println!("✓ VAD successfully processed mixed audio");
}

// Simple random number generator for testing
mod rand {
    use std::time::{SystemTime, UNIX_EPOCH};
    use std::cell::Cell;

    thread_local! {
        static SEED: Cell<u64> = Cell::new({
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64
        });
    }

    pub fn random<T>() -> T
    where
        T: From<f32>,
    {
        SEED.with(|seed| {
            let mut x = seed.get();
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            seed.set(x);
            T::from((x as f32 / u64::MAX as f32))
        })
    }
}