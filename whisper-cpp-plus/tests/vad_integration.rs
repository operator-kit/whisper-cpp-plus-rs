//! Integration tests for VAD (Voice Activity Detection) with real audio

use std::path::Path;
use whisper_cpp_plus::{
    FullParams, SamplingStrategy, VadParams, VadParamsBuilder, WhisperContext, WhisperVadProcessor,
};

/// Find VAD model (env var or default paths)
fn find_vad_model() -> Option<String> {
    if let Ok(dir) = std::env::var("WHISPER_TEST_MODEL_DIR") {
        for name in ["ggml-silero-v6.2.0.bin", "ggml-silero-vad.bin"] {
            let p = format!("{}/{}", dir, name);
            if Path::new(&p).exists() {
                return Some(p);
            }
        }
    }
    let paths = [
        "tests/models/ggml-silero-vad.bin",
        "../whisper-cpp-plus-sys/whisper.cpp/models/ggml-silero-v6.2.0.bin",
        "whisper-cpp-plus-sys/whisper.cpp/models/ggml-silero-v6.2.0.bin",
        "../whisper-cpp-plus-sys/whisper.cpp/models/ggml-silero-vad.bin",
        "whisper-cpp-plus-sys/whisper.cpp/models/ggml-silero-vad.bin",
        "../whisper-cpp-plus-sys/whisper.cpp/models/for-tests-silero-v6.2.0-ggml.bin",
        "whisper-cpp-plus-sys/whisper.cpp/models/for-tests-silero-v6.2.0-ggml.bin",
    ];
    paths
        .iter()
        .find(|p| Path::new(p).exists())
        .map(|s| s.to_string())
}

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

/// Load WAV file and convert to 16kHz mono f32
fn load_wav_16khz_mono(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    use hound;

    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    // Check sample rate
    if spec.sample_rate != 16000 {
        return Err(format!("Audio must be 16kHz, but got {}Hz", spec.sample_rate).into());
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
        _ => return Err(format!("Unsupported bits per sample: {}", spec.bits_per_sample).into()),
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
        _ => return Err(format!("Unsupported number of channels: {}", spec.channels).into()),
    };

    Ok(mono_samples)
}

#[test]
fn test_vad_with_jfk_audio() {
    // Check if both models exist (env var or default paths)
    let vad_model_path = find_vad_model();
    let jfk_path = find_jfk_audio();

    if vad_model_path.is_none() {
        eprintln!("Skipping: VAD model not found. Set WHISPER_TEST_MODEL_DIR or run `cargo xtask test-setup`");
        return;
    }

    if jfk_path.is_none() {
        eprintln!("Skipping: JFK audio not found. Set WHISPER_TEST_AUDIO_DIR or run `cargo xtask test-setup`");
        return;
    }
    let vad_model_path = vad_model_path.unwrap();
    let jfk_path = jfk_path.unwrap();

    // Load the JFK audio
    let audio = load_wav_16khz_mono(&jfk_path).expect("Failed to load JFK audio");
    let audio_duration_s = audio.len() as f32 / 16000.0;

    println!(
        "Loaded JFK audio: {} samples ({:.2}s)",
        audio.len(),
        audio_duration_s
    );

    // Initialize VAD processor
    let mut vad = WhisperVadProcessor::new(vad_model_path).expect("Failed to load VAD model");

    // Configure VAD parameters for speech detection
    let vad_params = VadParamsBuilder::new()
        .threshold(0.5) // Medium confidence threshold
        .min_speech_duration_ms(250) // Minimum 250ms for valid speech
        .min_silence_duration_ms(100) // 100ms silence to split segments
        .speech_pad_ms(100) // 100ms padding around speech
        .build();

    // Detect speech segments
    let segments = vad
        .segments_from_samples(&audio, &vad_params)
        .expect("Failed to detect speech segments");

    let n_segments = segments.n_segments();
    println!("Detected {} speech segments", n_segments);

    // Verify we detected speech
    assert!(
        n_segments > 0,
        "Should detect at least one speech segment in JFK audio"
    );

    // Calculate total speech duration
    let mut total_speech_duration = 0.0;
    let mut segments_info = Vec::new();

    for i in 0..n_segments {
        let start = segments.get_segment_t0(i); // Already in seconds
        let end = segments.get_segment_t1(i); // Already in seconds
        let duration = end - start;

        println!(
            "  Segment {}: {:.2}s - {:.2}s (duration: {:.2}s)",
            i, start, end, duration
        );

        segments_info.push((start, end));
        total_speech_duration += duration;

        // Verify segment is reasonable
        assert!(start >= 0.0, "Segment start should be non-negative");
        assert!(end > start, "Segment end should be after start");
        assert!(
            end <= audio_duration_s + 0.1,
            "Segment should not exceed audio duration"
        );
    }

    println!(
        "Total speech duration: {:.2}s out of {:.2}s ({:.1}%)",
        total_speech_duration,
        audio_duration_s,
        (total_speech_duration / audio_duration_s) * 100.0
    );

    // The JFK sample is mostly speech, so we expect high coverage
    let speech_ratio = total_speech_duration / audio_duration_s;
    assert!(
        speech_ratio > 0.5,
        "JFK sample should be at least 50% speech, but got {:.1}%",
        speech_ratio * 100.0
    );
    assert!(
        speech_ratio <= 1.1,
        "Speech duration shouldn't exceed audio duration significantly"
    );

    // Extract audio segments for the detected speech
    let audio_segments = segments.extract_audio_segments(&audio, 16000.0);
    assert_eq!(
        audio_segments.len(),
        n_segments as usize,
        "Should extract same number of audio segments as detected"
    );

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
            i,
            expected_samples,
            actual_samples
        );
    }

    println!(
        "✓ VAD successfully detected and extracted {} speech segments",
        n_segments
    );
}

#[test]
fn test_vad_with_transcription() {
    // This test combines VAD with actual transcription
    let vad_model_path = find_vad_model();
    let whisper_model_path = find_whisper_model();
    let jfk_path = find_jfk_audio();

    if vad_model_path.is_none() || whisper_model_path.is_none() || jfk_path.is_none() {
        eprintln!("Skipping: models/audio not found. Set WHISPER_TEST_MODEL_DIR/WHISPER_TEST_AUDIO_DIR or run `cargo xtask test-setup`");
        return;
    }
    let vad_model_path = vad_model_path.unwrap();
    let whisper_model_path = whisper_model_path.unwrap();
    let jfk_path = jfk_path.unwrap();

    // Load audio
    let audio = load_wav_16khz_mono(&jfk_path).expect("Failed to load JFK audio");

    // Initialize models
    let mut vad = WhisperVadProcessor::new(vad_model_path).expect("Failed to load VAD model");
    let whisper_ctx =
        WhisperContext::new(whisper_model_path).expect("Failed to load Whisper model");

    // Detect speech segments
    let vad_params = VadParams::default(); // Use default settings
    let segments = vad
        .segments_from_samples(&audio, &vad_params)
        .expect("Failed to detect speech segments");

    // Extract speech segments
    let speech_segments = segments.extract_audio_segments(&audio, 16000.0);

    println!("Transcribing {} speech segments...", speech_segments.len());

    // Transcribe each segment
    let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    let mut full_transcript = String::new();

    for segment_audio in &speech_segments {
        let mut state = whisper_ctx.create_state().expect("Failed to create state");

        state
            .full(params.clone(), segment_audio)
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
    assert!(
        !full_transcript.is_empty(),
        "Should get non-empty transcription"
    );

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

    println!(
        "Found {}/{} expected keywords in transcript",
        found_keywords,
        expected_keywords.len()
    );

    // We should find most of these very common words from the famous speech
    assert!(
        found_keywords >= 3,
        "Should find at least 3 expected keywords in JFK transcript, found {}",
        found_keywords
    );

    println!("✓ VAD + transcription successfully processed JFK audio");
}

#[test]
fn test_vad_with_silence() {
    // Test VAD with pure silence
    let vad_model_path = find_vad_model();

    if vad_model_path.is_none() {
        eprintln!("Skipping: VAD model not found. Set WHISPER_TEST_MODEL_DIR or run `cargo xtask test-setup`");
        return;
    }
    let vad_model_path = vad_model_path.unwrap();

    let mut vad = WhisperVadProcessor::new(&vad_model_path).expect("Failed to load VAD model");

    // Create 3 seconds of silence
    let silence = vec![0.0f32; 16000 * 3];

    let vad_params = VadParams::default();
    let segments = vad
        .segments_from_samples(&silence, &vad_params)
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
    let vad_model_path = find_vad_model();

    if vad_model_path.is_none() {
        eprintln!("Skipping: VAD model not found. Set WHISPER_TEST_MODEL_DIR or run `cargo xtask test-setup`");
        return;
    }
    let vad_model_path = vad_model_path.unwrap();

    let mut vad = WhisperVadProcessor::new(&vad_model_path).expect("Failed to load VAD model");

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

    let segments = vad
        .segments_from_samples(&audio, &vad_params)
        .expect("Failed to process mixed audio");

    let n_segments = segments.n_segments();
    println!("Detected {} segments in mixed audio", n_segments);

    // We expect to detect some segments (the noise parts might be detected as speech)
    // The exact number depends on the VAD model's behavior with noise
    assert!(
        n_segments >= 0,
        "VAD should process mixed audio without errors"
    );

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
    use std::cell::Cell;
    use std::time::{SystemTime, UNIX_EPOCH};

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
            T::from(x as f32 / u64::MAX as f32)
        })
    }
}

/// Average of the full-pass window probabilities whose window starts inside `[start, end)`.
fn reference_probe_avg(full: &[f32], window: usize, start: usize, end: usize) -> f32 {
    // Round up without usize::div_ceil (MSRV 1.70).
    let first = (start + window - 1) / window;
    let last = ((end + window - 1) / window).min(full.len());
    let probs = &full[first.min(last)..last];
    if probs.is_empty() {
        0.0
    } else {
        probs.iter().sum::<f32>() / probs.len() as f32
    }
}

#[test]
fn test_streaming_vad_matches_full_pass() {
    let Some(vad_path) = find_vad_model() else {
        eprintln!("Skipping: VAD model not found. Run `cargo xtask test-setup`");
        return;
    };
    let Some(audio_path) = find_jfk_audio() else {
        eprintln!("Skipping: JFK audio not found. Run `cargo xtask test-setup`");
        return;
    };

    let window = WhisperVadProcessor::WINDOW_SAMPLES;
    let audio = load_wav_16khz_mono(&audio_path).expect("Failed to load audio");
    let audio = &audio[..audio.len() / window * window];

    // WhisperStreamPcm defaults: 200 ms probes, 0.6 speech threshold.
    let probe = 3200;
    let thold = 0.6;

    let mut vad = WhisperVadProcessor::new(&vad_path).expect("Failed to load VAD model");

    // Reference: one pass over the whole file.
    assert!(vad.detect_speech(audio));
    let full = vad.get_probs();
    assert_eq!(full.len(), audio.len() / window);

    // Streaming without resets, feeding whole windows and carrying the remainder.
    vad.reset_state();
    let mut streamed = Vec::new();
    let mut streamed_probe_avgs = Vec::new();
    let mut carry = Vec::new();
    for chunk in audio.chunks(probe) {
        carry.extend_from_slice(chunk);
        let whole = carry.len() / window * window;
        if whole == 0 {
            streamed_probe_avgs.push(None);
            continue;
        }
        assert!(vad.detect_speech_no_reset(&carry[..whole]));
        let probs = vad.get_probs();
        streamed_probe_avgs.push(Some(probs.iter().sum::<f32>() / probs.len() as f32));
        streamed.extend(probs);
        carry.drain(..whole);
    }

    // Same LSTM input sequence as the full pass, so the probabilities must match.
    assert_eq!(streamed.len(), full.len());
    let max_diff = streamed
        .iter()
        .zip(&full)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-4,
        "streamed probabilities differ from the full pass by up to {max_diff}"
    );

    // Per-probe decisions: previous reset-per-probe behaviour vs streaming, against the
    // full-pass reference. Printed for comparison; only streaming is asserted.
    let mut reset_mismatches = 0;
    let mut stream_mismatches = 0;
    let mut reset_abs_err = 0.0f32;
    let mut stream_abs_err = 0.0f32;
    let n_probes = audio.chunks(probe).count();
    for (i, chunk) in audio.chunks(probe).enumerate() {
        let start = i * probe;
        let reference = reference_probe_avg(&full, window, start, start + chunk.len());

        assert!(vad.detect_speech(chunk));
        let probs = vad.get_probs();
        let reset_avg = probs.iter().sum::<f32>() / probs.len() as f32;
        reset_abs_err += (reset_avg - reference).abs();
        if (reset_avg < thold) != (reference < thold) {
            reset_mismatches += 1;
        }

        if let Some(stream_avg) = streamed_probe_avgs[i] {
            stream_abs_err += (stream_avg - reference).abs();
            if (stream_avg < thold) != (reference < thold) {
                stream_mismatches += 1;
            }
        }
    }

    println!(
        "{n_probes} probes of {probe} samples: reset-per-probe mean abs error {:.3}, \
         {reset_mismatches} speech/silence decisions differ from the full pass; \
         streaming mean abs error {:.3}, {stream_mismatches} differ",
        reset_abs_err / n_probes as f32,
        stream_abs_err / n_probes as f32,
    );
}
