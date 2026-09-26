use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use whisper_cpp_plus::{FullParams, SamplingStrategy, WhisperContext, WhisperState};

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
fn test_jfk_segment_times_are_milliseconds() {
    let Some(model_path) = find_whisper_model() else {
        eprintln!(
            "Skipping: model not found. Set WHISPER_TEST_MODEL_DIR or run `cargo xtask test-setup`"
        );
        return;
    };
    let Some(audio_path) = find_jfk_audio() else {
        eprintln!("Skipping: JFK audio not found. Set WHISPER_TEST_AUDIO_DIR or run `cargo xtask test-setup`");
        return;
    };

    let audio = load_wav_file(&audio_path).expect("Failed to load JFK audio");
    let audio_ms = audio.len() as i64 * 1000 / 16000;

    let ctx = WhisperContext::new(&model_path).expect("Failed to load model");
    let result = ctx
        .transcribe_with_full_params(
            &audio,
            FullParams::new(SamplingStrategy::Greedy { best_of: 1 }),
        )
        .expect("Failed to transcribe");

    let last = result
        .segments
        .last()
        .expect("Should have at least one segment");

    // jfk.wav is ~11 s. whisper.cpp reports centiseconds; if they leaked through unconverted,
    // the last segment would end around 1100 instead of 11000.
    assert!(
        last.end_ms > audio_ms * 3 / 4 && last.end_ms <= audio_ms + 1000,
        "last segment ends at {} ms, expected close to the {} ms audio length",
        last.end_ms,
        audio_ms
    );
    assert!((last.end_seconds() - last.end_ms as f64 / 1000.0).abs() < f64::EPSILON);
}

#[test]
fn test_state_getters_reject_out_of_range_indices() {
    let Some(model_path) = find_whisper_model() else {
        eprintln!(
            "Skipping: model not found. Set WHISPER_TEST_MODEL_DIR or run `cargo xtask test-setup`"
        );
        return;
    };
    let Some(audio_path) = find_jfk_audio() else {
        eprintln!("Skipping: JFK audio not found. Set WHISPER_TEST_AUDIO_DIR or run `cargo xtask test-setup`");
        return;
    };

    let audio = load_wav_file(&audio_path).expect("Failed to load JFK audio");
    let ctx = WhisperContext::new(&model_path).expect("Failed to load model");
    let mut state = WhisperState::new(&ctx).expect("Failed to create state");
    state
        .full(
            FullParams::new(SamplingStrategy::Greedy { best_of: 1 }),
            &audio,
        )
        .expect("Failed to transcribe");

    let n_segments = state.full_n_segments();
    assert!(n_segments > 0);
    let n_tokens = state.full_n_tokens(0);
    assert!(n_tokens > 0);

    // In-range access works.
    assert!(state.full_get_segment_text(0).is_ok());
    assert!(state.full_get_token_text(0, 0).is_ok());
    assert!(state.full_get_token_data(0, 0).is_some());
    let no_speech = state.full_get_segment_no_speech_prob(0);
    assert!((0.0..=1.0).contains(&no_speech));

    // Result/Option getters report out-of-range indices.
    assert!(state.full_get_segment_text(n_segments).is_err());
    assert!(state.full_get_segment_text(-1).is_err());
    assert!(state.full_get_token_text(0, n_tokens).is_err());
    assert!(state.full_get_token_text(n_segments, 0).is_err());
    assert!(state.full_get_token_data(0, n_tokens).is_none());
    assert!(state.full_get_token_data(0, -1).is_none());

    // Plain-value getters panic instead of reading out of bounds in C.
    let panics = |f: &dyn Fn()| catch_unwind(AssertUnwindSafe(f)).is_err();
    assert!(panics(&|| {
        state.full_get_segment_timestamps(n_segments);
    }));
    assert!(panics(&|| {
        state.full_get_segment_speaker_turn_next(-1);
    }));
    assert!(panics(&|| {
        state.full_get_segment_no_speech_prob(n_segments);
    }));
    assert!(panics(&|| {
        state.full_n_tokens(n_segments);
    }));
    assert!(panics(&|| {
        state.full_get_token_id(0, n_tokens);
    }));
    assert!(panics(&|| {
        state.full_get_token_prob(0, n_tokens);
    }));
}

/// Loads the model and jfk.wav, or returns `None` (skip) if either is missing.
fn jfk_fixture() -> Option<(WhisperContext, Vec<f32>)> {
    let Some(model_path) = find_whisper_model() else {
        eprintln!(
            "Skipping: model not found. Set WHISPER_TEST_MODEL_DIR or run `cargo xtask test-setup`"
        );
        return None;
    };
    let Some(audio_path) = find_jfk_audio() else {
        eprintln!("Skipping: JFK audio not found. Set WHISPER_TEST_AUDIO_DIR or run `cargo xtask test-setup`");
        return None;
    };
    let audio = load_wav_file(&audio_path).expect("Failed to load JFK audio");
    let ctx = WhisperContext::new(&model_path).expect("Failed to load model");
    Some((ctx, audio))
}

fn greedy_params() -> FullParams {
    FullParams::new(SamplingStrategy::Greedy { best_of: 1 })
}

fn assert_segments_ordered(result: &whisper_cpp_plus::TranscriptionResult) {
    for (i, segment) in result.segments.iter().enumerate() {
        assert!(
            segment.start_ms <= segment.end_ms,
            "segment {i} ends before it starts: {segment:?}"
        );
        if i > 0 {
            let previous = &result.segments[i - 1];
            assert!(
                segment.start_ms >= previous.end_ms,
                "segment {i} overlaps the previous one: {previous:?} then {segment:?}"
            );
        }
    }
}

#[test]
fn test_full_parallel_merges_chunks_on_original_timeline() {
    let Some((ctx, audio)) = jfk_fixture() else {
        return;
    };
    let audio_ms = audio.len() as i64 * 1000 / 16000;
    let split_ms = audio_ms / 2;

    let result = ctx
        .full_parallel(greedy_params(), &audio, 2)
        .expect("full_parallel failed");
    println!("full_parallel(2): {:?}", result.segments);

    assert!(!result.segments.is_empty());
    assert_segments_ordered(&result);

    // The second chunk's segments must be shifted past the split point.
    assert!(
        result.segments.iter().any(|s| s.start_ms >= split_ms),
        "no segment starts after the {split_ms} ms split: {:?}",
        result.segments
    );
    let last = result.segments.last().unwrap();
    assert!(
        last.end_ms > audio_ms * 3 / 4 && last.end_ms <= audio_ms,
        "last segment ends at {} ms, expected close to (and not past) {audio_ms} ms",
        last.end_ms
    );
    // Segments from the first chunk are clamped to it, so they can't overrun the split.
    for segment in result.segments.iter().filter(|s| s.start_ms < split_ms) {
        assert!(
            segment.end_ms <= split_ms,
            "first-chunk segment overruns the {split_ms} ms split: {segment:?}"
        );
    }

    let text = result.text.to_lowercase();
    assert!(
        text.contains("americans"),
        "missing first-chunk text: {text}"
    );
    assert!(
        text.contains("your country"),
        "missing second-chunk text: {text}"
    );
}

#[test]
fn test_full_parallel_respects_offset() {
    let Some((ctx, audio)) = jfk_fixture() else {
        return;
    };
    let audio_ms = audio.len() as i64 * 1000 / 16000;
    let offset_ms = 2000;
    // Chunks split the audio after the offset: the second starts halfway through the rest.
    let split_ms = offset_ms + (audio_ms - offset_ms) / 2;

    let result = ctx
        .full_parallel(greedy_params().offset_ms(offset_ms as i32), &audio, 2)
        .expect("full_parallel failed");
    println!(
        "full_parallel(2, offset {offset_ms} ms): {:?}",
        result.segments
    );

    assert!(!result.segments.is_empty());
    assert_segments_ordered(&result);
    assert!(
        result.segments[0].start_ms >= offset_ms - 100,
        "first segment starts before the offset: {:?}",
        result.segments[0]
    );
    assert!(
        result.segments.iter().any(|s| s.start_ms >= split_ms),
        "no segment starts after the {split_ms} ms split: {:?}",
        result.segments
    );
}

#[test]
fn test_full_parallel_single_chunk_matches_full() {
    let Some((ctx, audio)) = jfk_fixture() else {
        return;
    };

    let single = ctx
        .transcribe_with_full_params(&audio, greedy_params())
        .expect("transcription failed");
    let parallel = ctx
        .full_parallel(greedy_params(), &audio, 1)
        .expect("full_parallel failed");
    assert_eq!(parallel.text, single.text);
    assert_eq!(parallel.segments.len(), single.segments.len());

    // Audio too short to split into chunks falls back to a single transcription.
    let short = &audio[..3];
    let fallback = ctx
        .full_parallel(greedy_params(), short, 4)
        .expect("full_parallel on short audio failed");
    let expected = ctx
        .transcribe_with_full_params(short, greedy_params())
        .expect("transcription failed");
    assert_eq!(fallback.text, expected.text);
}

#[test]
fn test_full_parallel_rejects_invalid_input() {
    let Some((ctx, audio)) = jfk_fixture() else {
        return;
    };
    assert!(ctx.full_parallel(greedy_params(), &audio, 0).is_err());
    assert!(ctx.full_parallel(greedy_params(), &[], 2).is_err());
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
    let mono_samples = [0.1, 0.2, 0.3, 0.4];
    let stereo_samples = [0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4]; // L, R, L, R...

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
