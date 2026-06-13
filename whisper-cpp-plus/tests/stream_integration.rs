mod common;

use common::TestModels;
use std::path::Path;
use whisper_cpp_plus::{
    FullParams, SamplingStrategy, WhisperContext, WhisperStream, WhisperStreamConfig,
};

fn load_wav_as_f32<P: AsRef<Path>>(path: P) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    assert_eq!(spec.sample_rate, 16000, "test audio must be 16kHz");

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => match spec.bits_per_sample {
            16 => reader
                .samples::<i16>()
                .map(|sample| sample.map(|value| value as f32 / i16::MAX as f32))
                .collect::<Result<Vec<_>, _>>()?,
            bits => return Err(format!("unsupported int bit depth: {bits}").into()),
        },
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
    };

    if spec.channels == 1 {
        return Ok(samples);
    }

    Ok(samples
        .chunks(spec.channels as usize)
        .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
        .collect())
}

fn assert_jfk_keywords(text: &str) {
    let lower = text.to_lowercase();
    let keywords = ["ask", "not", "what", "country", "you"];
    let found = keywords
        .iter()
        .filter(|keyword| lower.contains(**keyword))
        .count();

    assert!(
        found >= 3,
        "expected at least 3 JFK keywords, found {found}; transcript: {text}"
    );
}

#[test]
fn test_whisper_stream_flush_transcribes_jfk_fixed_step() {
    let Some(model_path) = TestModels::tiny_en() else {
        eprintln!("Skipping: whisper model not found. Run `cargo xtask test-setup`");
        return;
    };
    let Some(audio_path) = TestModels::jfk_audio() else {
        eprintln!("Skipping: jfk.wav not found. Run `cargo xtask test-setup`");
        return;
    };

    let audio = load_wav_as_f32(&audio_path).expect("failed to load JFK audio");
    let ctx = WhisperContext::new(&model_path).unwrap();
    let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 }).language("en");
    let config = WhisperStreamConfig {
        step_ms: 5000,
        length_ms: 10000,
        keep_ms: 200,
        no_context: true,
        ..Default::default()
    };

    let mut stream = WhisperStream::with_config(&ctx, params, config).unwrap();
    stream.feed_audio(&audio);

    let segments = stream.flush().expect("WhisperStream::flush failed");
    let text = segments
        .iter()
        .map(|segment| segment.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    assert_eq!(stream.buffer_size(), 0);
    assert_eq!(stream.processed_samples(), audio.len() as i64);
    assert!(!segments.is_empty(), "stream should produce segments");
    assert!(!text.trim().is_empty(), "stream should produce text");
    assert_jfk_keywords(&text);
}
