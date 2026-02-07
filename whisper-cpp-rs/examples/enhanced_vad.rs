//! Example demonstrating enhanced VAD with segment aggregation

use std::path::Path;
use whisper_cpp_rs::{WhisperContext, TranscriptionParams};
use whisper_cpp_rs::enhanced::vad::{
    EnhancedWhisperVadProcessor, EnhancedVadParamsBuilder
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Check for model files
    let model_path = "tests/models/ggml-base.en.bin";
    let vad_model_path = "tests/models/ggml-silero-vad.bin";

    if !Path::new(model_path).exists() {
        eprintln!("Model file not found at: {}", model_path);
        eprintln!("Please download a Whisper model from https://huggingface.co/ggerganov/whisper.cpp");
        return Ok(());
    }

    if !Path::new(vad_model_path).exists() {
        eprintln!("VAD model file not found at: {}", vad_model_path);
        eprintln!("Please download the Silero VAD model");
        return Ok(());
    }

    println!("Loading models...");

    // Load models
    let ctx = WhisperContext::new(model_path)?;
    let mut vad = EnhancedWhisperVadProcessor::new(vad_model_path)?;

    // Configure enhanced VAD with segment aggregation
    let vad_params = EnhancedVadParamsBuilder::new()
        .threshold(0.5)
        .max_segment_duration(30.0)  // Aggregate up to 30 seconds
        .merge_segments(true)         // Merge adjacent segments
        .min_gap_ms(100)              // Minimum 100ms gap to keep segments separate
        .speech_pad_ms(400)           // Add 400ms padding around speech
        .build();

    println!("Enhanced VAD Parameters:");
    println!("  - Max segment duration: {} seconds", vad_params.max_segment_duration_s);
    println!("  - Merge segments: {}", vad_params.merge_segments);
    println!("  - Min gap: {} ms", vad_params.min_gap_ms);
    println!("  - Speech padding: {} ms", vad_params.base.speech_pad_ms);

    // Load audio (you would load real audio here)
    let audio = load_audio_example()?;

    println!("\nProcessing audio with enhanced VAD...");

    // Step 1: Preprocess with enhanced VAD
    let chunks = vad.process_with_aggregation(&audio, &vad_params)?;

    println!("Found {} speech chunks after aggregation", chunks.len());

    // Configure transcription parameters
    let params = TranscriptionParams::builder()
        .language("en")
        .enable_timestamps()
        .build();

    // Step 2: Transcribe each chunk
    let mut full_text = String::new();

    for (i, chunk) in chunks.iter().enumerate() {
        println!("\nProcessing chunk {} [{:.2}s - {:.2}s], duration: {:.2}s",
            i + 1,
            chunk.offset_seconds,
            chunk.offset_seconds + chunk.duration_seconds,
            chunk.duration_seconds
        );

        // Option A: Standard transcription
        let text = ctx.transcribe(&chunk.audio)?;

        // Option B: Enhanced transcription with fallback (uncomment to use)
        // let result = ctx.transcribe_with_params_enhanced(&chunk.audio, params.clone())?;
        // let text = result.text;

        println!("  Text: {}", text);

        if !full_text.is_empty() {
            full_text.push(' ');
        }
        full_text.push_str(&text);
    }

    println!("\n=== Full Transcription ===");
    println!("{}", full_text);

    // Show efficiency gains
    let original_duration = audio.len() as f32 / 16000.0;
    let processed_duration: f32 = chunks.iter()
        .map(|c| c.duration_seconds)
        .sum();

    println!("\n=== Efficiency Stats ===");
    println!("Original audio duration: {:.2}s", original_duration);
    println!("Processed audio duration: {:.2}s", processed_duration);
    println!("Reduction: {:.1}%", (1.0 - processed_duration / original_duration) * 100.0);

    Ok(())
}

fn load_audio_example() -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // Try multiple locations for the audio file
    let paths = vec![
        "vendor/whisper.cpp/samples/jfk.wav",
        "../vendor/whisper.cpp/samples/jfk.wav",
        "samples/audio.wav",
    ];

    for audio_path in &paths {
        if Path::new(audio_path).exists() {
            println!("Loading audio from: {}", audio_path);
            return load_wav_file(audio_path);
        }
    }

    eprintln!("\nError: No audio files found!");
    eprintln!("Please provide audio at one of these locations:");
    for path in &paths {
        eprintln!("  - {}", path);
    }
    eprintln!("\nNote: Synthetic audio generation was removed as it doesn't produce meaningful VAD results.");
    Err("No audio files found for enhanced VAD example".into())
}

fn load_wav_file(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    use hound;

    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    // Check if we need to resample
    if spec.sample_rate != 16000 {
        eprintln!("Warning: Audio sample rate is {}Hz, expected 16000Hz", spec.sample_rate);
        eprintln!("Resampling may be needed for accurate results");
    }

    if spec.channels != 1 {
        eprintln!("Warning: Audio has {} channels, expected mono", spec.channels);
        eprintln!("Using first channel only");
    }

    let samples: Vec<f32> = reader
        .samples::<i16>()
        .step_by(spec.channels as usize) // Take only first channel if stereo
        .map(|s| s.unwrap() as f32 / 32768.0) // Convert to f32 [-1, 1]
        .collect();

    Ok(samples)
}
