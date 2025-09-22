use whisper_cpp_rs::WhisperContext;

fn main() {
    println!("Starting minimal test...");

    // Try to load model
    let model_path = "tests/models/ggml-tiny.en.bin";
    println!("Loading model from: {}", model_path);

    match WhisperContext::new(model_path) {
        Ok(ctx) => {
            println!("Model loaded successfully!");
            println!("Model info:");
            println!("  - Vocabulary size: {}", ctx.n_vocab());
            println!("  - Audio context: {}", ctx.n_audio_ctx());
            println!("  - Text context: {}", ctx.n_text_ctx());
            println!("  - Multilingual: {}", ctx.is_multilingual());
            println!("Model will be dropped now...");
        }
        Err(e) => {
            println!("Failed to load model: {}", e);
        }
    }

    println!("Test completed!");
}