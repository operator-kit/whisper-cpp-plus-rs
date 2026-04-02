use std::path::{Path, PathBuf};
use whisper_cpp_plus::WhisperContext;

fn main() {
    println!("Starting minimal test...");

    let model_path = match find_model("ggml-tiny.en.bin") {
        Some(p) => p,
        None => {
            println!("Model not found. Run: cargo xtask test-setup");
            return;
        }
    };
    println!("Loading model from: {:?}", model_path);

    match WhisperContext::new(&model_path) {
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

fn find_model(name: &str) -> Option<PathBuf> {
    for env_var in ["WHISPER_TEST_MODEL_DIR", "WHISPER_MODEL_PATH"] {
        if let Ok(dir) = std::env::var(env_var) {
            let path = Path::new(&dir).join(name);
            if path.exists() {
                return Some(path);
            }
        }
    }
    let paths = [
        format!("tests/models/{}", name),
        format!("whisper-cpp-plus/tests/models/{}", name),
        format!("../whisper-cpp-plus-sys/whisper.cpp/models/{}", name),
        format!("whisper-cpp-plus-sys/whisper.cpp/models/{}", name),
    ];
    paths
        .iter()
        .find(|p| Path::new(p).exists())
        .map(PathBuf::from)
}
