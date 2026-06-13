use std::path::{Path, PathBuf};

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn workspace_dir() -> Option<PathBuf> {
    manifest_dir().parent().map(Path::to_path_buf)
}

fn find_in_dirs(name: &str, dirs: impl IntoIterator<Item = PathBuf>) -> Option<PathBuf> {
    dirs.into_iter()
        .map(|dir| dir.join(name))
        .find(|path| path.exists())
}

pub(crate) fn whisper_model(name: &str) -> Option<PathBuf> {
    let manifest = manifest_dir();
    let mut dirs = Vec::new();

    if let Ok(dir) = std::env::var("WHISPER_TEST_MODEL_DIR") {
        dirs.push(PathBuf::from(dir));
    }

    dirs.push(manifest.join("tests/models"));
    dirs.push(manifest.join("../whisper-cpp-plus-sys/whisper.cpp/models"));

    if let Some(workspace) = workspace_dir() {
        dirs.push(workspace.join("whisper-cpp-plus-sys/whisper.cpp/models"));
    }

    find_in_dirs(name, dirs)
}

pub(crate) fn tiny_en() -> Option<PathBuf> {
    whisper_model("ggml-tiny.en.bin").or_else(|| whisper_model("for-tests-ggml-tiny.en.bin"))
}

pub(crate) fn vad() -> Option<PathBuf> {
    whisper_model("ggml-silero-v6.2.0.bin")
        .or_else(|| whisper_model("ggml-silero-vad.bin"))
        .or_else(|| whisper_model("for-tests-silero-v6.2.0-ggml.bin"))
}

pub(crate) fn note_missing_fixture(name: &str) {
    if std::env::var_os("CI").is_some() {
        panic!("{name} fixture not found; run `cargo xtask test-setup` before tests");
    }

    eprintln!("Skipping: {name} fixture not found. Run `cargo xtask test-setup`");
}
