use std::env;
use std::path::{Path, PathBuf};

#[path = "cuda_detect.rs"]
mod cuda_detect;

/// Pinned commit from rmorse/whisper.cpp (stream-pcm branch, based on v1.8.3)
const WHISPER_CPP_VERSION: &str = "02de44819ba4f9cf3f3d2a4adcfc2c5130d7140a";
const WHISPER_CPP_REPO: &str = "rmorse/whisper.cpp";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=WHISPER_PREBUILT_PATH");
    println!("cargo:rerun-if-env-changed=WHISPER_CPP_SOURCE_DIR");
    for var in &cuda_detect::CUDA_PATH_ENV_VARS {
        println!("cargo:rerun-if-env-changed={}", var);
    }

    // docs.rs builds in a network-isolated container - skip compilation and generate stubs
    if env::var("DOCS_RS").is_ok() {
        println!("cargo:warning=docs.rs build detected, generating stub bindings only");
        generate_stub_bindings();
        return;
    }

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_else(|_| "unknown".to_string());

    let prebuilt_dir = check_and_use_prebuilt(&target_os);

    if let Some(ref dir) = prebuilt_dir {
        // Prebuilt path: link whisper + probe ggml satellite libs
        println!("cargo:rustc-link-lib=static=whisper");
        link_prebuilt_ggml_libs(dir, &target_os);
    } else {
        // CMake build path
        build_with_cmake(&target_os);
    }

    link_platform_libs(&target_os);
    link_accelerator_libs(&target_os);
    build_quantize_wrapper();
    generate_bindings();
}

// ---------------------------------------------------------------------------
// CMake build
// ---------------------------------------------------------------------------

fn build_with_cmake(target_os: &str) {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let whisper_root = get_whisper_source(&out);

    let mut config = cmake::Config::new(&whisper_root);
    config
        .profile("Release")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("WHISPER_BUILD_TESTS", "OFF")
        .define("WHISPER_BUILD_EXAMPLES", "OFF")
        .pic(true);

    // Feature-gated CMake flags (cmake handles toolkit discovery internally)
    if cfg!(feature = "cuda") {
        config.define("GGML_CUDA", "ON");
    }
    if cfg!(feature = "metal") {
        config.define("GGML_METAL", "ON");
        config.define("GGML_METAL_EMBED_LIBRARY", "ON");
    } else {
        config.define("GGML_METAL", "OFF");
    }
    if cfg!(feature = "openblas") {
        config.define("GGML_BLAS", "ON");
    }

    // Windows-specific
    if target_os == "windows" {
        config.cxxflag("/utf-8");
    }

    // Allow env var passthrough (CMAKE_*, WHISPER_*)
    for (key, value) in env::vars() {
        if key.starts_with("CMAKE_")
            || (key.starts_with("WHISPER_")
                && key != "WHISPER_PREBUILT_PATH"
                && key != "WHISPER_NO_AVX"
                && key != "WHISPER_TEST_MODEL_DIR"
                && key != "WHISPER_TEST_AUDIO_DIR")
        {
            config.define(&key, &value);
        }
    }

    let destination = config.build();

    // CMake scatters outputs across subdirs — add them all to link search
    add_link_search_path_recursive(&out.join("build"));
    println!(
        "cargo:rustc-link-search=native={}",
        destination.join("lib").display()
    );

    // Link produced static libs
    println!("cargo:rustc-link-lib=static=whisper");
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-base");
    println!("cargo:rustc-link-lib=static=ggml-cpu");

    // On macOS, BLAS is enabled by default using Apple's Accelerate framework
    if target_os == "macos" {
        println!("cargo:rustc-link-lib=static=ggml-blas");
    }

    if cfg!(feature = "cuda") {
        println!("cargo:rustc-link-lib=static=ggml-cuda");
    }
    if cfg!(feature = "metal") {
        println!("cargo:rustc-link-lib=static=ggml-metal");
    }
}

/// Recursively copy a directory, skipping `.git` entries.
fn copy_dir_filtered(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).expect("failed to create destination directory");
    for entry in std::fs::read_dir(src).expect("failed to read source directory") {
        let entry = entry.expect("failed to read directory entry");
        let name = entry.file_name();
        if name == ".git" {
            continue;
        }
        let src_path = entry.path();
        let dst_path = dst.join(&name);
        let ft = entry.file_type().expect("failed to get file type");
        if ft.is_dir() {
            copy_dir_filtered(&src_path, &dst_path);
        } else if ft.is_file() {
            std::fs::copy(&src_path, &dst_path).unwrap_or_else(|e| {
                panic!(
                    "failed to copy {} -> {}: {}",
                    src_path.display(),
                    dst_path.display(),
                    e
                )
            });
        }
        // Skip symlinks and other special files
    }
}

/// Get whisper.cpp source: env override, local submodule, or download from GitHub
fn get_whisper_source(out_dir: &Path) -> PathBuf {
    let whisper_root = out_dir.join("whisper.cpp");

    if whisper_root.exists() {
        return whisper_root;
    }

    // Check env var override first
    if let Ok(source_dir) = env::var("WHISPER_CPP_SOURCE_DIR") {
        let src = PathBuf::from(&source_dir);
        if src.join("include/whisper.h").exists() {
            println!("cargo:warning=Using WHISPER_CPP_SOURCE_DIR: {}", source_dir);
            copy_dir_filtered(&src, &whisper_root);
            return whisper_root;
        }
        panic!(
            "WHISPER_CPP_SOURCE_DIR set but whisper.h not found at {}/include/whisper.h",
            source_dir
        );
    }

    // Check local submodule (inside sys crate for dev)
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let bundled_path = manifest_dir.join("whisper.cpp");
    if bundled_path.join("include/whisper.h").exists() {
        copy_dir_filtered(&bundled_path, &whisper_root);
        return whisper_root;
    }

    // Download from GitHub
    let url = format!(
        "https://github.com/{}/archive/{}.tar.gz",
        WHISPER_CPP_REPO, WHISPER_CPP_VERSION
    );

    println!("cargo:warning=Downloading whisper.cpp from {}", url);

    let tarball_path = out_dir.join("whisper.cpp.tar.gz");
    let resp = ureq::get(&url)
        .call()
        .expect("failed to download whisper.cpp");
    let mut file = std::fs::File::create(&tarball_path).unwrap();
    std::io::copy(&mut resp.into_reader(), &mut file).unwrap();

    // Extract
    let tar_gz = std::fs::File::open(&tarball_path).unwrap();
    let tar = flate2::read::GzDecoder::new(tar_gz);
    let mut archive = tar::Archive::new(tar);
    archive.unpack(out_dir).unwrap();

    // Rename extracted folder (whisper.cpp-{commit} -> whisper.cpp)
    std::fs::rename(
        out_dir.join(format!("whisper.cpp-{}", WHISPER_CPP_VERSION)),
        &whisper_root,
    )
    .unwrap();

    // Cleanup tarball
    let _ = std::fs::remove_file(&tarball_path);

    whisper_root
}

/// Recursively add all subdirectories of `dir` to the native link search path.
/// CMake places .lib/.a files in various nested directories.
fn add_link_search_path_recursive(dir: &Path) {
    if !dir.exists() {
        return;
    }
    println!("cargo:rustc-link-search=native={}", dir.display());
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                add_link_search_path_recursive(&entry.path());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Quantize wrapper (cc crate — our own C++ file, not part of CMake build)
// ---------------------------------------------------------------------------

fn build_quantize_wrapper() {
    #[cfg(feature = "quantization")]
    {
        let out = PathBuf::from(env::var("OUT_DIR").unwrap());
        let whisper_src = get_whisper_source(&out);

        cc::Build::new()
            .cpp(true)
            .std("c++17")
            .include(whisper_src.join("include"))
            .include(whisper_src.join("ggml/include"))
            .include(whisper_src.join("examples"))
            .file(whisper_src.join("examples/common.cpp"))
            .file(whisper_src.join("examples/common-ggml.cpp"))
            .file("src/quantize_wrapper.cpp")
            .compile("quantize_wrapper");
    }
}

// ---------------------------------------------------------------------------
// Prebuilt library detection
// ---------------------------------------------------------------------------

/// Check for prebuilt library. Returns `Some(dir)` if found.
fn check_and_use_prebuilt(target_os: &str) -> Option<PathBuf> {
    let lib_name = if target_os == "windows" {
        "whisper.lib"
    } else {
        "libwhisper.a"
    };

    // Check WHISPER_PREBUILT_PATH env var
    if let Ok(prebuilt_path) = env::var("WHISPER_PREBUILT_PATH") {
        let dir = PathBuf::from(&prebuilt_path);
        let lib_path = dir.join(lib_name);
        if lib_path.exists() {
            println!(
                "cargo:warning=Using prebuilt whisper library from: {}",
                prebuilt_path
            );
            println!("cargo:rustc-link-search=native={}", prebuilt_path);
            return Some(dir);
        }
    }

    // Check prebuilt/ directory in project root
    let target = env::var("TARGET").unwrap_or_default();
    let profile = env::var("PROFILE").unwrap_or_else(|_| "release".to_string());
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let prebuilt_dir = Path::new(&manifest_dir)
        .parent()
        .map(|p| p.join("prebuilt").join(&target).join(&profile))
        .unwrap_or_else(|| Path::new("../prebuilt").join(&target).join(&profile));

    let lib_path = prebuilt_dir.join(lib_name);
    if lib_path.exists() {
        let abs_path = lib_path
            .parent()
            .unwrap()
            .canonicalize()
            .unwrap_or_else(|_| prebuilt_dir.clone());
        println!(
            "cargo:warning=Using prebuilt whisper library from: {}",
            abs_path.display()
        );
        println!("cargo:rustc-link-search=native={}", abs_path.display());
        return Some(abs_path);
    }

    // Check system paths (Unix only)
    if target_os != "windows" {
        let system_paths = ["/usr/local/lib", "/usr/lib", "/opt/homebrew/lib"];
        for path in &system_paths {
            let lib_path = Path::new(path).join("libwhisper.a");
            if lib_path.exists() {
                println!("cargo:warning=Using system whisper library from: {}", path);
                println!("cargo:rustc-link-search=native={}", path);
                return Some(PathBuf::from(path));
            }
        }
    }

    None
}

/// Probe prebuilt dir for ggml satellite libraries (produced by CMake builds).
fn link_prebuilt_ggml_libs(dir: &Path, target_os: &str) {
    let ext = if target_os == "windows" { "lib" } else { "a" };

    let optional_libs = ["ggml", "ggml-base", "ggml-cpu"];
    for lib in &optional_libs {
        let filename = if target_os == "windows" {
            format!("{}.{}", lib, ext)
        } else {
            format!("lib{}.{}", lib, ext)
        };
        if dir.join(&filename).exists() {
            println!("cargo:rustc-link-lib=static={}", lib);
        }
    }

    #[cfg(feature = "cuda")]
    {
        let cuda_filename = if target_os == "windows" {
            format!("ggml-cuda.{}", ext)
        } else {
            format!("libggml-cuda.{}", ext)
        };
        if dir.join(&cuda_filename).exists() {
            println!("cargo:rustc-link-lib=static=ggml-cuda");
        } else {
            panic!(
                "\n\n\
                 ======================================================\n\
                 CUDA feature enabled but ggml-cuda.{} not found in:\n\
                 {}\n\n\
                 Build whisper.cpp with CMake + -DGGML_CUDA=1 and copy\n\
                 all .lib/.a files to the prebuilt directory.\n\
                 ======================================================",
                ext,
                dir.display()
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Platform / accelerator linking
// ---------------------------------------------------------------------------

/// Platform-specific system libraries.
fn link_platform_libs(target_os: &str) {
    match target_os {
        "windows" => {
            println!("cargo:rustc-link-lib=ws2_32");
            println!("cargo:rustc-link-lib=bcrypt");
            println!("cargo:rustc-link-lib=advapi32");
            println!("cargo:rustc-link-lib=userenv");
        }
        "macos" => {
            println!("cargo:rustc-link-lib=framework=Accelerate");
            println!("cargo:rustc-link-lib=framework=Foundation");
        }
        "linux" => {
            println!("cargo:rustc-link-lib=m");
            println!("cargo:rustc-link-lib=pthread");
            println!("cargo:rustc-link-lib=stdc++");
        }
        _ => {}
    }
}

/// Accelerator libraries (CUDA toolkit, Metal frameworks, OpenBLAS).
fn link_accelerator_libs(_target_os: &str) {
    #[cfg(feature = "cuda")]
    {
        // CMake compiles ggml-cuda, but the final binary still needs
        // the CUDA toolkit runtime libs at link time.
        if cuda_detect::cuda_roots_from_env().is_empty() {
            println!(
                "cargo:warning=No CUDA path env var set ({:?}), probing standard locations",
                cuda_detect::CUDA_PATH_ENV_VARS
            );
        }
        let search_paths = cuda_detect::cuda_lib_search_paths();
        if search_paths.is_empty() {
            panic!(
                "\n\n\
                 ======================================================\n\
                 CUDA toolkit not found.\n\n\
                 Set one of these environment variables:\n\
                 - CUDA_PATH\n\
                 - CUDA_HOME\n\
                 - CUDA_ROOT\n\
                 - CUDA_TOOLKIT_ROOT_DIR\n\n\
                 Or install CUDA toolkit to a standard location:\n\
                 - Windows: C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\vX.Y\n\
                 - Linux: /usr/local/cuda\n\
                 ======================================================"
            );
        }
        for path in &search_paths {
            println!("cargo:rustc-link-search=native={}", path.display());
        }
        println!("cargo:rustc-link-lib=cudart_static");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cublasLt");
        println!("cargo:rustc-link-lib=cuda");
    }

    #[cfg(feature = "metal")]
    {
        if _target_os == "macos" {
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=MetalKit");
            println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
        }
    }

    #[cfg(feature = "openblas")]
    {
        println!("cargo:rustc-link-lib=openblas");
    }
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

/// Generate stub bindings for docs.rs (network-isolated, can't download whisper.cpp)
fn generate_stub_bindings() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let stub_bindings = r#"
// Stub bindings for docs.rs documentation build.
// This crate requires whisper.cpp which cannot be built in docs.rs's sandbox.
// For actual usage, build locally or see the repository.

pub type whisper_context = core::ffi::c_void;
pub type whisper_state = core::ffi::c_void;
pub type whisper_token = i32;
pub type whisper_pos = i64;

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct whisper_context_params {
    pub use_gpu: bool,
    pub flash_attn: bool,
    pub gpu_device: core::ffi::c_int,
    pub dtw_token_timestamps: bool,
    pub dtw_aheads_preset: core::ffi::c_int,
    pub dtw_n_top: core::ffi::c_int,
    pub dtw_aheads: whisper_aheads,
    pub dtw_mem_size: usize,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct whisper_aheads {
    pub n_heads: usize,
    pub heads: *const whisper_ahead,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct whisper_ahead {
    pub n_text_layer: core::ffi::c_int,
    pub n_head: core::ffi::c_int,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct whisper_full_params {
    pub strategy: core::ffi::c_int,
    pub n_threads: core::ffi::c_int,
    pub n_max_text_ctx: core::ffi::c_int,
    pub offset_ms: core::ffi::c_int,
    pub duration_ms: core::ffi::c_int,
    pub translate: bool,
    pub no_context: bool,
    pub no_timestamps: bool,
    pub single_segment: bool,
    pub print_special: bool,
    pub print_progress: bool,
    pub print_realtime: bool,
    pub print_timestamps: bool,
    pub token_timestamps: bool,
    pub thold_pt: f32,
    pub thold_ptsum: f32,
    pub max_len: core::ffi::c_int,
    pub split_on_word: bool,
    pub max_tokens: core::ffi::c_int,
    pub debug_mode: bool,
    pub audio_ctx: core::ffi::c_int,
    pub tdrz_enable: bool,
    pub suppress_regex: *const core::ffi::c_char,
    pub initial_prompt: *const core::ffi::c_char,
    pub prompt_tokens: *const whisper_token,
    pub prompt_n_tokens: core::ffi::c_int,
    pub language: *const core::ffi::c_char,
    pub detect_language: bool,
    pub suppress_blank: bool,
    pub suppress_nst: bool,
    pub temperature: f32,
    pub max_initial_ts: f32,
    pub length_penalty: f32,
    pub temperature_inc: f32,
    pub entropy_thold: f32,
    pub logprob_thold: f32,
    pub no_speech_thold: f32,
    pub greedy: whisper_full_params__bindgen_ty_1,
    pub beam_search: whisper_full_params__bindgen_ty_2,
    pub new_segment_callback: Option<unsafe extern "C" fn()>,
    pub new_segment_callback_user_data: *mut core::ffi::c_void,
    pub progress_callback: Option<unsafe extern "C" fn()>,
    pub progress_callback_user_data: *mut core::ffi::c_void,
    pub encoder_begin_callback: Option<unsafe extern "C" fn()>,
    pub encoder_begin_callback_user_data: *mut core::ffi::c_void,
    pub abort_callback: Option<unsafe extern "C" fn()>,
    pub abort_callback_user_data: *mut core::ffi::c_void,
    pub logits_filter_callback: Option<unsafe extern "C" fn()>,
    pub logits_filter_callback_user_data: *mut core::ffi::c_void,
    pub grammar_rules: *const *const core::ffi::c_void,
    pub n_grammar_rules: usize,
    pub i_start_rule: usize,
    pub grammar_penalty: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct whisper_full_params__bindgen_ty_1 {
    pub best_of: core::ffi::c_int,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct whisper_full_params__bindgen_ty_2 {
    pub beam_size: core::ffi::c_int,
    pub patience: f32,
}

pub const WHISPER_SAMPLE_RATE: u32 = 16000;
pub const WHISPER_N_FFT: u32 = 400;
pub const WHISPER_HOP_LENGTH: u32 = 160;
pub const WHISPER_CHUNK_SIZE: u32 = 30;

// Sampling strategy enum
pub const whisper_sampling_strategy_WHISPER_SAMPLING_GREEDY: core::ffi::c_int = 0;
pub const whisper_sampling_strategy_WHISPER_SAMPLING_BEAM_SEARCH: core::ffi::c_int = 1;

// Stub function declarations (not callable, just for docs)
extern "C" {
    // Context initialization/cleanup
    pub fn whisper_init_from_file_with_params(
        path: *const core::ffi::c_char,
        params: whisper_context_params,
    ) -> *mut whisper_context;
    pub fn whisper_init_from_buffer_with_params(
        buffer: *const core::ffi::c_void,
        buffer_size: usize,
        params: whisper_context_params,
    ) -> *mut whisper_context;
    pub fn whisper_free(ctx: *mut whisper_context);
    pub fn whisper_init_state(ctx: *mut whisper_context) -> *mut whisper_state;
    pub fn whisper_free_state(state: *mut whisper_state);
    pub fn whisper_ctx_init_openvino_encoder(
        ctx: *mut whisper_context,
        model_path: *const core::ffi::c_char,
        device: *const core::ffi::c_char,
        cache_dir: *const core::ffi::c_char,
    ) -> core::ffi::c_int;

    // Context info
    pub fn whisper_context_default_params() -> whisper_context_params;
    pub fn whisper_n_vocab(ctx: *mut whisper_context) -> core::ffi::c_int;
    pub fn whisper_n_audio_ctx(ctx: *mut whisper_context) -> core::ffi::c_int;
    pub fn whisper_n_text_ctx(ctx: *mut whisper_context) -> core::ffi::c_int;
    pub fn whisper_n_audio_state(ctx: *mut whisper_context) -> core::ffi::c_int;
    pub fn whisper_n_text_state(ctx: *mut whisper_context) -> core::ffi::c_int;
    pub fn whisper_n_text_head(ctx: *mut whisper_context) -> core::ffi::c_int;
    pub fn whisper_n_text_layer(ctx: *mut whisper_context) -> core::ffi::c_int;
    pub fn whisper_n_mels(ctx: *mut whisper_context) -> core::ffi::c_int;
    pub fn whisper_n_len(ctx: *mut whisper_context) -> core::ffi::c_int;
    pub fn whisper_n_len_from_state(state: *mut whisper_state) -> core::ffi::c_int;
    pub fn whisper_is_multilingual(ctx: *mut whisper_context) -> core::ffi::c_int;

    // Language
    pub fn whisper_lang_max_id() -> core::ffi::c_int;
    pub fn whisper_lang_id(lang: *const core::ffi::c_char) -> core::ffi::c_int;
    pub fn whisper_lang_str(id: core::ffi::c_int) -> *const core::ffi::c_char;
    pub fn whisper_lang_str_full(id: core::ffi::c_int) -> *const core::ffi::c_char;
    pub fn whisper_lang_auto_detect(
        ctx: *mut whisper_context,
        offset_ms: core::ffi::c_int,
        n_threads: core::ffi::c_int,
        lang_probs: *mut f32,
    ) -> core::ffi::c_int;
    pub fn whisper_lang_auto_detect_with_state(
        ctx: *mut whisper_context,
        state: *mut whisper_state,
        offset_ms: core::ffi::c_int,
        n_threads: core::ffi::c_int,
        lang_probs: *mut f32,
    ) -> core::ffi::c_int;

    // Transcription
    pub fn whisper_full_default_params(strategy: core::ffi::c_int) -> whisper_full_params;
    pub fn whisper_full(
        ctx: *mut whisper_context,
        params: whisper_full_params,
        samples: *const f32,
        n_samples: core::ffi::c_int,
    ) -> core::ffi::c_int;
    pub fn whisper_full_with_state(
        ctx: *mut whisper_context,
        state: *mut whisper_state,
        params: whisper_full_params,
        samples: *const f32,
        n_samples: core::ffi::c_int,
    ) -> core::ffi::c_int;
    pub fn whisper_full_parallel(
        ctx: *mut whisper_context,
        params: whisper_full_params,
        samples: *const f32,
        n_samples: core::ffi::c_int,
        n_processors: core::ffi::c_int,
    ) -> core::ffi::c_int;

    // Segment results
    pub fn whisper_full_lang_id(ctx: *mut whisper_context) -> core::ffi::c_int;
    pub fn whisper_full_lang_id_from_state(state: *mut whisper_state) -> core::ffi::c_int;
    pub fn whisper_full_n_segments(ctx: *mut whisper_context) -> core::ffi::c_int;
    pub fn whisper_full_n_segments_from_state(state: *mut whisper_state) -> core::ffi::c_int;
    pub fn whisper_full_get_segment_t0(ctx: *mut whisper_context, i_segment: core::ffi::c_int) -> i64;
    pub fn whisper_full_get_segment_t0_from_state(state: *mut whisper_state, i_segment: core::ffi::c_int) -> i64;
    pub fn whisper_full_get_segment_t1(ctx: *mut whisper_context, i_segment: core::ffi::c_int) -> i64;
    pub fn whisper_full_get_segment_t1_from_state(state: *mut whisper_state, i_segment: core::ffi::c_int) -> i64;
    pub fn whisper_full_get_segment_text(ctx: *mut whisper_context, i_segment: core::ffi::c_int) -> *const core::ffi::c_char;
    pub fn whisper_full_get_segment_text_from_state(state: *mut whisper_state, i_segment: core::ffi::c_int) -> *const core::ffi::c_char;
    pub fn whisper_full_get_segment_speaker_turn_next(ctx: *mut whisper_context, i_segment: core::ffi::c_int) -> bool;
    pub fn whisper_full_get_segment_speaker_turn_next_from_state(state: *mut whisper_state, i_segment: core::ffi::c_int) -> bool;
    pub fn whisper_full_get_segment_no_speech_prob(ctx: *mut whisper_context, i_segment: core::ffi::c_int) -> f32;
    pub fn whisper_full_get_segment_no_speech_prob_from_state(state: *mut whisper_state, i_segment: core::ffi::c_int) -> f32;

    // Token results
    pub fn whisper_full_n_tokens(ctx: *mut whisper_context, i_segment: core::ffi::c_int) -> core::ffi::c_int;
    pub fn whisper_full_n_tokens_from_state(state: *mut whisper_state, i_segment: core::ffi::c_int) -> core::ffi::c_int;
    pub fn whisper_full_get_token_text(ctx: *mut whisper_context, i_segment: core::ffi::c_int, i_token: core::ffi::c_int) -> *const core::ffi::c_char;
    pub fn whisper_full_get_token_text_from_state(ctx: *mut whisper_context, state: *mut whisper_state, i_segment: core::ffi::c_int, i_token: core::ffi::c_int) -> *const core::ffi::c_char;
    pub fn whisper_full_get_token_id(ctx: *mut whisper_context, i_segment: core::ffi::c_int, i_token: core::ffi::c_int) -> whisper_token;
    pub fn whisper_full_get_token_id_from_state(state: *mut whisper_state, i_segment: core::ffi::c_int, i_token: core::ffi::c_int) -> whisper_token;
    pub fn whisper_full_get_token_p(ctx: *mut whisper_context, i_segment: core::ffi::c_int, i_token: core::ffi::c_int) -> f32;
    pub fn whisper_full_get_token_p_from_state(state: *mut whisper_state, i_segment: core::ffi::c_int, i_token: core::ffi::c_int) -> f32;
    pub fn whisper_full_get_token_data(ctx: *mut whisper_context, i_segment: core::ffi::c_int, i_token: core::ffi::c_int) -> whisper_token_data;
    pub fn whisper_full_get_token_data_from_state(state: *mut whisper_state, i_segment: core::ffi::c_int, i_token: core::ffi::c_int) -> whisper_token_data;

    // Token utilities
    pub fn whisper_token_to_str(ctx: *mut whisper_context, token: whisper_token) -> *const core::ffi::c_char;
    pub fn whisper_token_eot(ctx: *mut whisper_context) -> whisper_token;
    pub fn whisper_token_sot(ctx: *mut whisper_context) -> whisper_token;
    pub fn whisper_token_prev(ctx: *mut whisper_context) -> whisper_token;
    pub fn whisper_token_solm(ctx: *mut whisper_context) -> whisper_token;
    pub fn whisper_token_not(ctx: *mut whisper_context) -> whisper_token;
    pub fn whisper_token_beg(ctx: *mut whisper_context) -> whisper_token;
    pub fn whisper_token_lang(ctx: *mut whisper_context, lang_id: core::ffi::c_int) -> whisper_token;
    pub fn whisper_token_translate(ctx: *mut whisper_context) -> whisper_token;
    pub fn whisper_token_transcribe(ctx: *mut whisper_context) -> whisper_token;

    // Timing
    pub fn whisper_print_timings(ctx: *mut whisper_context);
    pub fn whisper_reset_timings(ctx: *mut whisper_context);
    pub fn whisper_print_system_info() -> *const core::ffi::c_char;

    // VAD
    pub fn whisper_vad_init(ctx: *mut whisper_context, model_path: *const core::ffi::c_char) -> *mut whisper_vad_context;
    pub fn whisper_vad_init_from_buffer(ctx: *mut whisper_context, buffer: *const core::ffi::c_void, buffer_size: usize) -> *mut whisper_vad_context;
    pub fn whisper_vad_init_with_params(model_path: *const core::ffi::c_char, params: whisper_vad_context_params) -> *mut whisper_vad_context;
    pub fn whisper_vad_init_from_buffer_with_params(buffer: *const core::ffi::c_void, buffer_size: usize, params: whisper_vad_context_params) -> *mut whisper_vad_context;
    pub fn whisper_vad_init_from_file_with_params(model_path: *const core::ffi::c_char, params: whisper_vad_context_params) -> *mut whisper_vad_context;
    pub fn whisper_vad_free(vad_ctx: *mut whisper_vad_context);
    pub fn whisper_vad_default_params() -> whisper_vad_params;
    pub fn whisper_vad_default_context_params() -> whisper_vad_context_params;
    pub fn whisper_vad_detect_speech(vad_ctx: *mut whisper_vad_context, samples: *const f32, n_samples: core::ffi::c_int) -> bool;
    pub fn whisper_vad_n_probs(vad_ctx: *mut whisper_vad_context) -> core::ffi::c_int;
    pub fn whisper_vad_probs(vad_ctx: *mut whisper_vad_context) -> *const f32;
    pub fn whisper_vad_segments_from_probs(vad_ctx: *mut whisper_vad_context, params: whisper_vad_params) -> *mut whisper_vad_segments;
    pub fn whisper_vad_segments_from_samples(vad_ctx: *mut whisper_vad_context, params: whisper_vad_params, samples: *const f32, n_samples: core::ffi::c_int) -> *mut whisper_vad_segments;
    pub fn whisper_vad_segments_free(segments: *mut whisper_vad_segments);
    pub fn whisper_vad_free_segments(segments: *mut whisper_vad_segments);
    pub fn whisper_vad_segments_n_segments(segments: *mut whisper_vad_segments) -> core::ffi::c_int;
    pub fn whisper_vad_segments_get_segment_t0(segments: *mut whisper_vad_segments, i_segment: core::ffi::c_int) -> f32;
    pub fn whisper_vad_segments_get_segment_t1(segments: *mut whisper_vad_segments, i_segment: core::ffi::c_int) -> f32;
}

// Additional type stubs
pub type whisper_vad_context = core::ffi::c_void;
pub type whisper_vad_segments = core::ffi::c_void;

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct whisper_vad_context_params {
    pub use_gpu: bool,
    pub gpu_device: core::ffi::c_int,
    pub n_threads: core::ffi::c_int,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct whisper_vad_params {
    pub threshold: f32,
    pub min_speech_duration_ms: core::ffi::c_int,
    pub min_silence_duration_ms: core::ffi::c_int,
    pub max_speech_duration_s: f32,
    pub speech_pad_ms: core::ffi::c_int,
    pub samples_overlap: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct whisper_token_data {
    pub id: whisper_token,
    pub tid: whisper_token,
    pub p: f32,
    pub plog: f32,
    pub pt: f32,
    pub ptsum: f32,
    pub t0: i64,
    pub t1: i64,
    pub t_dtw: i64,
    pub vlen: f32,
}
"#;
    std::fs::write(out_dir.join("bindings.rs"), stub_bindings)
        .expect("Failed to write stub bindings");
}

fn generate_bindings() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let whisper_src = get_whisper_source(&out_dir);
    let header = whisper_src.join("include/whisper.h");

    println!("cargo:rerun-if-changed={}", header.display());

    let bindings = bindgen::Builder::default()
        .header(header.to_str().unwrap())
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=c++11")
        .clang_arg(format!("-I{}", whisper_src.join("include").display()))
        .clang_arg(format!("-I{}", whisper_src.join("ggml/include").display()))
        .allowlist_function("whisper_.*")
        .allowlist_type("whisper_.*")
        .allowlist_var("WHISPER_.*")
        .opaque_type("std::.*")
        .opaque_type("std::.*::.*")
        .use_core()
        .ctypes_prefix("::core::ffi")
        .layout_tests(true)
        .derive_default(true)
        .derive_debug(true)
        .generate()
        .expect("Unable to generate bindings");

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
