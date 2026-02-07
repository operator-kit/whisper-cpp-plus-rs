use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../vendor/whisper.cpp");
    println!("cargo:rerun-if-env-changed=WHISPER_PREBUILT_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

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
    let whisper_root = out.join("whisper.cpp");

    // Copy vendor source to OUT_DIR to avoid polluting the vendor directory.
    // Uses a filtered copy that skips .git (submodule files Windows can't access).
    if !whisper_root.exists() {
        let src = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
            .join("../vendor/whisper.cpp");
        let src = src.canonicalize().expect("vendor/whisper.cpp not found");
        copy_dir_filtered(&src, &whisper_root);
    }

    let mut config = cmake::Config::new(&whisper_root);
    config
        .profile("Release")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("WHISPER_BUILD_TESTS", "OFF")
        .define("WHISPER_BUILD_EXAMPLES", "OFF")
        .pic(true);

    // Feature-gated CMake flags
    if cfg!(feature = "cuda") {
        config.define("GGML_CUDA", "ON");
        // Help CMake find the CUDA toolkit — MSBuild integration alone isn't enough
        if let Some(cuda_path) = find_cuda_path() {
            config.define("CMAKE_CUDA_TOOLKIT_ROOT_DIR", &cuda_path);
            config.define("CUDAToolkit_ROOT", &cuda_path);
        }
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

    if cfg!(feature = "cuda") {
        println!("cargo:rustc-link-lib=static=ggml-cuda");
    }
    if cfg!(feature = "metal") {
        println!("cargo:rustc-link-lib=static=ggml-metal");
    }
}

/// Try to find CUDA toolkit path from env vars or standard locations.
/// Returns None if not found (cmake may still find it via its own detection).
fn find_cuda_path() -> Option<String> {
    if let Ok(p) = env::var("CUDA_PATH") {
        if PathBuf::from(&p).exists() {
            return Some(p);
        }
    }
    if let Ok(p) = env::var("CUDA_HOME") {
        if PathBuf::from(&p).exists() {
            return Some(p);
        }
    }
    // Windows standard install
    let base = PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA");
    if base.exists() {
        if let Ok(entries) = std::fs::read_dir(&base) {
            let mut versions: Vec<PathBuf> = entries
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.is_dir())
                .collect();
            versions.sort();
            if let Some(latest) = versions.last() {
                return Some(latest.to_string_lossy().to_string());
            }
        }
    }
    // Linux standard paths
    for p in &["/usr/local/cuda", "/usr/lib/cuda", "/opt/cuda"] {
        if PathBuf::from(p).exists() {
            return Some(p.to_string());
        }
    }
    None
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
                panic!("failed to copy {} -> {}: {}", src_path.display(), dst_path.display(), e)
            });
        }
        // Skip symlinks and other special files
    }
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
        cc::Build::new()
            .cpp(true)
            .std("c++17")
            .include("../vendor/whisper.cpp/include")
            .include("../vendor/whisper.cpp/ggml/include")
            .include("../vendor/whisper.cpp/examples")
            .file("../vendor/whisper.cpp/examples/common.cpp")
            .file("../vendor/whisper.cpp/examples/common-ggml.cpp")
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

    // ggml satellite libs that CMake may produce
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

    // CUDA satellite lib
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
        // CUDA toolkit libs still needed at link time.
        // CMake handles CUDA discovery during build, but we still need
        // to tell rustc where the toolkit libs are.
        let cuda_path = find_cuda_toolkit();
        let lib_dir = if _target_os == "windows" {
            cuda_path.join("lib").join("x64")
        } else {
            cuda_path.join("lib64")
        };
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
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

/// Locate CUDA toolkit installation directory, or panic with instructions.
#[cfg(feature = "cuda")]
fn find_cuda_toolkit() -> PathBuf {
    if let Some(p) = find_cuda_path() {
        println!("cargo:warning=Using CUDA toolkit from: {}", p);
        return PathBuf::from(p);
    }
    panic!(
        "\n\n\
         ======================================================\n\
         CUDA toolkit not found.\n\n\
         Set one of these environment variables:\n\
         - CUDA_PATH (checked first)\n\
         - CUDA_HOME\n\n\
         Or install CUDA toolkit to a standard location:\n\
         - Windows: C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\vX.Y\n\
         - Linux: /usr/local/cuda\n\
         ======================================================"
    );
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

fn generate_bindings() {
    println!("cargo:rerun-if-changed=../vendor/whisper.cpp/include/whisper.h");

    let bindings = bindgen::Builder::default()
        .header("../vendor/whisper.cpp/include/whisper.h")
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=c++11")
        .clang_arg("-I../vendor/whisper.cpp/include")
        .clang_arg("-I../vendor/whisper.cpp/ggml/include")
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

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
