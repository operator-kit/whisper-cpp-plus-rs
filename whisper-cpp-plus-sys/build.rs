use std::env;
use std::path::{Path, PathBuf};

#[path = "cuda_detect.rs"]
mod cuda_detect;

/// Pinned commit from rmorse/whisper.cpp (stream-pcm branch, based on upstream master after the v1.9.4 release)
const WHISPER_CPP_VERSION: &str = "de8fb5fda8b25837a2ba0034c8c24223a6fd6c6c";
const WHISPER_CPP_REPO: &str = "rmorse/whisper.cpp";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=WHISPER_PREBUILT_PATH");
    println!("cargo:rerun-if-env-changed=WHISPER_CPP_SOURCE_DIR");
    println!("cargo:rerun-if-env-changed=MACOSX_DEPLOYMENT_TARGET");
    println!("cargo:rerun-if-env-changed=DOCS_RS");
    for var in &cuda_detect::CUDA_PATH_ENV_VARS {
        println!("cargo:rerun-if-env-changed={}", var);
    }

    // docs.rs builds in a network-isolated container: skip compiling whisper.cpp and generate
    // bindings from the public headers packaged with the crate.
    if env::var("DOCS_RS").is_ok() {
        println!(
            "cargo:warning=docs.rs build detected, generating bindings from packaged headers only"
        );
        let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        generate_bindings(&manifest_dir.join("whisper.cpp"));
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

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    generate_bindings(&get_whisper_source(&out));
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

    if target_os == "macos" {
        if let Ok(deploy_target) = env::var("MACOSX_DEPLOYMENT_TARGET") {
            config.define("CMAKE_OSX_DEPLOYMENT_TARGET", deploy_target);
        }
    }

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

    // ggml may build a BLAS backend on macOS (Accelerate) even if we didn't
    // explicitly enable OpenBLAS. Link it if CMake produced it.
    let blas_filename = if target_os == "windows" {
        "ggml-blas.lib".to_string()
    } else {
        "libggml-blas.a".to_string()
    };
    if file_exists_recursive(&out.join("build"), &blas_filename) {
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

    // Check local submodule (inside sys crate for dev). The published crate only ships the
    // public headers under whisper.cpp/, so require the full source tree here.
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let bundled_path = manifest_dir.join("whisper.cpp");
    if bundled_path.join("CMakeLists.txt").exists() && bundled_path.join("src/whisper.cpp").exists()
    {
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

/// Search recursively for `filename` under `dir`.
fn file_exists_recursive(dir: &Path, filename: &str) -> bool {
    if !dir.exists() {
        return false;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() {
            if path.file_name().and_then(|s| s.to_str()) == Some(filename) {
                return true;
            }
        } else if path.is_dir() && file_exists_recursive(&path, filename) {
            return true;
        }
    }
    false
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
            .flag_if_supported("-Wno-switch")
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

    let optional_libs = ["ggml", "ggml-base", "ggml-cpu", "ggml-blas"];
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

    #[cfg(feature = "metal")]
    {
        let metal_filename = if target_os == "windows" {
            format!("ggml-metal.{}", ext)
        } else {
            format!("libggml-metal.{}", ext)
        };
        if dir.join(&metal_filename).exists() {
            println!("cargo:rustc-link-lib=static=ggml-metal");
        } else {
            panic!(
                "\n\n\
                 ======================================================\n\
                 Metal feature enabled but ggml-metal.{} not found in:\n\
                 {}\n\n\
                 Build whisper.cpp with CMake + -DGGML_METAL=1 and copy\n\
                 all .lib/.a files to the prebuilt directory, or unset\n\
                 WHISPER_PREBUILT_PATH to build whisper.cpp from source.\n\
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

/// Generate bindings from whisper.cpp's public headers. `whisper_src` only needs the `include/`
/// and `ggml/include/` directories, so this also works on docs.rs with the packaged headers.
fn generate_bindings(whisper_src: &Path) {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let header = whisper_src.join("include/whisper.h");
    if !header.exists() {
        panic!(
            "whisper.h not found at {}; whisper.cpp headers are missing",
            header.display()
        );
    }

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
