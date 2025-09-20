use std::env;
use std::path::PathBuf;

fn main() {
    // Tell cargo to rerun if the build script changes
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../vendor/whisper.cpp");
    println!("cargo:rerun-if-changed=src/backend_stubs.cpp");

    // Get output directory
    let out_dir = env::var("OUT_DIR").unwrap();

    // Get target OS for platform-specific configuration
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_else(|_| "unknown".to_string());
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_else(|_| "unknown".to_string());

    // Build whisper.cpp using cc crate
    build_whisper_cpp(&target_os, &target_arch);

    // Generate bindings using bindgen
    generate_bindings();

    // Important: Add linking instructions
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=whisper");

    // Windows-specific libraries
    if target_os == "windows" {
        println!("cargo:rustc-link-lib=ws2_32");
        println!("cargo:rustc-link-lib=bcrypt");
        println!("cargo:rustc-link-lib=advapi32");
        println!("cargo:rustc-link-lib=userenv");
    }
}

fn build_whisper_cpp(target_os: &str, target_arch: &str) {
    let mut build = cc::Build::new();

    // Configure C++ compilation
    build.cpp(true);

    // Set C++ standard - ggml-backend-reg.cpp requires C++17 for std::filesystem
    if cfg!(target_env = "msvc") {
        build.std("c++17");
    } else {
        build.std("c++17");
    }

    // Add include directories
    build.include("../vendor/whisper.cpp")
        .include("../vendor/whisper.cpp/include")
        .include("../vendor/whisper.cpp/ggml/include")
        .include("../vendor/whisper.cpp/ggml/src")
        .include("../vendor/whisper.cpp/ggml/src/ggml-cpu");

    // Core source files
    build.file("../vendor/whisper.cpp/src/whisper.cpp")
        // Core GGML files
        .file("../vendor/whisper.cpp/ggml/src/ggml.c")
        .file("../vendor/whisper.cpp/ggml/src/ggml.cpp")
        .file("../vendor/whisper.cpp/ggml/src/ggml-alloc.c")
        .file("../vendor/whisper.cpp/ggml/src/ggml-backend.cpp")
        .file("../vendor/whisper.cpp/ggml/src/ggml-backend-reg.cpp")
        .file("../vendor/whisper.cpp/ggml/src/ggml-threading.cpp")
        .file("../vendor/whisper.cpp/ggml/src/ggml-quants.c")
        .file("../vendor/whisper.cpp/ggml/src/ggml-opt.cpp")
        .file("../vendor/whisper.cpp/ggml/src/gguf.cpp")
        // CPU backend core files
        .file("../vendor/whisper.cpp/ggml/src/ggml-cpu/ggml-cpu.c")
        .file("../vendor/whisper.cpp/ggml/src/ggml-cpu/ggml-cpu.cpp")
        .file("../vendor/whisper.cpp/ggml/src/ggml-cpu/binary-ops.cpp")
        .file("../vendor/whisper.cpp/ggml/src/ggml-cpu/unary-ops.cpp")
        .file("../vendor/whisper.cpp/ggml/src/ggml-cpu/ops.cpp")
        .file("../vendor/whisper.cpp/ggml/src/ggml-cpu/quants.c")
        .file("../vendor/whisper.cpp/ggml/src/ggml-cpu/traits.cpp")
        .file("../vendor/whisper.cpp/ggml/src/ggml-cpu/vec.cpp")
        .file("../vendor/whisper.cpp/ggml/src/ggml-cpu/repack.cpp")
        .file("../vendor/whisper.cpp/ggml/src/ggml-cpu/hbm.cpp")
        // AMX (Advanced Matrix Extensions) files
        .file("../vendor/whisper.cpp/ggml/src/ggml-cpu/amx/amx.cpp")
        .file("../vendor/whisper.cpp/ggml/src/ggml-cpu/amx/mmq.cpp");

    // Common compiler flags
    build.flag_if_supported("-fPIC");

    // Add definitions for memory alignment and features
    build.define("_ALIGNAS_SUPPORTED", None);
    build.define("GGML_USE_CPU", None);

    // Platform-specific configuration
    match target_os {
        "macos" => {
            build.define("GGML_USE_ACCELERATE", None);

            // Link Accelerate framework
            println!("cargo:rustc-link-lib=framework=Accelerate");
            println!("cargo:rustc-link-lib=framework=Foundation");

            // Add Metal support if feature is enabled
            #[cfg(feature = "metal")]
            {
                build.define("GGML_USE_METAL", None);
                build.define("GGML_METAL_EMBED_LIBRARY", None);
                build.file("../vendor/whisper.cpp/ggml/src/ggml-metal.m");
                println!("cargo:rustc-link-lib=framework=Metal");
                println!("cargo:rustc-link-lib=framework=MetalKit");
                println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
            }
        }
        "windows" => {
            build.define("_CRT_SECURE_NO_WARNINGS", None);
            build.define("WIN32_LEAN_AND_MEAN", None);

            // Windows needs explicit linking for some libraries - handled in main()

            // Use MSVC-specific optimizations
            if cfg!(target_env = "msvc") {
                build.flag("/O2");
                build.flag("/arch:AVX2");
                build.flag("/MT");  // Static runtime to avoid DLL issues
            }
        }
        "linux" => {
            // Link math library on Linux
            println!("cargo:rustc-link-lib=m");
            println!("cargo:rustc-link-lib=pthread");

            // Enable OpenBLAS if feature is enabled
            #[cfg(feature = "openblas")]
            {
                build.define("GGML_USE_OPENBLAS", None);
                println!("cargo:rustc-link-lib=openblas");
            }
        }
        _ => {}
    }

    // Architecture-specific optimizations
    match target_arch {
        "x86_64" => {
            // Add x86-specific CPU backend files
            build.file("../vendor/whisper.cpp/ggml/src/ggml-cpu/arch/x86/quants.c")
                .file("../vendor/whisper.cpp/ggml/src/ggml-cpu/arch/x86/repack.cpp")
                .file("../vendor/whisper.cpp/ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp");

            // Enable AVX/AVX2 if available
            if !env::var("WHISPER_NO_AVX").is_ok() {
                build.flag_if_supported("-mavx");
                build.flag_if_supported("-mavx2");
                build.flag_if_supported("-mfma");
                build.flag_if_supported("-mf16c");
            }
        }
        "aarch64" => {
            // ARM NEON optimizations
            build.flag_if_supported("-mfpu=neon");
        }
        _ => {}
    }

    // CUDA support
    #[cfg(feature = "cuda")]
    {
        build.define("GGML_USE_CUDA", None);
        build.cuda(true);
        build.file("../vendor/whisper.cpp/ggml/src/ggml-cuda.cu");
        println!("cargo:rustc-link-lib=cuda");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cudart");
    }

    // Debug/Release specific flags
    let profile = env::var("PROFILE").unwrap_or_else(|_| "release".to_string());
    match profile.as_str() {
        "debug" => {
            build.opt_level(0);
            build.define("DEBUG", None);
        }
        _ => {
            build.opt_level(3);
            build.define("NDEBUG", None);
        }
    }

    // Compile the library
    build.compile("whisper");
}

fn generate_bindings() {
    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=../vendor/whisper.cpp/include/whisper.h");

    let bindings = bindgen::Builder::default()
        // The input header we would like to generate bindings for
        .header("../vendor/whisper.cpp/include/whisper.h")
        // Tell bindgen this is a C++ header
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=c++11")
        // Include paths
        .clang_arg("-I../vendor/whisper.cpp/include")
        .clang_arg("-I../vendor/whisper.cpp/ggml/include")
        // Only generate bindings for whisper functions and types
        .allowlist_function("whisper_.*")
        .allowlist_type("whisper_.*")
        .allowlist_var("WHISPER_.*")
        // Don't generate bindings for C++ STL types
        .opaque_type("std::.*")
        .opaque_type("std::.*::.*")
        // Use core instead of std in generated code
        .use_core()
        .ctypes_prefix("::core::ffi")
        // Generate layout tests
        .layout_tests(true)
        // Derive common traits
        .derive_default(true)
        .derive_debug(true)
        // Generate the bindings
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}