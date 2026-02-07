use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "xtask")]
#[command(about = "Build automation for whisper-cpp-wrapper")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build precompiled whisper library for caching
    Prebuild {
        /// Build profile (debug or release)
        #[arg(long, default_value = "release")]
        profile: String,

        /// Target triple (auto-detected if not specified)
        #[arg(long)]
        target: Option<String>,

        /// Force rebuild even if library exists
        #[arg(long)]
        force: bool,
    },

    /// Clean prebuilt libraries
    Clean,

    /// Show information about prebuilt libraries
    Info,

    /// Download test models (whisper tiny.en + Silero VAD)
    TestSetup {
        /// Force re-download even if models exist
        #[arg(long)]
        force: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Prebuild { profile, target, force } => {
            prebuild(&profile, target, force)?;
        }
        Commands::Clean => {
            clean()?;
        }
        Commands::Info => {
            info()?;
        }
        Commands::TestSetup { force } => {
            test_setup(force)?;
        }
    }

    Ok(())
}

fn prebuild(profile: &str, target: Option<String>, force: bool) -> Result<()> {
    // Detect target if not specified
    let target = target.unwrap_or_else(|| {
        detect_target().unwrap_or_else(|| "unknown".to_string())
    });

    println!("Building prebuilt whisper library:");
    println!("  Target: {}", target);
    println!("  Profile: {}", profile);

    // Create prebuilt directory
    let prebuilt_dir = project_root()?.join("prebuilt").join(&target).join(profile);
    fs::create_dir_all(&prebuilt_dir)
        .context("Failed to create prebuilt directory")?;

    // Determine library name based on platform
    let lib_name = if target.contains("windows") {
        "whisper.lib"
    } else {
        "libwhisper.a"
    };

    let lib_path = prebuilt_dir.join(lib_name);

    // Check if library already exists
    if lib_path.exists() && !force {
        println!("Library already exists at: {}", lib_path.display());
        println!("Use --force to rebuild");
        return Ok(());
    }

    // Build the library using cc crate
    println!("Building whisper.cpp...");
    build_whisper_cpp(&target, profile, &prebuilt_dir)?;

    // Verify the library was created
    if lib_path.exists() {
        let size = fs::metadata(&lib_path)?.len();
        println!("✅ Successfully built {} ({:.2} MB)", lib_name, size as f64 / 1_048_576.0);
        println!("📁 Location: {}", prebuilt_dir.display());
        println!();
        println!("To use this prebuilt library in your project:");
        println!();
        println!("1. Set environment variable:");
        println!("   export WHISPER_PREBUILT_PATH={}", prebuilt_dir.display());
        println!();
        println!("2. Or add to your project's .cargo/config.toml:");
        println!("   [env]");
        println!("   WHISPER_PREBUILT_PATH = \"{}\"", prebuilt_dir.display());
    } else {
        anyhow::bail!("Failed to create library file");
    }

    Ok(())
}

fn build_whisper_cpp(target: &str, profile: &str, output_dir: &Path) -> Result<()> {
    // Set environment variables needed by cc crate
    if env::var("HOST").is_err() {
        if let Some(host) = detect_host() {
            env::set_var("HOST", host);
        }
    }
    if env::var("TARGET").is_err() && target != "unknown" {
        env::set_var("TARGET", target);
    }

    let mut build = cc::Build::new();

    // Configure C++ compilation
    build.cpp(true);
    build.std("c++17");

    let vendor_path = project_root()?.join("vendor/whisper.cpp");

    // Add include directories
    build.include(&vendor_path)
        .include(vendor_path.join("include"))
        .include(vendor_path.join("ggml/include"))
        .include(vendor_path.join("ggml/src"))
        .include(vendor_path.join("ggml/src/ggml-cpu"));

    // Add source files
    build.file(vendor_path.join("src/whisper.cpp"))
        // Core GGML files
        .file(vendor_path.join("ggml/src/ggml.c"))
        .file(vendor_path.join("ggml/src/ggml.cpp"))
        .file(vendor_path.join("ggml/src/ggml-alloc.c"))
        .file(vendor_path.join("ggml/src/ggml-backend.cpp"))
        .file(vendor_path.join("ggml/src/ggml-backend-reg.cpp"))
        .file(vendor_path.join("ggml/src/ggml-threading.cpp"))
        .file(vendor_path.join("ggml/src/ggml-quants.c"))
        .file(vendor_path.join("ggml/src/ggml-opt.cpp"))
        .file(vendor_path.join("ggml/src/gguf.cpp"))
        // CPU backend files
        .file(vendor_path.join("ggml/src/ggml-cpu/ggml-cpu.c"))
        .file(vendor_path.join("ggml/src/ggml-cpu/ggml-cpu.cpp"))
        .file(vendor_path.join("ggml/src/ggml-cpu/binary-ops.cpp"))
        .file(vendor_path.join("ggml/src/ggml-cpu/unary-ops.cpp"))
        .file(vendor_path.join("ggml/src/ggml-cpu/ops.cpp"))
        .file(vendor_path.join("ggml/src/ggml-cpu/quants.c"))
        .file(vendor_path.join("ggml/src/ggml-cpu/traits.cpp"))
        .file(vendor_path.join("ggml/src/ggml-cpu/vec.cpp"))
        .file(vendor_path.join("ggml/src/ggml-cpu/repack.cpp"))
        .file(vendor_path.join("ggml/src/ggml-cpu/hbm.cpp"))
        // AMX files
        .file(vendor_path.join("ggml/src/ggml-cpu/amx/amx.cpp"))
        .file(vendor_path.join("ggml/src/ggml-cpu/amx/mmq.cpp"));

    // Common flags
    build.flag_if_supported("-fPIC");
    build.define("_ALIGNAS_SUPPORTED", None);
    build.define("GGML_USE_CPU", None);
    build.define("WHISPER_VERSION", Some("\"1.8.3\""));
    build.define("GGML_VERSION", Some("\"0.9.5\""));
    build.define("GGML_COMMIT", Some("\"unknown\""));

    // Platform-specific configuration
    if target.contains("windows") {
        build.define("_CRT_SECURE_NO_WARNINGS", None);
        build.define("WIN32_LEAN_AND_MEAN", None);
        if target.contains("msvc") {
            build.flag("/O2");
            build.flag("/arch:AVX2");
            build.flag("/MT");
        }
    } else if target.contains("apple") || target.contains("darwin") {
        build.define("GGML_USE_ACCELERATE", None);
    } else if target.contains("linux") {
        // Linux-specific flags if needed
    }

    // Architecture-specific optimizations
    if target.contains("x86_64") {
        // Add x86-specific files
        build.file(vendor_path.join("ggml/src/ggml-cpu/arch/x86/quants.c"))
            .file(vendor_path.join("ggml/src/ggml-cpu/arch/x86/repack.cpp"))
            .file(vendor_path.join("ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp"));

        build.flag_if_supported("-mavx");
        build.flag_if_supported("-mavx2");
        build.flag_if_supported("-mfma");
        build.flag_if_supported("-mf16c");
    }

    // Profile-specific flags
    match profile {
        "debug" => {
            build.opt_level(0);
            build.define("DEBUG", None);
        }
        _ => {
            build.opt_level(3);
            build.define("NDEBUG", None);
        }
    }

    // Set output directory
    build.out_dir(output_dir);

    // Set the target
    if target != "unknown" {
        build.target(target);
    }

    // Compile the library
    build.compile("whisper");

    Ok(())
}

fn clean() -> Result<()> {
    let prebuilt_dir = project_root()?.join("prebuilt");

    if prebuilt_dir.exists() {
        println!("Removing prebuilt directory: {}", prebuilt_dir.display());
        fs::remove_dir_all(&prebuilt_dir)?;
        println!("✅ Cleaned prebuilt libraries");
    } else {
        println!("No prebuilt directory found");
    }

    Ok(())
}

fn info() -> Result<()> {
    let prebuilt_dir = project_root()?.join("prebuilt");

    if !prebuilt_dir.exists() {
        println!("No prebuilt libraries found");
        return Ok(());
    }

    println!("Prebuilt libraries:");
    println!();

    // Walk through all prebuilt libraries
    for target_entry in fs::read_dir(&prebuilt_dir)? {
        let target_entry = target_entry?;
        if target_entry.file_type()?.is_dir() {
            let target_name = target_entry.file_name();

            let target_str = target_name.to_string_lossy();
            let lib_name = if target_str.contains("windows") {
                "whisper.lib"
            } else {
                "libwhisper.a"
            };

            for profile_entry in fs::read_dir(target_entry.path())? {
                let profile_entry = profile_entry?;
                if profile_entry.file_type()?.is_dir() {
                    let profile_name = profile_entry.file_name();
                    let lib_path = profile_entry.path().join(lib_name);

                    if lib_path.exists() {
                        let size = fs::metadata(&lib_path)?.len();
                        println!("  {} / {} ({:.2} MB)",
                            target_str,
                            profile_name.to_string_lossy(),
                            size as f64 / 1_048_576.0
                        );
                        println!("    Path: {}", lib_path.display());
                    }
                }
            }
        }
    }

    Ok(())
}

fn test_setup(force: bool) -> Result<()> {
    let root = project_root()?;
    let models_dir = root.join("vendor").join("whisper.cpp").join("models");

    println!("Setting up test models in {}", models_dir.display());
    println!();

    let models: &[(&str, &str, &str, &str)] = &[
        ("ggml-tiny.en.bin", "download-ggml-model.cmd", "download-ggml-model.sh", "tiny.en"),
        ("ggml-silero-v6.2.0.bin", "download-vad-model.cmd", "download-vad-model.sh", "silero-v6.2.0"),
    ];

    for (filename, cmd_script, sh_script, arg) in models {
        let model_path = models_dir.join(filename);

        if model_path.exists() && !force {
            println!("  [skip] {} (already exists)", filename);
            continue;
        }

        if force && model_path.exists() {
            fs::remove_file(&model_path)?;
        }

        println!("  [download] {} ...", filename);

        let status = if cfg!(windows) {
            let script_path = models_dir.join(cmd_script);
            std::process::Command::new("cmd")
                .args(["/c", &script_path.to_string_lossy(), arg, &models_dir.to_string_lossy()])
                .status()
                .context(format!("Failed to run {}", cmd_script))?
        } else {
            let script_path = models_dir.join(sh_script);
            let script_str = script_path.to_string_lossy().to_string();
            let arg_str = arg.to_string();
            let models_str = models_dir.to_string_lossy().to_string();
            std::process::Command::new("bash")
                .args([&script_str, &arg_str, &models_str])
                .status()
                .context(format!("Failed to run {}", sh_script))?
        };

        if !status.success() {
            anyhow::bail!("Download failed for {}", filename);
        }
    }

    println!();
    println!("Done! Run tests with: cargo test --test stream_pcm_integration -- --nocapture");
    Ok(())
}

fn project_root() -> Result<PathBuf> {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR")
        .context("CARGO_MANIFEST_DIR not set")?;

    // The xtask crate is in project_root/xtask, so go up one level
    let root = Path::new(&manifest_dir).parent()
        .context("Failed to find project root")?
        .to_path_buf();

    Ok(root)
}

fn detect_target() -> Option<String> {
    // Try to get from environment
    if let Ok(target) = env::var("TARGET") {
        return Some(target);
    }

    // Detect based on current platform
    detect_host()
}

fn detect_host() -> Option<String> {
    let target = if cfg!(all(target_os = "windows", target_arch = "x86_64", target_env = "msvc")) {
        "x86_64-pc-windows-msvc"
    } else if cfg!(all(target_os = "windows", target_arch = "x86_64", target_env = "gnu")) {
        "x86_64-pc-windows-gnu"
    } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        "x86_64-unknown-linux-gnu"
    } else if cfg!(all(target_os = "macos", target_arch = "x86_64")) {
        "x86_64-apple-darwin"
    } else if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        "aarch64-apple-darwin"
    } else if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
        "aarch64-unknown-linux-gnu"
    } else {
        return None;
    };
    Some(target.to_string())
}