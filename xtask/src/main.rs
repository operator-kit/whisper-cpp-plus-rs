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

        /// Enable CUDA in prebuilt library
        #[arg(long)]
        cuda: bool,
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
        Commands::Prebuild {
            profile,
            target,
            force,
            cuda,
        } => {
            prebuild(&profile, target, force, cuda)?;
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

fn prebuild(profile: &str, target: Option<String>, force: bool, cuda: bool) -> Result<()> {
    let target = target.unwrap_or_else(|| detect_target().unwrap_or_else(|| "unknown".to_string()));

    println!("Building prebuilt whisper library:");
    println!("  Target: {}", target);
    println!("  Profile: {}", profile);
    if cuda {
        println!("  CUDA: enabled");
    }

    let prebuilt_dir = project_root()?.join("prebuilt").join(&target).join(profile);
    fs::create_dir_all(&prebuilt_dir).context("Failed to create prebuilt directory")?;

    let lib_name = if target.contains("windows") {
        "whisper.lib"
    } else {
        "libwhisper.a"
    };

    let lib_path = prebuilt_dir.join(lib_name);

    if lib_path.exists() && !force {
        println!("Library already exists at: {}", lib_path.display());
        println!("Use --force to rebuild");
        return Ok(());
    }

    println!("Building whisper.cpp with CMake...");
    build_whisper_cpp(&target, profile, &prebuilt_dir, cuda)?;

    if lib_path.exists() {
        let size = fs::metadata(&lib_path)?.len();
        println!(
            "Successfully built {} ({:.2} MB)",
            lib_name,
            size as f64 / 1_048_576.0
        );
        println!("Location: {}", prebuilt_dir.display());
        println!();
        println!("To use this prebuilt library:");
        println!();
        println!("1. Set environment variable:");
        println!("   export WHISPER_PREBUILT_PATH={}", prebuilt_dir.display());
        println!();
        println!("2. Or add to .cargo/config.toml:");
        println!("   [env]");
        println!("   WHISPER_PREBUILT_PATH = \"{}\"", prebuilt_dir.display());
    } else {
        anyhow::bail!("Failed to create library file");
    }

    Ok(())
}

fn build_whisper_cpp(target: &str, profile: &str, output_dir: &Path, cuda: bool) -> Result<()> {
    let root = project_root()?;
    let vendor_path = root.join("whisper-cpp-plus-sys/whisper.cpp");

    let mut config = cmake::Config::new(&vendor_path);
    config
        .profile(if profile == "debug" {
            "Debug"
        } else {
            "Release"
        })
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("WHISPER_BUILD_TESTS", "OFF")
        .define("WHISPER_BUILD_EXAMPLES", "OFF")
        .pic(true);

    if cuda {
        config.define("GGML_CUDA", "ON");
    }

    if target.contains("windows") {
        config.cxxflag("/utf-8");
    }

    if target != "unknown" {
        // cmake crate doesn't have a .target() — set via env
        env::set_var("TARGET", target);
    }

    let destination = config.build();

    // Copy produced libs to output_dir
    copy_built_libs(&destination, output_dir, target, cuda)?;

    Ok(())
}

/// Copy CMake-produced static libs from the build destination to the output directory.
fn copy_built_libs(cmake_dest: &Path, output_dir: &Path, target: &str, cuda: bool) -> Result<()> {
    let ext = if target.contains("windows") {
        "lib"
    } else {
        "a"
    };
    let prefix = if target.contains("windows") {
        ""
    } else {
        "lib"
    };

    let mut libs = vec!["whisper", "ggml", "ggml-base", "ggml-cpu"];
    if cuda {
        libs.push("ggml-cuda");
    }

    // Collect all lib files from cmake destination tree
    let mut lib_files: Vec<PathBuf> = Vec::new();
    collect_lib_files(cmake_dest, &mut lib_files, ext);

    for lib_name in &libs {
        let filename = format!("{}{}.{}", prefix, lib_name, ext);
        let found = lib_files.iter().find(|p| {
            p.file_name()
                .map(|f| f.to_string_lossy() == filename)
                .unwrap_or(false)
        });

        if let Some(src_path) = found {
            let dest_path = output_dir.join(&filename);
            fs::copy(src_path, &dest_path).with_context(|| {
                format!(
                    "Failed to copy {} to {}",
                    src_path.display(),
                    dest_path.display()
                )
            })?;
            let size = fs::metadata(&dest_path)?.len();
            println!("  {} ({:.2} MB)", filename, size as f64 / 1_048_576.0);
        } else {
            println!("  [warn] {} not found in build output", filename);
        }
    }

    Ok(())
}

/// Recursively collect all files with the given extension from a directory tree.
fn collect_lib_files(dir: &Path, files: &mut Vec<PathBuf>, ext: &str) {
    if !dir.exists() {
        return;
    }
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_lib_files(&path, files, ext);
            } else if path.extension().map(|e| e == ext).unwrap_or(false) {
                files.push(path);
            }
        }
    }
}

fn clean() -> Result<()> {
    let prebuilt_dir = project_root()?.join("prebuilt");

    if prebuilt_dir.exists() {
        println!("Removing prebuilt directory: {}", prebuilt_dir.display());
        fs::remove_dir_all(&prebuilt_dir)?;
        println!("Cleaned prebuilt libraries");
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

    for target_entry in fs::read_dir(&prebuilt_dir)? {
        let target_entry = target_entry?;
        if target_entry.file_type()?.is_dir() {
            let target_name = target_entry.file_name();
            let target_str = target_name.to_string_lossy();

            let ext = if target_str.contains("windows") {
                "lib"
            } else {
                "a"
            };
            let prefix = if target_str.contains("windows") {
                ""
            } else {
                "lib"
            };

            for profile_entry in fs::read_dir(target_entry.path())? {
                let profile_entry = profile_entry?;
                if profile_entry.file_type()?.is_dir() {
                    let profile_name = profile_entry.file_name();
                    let profile_dir = profile_entry.path();

                    // Check for whisper lib
                    let whisper_lib = format!("{}whisper.{}", prefix, ext);
                    let lib_path = profile_dir.join(&whisper_lib);
                    if !lib_path.exists() {
                        continue;
                    }

                    let size = fs::metadata(&lib_path)?.len();
                    println!(
                        "  {} / {} ({:.2} MB)",
                        target_str,
                        profile_name.to_string_lossy(),
                        size as f64 / 1_048_576.0
                    );
                    println!("    Path: {}", lib_path.display());

                    // List satellite libs
                    let satellites = ["ggml", "ggml-base", "ggml-cpu", "ggml-cuda"];
                    let mut found_satellites = Vec::new();
                    for sat in &satellites {
                        let sat_file = format!("{}{}.{}", prefix, sat, ext);
                        if profile_dir.join(&sat_file).exists() {
                            found_satellites.push(*sat);
                        }
                    }
                    if !found_satellites.is_empty() {
                        println!("    Satellites: {}", found_satellites.join(", "));
                    }
                }
            }
        }
    }

    Ok(())
}

fn test_setup(force: bool) -> Result<()> {
    let root = project_root()?;
    let models_dir = root
        .join("whisper-cpp-plus-sys")
        .join("whisper.cpp")
        .join("models");

    println!("Setting up test models in {}", models_dir.display());
    println!();

    let models: &[(&str, &str, &str, &str)] = &[
        (
            "ggml-tiny.en.bin",
            "download-ggml-model.cmd",
            "download-ggml-model.sh",
            "tiny.en",
        ),
        (
            "ggml-silero-v6.2.0.bin",
            "download-vad-model.cmd",
            "download-vad-model.sh",
            "silero-v6.2.0",
        ),
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
                .args([
                    "/c",
                    &script_path.to_string_lossy(),
                    arg,
                    &models_dir.to_string_lossy(),
                ])
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
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").context("CARGO_MANIFEST_DIR not set")?;

    let root = Path::new(&manifest_dir)
        .parent()
        .context("Failed to find project root")?
        .to_path_buf();

    Ok(root)
}

fn detect_target() -> Option<String> {
    if let Ok(target) = env::var("TARGET") {
        return Some(target);
    }
    detect_host()
}

fn detect_host() -> Option<String> {
    let target = if cfg!(all(
        target_os = "windows",
        target_arch = "x86_64",
        target_env = "msvc"
    )) {
        "x86_64-pc-windows-msvc"
    } else if cfg!(all(
        target_os = "windows",
        target_arch = "x86_64",
        target_env = "gnu"
    )) {
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
