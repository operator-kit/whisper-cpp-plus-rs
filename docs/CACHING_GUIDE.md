# Build Caching Guide

## Problem
When including this crate as a local dependency, whisper.cpp rebuilds every time (several minutes per build).

## Default Build

By default, `cargo build` invokes CMake to compile whisper.cpp from source. This works for all features including CUDA — just install the toolkit and `cargo build --features cuda`.

The first build takes several minutes (CMake configure + full C++ compile). Subsequent builds are cached by cargo unless the source changes.

## Optional: Prebuilt Libraries with xtask

For CI pipelines or to skip recompilation, you can prebuild and cache the static libraries. The `xtask` commands:

| Command | Description |
|---------|-------------|
| `cargo xtask prebuild` | Build and cache the whisper library |
| `cargo xtask info` | Show available prebuilt libraries |
| `cargo xtask clean` | Remove all prebuilt libraries |
| `cargo xtask test-setup` | Download test models (whisper tiny.en + Silero VAD) |

### Quick Start

```bash
# Build once, reuse on every subsequent cargo build
cargo xtask prebuild

# Verify it worked
cargo xtask info
```

The prebuilt libraries are stored at `prebuilt/{target}/{profile}/` and automatically detected by `whisper-cpp-plus-sys/build.rs` on subsequent builds. On macOS, the default xtask prebuild path produces a CPU/BLAS cache and disables Metal.

### Using the Prebuilt Library

#### Option 1: Automatic Detection (Recommended)
No configuration needed. The build system checks `prebuilt/{target}/{profile}/` automatically.

#### Option 2: Environment Variable
Set the path explicitly if you've moved the prebuilt library:

```bash
# Windows
set WHISPER_PREBUILT_PATH=C:\path\to\prebuilt\x86_64-pc-windows-msvc\release

# Unix/Mac
export WHISPER_PREBUILT_PATH=/path/to/prebuilt/x86_64-unknown-linux-gnu/release
```

#### Option 3: Project Configuration
Add to your project's `.cargo/config.toml`:

```toml
[env]
WHISPER_PREBUILT_PATH = "/path/to/prebuilt/x86_64-pc-windows-msvc/release"
```

### Prebuild Options

```bash
# Debug build
cargo xtask prebuild --profile debug

# Release build (default)
cargo xtask prebuild --profile release

# Specify target explicitly
cargo xtask prebuild --target aarch64-apple-darwin

# Force rebuild even if library exists.
# This removes the existing target/profile cache before rebuilding.
cargo xtask prebuild --force
```

### macOS Metal and Prebuilt Libraries

`cargo xtask prebuild` disables `GGML_METAL` on macOS, so the generated cache is intended for CPU/BLAS builds. For `--features metal`, either let the crate build whisper.cpp from source or point `WHISPER_PREBUILT_PATH` at a custom cache that includes `libggml-metal.a`. The build script fails early if Metal is enabled with an incomplete prebuilt cache.

### Test Setup

Download models required for integration tests:

```bash
cargo xtask test-setup

# Force re-download
cargo xtask test-setup --force
```

Downloads `ggml-tiny.en.bin` and `ggml-silero-v6.2.0.bin` into `whisper-cpp-plus-sys/whisper.cpp/models/`. Works on both Windows (`.cmd` scripts) and Unix (`.sh` scripts).

### Performance Impact

- **Without caching**: Full C++ compilation takes several minutes
- **With prebuilt library**: Build completes in <1 second

### How It Works

1. `cargo xtask prebuild` compiles whisper.cpp via CMake and stores static libraries in `prebuilt/`
2. `whisper-cpp-plus-sys/build.rs` checks for prebuilt libraries before invoking CMake:
   - First checks `WHISPER_PREBUILT_PATH` env var
   - Then checks `prebuilt/{target}/{profile}/` relative to project root
   - On Unix, also checks system paths (`/usr/local/lib`, `/usr/lib`, `/opt/homebrew/lib`)
3. If found, it links the prebuilt libraries instead of running CMake

### Verification

When a prebuilt library is used, you'll see during build:
```
warning: Using prebuilt whisper library from: /path/to/prebuilt
```

### Alternative Solutions

#### Cargo Workspace
If both projects are yours, combine them into a workspace to share build artifacts:

```toml
[workspace]
members = ["whisper-cpp-plus", "your-project"]
resolver = "2"
```

#### sccache
For additional caching across projects:

```bash
cargo install sccache
export RUSTC_WRAPPER=sccache
```

## CUDA Builds

### Direct Build (Recommended)

Install the CUDA toolkit and build directly:

```bash
cargo build --features cuda
```

The build script invokes CMake with `-DGGML_CUDA=1` automatically. CMake handles CUDA compiler discovery.

To set a specific GPU architecture:

```bash
CMAKE_CUDA_ARCHITECTURES=86 cargo build --features cuda
```

Any `CMAKE_*` environment variable is passed through to CMake.

### Prebuilt CUDA Libraries

For CI or repeated builds, use xtask:

```bash
cargo xtask prebuild --cuda
```

Or set `WHISPER_PREBUILT_PATH` to a directory containing pre-compiled static libs (whisper + ggml satellites + ggml-cuda).

### CUDA Toolkit Detection

The build script locates the CUDA toolkit in this order:

1. `CUDA_PATH` environment variable
2. `CUDA_HOME` environment variable
3. Standard paths:
   - Windows: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y` (latest version)
   - Linux: `/usr/local/cuda`, `/usr/lib/cuda`, `/opt/cuda`

### Common `CMAKE_CUDA_ARCHITECTURES` Values

| GPU | Architecture | Value |
|-----|-------------|-------|
| RTX 4090/4080/4070 | Ada Lovelace | `89` |
| RTX 3090/3080/3070 | Ampere | `86` |
| RTX 3060/3050 | Ampere | `86` |
| A100 | Ampere | `80` |
| RTX 2080/2070 | Turing | `75` |
| GTX 1080/1070 | Pascal | `61` |
| Multiple GPUs | Mixed | `"75;86;89"` |

### Troubleshooting

1. **Library not found**: Run `cargo xtask info` to see available prebuilt libraries
2. **Linking errors**: Ensure the prebuilt library matches your target architecture and compiler
3. **Outdated library**: Run `cargo xtask clean` then `cargo xtask prebuild` to rebuild

## Platform Support

Target auto-detection works on:
- Windows (MSVC and GNU, x86_64)
- macOS (x86_64 and ARM64)
- Linux (x86_64 and ARM64)

For other targets, use `--target` explicitly: `cargo xtask prebuild --target <triple>`
