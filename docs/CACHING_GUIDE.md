# Build Caching Guide

## Problem
When including this crate as a local dependency, whisper.cpp rebuilds every time (several minutes per build).

## Solution: Prebuilt Libraries with xtask

This project uses the `xtask` pattern for build automation. The primary commands:

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

The prebuilt library is stored at `prebuilt/{target}/{profile}/` and automatically detected by `whisper-sys/build.rs` on subsequent builds.

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

# Force rebuild even if library exists
cargo xtask prebuild --force
```

### Test Setup

Download models required for integration tests:

```bash
cargo xtask test-setup

# Force re-download
cargo xtask test-setup --force
```

Downloads `ggml-tiny.en.bin` and `ggml-silero-v6.2.0.bin` into `vendor/whisper.cpp/models/`. Works on both Windows (`.cmd` scripts) and Unix (`.sh` scripts).

### Performance Impact

- **Without caching**: Full C++ compilation takes several minutes
- **With prebuilt library**: Build completes in <1 second

### How It Works

1. `cargo xtask prebuild` compiles whisper.cpp via the `cc` crate and stores the static library in `prebuilt/`
2. `whisper-sys/build.rs` checks for prebuilt libraries before compiling:
   - First checks `WHISPER_PREBUILT_PATH` env var
   - Then checks `prebuilt/{target}/{profile}/` relative to project root
   - On Unix, also checks system paths (`/usr/local/lib`, `/usr/lib`, `/opt/homebrew/lib`)
3. If found, it links the prebuilt library instead of recompiling

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
members = ["whisper-cpp-rs", "your-project"]
resolver = "2"
```

#### sccache
For additional caching across projects:

```bash
cargo install sccache
export RUSTC_WRAPPER=sccache
```

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
