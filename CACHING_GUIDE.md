# Whisper-cpp-wrapper Caching Guide

## Problem
When including this crate as a local dependency, whisper.cpp rebuilds every time, which is time-consuming (several minutes per build).

## Solution: Prebuilt Libraries with xtask

This project uses the `xtask` pattern to provide cross-platform build automation for creating and managing prebuilt libraries.

### Quick Start

1. **Build the library once:**
   ```bash
   cargo xtask prebuild
   ```

   This creates a prebuilt library in `prebuilt/{target}/{profile}/` that will be automatically detected on subsequent builds.

2. **Check what prebuilt libraries are available:**
   ```bash
   cargo xtask info
   ```

3. **Clean prebuilt libraries if needed:**
   ```bash
   cargo xtask clean
   ```

### Using in Your Project

Once you've built the library with `cargo xtask prebuild`, there are two ways to use it:

#### Option 1: Automatic Detection (Recommended)
The build system will automatically detect and use prebuilt libraries in the standard location. No configuration needed!

#### Option 2: Environment Variable
Set the path explicitly if you've moved the prebuilt library:

```bash
# Windows
set WHISPER_PREBUILT_PATH=/path/to/whisper-cpp-rs\prebuilt\x86_64-pc-windows-msvc\release

# Unix/Mac
export WHISPER_PREBUILT_PATH=/path/to/whisper-cpp-rs/prebuilt/x86_64-unknown-linux-gnu/release
```

#### Option 3: Project Configuration
Add to your project's `.cargo/config.toml`:

```toml
[env]
WHISPER_PREBUILT_PATH = "/path/to/whisper-cpp-rs/prebuilt/x86_64-pc-windows-msvc/release"
```

### Advanced Options

#### Building for Different Profiles
```bash
# Debug build
cargo xtask prebuild --profile debug

# Release build (default)
cargo xtask prebuild --profile release
```

#### Cross-compilation
```bash
# Specify target explicitly
cargo xtask prebuild --target aarch64-apple-darwin
```

#### Force Rebuild
```bash
# Rebuild even if library exists
cargo xtask prebuild --force
```

### Performance Impact

- **Without caching**: Full C++ compilation takes several minutes
- **With prebuilt library**: Build completes in <1 second

### How It Works

1. `cargo xtask prebuild` compiles whisper.cpp once and stores it in `prebuilt/`
2. The `whisper-sys/build.rs` checks for prebuilt libraries before compiling
3. If found, it links the prebuilt library instead of recompiling

### Verification

When using a prebuilt library, you'll see this message during build:
```
warning: Using prebuilt whisper library from: /path/to/prebuilt
```

### Alternative Solutions

#### Use Cargo Workspace
If both projects are yours, combine them into a workspace to share build artifacts:

```toml
# In parent directory's Cargo.toml
[workspace]
members = ["whisper-cpp-rs", "your-project"]
resolver = "2"
```

#### Use sccache
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

The xtask solution is fully cross-platform and works on:
- Windows (MSVC and GNU)
- macOS (x86_64 and ARM64)
- Linux (x86_64 and ARM64)