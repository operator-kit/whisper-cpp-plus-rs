# Publishing Guide

Guide for publishing whisper-cpp-plus crates to crates.io.

For branch policy, PR flow, and release branch handling, see
[`CONTRIBUTING.md`](../CONTRIBUTING.md). This guide focuses on the concrete
release and crates.io publishing checklist.

## Pre-publish Checklist

### 1. Release Branch Sanity

Confirm the release commit is clean, pushed, and green in CI before publishing:

```bash
git status --short --branch
git log -1 --oneline --decorate

cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace -- --test-threads=1
cargo clippy -p whisper-cpp-plus --all-targets --features async -- -D warnings
cargo test -p whisper-cpp-plus --features async -- --test-threads=1
```

The macOS CI workflow should also be green on the release commit. For macOS-sensitive releases, verify:

```bash
cargo xtask test-setup
MACOSX_DEPLOYMENT_TARGET=14.0 cargo clippy -p whisper-cpp-plus --all-targets --features metal -- -D warnings
MACOSX_DEPLOYMENT_TARGET=14.0 cargo test -p whisper-cpp-plus --features metal -- --test-threads=1

cargo xtask clean
MACOSX_DEPLOYMENT_TARGET=14.0 cargo xtask prebuild --force
cargo xtask info
WHISPER_PREBUILT_PATH=prebuilt/<apple-target>/release cargo test -p whisper-cpp-plus-sys
WHISPER_PREBUILT_PATH=prebuilt/<apple-target>/release cargo test -p whisper-cpp-plus --test stream_pcm_integration -- --nocapture --test-threads=1
```

### 2. Version Bump

Update version in all locations:

```bash
# Workspace version (root Cargo.toml)
# whisper-cpp-plus-sys dependency version (whisper-cpp-plus/Cargo.toml)
# README.md examples (root + whisper-cpp-plus/)
# Doc comments (whisper-cpp-plus/src/quantize.rs)
```

Update `CHANGELOG.md` with a dated release entry before publishing.

### 3. Test docs.rs Build Locally

docs.rs runs in a **network-isolated container**, so it cannot download whisper.cpp at build time. The sys crate package therefore ships only whisper.cpp's public headers (`whisper.cpp/include/*.h`, `whisper.cpp/ggml/include/*.h`) plus whisper.cpp's `LICENSE`. When `build.rs` sees `DOCS_RS=1` it skips compiling whisper.cpp and runs the normal bindgen step against those headers, so docs.rs gets exact bindings with nothing to maintain by hand. The docs.rs image includes libclang (`clang`, `libclang-dev` in [crates-build-env](https://github.com/rust-lang/crates-build-env)).

Regular builds of the published crate ignore the packaged headers: `build.rs` only uses the bundled `whisper.cpp/` directory when it contains the full source tree (`CMakeLists.txt` and `src/whisper.cpp`), otherwise it downloads the pinned commit.

**Simulate docs.rs against the packaged crate** (headers only, no network):

```bash
cargo package -p whisper-cpp-plus-sys --no-verify
mkdir -p /tmp/wcp-docsrs && tar -xzf target/package/whisper-cpp-plus-sys-X.Y.Z.crate -C /tmp/wcp-docsrs
DOCS_RS=1 cargo doc --offline --no-deps \
  --manifest-path /tmp/wcp-docsrs/whisper-cpp-plus-sys-X.Y.Z/Cargo.toml \
  --target-dir /tmp/wcp-docsrs/target

# The high-level crate, as docs.rs builds it (all features):
DOCS_RS=1 cargo doc --offline --no-deps -p whisper-cpp-plus --all-features --target-dir /tmp/wcp-docsrs/target
```

On Windows, GNU tar needs `--force-local` for paths with a drive letter. Use a separate `--target-dir` so the `DOCS_RS` build script run doesn't invalidate your normal build cache.

### 4. Run Tests

```bash
cargo test -p whisper-cpp-plus
cargo test -p whisper-cpp-plus --features async
```

### 5. Verify Package Contents

```bash
cargo package -p whisper-cpp-plus-sys --list
cargo package -p whisper-cpp-plus --list
```

### 6. Dry-run Publishing

Always dry-run the package that is about to be published:

```bash
cargo publish -p whisper-cpp-plus-sys --dry-run
```

After `whisper-cpp-plus-sys` is published and appears in the crates.io index, dry-run the high-level crate:

```bash
cargo publish -p whisper-cpp-plus --dry-run
```

`whisper-cpp-plus` depends on the same-version `whisper-cpp-plus-sys` from crates.io. Its package verification will fail until that sys crate version exists in the crates.io index.

### Windows Package Verification

Some Windows antivirus tools block Cargo from executing temporary package verification build scripts. If package verification fails with `Access is denied`, create and whitelist a stable target directory, then rerun with `--target-dir`:

```powershell
New-Item -ItemType Directory -Force D:\cargo-package-verify-whisper-cpp-plus-rs\target
cargo publish -p whisper-cpp-plus-sys --dry-run --target-dir D:\cargo-package-verify-whisper-cpp-plus-rs\target
```

## Publishing

**Order matters** - sys crate must be published first:

```bash
# 1. Publish sys crate
cargo publish -p whisper-cpp-plus-sys

# 2. Wait for crates.io index to include whisper-cpp-plus-sys v0.1.X
cargo search whisper-cpp-plus-sys --limit 5

# 3. Dry-run main crate
cargo publish -p whisper-cpp-plus --dry-run

# 4. Publish main crate
cargo publish -p whisper-cpp-plus
```

## Git Tags & GitHub Releases

After both crates are live, create matching git tags and the GitHub release. Do not tag before both crates publish successfully.

```bash
# Tag current commit
git tag -a v0.1.X -m "v0.1.X: Brief description"
git push origin v0.1.X

# Create GitHub release
gh release create v0.1.X --title "v0.1.X" --notes "Release notes here"
```

## Verifying docs.rs Build

After publishing, monitor the docs.rs build:

1. Check build queue: https://docs.rs/releases/queue
2. View build status: https://docs.rs/crate/whisper-cpp-plus/VERSION/builds
3. If build fails, check the logs

### Common docs.rs Failures

| Error | Cause | Fix |
|-------|-------|-----|
| DNS resolution failed | Network access attempted | Ensure the `DOCS_RS` early return in `build.rs` runs before any download or CMake step |
| `whisper.h not found` | Headers missing from the package | Check the `include` list in `whisper-cpp-plus-sys/Cargo.toml` and `cargo package -p whisper-cpp-plus-sys --list` |
| `'<header>.h' file not found` | A packaged header includes a file outside the packaged directories | Add the directory to the sys crate's `include` list |
| Unable to find libclang | docs.rs image changed | Check [crates-build-env](https://github.com/rust-lang/crates-build-env) and open an issue there |

New FFI functions need no docs.rs-specific work: bindings are generated from the same headers everywhere.

## Yanking Bad Releases

If a release has critical issues:

```bash
cargo yank --version 0.1.X whisper-cpp-plus
cargo yank --version 0.1.X whisper-cpp-plus-sys
```

Note: Yanked versions can still be used by existing Cargo.lock files but won't be selected for new projects.
