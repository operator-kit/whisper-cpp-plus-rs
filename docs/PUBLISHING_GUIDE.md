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
cargo test --workspace -- --test-threads=1
cargo test -p whisper-cpp-plus --features async -- --test-threads=1
```

The macOS CI workflow should also be green on the release commit. For macOS-sensitive releases, verify:

```bash
cargo xtask test-setup
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

docs.rs runs in a **network-isolated container** - it cannot download dependencies at build time. Our `build.rs` detects `DOCS_RS=1` and generates stub bindings instead of compiling whisper.cpp.

**Test the stub bindings work:**

```bash
# Clean and rebuild with DOCS_RS simulation
export DOCS_RS=1
cargo clean -p whisper-cpp-plus-sys
cargo check -p whisper-cpp-plus

# Test docs generation
cargo doc -p whisper-cpp-plus --no-deps
```

If this fails, the stub bindings in `whisper-cpp-plus-sys/build.rs` (`generate_stub_bindings()`) need updating to include missing FFI symbols.

Unset `DOCS_RS` before running normal tests again.

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
3. If build fails, check logs and fix stub bindings

### Common docs.rs Failures

| Error | Cause | Fix |
|-------|-------|-----|
| DNS resolution failed | Network access attempted | Ensure `DOCS_RS` check in build.rs |
| Cannot find function X | Missing stub binding | Add function to `generate_stub_bindings()` |
| Type mismatch | Stub signature wrong | Match stub to actual usage in high-level crate |
| Inner attribute not permitted | `#![allow(...)]` in included file | Remove inner attrs from stub bindings |

## Stub Bindings Maintenance

When adding new FFI functions to the high-level crate, also add stubs:

1. Add function to `generate_stub_bindings()` in `whisper-cpp-plus-sys/build.rs`
2. Match the signature to how the high-level code calls it
3. Test with `DOCS_RS=1 cargo check -p whisper-cpp-plus`

## Yanking Bad Releases

If a release has critical issues:

```bash
cargo yank --version 0.1.X whisper-cpp-plus
cargo yank --version 0.1.X whisper-cpp-plus-sys
```

Note: Yanked versions can still be used by existing Cargo.lock files but won't be selected for new projects.
