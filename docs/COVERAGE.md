# Coverage

This project uses [`cargo-llvm-cov`](https://github.com/taiki-e/cargo-llvm-cov)
for local Rust coverage reports. Coverage is currently a diagnostic tool, not a
merge or release gate.

## One-Time Setup

Install the LLVM tools component and the coverage runner:

```bash
rustup component add llvm-tools-preview
cargo install cargo-llvm-cov --locked
```

Download the model fixtures before running model-backed tests:

```bash
cargo xtask test-setup
```

## Windows Antivirus Setup

Some Windows antivirus tools interfere with Cargo when coverage builds create
and execute files in temporary directories. If that happens, create a stable
target directory and temp directory, whitelist them in the antivirus tool, then
run coverage with those paths:

```powershell
New-Item -ItemType Directory -Force D:\cargo-coverage-whisper-cpp-plus-rs\target
New-Item -ItemType Directory -Force D:\cargo-coverage-whisper-cpp-plus-rs\tmp

$env:CARGO_TARGET_DIR='D:\cargo-coverage-whisper-cpp-plus-rs\target'
$env:TMP='D:\cargo-coverage-whisper-cpp-plus-rs\tmp'
$env:TEMP='D:\cargo-coverage-whisper-cpp-plus-rs\tmp'
```

Keep those environment variables set in the shell that runs `cargo llvm-cov`.

## Wrapper Crate Baseline

Use the high-level crate as the default local baseline. This keeps the report
focused on the public wrapper behavior and avoids including `xtask` utility
code in the main number.

```bash
cargo llvm-cov clean --workspace
cargo llvm-cov -p whisper-cpp-plus --text --summary-only -- --test-threads=1
```

To save a machine-readable report:

```bash
cargo llvm-cov -p whisper-cpp-plus --json --summary-only --output-path coverage-whisper-cpp-plus.json -- --test-threads=1
```

## Optional Features

Run optional feature coverage separately when touching feature-gated code. For
the async API:

```bash
cargo llvm-cov clean --workspace
cargo llvm-cov -p whisper-cpp-plus --features async --text --summary-only -- --test-threads=1
```

For CI providers or hosted coverage services, generate LCOV:

```bash
cargo llvm-cov -p whisper-cpp-plus --lcov --output-path lcov.info -- --test-threads=1
```

## Workspace Report

Use a workspace report when changes affect build automation, the sys crate, or
cross-crate behavior:

```bash
cargo llvm-cov clean --workspace
cargo llvm-cov --workspace --text --summary-only -- --test-threads=1
```

The workspace number includes `xtask`, so it is expected to be lower than the
high-level crate number unless `xtask` coverage is improved.

## Notes

- Do not combine `--all-targets` with `-- --test-threads=1`; Criterion benches
  receive the test harness flag and fail. Run benchmark checks separately.
- Keep the normal quality gates from `CONTRIBUTING.md` as the source of truth
  for merge readiness. Coverage helps identify blind spots, but it does not
  replace formatting, clippy, or model-backed tests.
