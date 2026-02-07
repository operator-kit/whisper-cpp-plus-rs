# xtask

Build automation tasks for the whisper-cpp-rs workspace.

## Running

```sh
cargo xtask <command>
```

## Commands

### `prebuild`

Build a precompiled whisper.cpp static library for caching. Avoids recompilation on every `cargo build`.

```sh
cargo xtask prebuild                    # release build, auto-detect target
cargo xtask prebuild --profile debug    # debug build
cargo xtask prebuild --target x86_64-pc-windows-msvc
cargo xtask prebuild --force            # rebuild even if exists
```

Output goes to `prebuilt/<target>/<profile>/`. To use it, set:
```sh
export WHISPER_PREBUILT_PATH=prebuilt/<target>/<profile>
```

### `clean`

Remove all prebuilt libraries.

```sh
cargo xtask clean
```

### `info`

List all prebuilt libraries with sizes and paths.

```sh
cargo xtask info
```

### `test-setup`

Download test models (whisper tiny.en + Silero VAD) into `vendor/whisper.cpp/models/`.

```sh
cargo xtask test-setup          # download if not present
cargo xtask test-setup --force  # re-download
```

After setup, run tests with:
```sh
cargo test --test stream_pcm_integration -- --nocapture
```

## License

MIT
