# whisper-sys

Low-level FFI bindings to [whisper.cpp](https://github.com/ggerganov/whisper.cpp) for Rust.

This is the foundation crate for [whisper-cpp-rs](https://github.com/Code-Amp/whisper-cpp-rs). For a safe, high-level API, use `whisper-cpp-rs` instead.

## Features

| Feature | Description |
|---------|-------------|
| `quantization` | Model quantization support |
| `cuda` | NVIDIA GPU acceleration (requires CUDA toolkit) |
| `metal` | Apple Metal acceleration (macOS) |
| `openblas` | OpenBLAS acceleration (Linux) |

## Build

The build script (`build.rs`) compiles whisper.cpp from source via the `cc` crate, or links a prebuilt library if available. Control via environment variables:

| Variable | Description |
|----------|-------------|
| `WHISPER_PREBUILT_PATH` | Path to prebuilt `whisper.lib`/`libwhisper.a` |
| `WHISPER_NO_AVX` | Disable AVX/AVX2 instructions |

### Platform requirements

- **Windows**: Visual Studio 2019+ (MSVC)
- **Linux**: GCC 9+ or Clang 11+, `build-essential`
- **macOS**: Xcode command line tools (Accelerate linked automatically)

## Usage

All functions are `unsafe`. Key rules:
- `whisper_context` must outlive all `whisper_state` instances
- Each thread needs its own `whisper_state`
- Copy strings from result pointers before freeing state
- Free states before context

```rust
use whisper_sys::*;
use std::ffi::CString;

unsafe {
    let params = whisper_context_default_params();
    let path = CString::new("model.bin").unwrap();
    let ctx = whisper_init_from_file_with_params(path.as_ptr(), params);
    assert!(!ctx.is_null());

    let state = whisper_init_state(ctx);
    // ... transcribe ...
    whisper_free_state(state);
    whisper_free(ctx);
}
```

## License

MIT
