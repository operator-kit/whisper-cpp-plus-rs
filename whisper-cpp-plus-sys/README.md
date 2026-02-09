# whisper-cpp-plus-sys

> **Pinned to whisper.cpp v1.8.3** (fork: [`rmorse/whisper.cpp`](https://github.com/rmorse/whisper.cpp), branch: `stream-pcm`)

Low-level FFI bindings to [whisper.cpp](https://github.com/ggerganov/whisper.cpp) for Rust.

This is the foundation crate for [whisper-cpp-plus](https://github.com/getpinch/whisper-cpp-plus-rs). For a safe, high-level API, use `whisper-cpp-plus` instead.

## Features

| Feature | Description |
|---------|-------------|
| `quantization` | Model quantization support |
| `cuda` | NVIDIA GPU acceleration (requires CUDA toolkit) |
| `metal` | Apple Metal acceleration (macOS) |
| `openblas` | OpenBLAS acceleration (Linux) |

## Build

The build script (`build.rs`) compiles whisper.cpp from source via the `cmake` crate, or links a prebuilt library if available. CMake is invoked automatically — GPU features like CUDA "just work" with the toolkit installed.

| Variable | Description |
|----------|-------------|
| `WHISPER_PREBUILT_PATH` | Path to prebuilt static libs (skips cmake build) |
| `WHISPER_NO_AVX` | Disable AVX/AVX2 instructions |
| `CUDA_PATH` | CUDA toolkit root (checked first for `cuda` feature) |
| `CUDA_HOME` | CUDA toolkit root (fallback) |
| `CMAKE_*` | Passed through to CMake (e.g. `CMAKE_CUDA_ARCHITECTURES`) |

### Platform requirements

- **All**: CMake 3.14+
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
use whisper_cpp_plus_sys::*;
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
