# Changelog

All notable changes to whisper-cpp-plus will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

#### Performance Optimization: WhisperStream State Reuse

- **Fixed major performance issue** where `WhisperStream::reset()` was creating a new `WhisperState` (500MB+ allocation for medium/large models) on every streaming session
- **Optimized `WhisperStream::reset()`** to reuse the existing `WhisperState` instead of recreating it
  - The whisper.cpp library automatically clears internal results when starting a new transcription
  - This matches the behavior of whisper.cpp's own streaming implementation
- **Added `WhisperStream::recreate_state()`** method for explicit state recreation when needed (e.g., after errors or when switching between very different audio sources)
- **Performance impact**: Eliminates repeated large memory allocations, significantly improving streaming performance especially for larger models

### Documentation

- Added comprehensive documentation explaining the state reuse optimization in `stream.rs`
- Added unit tests to verify state reuse behavior

### Technical Details

The issue was identified by a user who noticed that each call to `start_streaming()` would trigger a new `whisper_init_state` allocation, even though the context was being reused. The root cause was in `WhisperStream::reset()` which was calling `self.state = self.context.create_state()?` every time.

The fix aligns our implementation with whisper.cpp's approach where `whisper_full_with_state()` clears the results (`result_all.clear()`) but keeps the allocated state memory intact. This is both safe and performant.

**Migration**: No changes required for existing code. The optimization is transparent to users.

## [0.1.0] - Previous Release

Initial release of whisper-cpp-plus with full Rust bindings to whisper.cpp.