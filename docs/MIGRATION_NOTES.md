# Migration Notes

## WhisperStream Performance Optimization (Sept 2024)

### What Changed

The `WhisperStream::reset()` method now reuses the internal `WhisperState` instead of recreating it. This is a **transparent optimization** that requires no code changes.

### Before (Old Behavior)
```rust
// reset() would internally do:
self.state = self.context.create_state()?;  // 500MB+ allocation!
```

### After (New Behavior)
```rust
// reset() now only clears buffers, keeping state alive
self.buffer.clear();
// State is reused - no new allocation
```

### Benefits
- **Memory**: Saves 500MB+ allocations per reset for medium/large models
- **Performance**: Eliminates allocation overhead between streaming sessions
- **Compatibility**: Fully backward compatible - no API changes

### When You Might Need `recreate_state()`

In rare cases where you need to force a complete state recreation:

```rust
// After an error that might have corrupted state
stream.recreate_state()?;

// When switching between very different audio sources
stream.recreate_state()?;

// Normal reset for typical streaming sessions
stream.reset()?;  // Efficient - reuses state
```

### Technical Details

This optimization aligns with whisper.cpp's internal behavior where `whisper_full_with_state()` clears results but preserves the allocated state structure. The state's internal results are automatically cleared at the start of each transcription, making reuse both safe and efficient.