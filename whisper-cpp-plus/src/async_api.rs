//! Async API for non-blocking transcription
//!
//! This module provides async wrappers around the synchronous whisper.cpp API,
//! enabling non-blocking transcription in async Rust applications.

#[cfg(feature = "async")]
use crate::{
    context::WhisperContext, error::Result, params::FullParams,
    state::WhisperState, stream::{StreamConfig, WhisperStream}, Segment, TranscriptionResult,
};
#[cfg(feature = "async")]
use std::sync::Arc;
#[cfg(feature = "async")]
use tokio::sync::{mpsc, oneshot, Mutex};
#[cfg(feature = "async")]
use tokio::task;

#[cfg(feature = "async")]
impl WhisperContext {
    /// Transcribe audio asynchronously using default parameters
    pub async fn transcribe_async(&self, audio: Vec<f32>) -> Result<String> {
        let ctx = self.clone();
        task::spawn_blocking(move || ctx.transcribe(&audio))
            .await
            .map_err(|e| crate::WhisperError::TranscriptionError(e.to_string()))?
    }

    /// Transcribe audio asynchronously with custom parameters
    pub async fn transcribe_with_params_async(
        &self,
        audio: Vec<f32>,
        params: crate::TranscriptionParams,
    ) -> Result<TranscriptionResult> {
        let ctx = self.clone();
        task::spawn_blocking(move || ctx.transcribe_with_params(&audio, params))
            .await
            .map_err(|e| crate::WhisperError::TranscriptionError(e.to_string()))?
    }

    /// Create an async state for manual control
    pub async fn create_state_async(&self) -> Result<WhisperState> {
        let ctx = self.clone();
        task::spawn_blocking(move || ctx.create_state())
            .await
            .map_err(|e| crate::WhisperError::TranscriptionError(e.to_string()))?
    }
}

/// An async streaming transcriber with channels for audio input
#[cfg(feature = "async")]
pub struct AsyncWhisperStream {
    /// Channel for sending audio to the processing task
    audio_tx: mpsc::Sender<AudioCommand>,
    /// Channel for receiving transcribed segments
    segment_rx: mpsc::Receiver<Vec<Segment>>,
    /// Handle to the background processing task
    handle: task::JoinHandle<Result<()>>,
}

#[cfg(feature = "async")]
enum AudioCommand {
    Feed(Vec<f32>),
    Flush(oneshot::Sender<Vec<Segment>>),
    Stop,
}

#[cfg(feature = "async")]
impl AsyncWhisperStream {
    /// Create a new async streaming transcriber
    pub fn new(
        context: WhisperContext,
        params: FullParams,
    ) -> Result<Self> {
        Self::with_config(context, params, StreamConfig::default())
    }

    /// Create a new async streaming transcriber with custom configuration
    pub fn with_config(
        context: WhisperContext,
        params: FullParams,
        config: StreamConfig,
    ) -> Result<Self> {
        let (audio_tx, mut audio_rx) = mpsc::channel::<AudioCommand>(100);
        let (segment_tx, segment_rx) = mpsc::channel::<Vec<Segment>>(100);

        let handle = task::spawn_blocking(move || {
            let mut stream = WhisperStream::with_config(&context, params, config)?;

            while let Some(cmd) = audio_rx.blocking_recv() {
                match cmd {
                    AudioCommand::Feed(audio) => {
                        stream.feed_audio(&audio);

                        // Process pending audio
                        let segments = stream.process_pending()?;
                        if !segments.is_empty() {
                            // Try to send segments, ignore if receiver dropped
                            let _ = segment_tx.blocking_send(segments);
                        }
                    }
                    AudioCommand::Flush(response) => {
                        let segments = stream.flush()?;
                        let _ = response.send(segments);
                    }
                    AudioCommand::Stop => break,
                }
            }

            Ok(())
        });

        Ok(Self {
            audio_tx,
            segment_rx,
            handle,
        })
    }

    /// Feed audio samples to the stream
    pub async fn feed_audio(&self, audio: Vec<f32>) -> Result<()> {
        self.audio_tx
            .send(AudioCommand::Feed(audio))
            .await
            .map_err(|_| crate::WhisperError::TranscriptionError("Stream closed".into()))
    }

    /// Receive transcribed segments if available
    pub async fn recv_segments(&mut self) -> Option<Vec<Segment>> {
        self.segment_rx.recv().await
    }

    /// Try to receive segments without blocking
    pub fn try_recv_segments(&mut self) -> Option<Vec<Segment>> {
        self.segment_rx.try_recv().ok()
    }

    /// Flush the stream and get all pending segments
    pub async fn flush(&self) -> Result<Vec<Segment>> {
        let (tx, rx) = oneshot::channel();
        self.audio_tx
            .send(AudioCommand::Flush(tx))
            .await
            .map_err(|_| crate::WhisperError::TranscriptionError("Stream closed".into()))?;
        rx.await
            .map_err(|_| crate::WhisperError::TranscriptionError("Failed to flush".into()))
    }

    /// Stop the stream gracefully
    pub async fn stop(self) -> Result<()> {
        let _ = self.audio_tx.send(AudioCommand::Stop).await;
        self.handle
            .await
            .map_err(|e| crate::WhisperError::TranscriptionError(e.to_string()))?
    }
}

/// A shared async stream for multiple producers
#[cfg(feature = "async")]
pub struct SharedAsyncStream {
    inner: Arc<Mutex<AsyncStreamInner>>,
}

#[cfg(feature = "async")]
struct AsyncStreamInner {
    stream: WhisperStream,
    pending_segments: Vec<Segment>,
}

#[cfg(feature = "async")]
impl SharedAsyncStream {
    /// Create a new shared async stream
    pub async fn new(
        context: &WhisperContext,
        params: FullParams,
        config: StreamConfig,
    ) -> Result<Self> {
        let stream = WhisperStream::with_config(context, params, config)?;

        Ok(Self {
            inner: Arc::new(Mutex::new(AsyncStreamInner {
                stream,
                pending_segments: Vec::new(),
            })),
        })
    }

    /// Feed audio and get segments atomically
    pub async fn feed_and_process(&self, audio: Vec<f32>) -> Result<Vec<Segment>> {
        let mut inner = self.inner.lock().await;

        // Feed audio
        inner.stream.feed_audio(&audio);

        // Process pending
        let segments = inner.stream.process_pending()?;

        // Store segments
        inner.pending_segments.extend(segments.clone());

        Ok(segments)
    }

    /// Drain all pending segments
    pub async fn drain_segments(&self) -> Vec<Segment> {
        let mut inner = self.inner.lock().await;
        std::mem::take(&mut inner.pending_segments)
    }

    /// Flush the stream
    pub async fn flush(&self) -> Result<Vec<Segment>> {
        let mut inner = self.inner.lock().await;
        let segments = inner.stream.flush()?;
        inner.pending_segments.extend(segments.clone());
        Ok(segments)
    }
}

#[cfg(all(test, feature = "async"))]
mod tests {
    use super::*;
    use crate::SamplingStrategy;
    use std::path::Path;

    #[tokio::test]
    async fn test_async_transcribe() {
        let model_path = "tests/models/ggml-tiny.en.bin";
        if Path::new(model_path).exists() {
            let ctx = WhisperContext::new(model_path).unwrap();
            let audio = vec![0.0f32; 16000]; // 1 second of silence

            let result = ctx.transcribe_async(audio).await;
            assert!(result.is_ok());
        }
    }

    #[tokio::test]
    async fn test_async_stream() {
        let model_path = "tests/models/ggml-tiny.en.bin";
        if Path::new(model_path).exists() {
            let ctx = WhisperContext::new(model_path).unwrap();
            let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

            let stream = AsyncWhisperStream::new(ctx, params);
            assert!(stream.is_ok());

            let stream = stream.unwrap();

            // Feed some audio
            let audio = vec![0.0f32; 16000];
            let result = stream.feed_audio(audio).await;
            assert!(result.is_ok());

            // Stop the stream
            let result = stream.stop().await;
            assert!(result.is_ok());
        }
    }

    #[tokio::test]
    async fn test_shared_stream() {
        let model_path = "tests/models/ggml-tiny.en.bin";
        if Path::new(model_path).exists() {
            let ctx = WhisperContext::new(model_path).unwrap();
            let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

            let stream = SharedAsyncStream::new(&ctx, params, StreamConfig::default()).await;
            assert!(stream.is_ok());

            let stream = stream.unwrap();

            // Feed audio from multiple tasks
            let stream1 = stream.clone();
            let handle1 = tokio::spawn(async move {
                let audio = vec![0.0f32; 16000];
                stream1.feed_and_process(audio).await
            });

            let stream2 = stream.clone();
            let handle2 = tokio::spawn(async move {
                let audio = vec![0.0f32; 16000];
                stream2.feed_and_process(audio).await
            });

            // Wait for both
            let result1 = handle1.await.unwrap();
            let result2 = handle2.await.unwrap();

            assert!(result1.is_ok());
            assert!(result2.is_ok());
        }
    }
}

// Implement Clone for SharedAsyncStream
#[cfg(feature = "async")]
impl Clone for SharedAsyncStream {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}