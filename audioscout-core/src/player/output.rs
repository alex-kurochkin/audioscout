//! Real audio output for the player (TZ §5.2): rodio `Sink` + a metering
//! source over the loaded document's PCM buffer.
//!
//! The `OutputStream` is created once and kept alive by the owner (dropping
//! it closes the sound device). Every transport mutation (play / pause /
//! stop / seek / volume) is applied synchronously against the live sink, so
//! calls arriving "before full initialization" are safe no-ops — a missing
//! or failed output device never panics: playback degrades to silent
//! bookkeeping and the transport state machine still advances.
//!
//! In unit tests (`cfg(test)`) no real device is opened at all: the output
//! runs in a deterministic simulated mode where [`AudioOutput::advance`]
//! consumes frames on demand, which keeps the transport logic testable
//! without audio hardware.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Half-chunk size for the pull-based metering source (1024 samples per
/// block is small enough for responsive meters, large enough to amortize
/// the mutex lock on the stats accumulator).
const BLOCK_SAMPLES: usize = 1024;

/// Shared counters written by the audio thread on every consumed chunk and
/// read by the UI playhead poll (§1.2: 50 ms cadence).
#[derive(Debug, Default)]
pub struct SinkProgress {
    /// Frames consumed since the start of the current slice append.
    pub played_frames: AtomicU64,
    /// Set when the appended slice has been fully consumed; cleared on
    /// every new append so a stale flag can never stop a fresh session.
    pub finished: AtomicBool,
}

impl SinkProgress {
    fn add_samples(&self, n: u64, channels: u64) {
        if channels == 0 {
            return;
        }
        self.played_frames.fetch_add(n / channels, Ordering::Relaxed);
    }
    fn reset(&self) {
        self.played_frames.store(0, Ordering::Relaxed);
        self.finished.store(false, Ordering::Relaxed);
    }
}

/// Pull-based source that forwards an interleaved PCM slice while
/// - counting consumed frames (playhead position),
/// - feeding per-channel peak/RMS statistics into the shared
///   [`MeteringSource`] before the samples reach the `Sink` (§3.5.1:
///   "собственный rodio Source, оборачивающий SamplesBuffer, снимает
///   статистику блоков сэмплов до Sink и прокидывает сэмплы дальше").
///
/// Volume is applied by the Controller through `Sink::set_volume` — a live
/// change without restarting the source (§3.5) — so this path forwards the
/// samples untouched.
struct MeteredSlice {
    data: Arc<Vec<f32>>,
    pos: usize,
    channels: usize,
    sample_rate: u32,
    progress: Arc<SinkProgress>,
    stats: Arc<Mutex<crate::player::MeteringSource>>,
    pending: VecDeque<f32>,
}

/// Consume up to `frames` more frames from the slice: count them, run the
/// metering block statistics and refill the pending-sample queue. Shared by
/// the real pull-based iterator and the deterministic test simulation.
fn consume_block(
    data: &[f32],
    pos: &mut usize,
    channels: usize,
    progress: &SinkProgress,
    stats: &Mutex<crate::player::MeteringSource>,
    pending: &mut VecDeque<f32>,
    frames: usize,
) {
    let want_samples = frames.saturating_mul(channels).min(BLOCK_SAMPLES);
    if want_samples == 0 {
        return;
    }
    let take = (data.len() - *pos).min(want_samples);
    if take == 0 {
        return;
    }
    let block = &data[*pos..*pos + take];
    *pos += take;
    progress.add_samples(block.len() as u64, channels as u64);
    if let Ok(mut m) = stats.lock() {
        m.process_samples(block);
    }
    pending.extend(block.iter().copied());
}

impl Iterator for MeteredSlice {
    type Item = f32;
    fn next(&mut self) -> Option<f32> {
        loop {
            if let Some(v) = self.pending.pop_front() {
                return Some(v);
            }
            if self.pos >= self.data.len() {
                self.progress.finished.store(true, Ordering::Relaxed);
                return None;
            }
            consume_block(
                &self.data,
                &mut self.pos,
                self.channels,
                &self.progress,
                &self.stats,
                &mut self.pending,
                BLOCK_SAMPLES / self.channels.max(1),
            );
        }
    }
}

impl rodio::Source for MeteredSlice {
    fn current_frame_len(&self) -> Option<usize> {
        Some(self.channels)
    }
    fn channels(&self) -> u16 {
        self.channels as u16
    }
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    fn total_duration(&self) -> Option<std::time::Duration> {
        None
    }
}

/// Owns the rodio stream/sink and exposes slice-append playback. Looping is
/// handled at the Controller level: on natural end-of-slice the controller
/// re-appends the [start,end) range while armed (live retarget without
/// touching the OutputStream, §5.2).
pub struct AudioOutput {
    // Kept alive for the lifetime of the object: dropping OutputStream
    // closes the audio device (TZ §5.2).
    _stream: Option<rodio::OutputStream>,
    handle: Option<rodio::OutputStreamHandle>,
    sink: Option<Arc<rodio::Sink>>,
    progress: Arc<SinkProgress>,
    stats: Arc<Mutex<crate::player::MeteringSource>>,
    session_start_frame: Arc<AtomicU64>,
    device_failed: bool,
    #[cfg(test)]
    sim: Option<SimPlayback>,
}

/// Deterministic playback used in unit tests instead of a real device.
#[cfg(test)]
struct SimPlayback {
    data: Vec<f32>,
    pos: usize,
    channels: usize,
    paused: bool,
}

impl Default for AudioOutput {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioOutput {
    /// Open the default output device once. Failure is non-fatal (see the
    /// module docs): all later calls degrade to silent bookkeeping.
    pub fn new() -> Self {
        let progress = Arc::new(SinkProgress::default());
        let stats = Arc::new(Mutex::new(crate::player::MeteringSource::new(2)));
        let session_start_frame = Arc::new(AtomicU64::new(0));
        #[cfg(test)]
        {
            return Self {
                _stream: None,
                handle: None,
                sink: None,
                progress,
                stats,
                session_start_frame,
                device_failed: false,
                sim: None,
            };
        }
        #[cfg(not(test))]
        match rodio::OutputStream::try_default() {
            Ok((stream, handle)) => Self {
                _stream: Some(stream),
                handle: Some(handle),
                sink: None,
                progress,
                stats,
                session_start_frame,
                device_failed: false,
            },
            Err(e) => {
                tracing::warn!("Audio output unavailable (silent fallback): {}", e);
                Self {
                    _stream: None,
                    handle: None,
                    sink: None,
                    progress,
                    stats,
                    session_start_frame,
                    device_failed: true,
                }
            }
        }
    }

    pub fn is_device_available(&self) -> bool {
        !self.device_failed
    }

    /// Absolute playhead position in frames: session start plus the frames
    /// consumed by the audio thread since the last append.
    pub fn position_frames(&self) -> u64 {
        let base = self.session_start_frame.load(Ordering::Relaxed);
        base.saturating_add(self.progress.played_frames.load(Ordering::Relaxed))
    }

    /// Play the interleaved PCM slice `[start_frame, end_frame)` of `data`.
    /// Whatever currently plays is dropped *before* the new slice is
    /// appended ("дропнуть текущий источник ДО подачи нового", §5.3).
    pub fn play_slice(&mut self, data: &[f32], channels: u16, sample_rate: u32, start_frame: u64, end_frame: u64) {
        self.stop_sink();
        let ch = channels.max(1) as usize;
        self.session_start_frame.store(start_frame, Ordering::Relaxed);
        self.progress.reset();
        *self.stats.lock().unwrap() = crate::player::MeteringSource::new(channels.max(1));

        let frames_total = data.len() / ch;
        let s = (start_frame as usize).min(frames_total);
        let e = (end_frame as usize).min(frames_total);
        if e <= s {
            // Empty slice: mark finished immediately so the controller can
            // run its loop/end logic on the next tick.
            self.progress.finished.store(true, Ordering::Relaxed);
            return;
        }
        let owned: Vec<f32> = data[s * ch..e * ch].to_vec();

        #[cfg(test)]
        {
            self.sim = Some(SimPlayback { data: owned, pos: 0, channels: ch, paused: false });
            return;
        }
        #[cfg(not(test))]
        if let Some(handle) = &self.handle {
            match rodio::Sink::connect_new(handle) {
                Ok(sink) => {
                    let src = MeteredSlice {
                        data: Arc::new(owned),
                        pos: 0,
                        channels: ch,
                        sample_rate: sample_rate.max(1),
                        progress: self.progress.clone(),
                        stats: self.stats.clone(),
                        pending: VecDeque::new(),
                    };
                    sink.append(src);
                    self.sink = Some(Arc::new(sink));
                }
                Err(err) => {
                    tracing::warn!("Failed to connect sink: {}", err);
                    self.progress.finished.store(true, Ordering::Relaxed);
                }
            }
        } else {
            // No device available: simulate completion so the transport
            // state machine stays deterministic in headless environments.
            self.progress.finished.store(true, Ordering::Relaxed);
        }
    }

    /// Freeze playback keeping the rendered position (§5.3 pause).
    pub fn pause(&mut self) {
        if let Some(sink) = &self.sink {
            sink.pause();
        }
        #[cfg(test)]
        if let Some(sim) = &mut self.sim {
            sim.paused = true;
        }
    }

    /// Resume after pause without re-appending anything.
    pub fn resume(&mut self) {
        if let Some(sink) = &self.sink {
            sink.play();
        }
        #[cfg(test)]
        if let Some(sim) = &mut self.sim {
            sim.paused = false;
        }
    }

    /// Immediate silence; rewind semantics live in the Controller (§5.3).
    pub fn stop(&mut self) {
        self.stop_sink();
    }

    fn stop_sink(&mut self) {
        if let Some(sink) = self.sink.take() {
            sink.clear();
            sink.stop();
        }
        #[cfg(test)]
        {
            self.sim = None;
        }
    }

    /// Live volume change without restarting the source (§3.5): rodio
    /// `Sink::set_volume` applies immediately to in-flight audio.
    pub fn set_volume(&mut self, vol: f32) {
        let v = vol.clamp(0.0, 1.0);
        if let Some(sink) = &self.sink {
            sink.set_volume(v);
        }
    }

    /// Snapshot-and-clear of the metering block (§3.5.1: the UI consumes
    /// each 50 ms tick, then the accumulator restarts).
    pub fn take_meter_data(&self) -> crate::player::MeterData {
        let mut m = self.stats.lock().unwrap();
        let d = crate::player::MeterData {
            peak_l: m.peak_l,
            peak_r: m.peak_r,
            rms_l: if m.sample_count > 0 { (m.rms_sum_l / m.sample_count as f64).sqrt() as f32 } else { 0.0 },
            rms_r: if m.sample_count > 0 { (m.rms_sum_r / m.sample_count as f64).sqrt() as f32 } else { 0.0 },
        };
        m.reset();
        d
    }

    /// Drain event: true exactly once per natural end-of-slice.
    pub fn take_finished(&self) -> bool {
        self.progress.finished.swap(false, Ordering::Relaxed)
    }

    /// Test-only: deterministically consume `frames` of playback (the role
    /// the audio thread plays with a real device). Returns the number of
    /// frames actually consumed; sets `finished` when the slice drains.
    #[cfg(test)]
    pub fn advance(&mut self, frames: usize) -> usize {
        let Some(sim) = &mut self.sim else { return 0 };
        if sim.paused || sim.pos >= sim.data.len() {
            if sim.pos >= sim.data.len() {
                self.progress.finished.store(true, Ordering::Relaxed);
            }
            return 0;
        }
        let mut pending = VecDeque::new();
        let before = sim.pos;
        consume_block(&sim.data, &mut sim.pos, sim.channels, &self.progress, &self.stats, &mut pending, frames);
        let consumed = (sim.pos - before) / sim.channels.max(1);
        if sim.pos >= sim.data.len() {
            self.progress.finished.store(true, Ordering::Relaxed);
        }
        consumed
    }
}
