//! Audio output device management (tz §5.2).
//!
//! `rodio::OutputStream` is created once at startup and kept alive for the
//! whole session — dropping it closes the audio stream (§5.2: "drop = закрытие
//! звука"). Playback itself runs through a dedicated controller thread that
//! owns the `rodio::Sink` + `SamplesBuffer<f32>` slice, so transport commands
//! arriving from the UI frame loop are serialized in one place ("Seek во время
//! игры: дропнуть текущий источник ДО подачи нового + append нового среза" —
//! §5.3). The playhead position and the volume-meter statistics are exported
//! through atomics / a mutex and polled by the UI every 50 ms (§1.2, §5.3).

use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, Mutex,
};

use crossbeam_channel::{bounded, Receiver, Sender};

use super::controller::Controller;
use super::{CursorSource, MeterData, TransportState};
use crate::{Document, UiEvent};

/// Live, lock-free snapshot of the output side shared with the UI. Every
/// field is written only by the controller thread and read by the UI poll.
#[derive(Debug)]
pub struct SharedOutput {
    /// Absolute sample index (frame * channels + channel) consumed by the
    /// Sink since the buffer was (re)created. Playhead polling divides this
    /// by the channel count to get the current frame (§5.3: пул ±50 мс).
    pub consumed_samples: AtomicU64,
    /// Frame index where the current playback session started — used for the
    /// rewind-to-session-start semantics of Stop (§5.3).
    pub session_start_frame: AtomicU64,
    /// Set by the controller while a source is attached to the sink. The UI
    /// treats "playing && !source_active" as natural end-of-session.
    pub source_active: AtomicBool,
    /// Volume-meter snapshot produced by `MeteringSource` before samples
    /// reach the Sink (§3.5.1). Read (and reset) once per UI poll tick.
    pub meter: Mutex<MeterData>,
    /// Frame offset where the currently appended slice started — added to
    /// the Sink's relative get_pos() clock to obtain an absolute frame.
    pub pos_offset: AtomicU64,
}

impl Default for SharedOutput {
    fn default() -> Self {
        Self {
            consumed_samples: AtomicU64::new(0),
            session_start_frame: AtomicU64::new(0),
            source_active: AtomicBool::new(false),
            meter: Mutex::new(MeterData::default()),
            pos_offset: AtomicU64::new(0),
        }
    }
}

impl SharedOutput {
    /// Current playhead in frames. `Sink::get_pos()` counts individual
    /// channel samples, so it must be divided by the channel count.
    pub fn current_frame(&self, channels: u16) -> u64 {
        let ch = channels.max(1) as u64;
        self.consumed_samples.load(Ordering::Relaxed) / ch
    }
}

/// Commands accepted by the output controller thread. All transport actions
/// are plain data so the UI thread never touches rodio objects directly.
enum Cmd {
    /// Load a new document into the player: stops any output, resets the
    /// resume point and remembers the fresh PCM buffer (§5.1).
    Load { doc: Arc<Document> },
    /// Start playing [start,end); `end <= start` means "up to EOF" (§5.3).
    Play { start: u64, end: u64 },
    /// Freeze position and in-flight PCM (§5.3) — implemented as drop of the
    /// current source; the resume frame is kept by the controller.
    Pause,
    /// Stop + rewind the playhead to the session start point (§5.3).
    Stop,
    /// Re-append the slice starting at `frame`; clamped into the active
    /// selection/loop range when one exists (§5.3 click-to-seek).
    Seek { frame: u64, source: CursorSource },
    /// Toggle the loop flag; live-retarget without re-creating the Sink
    /// unless the playhead has to jump back into the loop window (§3.5).
    SetLoop { enabled: bool },
    /// Live volume change via `Sink::set_volume` — no restart needed (§5.2).
    SetVolume(f32),
    /// Quit the controller thread (app shutdown).
    Shutdown,
}

/// Owns the `rodio::OutputStream` AND its `OutputStreamHandle` for the entire
/// application lifetime (§5.2). Dropping this handle closes the audio device.
///
/// BUGFIX (play appended but silent, playhead frozen at 0): in cpal 0.15 the
/// stream only keeps running while a live clone of its `StreamHandle` exists —
/// every handle drop triggers `StreamId::destroy()`. We used to keep only the
/// `OutputStream` and let the handle returned by `try_default()` die right
/// after `Sink::try_new`, so the ALSA stream was destroyed immediately: the
/// Sink happily "played" into a dead endpoint (append OK, zero samples
/// consumed, `get_pos()` stuck). Keeping one handle alive next to the stream
/// prevents the destruction.
pub struct OutputStreamHolder {
    _stream: rodio::OutputStream,
    _handle: rodio::OutputStreamHandle,
}

/// Output-side player: spawns the controller thread on creation and talks to
/// it through a bounded command channel. Safe to call before anything is
/// loaded — commands without a document are ignored (§1.2: "нажатия раньше
/// времени не падают").
pub struct RodioOutput {
    tx: Sender<Cmd>,
    pub shared: Arc<SharedOutput>,
    controller: Arc<Mutex<Controller>>,
    _holder: Arc<OutputStreamHolder>,
}

impl RodioOutput {
    /// Creates the single application-wide `OutputStream`. If the audio
    /// device cannot be opened, returns `None`: the app stays functional for
    /// browsing/editing, transport state simply never advances (graceful
    /// degradation, §0 runtime-dependency note about libasound2).
    pub fn new(controller: Arc<Mutex<Controller>>, event_tx: Sender<UiEvent>) -> Option<Self> {
        // DEBUG (playback diagnostics): every early return below is a hard
        // reason why playback can never start — log which stage failed.
        //
        // BUGFIX (silent playback on Debian/ALSA): cpal enumerates every ALSA
        // card and `default_output_device()` picks the first *usable* one —
        // typically the HDMI output of a graphics card ("built-in Audio
        // Digital Stereo (HDMI)"), which is wired to no speakers. The stream
        // then runs perfectly (samples consumed, playhead advancing) but the
        // user hears nothing. Prefer the ALSA "default" device explicitly:
        // its cpal DeviceName is exactly "default", and it routes through
        // /usr/share/alsa alsa-lib → dmix/default.pcm — i.e. the card the
        // desktop sound settings point at. Fall back to the system default
        // when no such device exists (non-ALSA backends).
        //
        // `rodio::OutputStream` is `#[deprecated]` in 0.19 (module-level), so
        // this fn opts out of the deprecation warning while keeping the
        // existing, already-working stream lifecycle code intact.
        #[allow(deprecated)]
        let mut chosen: Option<(rodio::OutputStream, rodio::OutputStreamHandle)> = None;
        // DEBUG: dump every output device rodio enumerates
        // (`rodio::supported_output_devices()`), and mark which one the
        // ALSA-'default' preference picks.
        for (device, name) in list_output_devices() {
            tracing::info!("[output] device enumerated: \"{name}\"");
            let _ = &device;
        }
        if let Some((device, name)) = find_alsa_default_device() {
            match rodio::OutputStream::try_from_device(&device) {
                Ok((stream, handle)) => {
                    tracing::info!("[output] opening ALSA 'default' device \"{name}\"");
                    chosen = Some((stream, handle));
                }
                Err(e) => tracing::warn!("[output] try_from_device(\"{name}\") FAILED: {e} — falling back to system default device"),
            }
        } else {
            tracing::debug!("[output] no ALSA 'default' device among host outputs — using system default device");
        }
        let (stream, handle) = match chosen {
            Some(pair) => pair,
            None => match rodio::OutputStream::try_default() {
                Ok(v) => v,
                Err(e) => {
                    tracing::error!("[output] OutputStream::try_default FAILED: {e} — audio device unavailable, playback will never start");
                    return None;
                }
            },
        };
        // rodio 0.19 API: `Sink::connect_new` does not exist — construct via
        // `Sink::try_new(&OutputStreamHandle)` (the error variant can only
        // happen if the stream died between calls; fall back to silence then).
        // NOTE: 0.19 has no `set_speed_of_playback` — playback rate stays 1.0.
        let sink = match rodio::Sink::try_new(&handle) {
            Ok(s) => s,
            Err(e) => {
                tracing::error!("[output] Sink::try_new FAILED: {e} — playback will never start");
                return None;
            }
        };
        tracing::info!("[output] OutputStream + Sink created OK");

        let shared = Arc::new(SharedOutput::default());
        let (tx, rx) = bounded::<Cmd>(16);
        // Keep a clone of the stream handle alive for the whole session —
        // see the BUGFIX note on `OutputStreamHolder` (dropping every handle
        // destroys the cpal stream and mutes playback completely).
        let holder = Arc::new(OutputStreamHolder {
            _stream: stream,
            _handle: handle.clone(),
        });

        let join = std::thread::Builder::new()
            .name("player-controller".into())
            .spawn({
                let shared = shared.clone();
                let controller = controller.clone();
                move || controller_thread(sink, rx, controller, shared, event_tx)
            });
        match join {
            Ok(_) => tracing::info!("[output] controller thread spawned OK"),
            Err(e) => {
                tracing::error!("[output] controller thread spawn FAILED: {e} — playback will never start");
                return None;
            }
        }

        Some(Self { tx, shared, controller, _holder: holder })
    }

    pub fn load_document(&self, doc: Arc<Document>) {
        tracing::debug!("[output] send Cmd::Load frames={} sr={} ch={}", doc.total_frames(), doc.sample_rate, doc.channels);
        if let Err(e) = self.tx.send(Cmd::Load { doc }) {
            tracing::error!("[output] Cmd::Load send FAILED: {e} — controller thread is gone");
        }
    }

    pub fn play(&self, start: u64, end: u64) {
        tracing::debug!("[output] send Cmd::Play start={start} end={end}");
        if let Err(e) = self.tx.send(Cmd::Play { start, end }) {
            tracing::error!("[output] Cmd::Play send FAILED: {e} — controller thread is gone, playback cannot start");
        }
    }

    pub fn pause(&self) {
        let _ = self.tx.send(Cmd::Pause);
    }

    pub fn stop(&self) {
        let _ = self.tx.send(Cmd::Stop);
    }

    pub fn seek(&self, frame: u64, source: CursorSource) {
        let _ = self.tx.send(Cmd::Seek { frame, source });
    }

    pub fn set_loop(&self, enabled: bool) {
        let _ = self.tx.send(Cmd::SetLoop { enabled });
    }

    pub fn set_volume(&self, vol: f32) {
        let _ = self.tx.send(Cmd::SetVolume(vol));
    }

    /// Snapshot of the shared UI-facing transport state machine (§5.3). The
    /// output thread mirrors its progress into this `Controller`; the UI reads
    /// it here or through the same `Arc` it passed to `RodioOutput::new`.
    pub fn controller_state(&self) -> TransportState {
        self.controller.lock().unwrap_or_else(|p| p.into_inner()).state()
    }
}

impl Drop for RodioOutput {
    fn drop(&mut self) {
        let _ = self.tx.send(Cmd::Shutdown);
    }
}

/// Interleaved PCM slice [start_frame, end_frame) × channels, taken from the
/// document buffer (§5.2: "SamplesBuffer<f32> ... срез [start,end) × каналов").
/// rodio 0.19: `SamplesBuffer<S>` requires `S: Sample`, i.e. the element type
/// itself (`f32`) — NOT an iterator. The data argument is `D: Into<Vec<S>>`,
/// so we move the plain `Vec<f32>` in directly.
fn slice_to_buffer(doc: &Document, start_frame: u64, end_frame: u64) -> rodio::buffer::SamplesBuffer<f32> {
    let ch = doc.channels.max(1) as usize;
    let total = doc.total_frames();
    let s = start_frame.min(total) as usize;
    let e = end_frame.clamp(s as u64 + 1, total) as usize;
    let data: Vec<f32> = doc.samples[s * ch..e * ch].to_vec();
    rodio::buffer::SamplesBuffer::new(ch as u16, doc.sample_rate, data)
}

/// Wraps a `SamplesBuffer` and taps every sample for the volume-meter
/// *before* forwarding them to the Sink (§3.5.1: "собственный rodio Source,
/// оборачивающий SamplesBuffer ... снимает статистику блоков до Sink и
/// прокидывает сэмплы дальше"). Also publishes the consumed-sample counter
/// used for playhead polling.
struct MeteredSource {
    inner: rodio::buffer::SamplesBuffer<f32>,
    channels: u16,
    shared: Arc<SharedOutput>,
    // Per-channel RMS accumulators (f64 over the current meter window).
    rms_sum_l: f64,
    rms_sum_r: f64,
    rms_count: usize,
}

impl Iterator for MeteredSource {
    type Item = f32;
    #[inline]
    fn next(&mut self) -> Option<f32> {
        // `SamplesBuffer<f32>: Iterator<Item = f32>` (rodio 0.19), so the
        // element type is inferred concretely here — no annotation needed.
        match self.inner.next() {
            Some(v) => {
                // Absolute peak per channel + running RMS sums, updated on
                // every sample so the UI poll sees a fresh window (§3.5.1).
                let idx = self.shared.consumed_samples.fetch_add(1, Ordering::Relaxed);
                let av = v.abs();
                let is_left = self.channels < 2 || (idx % self.channels as u64) == 0;
                {
                    let mut meter = self.shared.meter.lock().unwrap_or_else(|p| p.into_inner());
                    if is_left {
                        if av > meter.peak_l { meter.peak_l = av; }
                        self.rms_sum_l += (v as f64).powi(2);
                    } else {
                        if av > meter.peak_r { meter.peak_r = av; }
                        self.rms_sum_r += (v as f64).powi(2);
                    }
                }
                self.rms_count += 1;
                if self.rms_count >= 512 {
                    // Publish the RMS block stats and reset the window.
                    let mut meter = self.shared.meter.lock().unwrap_or_else(|p| p.into_inner());
                    let n = self.rms_count.max(1) as f64;
                    meter.rms_l = (self.rms_sum_l / n).sqrt() as f32;
                    meter.rms_r = (self.rms_sum_r / n).sqrt() as f32;
                    self.rms_sum_l = 0.0;
                    self.rms_sum_r = 0.0;
                    self.rms_count = 0;
                }
                Some(v)
            }
            None => None,
        }
    }
}

// rodio 0.19 `Source` inherits `Iterator`; the metadata getters resolve on
// `SamplesBuffer<f32>` once `S: Sample` holds (which f32 does). All four
// overrides simply forward to the wrapped buffer.
impl rodio::Source for MeteredSource {
    fn current_frame_len(&self) -> Option<usize> { self.inner.current_frame_len() }
    fn channels(&self) -> u16 { self.inner.channels() }
    fn sample_rate(&self) -> u32 { self.inner.sample_rate() }
    fn total_duration(&self) -> Option<std::time::Duration> { self.inner.total_duration() }
}

/// The single output-controller thread: owns the Sink and serializes all
/// transport mutations (§5.3: "stop→append последовательно в одном потоке
/// контроллера").
fn controller_thread(
    sink: rodio::Sink,
    rx: Receiver<Cmd>,
    controller: Arc<Mutex<Controller>>,
    shared: Arc<SharedOutput>,
    event_tx: Sender<UiEvent>,
) {
    // Document currently loaded into the player. Kept as Arc so slices can be
    // copied cheaply when a new source is appended.
    let mut doc: Option<Arc<Document>> = None;
    // Active session bounds and flags (mirrored into the Controller for the
    // UI-facing state machine).
    let mut sel_start: u64 = 0;
    let mut sel_end: u64 = 0;
    let mut loop_enabled = false;
    let mut last_pos_secs: f64 = -1.0;
    // BUGFIX (play never starts): a Play that arrives before any document is
    // loaded is parked here and applied by the next Load (§5.1/§5.3).
    let mut pending_play: Option<(u64, u64)> = None;
    // DEBUG (playback diagnostics): cadence limiter for the "stuck playhead"
    // warning below so it fires at most once per second while Playing.
    let mut last_stuck_warn = std::time::Instant::now();
    let mut last_seen_pos_frame: u64 = u64::MAX;
    let mut stuck_since = std::time::Instant::now();

    // Append the [start,end) slice of the current document to the sink and
    // mirror the new session into the shared state (§5.3). Shared by the
    // Play arm and by the deferred play applied on Load.
    macro_rules! apply_play {
        ($start:expr, $end:expr) => {{
            if let Some(d) = &doc {
                let total = d.total_frames();
                // end<=start ⇒ play to EOF (§5.3).
                let (s, e) = if $end > $start { ($start.min(total), $end.min(total)) } else { ($start.min(total), total) };
                if e > s {
                    sink.stop();
                    // BUGFIX (playhead frozen at 0, UI looks like playback
                    // never started): the consumed-sample counter must be
                    // seeded from the slice start in RAW SAMPLE units. The old
                    // code stored `s * channels` while MeteredSource counts
                    // one increment per sample — for stereo this made every
                    // position readout half of the true frame index, and any
                    // later unit mismatch pinned the cursor at 0.
                    shared.consumed_samples.store(s * d.channels.max(1) as u64, Ordering::Relaxed);
                    shared.pos_offset.store(s, Ordering::Relaxed);
                    shared.session_start_frame.store(s, Ordering::Relaxed);
                    sel_start = s;
                    sel_end = e;
                    let src = MeteredSource {
                        inner: slice_to_buffer(d, s, e),
                        channels: d.channels.max(1),
                        shared: shared.clone(),
                        rms_sum_l: 0.0,
                        rms_sum_r: 0.0,
                        rms_count: 0,
                    };
                    sink.append(src);
                    shared.source_active.store(true, Ordering::Relaxed);
                    let mut ctrl = controller.lock().unwrap_or_else(|p| p.into_inner());
                    ctrl.mark_playing(s, e);
                    sink.set_volume(ctrl.volume());
                    loop_enabled = ctrl.is_looping();
                    drop(ctrl);
                    tracing::info!("[output] PLAY appended slice [{s},{e}) of {total} frames sr={} ch={} vol — source active", d.sample_rate, d.channels.max(1));
                } else {
                    // DEBUG: empty playable range — this is exactly the case
                    // where the button toggles but nothing ever plays.
                    tracing::error!("[output] PLAY DROPPED: empty range start={} end={} total={}", $start, $end, total);
                }
            } else {
                tracing::error!("[output] PLAY with no document loaded — cannot append");
            }
        }};
    }

    'outer: loop {
        // Block until something happens: a command, or the current source
        // finishing (poll cadence ~50 ms, §5.3).
        let cmd_opt = if sink.empty() {
            rx.recv().ok()
        } else {
            rx.recv_timeout(std::time::Duration::from_millis(50)).ok()
        };

        match cmd_opt {
            Some(Cmd::Load { doc: new_doc }) => {
                tracing::debug!("[output] recv Cmd::Load frames={} sr={} ch={}", new_doc.total_frames(), new_doc.sample_rate, new_doc.channels);
                sink.stop(); // drop the current source BEFORE loading (§5.3)
                // A Play deferred while no document existed is applied to the
                // freshly loaded buffer now (BUGFIX: play never started).
                let deferred = pending_play.take();
                if deferred.is_some() {
                    tracing::info!("[output] applying DEFERRED play {:?} on Load", deferred);
                }
                doc = Some(new_doc);
                sel_start = 0;
                sel_end = 0;
                shared.session_start_frame.store(0, Ordering::Relaxed);
                shared.source_active.store(false, Ordering::Relaxed);
                let mut ctrl = controller.lock().unwrap_or_else(|p| p.into_inner());
                ctrl.reset();
                sink.set_volume(ctrl.volume());
                loop_enabled = ctrl.is_looping();
                drop(ctrl);
                if let Some((ps, pe)) = deferred {
                    apply_play!(ps, pe);
                }
            }
            Some(Cmd::Play { start, end }) => {
                // BUGFIX (play never starts): a Play sent before any document
                // was loaded used to be silently dropped here — the UI had
                // already switched its state machine to Playing, so the file
                // appeared frozen with no audio. Keep the command pending and
                // apply it as soon as Load delivers the PCM buffer (§5.1:
                // "stop предыдущего вывода + сброс resume point" happen in
                // the Load arm below).
                if doc.is_none() {
                    tracing::warn!("[output] recv Cmd::Play start={start} end={end} BEFORE any Load — parked as pending_play");
                    pending_play = Some((start, end));
                } else {
                    tracing::debug!("[output] recv Cmd::Play start={start} end={end}");
                    apply_play!(start, end);
                }
            }
            Some(Cmd::Pause) => {
                pending_play = None; // a parked Play must not survive Pause
                sink.stop();
                shared.source_active.store(false, Ordering::Relaxed);
                let mut ctrl = controller.lock().unwrap_or_else(|p| p.into_inner());
                // Transition the state machine FIRST (it snapshots the frozen
                // resume point), then overwrite with the measured audio-clock
                // position (§5.3 pause freezes position and in-flight PCM).
                ctrl.pause();
                ctrl.freeze_resume_at(shared.current_frame(channels_of(&doc)));
            }
            Some(Cmd::Stop) => {
                pending_play = None; // ... or Stop (§5.3 stop = silence)
                sink.stop();
                shared.source_active.store(false, Ordering::Relaxed);
                let start = shared.session_start_frame.load(Ordering::Relaxed);
                let mut ctrl = controller.lock().unwrap_or_else(|p| p.into_inner());
                let ch = channels_of(&doc);
                shared.consumed_samples.store(start * ch as u64, Ordering::Relaxed);
                ctrl.stop_at_session_start(start);
            }
            Some(Cmd::Seek { frame, source: _source }) => {
                if let Some(d) = &doc {
                    let ch = d.channels.max(1);
                    // Click-to-seek outside a selection never extends it; the
                    // session keeps running up to its original boundary (§5.3).
                    let target = if sel_end > sel_start { frame.clamp(sel_start, sel_end.saturating_sub(1)) } else { frame };
                    let target = target.min(d.total_frames());
                    sink.stop(); // drop the current source BEFORE appending the new slice
                    shared.consumed_samples.store(target * ch as u64, Ordering::Relaxed);
                    shared.pos_offset.store(target, Ordering::Relaxed);
                    {
                        let mut ctrl = controller.lock().unwrap_or_else(|p| p.into_inner());
                        ctrl.mark_seek(target);
                        if ctrl.is_playing() {
                            let (s, e) = (target, if sel_end > sel_start { sel_end } else { d.total_frames() });
                            let src = MeteredSource {
                                inner: slice_to_buffer(d, s, e),
                                channels: ch,
                                shared: shared.clone(),
                                rms_sum_l: 0.0,
                                rms_sum_r: 0.0,
                                rms_count: 0,
                            };
                            sink.append(src);
                            shared.source_active.store(true, Ordering::Relaxed);
                            sel_start = s;
                            sel_end = e;
                        } else {
                            shared.source_active.store(false, Ordering::Relaxed);
                        }
                    }
                }
            }
            Some(Cmd::SetLoop { enabled }) => {
                loop_enabled = enabled;
                let mut ctrl = controller.lock().unwrap_or_else(|p| p.into_inner());
                ctrl.set_loop(enabled);
                // Live retarget: if we are mid-session and looping, snap the
                // playhead back to the loop start without glitch-restart of
                // the Sink itself (§3.5).
                if enabled && ctrl.state() == TransportState::Playing && shared.source_active.load(Ordering::Relaxed) {
                    if let Some(d) = &doc {
                        let ch = d.channels.max(1);
                        sink.stop();
                        shared.consumed_samples.store(sel_start * ch as u64, Ordering::Relaxed);
                        shared.pos_offset.store(sel_start, Ordering::Relaxed);
                        let src = MeteredSource {
                            inner: slice_to_buffer(d, sel_start, sel_end),
                            channels: ch,
                            shared: shared.clone(),
                            rms_sum_l: 0.0,
                            rms_sum_r: 0.0,
                            rms_count: 0,
                        };
                        sink.append(src);
                    }
                }
            }
            Some(Cmd::SetVolume(vol)) => {
                // Live gain change inside the app — no restart (§3.5/§5.2).
                let mut ctrl = controller.lock().unwrap_or_else(|p| p.into_inner());
                ctrl.set_volume(vol);
                sink.set_volume(ctrl.volume());
            }
            Some(Cmd::Shutdown) | None => break 'outer,
        }

        // ---- poll section: runs after every command or timeout tick ----
        let (pos_frame, finished) = {
            let ctrl = controller.lock().unwrap_or_else(|p| p.into_inner());
            // rodio 0.19: Sink::get_pos() returns the elapsed playback
            // Duration of the current source (§5.3 playhead polling);
            // convert it to a frame index via the sample rate and add the
            // offset where the appended slice started.
            let sr = doc.as_ref().map(|d| d.sample_rate.max(1) as f64).unwrap_or(44100.0);
            let pos_secs = sink.get_pos().as_secs_f64();
            let pos = (pos_secs * sr) as u64 + shared.pos_offset.load(Ordering::Relaxed);
            let playing = ctrl.state() == TransportState::Playing;
            let active = shared.source_active.load(Ordering::Relaxed);
            (pos, playing && active && sink.empty())
        };

        // DEBUG (playback diagnostics): if the state machine says Playing and
        // a source is active but the audio clock does not move for >1s, the
        // cpal stream is almost certainly dead (e.g. OutputStreamHandle was
        // dropped — see OutputStreamHolder BUGFIX). Log the raw clock values
        // so the stall is attributable from the console alone.
        {
            let ctrl = controller.lock().unwrap_or_else(|p| p.into_inner());
            let playing = ctrl.state() == TransportState::Playing;
            drop(ctrl);
            let active = shared.source_active.load(Ordering::Relaxed);
            if playing && active {
                if pos_frame != last_seen_pos_frame {
                    last_seen_pos_frame = pos_frame;
                    stuck_since = std::time::Instant::now();
                } else if stuck_since.elapsed() > std::time::Duration::from_secs(1)
                    && last_stuck_warn.elapsed() > std::time::Duration::from_secs(1)
                {
                    last_stuck_warn = std::time::Instant::now();
                    tracing::error!(
                        "[output] playhead STUCK at frame {pos_frame} for >1s while Playing \
                         (sink.get_pos()={:.4}s empty={} consumed={} pos_offset={}) \
                         — audio stream likely destroyed/muted",
                        sink.get_pos().as_secs_f64(),
                        sink.empty(),
                        shared.consumed_samples.load(Ordering::Relaxed),
                        shared.pos_offset.load(Ordering::Relaxed),
                    );
                }
            } else {
                last_seen_pos_frame = u64::MAX;
                stuck_since = std::time::Instant::now();
            }
        }

        if let Some(d) = &doc {
            let sr = d.sample_rate.max(1) as f64;
            let secs = pos_frame as f64 / sr;
            if (secs - last_pos_secs).abs() >= 0.03 {
                last_pos_secs = secs;
                let _ = event_tx.send(UiEvent::PlayheadUpdate { position_secs: secs });
            }
        }

        // Volume-meter update: publish the tapped peaks, then reset them so
        // each UI poll tick sees a fresh window (§3.5.1: обновление с частотой
        // кадров UI из тика playhead).
        {
            let snapshot = {
                let mut m = shared.meter.lock().unwrap_or_else(|p| p.into_inner());
                let snap = *m;
                *m = MeterData::default();
                snap
            };
            let _ = event_tx.send(UiEvent::VolumeMeterUpdate {
                peak_l: snapshot.peak_l,
                peak_r: snapshot.peak_r,
                rms_l: snapshot.rms_l,
                rms_r: snapshot.rms_r,
            });
        }

        if finished {
            if loop_enabled && sel_end > sel_start {
                // Loop sessions never report natural completion — restart the
                // slice transparently (§5.3: "При Loop событие окончания
                // приходит только на ошибках").
                if let Some(d) = &doc {
                    let ch = d.channels.max(1);
                    shared.consumed_samples.store(sel_start * ch as u64, Ordering::Relaxed);
                    shared.pos_offset.store(sel_start, Ordering::Relaxed);
                    let src = MeteredSource {
                        inner: slice_to_buffer(d, sel_start, sel_end),
                        channels: ch,
                        shared: shared.clone(),
                        rms_sum_l: 0.0,
                        rms_sum_r: 0.0,
                        rms_count: 0,
                    };
                    sink.append(src);
                }
            } else {
                // Natural end of a non-loop session: rewind the playhead to
                // the session start point (§5.3).
                let start = shared.session_start_frame.load(Ordering::Relaxed);
                let ch = channels_of(&doc);
                shared.consumed_samples.store(start * ch as u64, Ordering::Relaxed);
                shared.pos_offset.store(start, Ordering::Relaxed);
                shared.source_active.store(false, Ordering::Relaxed);
                {
                    let mut ctrl = controller.lock().unwrap_or_else(|p| p.into_inner());
                    ctrl.finish_session_at(start);
                }
                let _ = event_tx.send(UiEvent::PlaybackFinished);
            }
        }
    }
}

fn channels_of(doc: &Option<Arc<Document>>) -> u16 {
    doc.as_ref().map(|d| d.channels.max(1)).unwrap_or(2)
}

/// Enumerate all output-capable devices through **cpal's public host API**
/// (re-exported verbatim by rodio as `rodio::cpal`).
///
/// Verified against the exact sources of rodio 0.19 / cpal 0.15.3:
/// * rodio's `lib.rs` contains `pub use cpal::{self, ...}` — so `rodio::cpal`
///   IS the cpal crate root, including its top-level free functions;
/// * cpal 0.15 has NO `host_ids()` function (that was cpal 0.11-era API) and
///   NO `Host::from_id` associated fn — the correct pair is the free
///   functions `cpal::available_hosts() -> Vec<HostId>` and
///   `cpal::host_from_id(HostId) -> Result<Host, HostUnavailable>`;
/// * rodio 0.19 has no `rodio::supported_output_devices()` at all (E0425),
///   so device enumeration must go through cpal directly;
/// * device listing needs the trait methods from `cpal::traits::HostTrait`
///   (`devices()`) and `cpal::traits::DeviceTrait` (`name()`,
///   `supported_output_configs()`) imported into scope — without them the
///   calls don't resolve and type inference fails (E0282).
///
/// Each device is probed with `supported_output_configs()` so only genuinely
/// output-capable devices are returned (on ALSA every card AND every extra
/// PCM like "hdmi:CARD=…" shows up; opening a capture-only or dead PCM would
/// silently kill playback). The returned `rodio::cpal::Device` is the same
/// type `rodio::OutputStream::try_from_device(&Device)` accepts.
fn list_output_devices() -> Vec<(rodio::cpal::Device, String)> {
    use rodio::cpal::traits::{DeviceTrait, HostTrait};
    let mut out = Vec::new();
    for host_id in rodio::cpal::available_hosts() {
        let host: rodio::cpal::Host = match rodio::cpal::host_from_id(host_id) {
            Ok(h) => h,
            Err(e) => {
                tracing::debug!("[output] host {host_id:?} unavailable: {e}");
                continue;
            }
        };
        let devices: rodio::cpal::Devices = match host.devices() {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!("[output] host {host_id:?} device enumeration failed: {e}");
                continue;
            }
        };
        for device in devices {
            let name: String = device.name().unwrap_or_else(|_| "<unnamed>".to_string());
            // Keep only devices that actually expose an output stream config.
            let outputs = match device.supported_output_configs() {
                Ok(cfgs) => cfgs.next().is_some(),
                Err(_) => false,
            };
            if outputs {
                out.push((device, name));
            }
        }
    }
    out
}

/// Find the ALSA "default" output device among the enumerated devices.
///
/// On Linux the ALSA backend exposes every alsa-lib PCM name as a separate
/// device, and the virtual "default" (which follows the user's desktop sound
/// settings / dmix route) is identifiable by its name. Returns the first
/// match; `None` on non-ALSA backends or enumeration failure (the caller
/// then falls back to `OutputStream::try_default`).
fn find_alsa_default_device() -> Option<(rodio::cpal::Device, String)> {
    for (device, name) in list_output_devices() {
        let lname = name.to_lowercase();
        // Match the ALSA virtual default ("default") and dmix-mapped variants.
        if lname == "default" || lname.contains("dmix") {
            return Some((device, name));
        }
    }
    None
}

