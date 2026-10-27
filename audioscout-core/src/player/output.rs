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
    /// BUGFIX (playback freezes ~0.5 s after start): the sample rate the cpal
    /// stream was actually opened with. Every source is resampled to this
    /// rate before being appended (`slice_to_buffer`), because rodio 0.19's
    /// dynamic mixer consumes samples at the *hardware* clock and never
    /// resamples `Sink::append`ed sources itself — a 22050 Hz source on a
    /// 48 kHz device drains its buffer ~2× faster than the audible window it
    /// represents and the queue goes silent right after the pre-buffered
    /// chunk (~0.5 s of sound, then freeze — exactly the reported symptom).
    pub stream_sample_rate: u32,
    /// Channel count of the opened config; sources are up/down mixed to it.
    pub stream_channels: u16,
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
    /// Previous value of the `consumed_samples` audio clock, observed by
    /// `poll_stall()` on each UI tick.
    last_consumed: u64,
    /// When the audio clock last moved (or playback was not active).
    last_clock_move: std::time::Instant,
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
        // BUGFIX (playback freezes ~0.5 s after start): the ALSA/Pulse device
        // is opened through `rodio::OutputStream::try_from_device`, which
        // always picks the FIRST supported config of the device. On Debian
        // with the "pulse" pseudo-device that config can be e.g. 192 kHz
        // while the file is 44.1/22.05 kHz — the sink then feeds a trickle of
        // underrun-glued samples and playback appears to freeze right after
        // starting. To make the real device format visible in the debug log
        // we probe it here (lazily, once, only for the chosen device).
        fn log_device_config(device: &rodio::cpal::Device) {
            // cpal 0.15: `DeviceTrait` lives in `cpal::traits`, not at the
            // crate root — importing it from the root is E0432.
            use rodio::cpal::traits::DeviceTrait;
            match device.default_output_config() {
                Ok(cfg) => tracing::info!(
                    "[output] device config: {} ch @ {} Hz",
                    cfg.channels(),
                    cfg.sample_rate().0
                ),
                Err(e) => tracing::warn!("[output] default_output_config failed: {e}"),
            }
        }
        // BUGFIX (silent playback on Debian/ALSA, part 2): the previous
        // strategy preferred the ALSA virtual device named "default". That is
        // WRONG on a typical desktop Debian: cpal's ALSA backend only lists
        // *hardware* PCMs ("hw:CARD=PCH,DEV=0", "plughw:…", "hdmi:…") — the
        // plain "default" never appears (see debug log: enumeration produced
        // lavrate/pulse/hw:* only). Worse, blindly opening hardware devices
        // probes them via `supported_output_configs()`, which makes alsa-lib
        // actually open every PCM: dmix/dsnoop plugins fail with
        // "unable to open slave", JACK attempts spawn server-connection spam —
        // that is the disk/console noise the user saw while hearing nothing.
        //
        // Correct order now:
        //   1) env override AUDIOSCOUT_ALSA_DEVICE (exact ALSA PCM name, e.g.
        //      "default", "pulse", "sysdefault:CARD=PCH");
        //   2) PulseAudio host when available — on Debian it is the sound
        //      server everything else routes through; opening it touches no
        //      raw PCMs at all;
        //   3) cpal/rodio system default device (last resort).
        // Device probing for logging uses `supported_configs` lazily and is
        // skipped entirely once a target is found.
        for (device, name) in list_output_devices_quiet() {
            tracing::debug!("[output] device enumerated: \"{name}\"");
            let _ = &device;
        }
        // BUGFIX (playback freezes ~0.5 s after start): the ALSA/Pulse device
        // was opened through `try_from_device`, which always picks the
        // device's FIRST supported config. When that config's sample format
        // is not f32 (e.g. S16 on some ALSA PCMs), rodio 0.19 feeds the cpal
        // callback raw f32 samples reinterpreted as the hardware format —
        // the stream plays garbage for the pre-buffered window (~0.5 s) and
        // then stalls, exactly matching the observed symptom. Fix: probe the
        // device's supported configs once at startup and prefer an f32
        // config (native rate first, then any f32 range containing it),
        // opening via `try_from_device_config`. If the device exposes no f32
        // format at all, fall back to plain `try_from_device` (rodio will
        // convert internally for its own decoders; our MeteredSource keeps
        // emitting f32, so the fallback path may misbehave — logged loudly).
        #[allow(deprecated)]
        let mut chosen: Option<(rodio::OutputStream, rodio::OutputStreamHandle)> = None;
        // Sample rate / channel count the stream must be opened with so the
        // controller thread can resample every appended source to match.
        let mut stream_format: (u32, u16) = (44100, 2);
        let target: Option<(rodio::cpal::Device, String)> = pick_output_device();
        if let Some((device, name)) = &target {
            log_device_config(device);
            match select_f32_output_config(device) {
                Some(cfg) => {
                    tracing::info!(
                        "[output] opening \"{name}\" with f32 config {} ch @ {} Hz",
                        cfg.channels(),
                        cfg.sample_rate().0
                    );
                    // BUGFIX (compilation): rodio 0.19's
                    // `try_from_device_config(&Device, SupportedStreamConfig)`
                    // takes a *reference* to the device and a *concrete*
                    // `SupportedStreamConfig` — not `device.clone()` and not
                    // `cfg.config()` (that returns the generic `StreamConfig`,
                    // E0308). `cfg` here is already `SupportedStreamConfig`.
                    stream_format = (cfg.sample_rate().0, cfg.channels());
                    #[allow(deprecated)]
                    match rodio::OutputStream::try_from_device_config(device, cfg) {
                        Ok((stream, handle)) => {
                            tracing::info!("[output] opened audio device \"{name}\" (explicit config)");
                            chosen = Some((stream, handle));
                        }
                        Err(e) => tracing::warn!("[output] try_from_device_config(\"{name}\") FAILED: {e} — trying try_from_device"),
                    }
                }
                None => tracing::warn!("[output] no f32 output config on \"{name}\" — falling back to device default format"),
            }
            if chosen.is_none() {
                #[allow(deprecated)]
                match rodio::OutputStream::try_from_device(device) {
                    Ok((stream, handle)) => {
                        tracing::info!("[output] opened audio device \"{name}\" (default config)");
                        chosen = Some((stream, handle));
                    }
                    Err(e) => tracing::warn!("[output] try_from_device(\"{name}\") FAILED: {e} — falling back to system default device"),
                }
            }
        } else {
            tracing::debug!("[output] no explicit device selected — using system default device");
        }
        #[allow(deprecated)]
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
            stream_sample_rate: stream_format.0,
            stream_channels: stream_format.1.max(1),
        });

        let join = std::thread::Builder::new()
            .name("player-controller".into())
            .spawn({
                let shared = shared.clone();
                let controller = controller.clone();
                move || controller_thread(sink, rx, controller, shared, event_tx, stream_format)
            });
        match join {
            Ok(_) => tracing::info!("[output] controller thread spawned OK"),
            Err(e) => {
                tracing::error!("[output] controller thread spawn FAILED: {e} — playback will never start");
                return None;
            }
        }

        Some(Self {
            tx,
            shared,
            controller,
            _holder: holder,
            last_consumed: 0,
            last_clock_move: std::time::Instant::now(),
        })
    }

    /// BUGFIX (playback silently freezes ~0.5 s after start with the cursor
    /// stuck): detect a dead audio stream and make the app recover instead of
    /// hanging in a fake "Playing" state. `MeteredSource::next()` runs inside
    /// the cpal callback and advances `consumed_samples`; if PulseAudio/ALSA
    /// drops the stream, the callback stops firing and the clock freezes —
    /// the only externally observable symptom of a dead endpoint. The UI poll
    /// calls this method every 50 ms tick; when transport says Playing, a
    /// source is attached, and the clock has not moved for >1 s, we pause the
    /// transport and report `true` once so the app can surface the error.
    pub fn poll_stall(&mut self) -> bool {
        let playing = self.controller_state() == TransportState::Playing;
        let active = self.shared.source_active.load(Ordering::Relaxed);
        let consumed = self.shared.consumed_samples.load(Ordering::Relaxed);
        if !playing || !active {
            self.last_consumed = consumed;
            self.last_clock_move = std::time::Instant::now();
            return false;
        }
        if consumed != self.last_consumed {
            self.last_consumed = consumed;
            self.last_clock_move = std::time::Instant::now();
            return false;
        }
        if self.last_clock_move.elapsed() > std::time::Duration::from_secs(1) {
            tracing::error!(
                "[output] STALL DETECTED: Playing & source active but audio clock \
                 frozen at consumed={consumed} for >1s — pausing transport \
                 (stream was likely dropped by PulseAudio/ALSA)"
            );
            // Reset first so the next call doesn't re-fire until the clock
            // moves again; then pause the Sink and the state machine.
            self.last_clock_move = std::time::Instant::now();
            let _ = self.tx.send(Cmd::Pause);
            if let Ok(mut ctrl) = self.controller.lock() {
                ctrl.pause();
            }
            return true;
        }
        false
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
fn slice_to_buffer(
    doc: &Document,
    start_frame: u64,
    end_frame: u64,
    stream_format: (u32, u16),
) -> rodio::buffer::SamplesBuffer<f32> {
    let ch = doc.channels.max(1) as usize;
    let total = doc.total_frames();
    let s = start_frame.min(total) as usize;
    let e = end_frame.clamp(s as u64 + 1, total) as usize;
    let src_rate = doc.sample_rate.max(1);
    let (dst_rate, dst_ch) = (stream_format.0.max(1), stream_format.1.max(1) as usize);
    let frames = e - s;

    // BUGFIX (playback freezes ~0.5 s after start): resample the slice to the
    // device's stream rate and mix to its channel count. rodio 0.19 consumes
    // appended sources at the hardware clock without resampling, so a file
    // whose rate differs from the opened config drains N× too fast/slow —
    // with the observed "half a second of sound, then everything stops".
    // Linear interpolation per channel; mono/stereo up/down-mix only.
    let out_frames = ((frames as u64 * dst_rate as u64) / src_rate as u64).max(1) as usize;
    let mut data: Vec<f32> = Vec::with_capacity(out_frames * dst_ch);
    let ratio = src_rate as f64 / dst_rate as f64;
    for i in 0..out_frames {
        let pos = (i as f64) * ratio;
        let i0 = (pos as usize).min(frames - 1);
        let i1 = (i0 + 1).min(frames - 1);
        let frac = (pos - i0 as f64) as f32;
        let a = &doc.samples[s * ch + i0 * ch..s * ch + i0 * ch + ch];
        let b = &doc.samples[s * ch + i1 * ch..s * ch + i1 * ch + ch];
        if dst_ch == 1 {
            // down-mix both interpolated frames to mono
            let l = a[0] + (b[0] - a[0]) * frac;
            let r = if ch > 1 { a[ch - 1] + (b[ch - 1] - a[ch - 1]) * frac } else { l };
            data.push((l + r) * 0.5);
        } else {
            for c in 0..dst_ch {
                let sc = c.min(ch - 1); // mono → duplicate, extra ch → last ch
                data.push(a[sc] + (b[sc] - a[sc]) * frac);
            }
        }
    }
    rodio::buffer::SamplesBuffer::new(dst_ch as u16, dst_rate, data)
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
    // Local mirror of the global `consumed_samples` clock. Incremented in
    // `next()` and flushed into the atomic only every 512 samples — see the
    // BUGFIX note below (atomic-per-sample starved the cpal callback).
    local_idx: u64,
    // Per-channel RMS accumulators (f64 over the current meter window).
    rms_sum_l: f64,
    rms_sum_r: f64,
    rms_count: usize,
    // Window peak holders (published to the shared meter with the window).
    peak_l: f32,
    peak_r: f32,
}

impl Iterator for MeteredSource {
    type Item = f32;
    #[inline]
    fn next(&mut self) -> Option<f32> {
        // `SamplesBuffer<f32>: Iterator<Item = f32>` (rodio 0.19), so the
        // element type is inferred concretely here — no annotation needed.
        match self.inner.next() {
            Some(v) => {
                // BUGFIX (playback dies ~0.5 s after start, disk hammering):
                // this iterator runs inside the cpal audio callback. A
                // contended `fetch_add` on the shared atomic AND a mutex lock
                // on EVERY sample made the callback overrun; PulseAudio then
                // dropped the stream after its first period (~0.5 s at
                // 44.1 kHz) — exactly the observed symptom. The hot path now
                // touches only thread-local state; the global clock and the
                // meter are published once per 512-sample window.
                let idx = self.local_idx;
                self.local_idx += 1;
                let av = v.abs();
                let is_left = self.channels < 2 || (idx % self.channels as u64) == 0;
                if is_left {
                    if av > self.peak_l { self.peak_l = av; }
                    self.rms_sum_l += (v as f64).powi(2);
                } else {
                    if av > self.peak_r { self.peak_r = av; }
                    self.rms_sum_r += (v as f64).powi(2);
                }
                self.rms_count += 1;
                if self.rms_count >= 512 {
                    self.publish();
                }
                Some(v)
            }
            None => {
                // Source exhausted — flush whatever remains in the window so
                // the final partial block still reaches the meter/clock.
                if self.rms_count > 0 {
                    self.publish();
                }
                None
            }
        }
    }
}

impl MeteredSource {
    /// Push the accumulated window into the shared atomics/mutex. Called at
    /// most once per 512 samples (~11 ms at 44.1 kHz stereo) — cheap enough
    /// for the audio callback even under contention.
    fn publish(&mut self) {
        // Advance the global clock by the number of samples produced since
        // the last publish (store-free: relative update keeps pause/resume
        // and cross-source appends consistent).
        self.shared
            .consumed_samples
            .fetch_add(self.rms_count as u64, Ordering::Relaxed);
        let n = self.rms_count.max(1) as f64;
        let (rms_l, rms_r, peak_l, peak_r) = (
            (self.rms_sum_l / n).sqrt() as f32,
            (self.rms_sum_r / n).sqrt() as f32,
            std::mem::take(&mut self.peak_l),
            std::mem::take(&mut self.peak_r),
        );
        if let Ok(mut meter) = self.shared.meter.try_lock() {
            meter.rms_l = rms_l;
            meter.rms_r = rms_r;
            if peak_l > meter.peak_l { meter.peak_l = peak_l; }
            if peak_r > meter.peak_r { meter.peak_r = peak_r; }
        }
        self.rms_sum_l = 0.0;
        self.rms_sum_r = 0.0;
        self.rms_count = 0;
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
    // (sample_rate, channels) the OutputStream was opened with — every slice
    // is resampled/mixed to this format before append (see slice_to_buffer).
    stream_format: (u32, u16),
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
                    shared.consumed_samples.store(stream_sample_at(d, s, stream_format), Ordering::Relaxed);
                    shared.pos_offset.store(s, Ordering::Relaxed);
                    shared.session_start_frame.store(s, Ordering::Relaxed);
                    sel_start = s;
                    sel_end = e;
                    let src = MeteredSource {
                        inner: slice_to_buffer(d, s, e, stream_format),
                        channels: d.channels.max(1),
                        shared: shared.clone(),
                        local_idx: 0,
                        rms_sum_l: 0.0,
                        rms_sum_r: 0.0,
                        rms_count: 0,
                        peak_l: 0.0,
                        peak_r: 0.0,
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
                {
                    let ch = channels_of(&doc);
                    // consumed_samples now counts STREAM samples; convert the
                    // raw counter back to a file-frame index for the state
                    // machine (see stream_frame_of).
                    let f = stream_frame_of(&shared.consumed_samples.load(Ordering::Relaxed), doc.as_deref(), ch, stream_format);
                    ctrl.freeze_resume_at(f);
                }
            }
            Some(Cmd::Stop) => {
                pending_play = None; // ... or Stop (§5.3 stop = silence)
                sink.stop();
                shared.source_active.store(false, Ordering::Relaxed);
                let start = shared.session_start_frame.load(Ordering::Relaxed);
                let mut ctrl = controller.lock().unwrap_or_else(|p| p.into_inner());
                // Seed the playhead clock in stream-sample units when a
                // document is loaded; without one fall back to file units.
                match &doc {
                    Some(d) => shared.consumed_samples.store(stream_sample_at(d, start, stream_format), Ordering::Relaxed),
                    None => shared.consumed_samples.store(start * channels_of(&doc) as u64, Ordering::Relaxed),
                }
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
                    shared.consumed_samples.store(frame_to_stream_samples(target, d.sample_rate, stream_format), Ordering::Relaxed);
                    shared.pos_offset.store(target, Ordering::Relaxed);
                    {
                        let mut ctrl = controller.lock().unwrap_or_else(|p| p.into_inner());
                        ctrl.mark_seek(target);
                        if ctrl.is_playing() {
                            let (s, e) = (target, if sel_end > sel_start { sel_end } else { d.total_frames() });
                            let src = MeteredSource {
                                inner: slice_to_buffer(d, s, e, stream_format),
                                channels: ch,
                                shared: shared.clone(),
                                local_idx: 0,
                                rms_sum_l: 0.0,
                                rms_sum_r: 0.0,
                                rms_count: 0,
                                peak_l: 0.0,
                                peak_r: 0.0,
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
                        shared.consumed_samples.store(frame_to_stream_samples(sel_start, d.sample_rate, stream_format), Ordering::Relaxed);
                        shared.pos_offset.store(sel_start, Ordering::Relaxed);
                        let src = MeteredSource {
                            inner: slice_to_buffer(d, sel_start, sel_end, stream_format),
                            channels: ch,
                            shared: shared.clone(),
                            local_idx: 0,
                            rms_sum_l: 0.0,
                            rms_sum_r: 0.0,
                            rms_count: 0,
                            peak_l: 0.0,
                            peak_r: 0.0,
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
                    shared.consumed_samples.store(frame_to_stream_samples(sel_start, d.sample_rate, stream_format), Ordering::Relaxed);
                    shared.pos_offset.store(sel_start, Ordering::Relaxed);
                    let src = MeteredSource {
                        inner: slice_to_buffer(d, sel_start, sel_end, stream_format),
                        channels: ch,
                        shared: shared.clone(),
                        local_idx: 0,
                        rms_sum_l: 0.0,
                        rms_sum_r: 0.0,
                        rms_count: 0,
                        peak_l: 0.0,
                        peak_r: 0.0,
                    };
                    sink.append(src);
                }
            } else {
                // Natural end of a non-loop session: rewind the playhead to
                // the session start point (§5.3).
                let start = shared.session_start_frame.load(Ordering::Relaxed);
                match &doc {
                    Some(d) => shared.consumed_samples.store(stream_sample_at(d, start, stream_format), Ordering::Relaxed),
                    None => shared.consumed_samples.store(start * channels_of(&doc) as u64, Ordering::Relaxed),
                }
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

/// File-frame → position in the `consumed_samples` clock. The clock counts
/// one increment per sample emitted by MeteredSource — i.e. samples of the
/// RESAMPLED stream slice (stream rate × stream channels), not of the source
/// document. Seeding it with file units (`frame * file_channels`) while the
/// meter advances in stream units skews the playhead whenever the two differ
/// (e.g. 22050 Hz file on a 48 kHz device).
fn stream_sample_at(doc: &Document, frame: u64, stream_format: (u32, u16)) -> u64 {
    let src_rate = doc.sample_rate.max(1) as u64;
    let dst_rate = stream_format.0.max(1) as u64;
    let dst_ch = stream_format.1.max(1) as u64;
    (frame * dst_rate / src_rate) * dst_ch
}

/// Inverse of `stream_sample_at`: raw counter → file-frame index. Falls back
/// to the legacy `counter / file_channels` formula when no document is known.
fn stream_frame_of(counter: &u64, doc: Option<&Document>, file_channels: u16, stream_format: (u32, u16)) -> u64 {
    match doc {
        Some(d) => {
            let src_rate = d.sample_rate.max(1) as u64;
            let dst_rate = stream_format.0.max(1) as u64;
            let dst_ch = stream_format.1.max(1) as u64;
            (*counter / dst_ch) * src_rate / dst_rate
        }
        None => *counter / file_channels.max(1) as u64,
    }
}

/// File-frame → position in the `consumed_samples` clock for call sites that
/// don't have a `Document` at hand (Seek / loop-restart in the controller
/// thread). The file's own sample rate is passed explicitly; all other math
/// is identical to `stream_sample_at`. Without this helper the three seek
/// paths referenced a non-existent function (E0425).
fn frame_to_stream_samples(frame: u64, file_sample_rate: u32, stream_format: (u32, u16)) -> u64 {
    let src_rate = file_sample_rate.max(1) as u64;
    let dst_rate = stream_format.0.max(1) as u64;
    let dst_ch = stream_format.1.max(1) as u64;
    (frame * dst_rate / src_rate) * dst_ch
}

/// Enumerate all devices through **cpal's public host API** (re-exported
/// verbatim by rodio as `rodio::cpal`). Verified against the exact sources of
/// rodio 0.19 / cpal 0.15.3:
/// * cpal 0.15 exposes `available_hosts() -> Vec<HostId>` and
///   `host_from_id(HostId) -> Result<Host, HostUnavailable>`;
/// * trait methods need `cpal::traits::{HostTrait, DeviceTrait}` in scope.
///
/// Devices are listed WITHOUT probing `supported_output_configs()` — on the
/// ALSA backend that probe opens every PCM via alsa-lib, spamming dmix/
/// dsnoop/JACK errors and touching hardware. Output-capability is checked
/// lazily per candidate inside `pick_output_device` instead.
fn list_output_devices_quiet() -> Vec<(rodio::cpal::Device, String)> {
    list_all_named_devices()
}

/// Does this device expose at least one output config? Lazy probe.
fn device_has_output(device: &rodio::cpal::Device) -> bool {
    use rodio::cpal::traits::DeviceTrait;
    match device.supported_output_configs() {
        Ok(mut cfgs) => cfgs.next().is_some(),
        Err(_) => false,
    }
}

/// BUGFIX (playback freezes ~0.5 s after start): pick a stream config for the
/// chosen device that matches what our sources actually emit — `f32` samples.
/// Probed ONCE at startup for the single selected device only (never during
/// enumeration, so no dmix/JACK error spam). Strategy:
///   1. the device's default config when its sample format is f32 (keeps the
///      desktop's own rate/format choice — Pulse resamples transparently);
///   2. otherwise the first supported config with `sample_format() == F32`
///      whose channel count is 1 or 2 (preferring 2);
///   3. `None` → caller falls back to `try_from_device`.
fn select_f32_output_config(device: &rodio::cpal::Device) -> Option<rodio::cpal::SupportedStreamConfig> {
    use rodio::cpal::traits::DeviceTrait;
    if let Ok(cfg) = device.default_output_config() {
        if cfg.sample_format() == rodio::cpal::SampleFormat::F32 {
            return Some(cfg);
        }
        tracing::debug!(
            "[output] default config format {:?} != F32 — scanning supported configs",
            cfg.sample_format()
        );
    }
    let cfgs = match device.supported_output_configs() {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!("[output] supported_output_configs failed: {e}");
            return None;
        }
    };
    let mut fallback: Option<rodio::cpal::SupportedStreamConfig> = None;
    for sc in cfgs {
        if sc.min_sample_rate() > sc.max_sample_rate() {
            continue;
        }
        if sc.channels() < 1 || sc.channels() > 2 {
            continue;
        }
        // SupportedStreamConfigBuf -> concrete SupportedStreamConfig at the
        // device's native rate (min == max for most ALSA/Pulse PCMs; for
        // range devices min_sample_rate is always supported).
        let concrete = sc.with_max_sample_rate();
        if concrete.sample_format() == rodio::cpal::SampleFormat::F32 {
            if concrete.channels() == 2 {
                return Some(concrete);
            }
            if fallback.is_none() {
                fallback = Some(concrete);
            }
        }
    }
    fallback
}

/// Choose the output device, in order of preference:
///   1. `AUDIOSCOUT_ALSA_DEVICE` env override — exact (case-insensitive)
///      substring match against device names. Lets the user force "default",
///      "pulse", "sysdefault:CARD=PCH", "hw:CARD=PCH,DEV=0", … without a
///      rebuild: `AUDIOSCOUT_ALSA_DEVICE=pulse target/debug/audioscout`.
///   2. The cpal **PulseAudio** host's default device — on Debian desktops
///      Pulse/pipewire owns the sound card; opening its default sink routes
///      through the same path as every other app and never touches raw PCMs
///      (no dmix/JACK error spam).
///   3. The cpal **ALSA** host device whose name contains "pulse" or equals
///      "default"/"dmix" (some cpal builds surface the pulse plugin through
///      the ALSA host instead).
///   4. `None` → caller falls back to `OutputStream::try_default()`.
fn pick_output_device() -> Option<(rodio::cpal::Device, String)> {
    use rodio::cpal::traits::{DeviceTrait, HostTrait};

    let all = list_all_named_devices();

    // 1) explicit override
    if let Ok(want) = std::env::var("AUDIOSCOUT_ALSA_DEVICE") {
        let w = want.trim().to_lowercase();
        if !w.is_empty() {
            for (device, name) in &all {
                if name.to_lowercase() == w || name.to_lowercase().contains(&w) {
                    if device_has_output(device) {
                        tracing::info!("[output] using AUDIOSCOUT_ALSA_DEVICE=\"{name}\"");
                        return Some((device.clone(), name.clone()));
                    }
                }
            }
            tracing::warn!("[output] AUDIOSCOUT_ALSA_DEVICE=\"{want}\" not found among devices — falling back to auto-pick");
        }
    }

    // 2) PulseAudio host default output device.
    //    cpal 0.15 with the *default* feature set (which is what rodio pulls
    //    in — the separate "pulse" feature is NOT enabled) has no
    //    `HostId::Pulse` variant: PulseAudio is reached through the ALSA
    //    plugin layer (`pcm.pulse`), i.e. a device named "pulse" under the
    //    ALSA host. That case is covered by step 3 below. Here we additionally
    //    try every host's `default_output_device()` (returns Option<Device>,
    //    not Result, in cpal 0.15) and accept it only if its name mentions
    //    pulse/pipewire — otherwise we might silently pick a raw hw device.
    for host_id in rodio::cpal::available_hosts() {
        let host = match rodio::cpal::host_from_id(host_id) {
            Ok(h) => h,
            Err(_) => continue,
        };
        if let Some(device) = host.default_output_device() {
            let name = rodio::cpal::traits::DeviceTrait::name(&device)
                .unwrap_or_else(|_| "<unnamed>".to_string());
            let lname = name.to_lowercase();
            let wants_pulse_route = lname.contains("pulse") || lname.contains("pipewire");
            if wants_pulse_route && device_has_output(&device) {
                tracing::info!("[output] auto-picked host-default device \"{name}\"");
                return Some((device, name));
            }
        }
    }

    // 3) ALSA-host virtual routings ("pulse", "default", "*dmix*")
    const PREFERRED: [&str; 3] = ["pulse", "default", "dmix"];
    for pref in PREFERRED {
        for (device, name) in &all {
            let lname = name.to_lowercase();
            let hit = if pref == "dmix" { lname.contains(pref) } else { lname == pref };
            if hit && device_has_output(device) {
                tracing::info!("[output] auto-picked device \"{name}\" (preference \"{pref}\")");
                return Some((device.clone(), name.clone()));
            }
        }
    }

    // nothing matched
    None
}

/// All devices of all hosts with their names (no config probing).
fn list_all_named_devices() -> Vec<(rodio::cpal::Device, String)> {
    use rodio::cpal::traits::{DeviceTrait, HostTrait};
    let mut out = Vec::new();
    for host_id in rodio::cpal::available_hosts() {
        let host = match rodio::cpal::host_from_id(host_id) {
            Ok(h) => h,
            Err(_) => continue,
        };
        let devices = match host.devices() {
            Ok(d) => d,
            Err(_) => continue,
        };
        for device in devices {
            let name = device.name().unwrap_or_else(|_| "<unnamed>".to_string());
            out.push((device, name));
        }
    }
    out
}

