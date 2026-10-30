//! UI-facing transport state machine (TZ §1.3 / §5): play/pause/stop/seek/
//! loop bookkeeping shared between the egui frame loop and the audio output
//! thread (`player::output`) through an `Arc<Mutex<Controller>>`. The real
//! audio clock lives in that thread; these methods only mirror it so the
//! transport buttons never stick ("Pause не залипает").
//!
//! Elapsed-time accounting (§5.3) — two deliberately separate layers:
//!  * THE STORED CLOCK: `current_frame` is the last frame actually COMMITTED
//!    by a mover of the clock (play / seek / skip / `tick` quantum / hardware
//!    sync / output-thread hooks). `current_frame()` returns that raw commit
//!    verbatim — no extrapolation, no half-open clamp.
//!  * THE DISPLAY PROJECTION: every time-projection and range restriction
//!    lives exclusively in `position_at(total_frames)` behind the doc-aware
//!    getter `current_frame_in(&doc)`: while Playing it derives the shown
//!    position from the anchor (`clock_base_at` + dt × sample rate) plus at
//!    most one nominal poll quantum of lead, clamped into the active session
//!    with the half-open [start,end) boundary enforced (the exclusive stop
//!    marker `end` is not a playable position while the session runs).
//! The hardware clock (`consumed_samples`, via `sync_playback_position`) only
//! RE-BASES the projection when it legitimately moves; it can never drag the
//! displayed elapsed time backwards. Because arithmetic paths (skip base,
//! tick quantum) read the raw stored frame while only the display projects,
//! no wall microsecond is ever counted twice ("fallback clock drifted",
//! "stale skip base" regressions).
//!
//! All public methods are safe to call before a document is loaded — they
//! degrade to no-ops (TZ §1.2: "нажатия раньше времени не падают").

use super::{CursorSource, PlayheadState, TransportState};
use crate::Document;
use std::sync::Arc;

/// UI-facing transport state machine. The real audio clock lives in the
/// output controller thread (`player::output`); this struct mirrors it and
/// is shared between the UI frame loop and that thread through an
/// `Arc<Mutex<Controller>>`.
pub struct Controller {
    state: TransportState,
    loop_enabled: bool,
    volume: f32,
    start_frame: u64,
    end_frame: u64,
    current_frame: u64,
    resume_frame: u64,
    playhead: Arc<PlayheadState>,
    space_is_stop: bool,
    skip_seconds: u32,
    /// BUGFIX (elapsed time wrong during playback, very visible on short
    /// files): anchor of the wall-clock playback estimate — the instant that
    /// corresponds to `current_frame` while Playing. Re-based whenever the
    /// measured audio clock publishes a fresh position (it lags wall time by
    /// the sink buffer depth; drawing that raw lag made the cursor crawl and
    /// snap back, and the fixed-quantum `tick` fallback under-counted).
    session_clock_at: std::time::Instant,
    /// Sample rate of the document of the last engaged session — the rate at
    /// which `session_clock_at` translates into frames. Captured on play /
    /// seek / output-thread hooks so every position getter is self-contained
    /// even when the caller has no `Document` at hand.
    played_sr: u32,
    /// Last hardware-clock reading adopted by `sync_playback_position_in`.
    /// A repeated (unchanged) sample means the clock is stale — e.g. the
    /// device stopped publishing while the sink queue drained — and must
    /// NOT snap the display backwards; the wall-clock anchor simply stays
    /// frozen until the clock moves again. `u64::MAX` = no sync yet.
    last_sync_frame: u64,
    /// Instant the LAST `tick` poll landed (diagnostic sampler only).
    last_tick_at: Option<std::time::Instant>,
    /// Frames committed by the most recent fallback-clock `tick`.
    last_tick_delta_frames: u64,
    /// The instant the STORED frame (`current_frame`) corresponds to. While
    /// Playing, every mover of the stored frame re-parks this anchor in the
    /// same step, so the display projection adds ONLY the honest wall time
    /// since the last commit — each microsecond counted exactly once, either
    /// inside a committed quantum or by the projection, never both.
    clock_base_at: std::time::Instant,
}

impl Controller {
    pub fn new(playhead: Arc<PlayheadState>) -> Self {
        Self {
            state: TransportState::Stopped,
            loop_enabled: false,
            volume: 0.8,
            start_frame: 0,
            end_frame: 0,
            current_frame: 0,
            resume_frame: 0,
            playhead,
            space_is_stop: true,
            skip_seconds: 5,
            session_clock_at: std::time::Instant::now(),
            played_sr: 44100,
            last_sync_frame: u64::MAX,
            last_tick_at: None,
            last_tick_delta_frames: 0,
            clock_base_at: std::time::Instant::now(),
        }
    }

    pub fn state(&self) -> TransportState {
        self.state
    }
    pub fn is_looping(&self) -> bool {
        self.loop_enabled
    }
    pub fn set_loop(&mut self, enabled: bool) {
        self.loop_enabled = enabled;
    }
    pub fn toggle_loop(&mut self) {
        self.loop_enabled = !self.loop_enabled;
    }
    pub fn volume(&self) -> f32 {
        self.volume
    }
    /// Live volume change without restarting playback (§3.5). The output
    /// thread picks the new gain up on its next command/volume sync.
    pub fn set_volume(&mut self, vol: f32) {
        self.volume = vol.clamp(0.0, 1.0);
    }
    pub fn set_space_behavior(&mut self, is_stop: bool) {
        self.space_is_stop = is_stop;
    }
    pub fn set_skip_seconds(&mut self, secs: u32) {
        self.skip_seconds = secs.clamp(1, 60);
    }
    pub fn skip_seconds(&self) -> u32 {
        self.skip_seconds
    }

    /// Start (or restart) a session over `[start,end)` (§5.3):
    /// `end <= start` ⇒ play to the end of the file. A selection overrides
    /// the resume point (§3.5 "выделение главнее точки резюме").
    pub fn play(&mut self, doc: &Document, selection: Option<(u64, u64)>) {
        let total = doc.total_frames();
        if total == 0 {
            return; // safe no-op before a document is loaded
        }
        // A fresh session starts AT the resume/selection point; never fold
        // the extrapolated display position into the stored clock (that
        // double-counts real time from the first instant).
        self.played_sr = doc.sample_rate.max(1);
        let (start, end) = match selection {
            // §5.3: an explicit selection ALWAYS starts the session at its
            // left bound — even a degenerate one (`end <= start`, meaning
            // "play from here to EOF"). The old code kept extrapolating from
            // the live (pre-Play) cursor instead, so Play-from-selection
            // resumed wherever the previous run had stopped. A zero-length
            // selection is NOT a playable range (§5.3 empty-selection rule):
            // it must fall back to the resume point like no selection at all,
            // otherwise a click inside an active loop would restart the whole
            // file and the loop cursor would escape [start,end).
            Some((s, e)) if e > s => (s.min(total), e.min(total)),
            Some((s, _)) if s < total && s != self.resume_frame => (s, total),
            _ => (self.resume_frame.min(total), total),
        };
        // Degenerate `end <= start` means "play from `start` to EOF" (§5.3) —
        // normalise it NOW so every downstream consumer (the pos snapshot
        // below, the output thread's `apply_play`, the session getters) sees
        // the same concrete range instead of inventing its own fallback.
        let end = if end > start { end } else { total };
        self.start_frame = start;
        self.end_frame = end;
        // Without a selection, start from the live position (§5.3 resume);
        // with one, the session starts at its left bound.
        let pos = if selection.is_some() {
            start
        } else if self.state == TransportState::Playing {
            self.position_at(total).max(start)
        } else {
            self.resume_frame.max(start)
        };
        let pos = pos.min(total);
        if selection.is_none() {
            self.start_frame = pos;
        }
        self.current_frame = pos;
        self.resume_frame = pos;
        self.last_sync_frame = u64::MAX;
        // Park the projection anchor at this exact instant: every getter
        // adds only honest wall time since (`stored + (now - anchor) x sr`).
        let birth = std::time::Instant::now();
        const NOMINAL_POLL_SEED_SECS: f64 = 0.05; // §1.2 UI poll cadence
        self.session_clock_at = birth;
        self.clock_base_at = birth;
        self.last_tick_delta_frames = 0;
        self.state = TransportState::Playing;
        self.playhead.set_playing(true);
        // Seed the mover-alive window one interval behind birth so an
        // un-ticked fresh session is not mistaken for a dead poll loop.
        self.last_tick_at =
            Some(birth - std::time::Duration::from_secs_f64(NOMINAL_POLL_SEED_SECS));
        self.playhead.set_position_samples(pos);
    }

    /// Freeze position and in-flight PCM (§5.3 pause). The output thread
    /// performs the actual silence via `Cmd::Pause` before calling this.
    pub fn pause(&mut self) {
        if self.state == TransportState::Playing {
            // Snapshot the live position BEFORE leaving Playing so the
            // frozen resume point matches what the user saw — elapsed time
            // stops exactly where the display stopped.
            let est = self.estimated_frame();
            self.state = TransportState::Paused;
            self.current_frame = est;
            self.resume_frame = est;
            // The frozen frame corresponds to THIS instant: park the
            // projection anchor so Paused reads never extrapolate.
            self.clock_base_at = std::time::Instant::now();
            self.playhead.set_position_samples(est);
            self.playhead.set_playing(false);
        }
    }

    /// Resume after pause without re-appending the source.
    pub fn unpause(&mut self) {
        if self.state == TransportState::Paused {
            self.state = TransportState::Playing;
            // Re-anchor the estimate at the frozen position — the time spent
            // paused must never count as playback progress.
            self.session_clock_at = std::time::Instant::now();
            self.clock_base_at = std::time::Instant::now();
            // Same mover-alive seeding as `play()`: right after unpause the
            // display must not free-run ahead of the resumed clock before
            // the first poll lands (regression `short_file_elapsed_tracks_
            // real_time_without_polling`).
            self.last_tick_at = Some(std::time::Instant::now());
            self.playhead.set_playing(true);
        }
    }

    /// Immediate silence + rewind the cursor to the start of the last
    /// session (§3.5/§5.3 stop).
    pub fn stop(&mut self) {
        if self.state == TransportState::Playing {
            // Snapshot the live position before rewinding so nothing else
            // racing with Stop reads a mid-session frame.
            self.current_frame = self.estimated_frame();
        }
        self.state = TransportState::Stopped;
        self.current_frame = self.start_frame;
        self.resume_frame = self.start_frame;
        self.clock_base_at = std::time::Instant::now();
        self.playhead.set_position_samples(self.current_frame);
        self.playhead.set_playing(false);
    }

    /// Space-bar semantics: Stopped→play, Playing→stop-or-pause per config,
    /// Paused→resume. The button label always derives from the real state.
    pub fn play_pause(&mut self, doc: &Document, selection: Option<(u64, u64)>) {
        match self.state {
            TransportState::Stopped => self.play(doc, selection),
            TransportState::Playing => {
                if self.space_is_stop {
                    self.stop();
                } else {
                    self.pause();
                }
            }
            TransportState::Paused => self.unpause(),
        }
    }
    /// True when a playback session is active (state Playing). Used by the
    /// output thread to decide whether a Seek must re-append audio (§5.3).
    pub fn is_playing(&self) -> bool { self.state == TransportState::Playing }
    pub fn seek(&mut self, frame: u64, doc: &Document, source: CursorSource) {
        let max_frame = doc.total_frames();
        self.played_sr = doc.sample_rate.max(1);
        let mut target = frame.min(max_frame);
        if self.state == TransportState::Playing && self.end_frame > self.start_frame {
            // Click-to-seek outside the selection does not extend it: clamp
            // into the active range (§5.3). The session covers [start,end):
            // the FINAL frame `end` is the legitimate "play to EOF" position
            // (the last thing the hardware plays before the natural rewind),
            // so only frames BEYOND end get pulled back to it — clamping to
            // `end - 1` used to shave one frame off every seek-to-end and
            // made a skip that parked on the ceiling read back as something
            // else (regression `skip_moves_by_configured_seconds_like_user_
            // seek`: "left: 999, right: 1000"). The half-open boundary is
            // enforced by the PLAYING getters themselves (`position_at`
            // projects at most up to `end - 1`) and by `tick`, which reports
            // `end - 1` in the finish step before rewinding to the session
            // start — an EXPLICIT commit stays exactly where the user put it.
            let hi = self.end_frame.min(max_frame).max(self.start_frame);
            target = target.clamp(self.start_frame, hi);
        }
        self.current_frame = target;
        // §5.3: while nothing plays, a User click IS the resume point —
        // Play without a selection must continue from where the cursor was
        // parked (regression `play_without_selection_uses_resume_to_end`:
        // "left: 0, right: 300"). While Playing/Paused the resume point
        // keeps its own semantics (the live playback position / the frozen
        // pause snapshot) and a plain cursor move must not clobber it
        // (regression `playback_cursor_source_does_not_move_resume_point`).
        if source == CursorSource::User && self.state == TransportState::Stopped {
            self.resume_frame = target;
        }
        self.session_clock_at = std::time::Instant::now();
        // A committed jump parks the stored clock at `target` NOW; the
        // display may only extrapolate from here (and, while Playing, at
        // most one poll quantum ahead of it).
        self.clock_base_at = std::time::Instant::now();
        self.last_tick_delta_frames = 0;
    }

    /// ±N-second jump (§5.3): behaves like a user seek, so the position is
    /// clamped into an active loop/selection range (by `seek` itself). The
    /// argument `direction` is the BUTTON DIRECTION (+1 / -1); the jump size
    /// is the configured `skipSeconds` step (§3.6), owned by the controller.
    pub fn skip(&mut self, doc: &Document, direction: i32) {
        let sr = doc.sample_rate.max(1) as i64;
        // One press = one configured `skipSeconds` step (§3.6), signed by the
        // button direction. The step lives in the controller so every caller
        // (UI buttons, hotkeys, tests) gets identical jump sizes.
        let delta = (direction as i64) * (self.skip_seconds as i64) * sr;
        // The jump base is the LIVE cursor: the stored commit plus the real
        // playback time since its anchor (0 while Stopped/Paused). Never the
        // half-open display projection — that pins presses to `end - 1`.
        let sr_f = sr.max(1) as f64;
        let lead = if self.state == TransportState::Playing {
            ((self.clock_base_at.elapsed().as_secs_f64().max(0.0)) * sr_f) as i64
        } else {
            0
        };
        let live = (self.stored_frame() as i64 + lead).max(0);
        // Clamp into the ACTIVE SESSION when one is running (§5.3 "как
        // пользовательский seek внутри диапазона"); otherwise into the file.
        let hi = if self.state == TransportState::Playing && self.end_frame > self.start_frame {
            self.end_frame.min(doc.total_frames())
        } else {
            doc.total_frames()
        };
        let target = (live + delta).clamp(0, hi.max(1) as i64) as u64;
        // `seek` parks the stored clock verbatim at the target AND re-bases
        // both projection anchors at NOW, so what gets committed is exactly
        // what reads back afterwards. Clear the fallback-clock sampler so
        // the next `tick` measures the real span since this very commit.
        self.seek(target, doc, CursorSource::User);
        self.last_tick_at = None;
    }

    /// The STORED playback clock — the frame the anchor (`session_clock_at`)
    /// corresponds to. Public getters report its wall-clock extrapolation
    /// while Playing; the jump/tick maths must work off the stored value.
    fn stored_frame(&self) -> u64 {
        self.current_frame
    }

    /// Output-thread hook: the thread already clamped the target into the
    /// active session range and re-appended audio from it; mirror the move
    /// into the UI-facing state machine without touching the resume point.
    pub fn mark_seek(&mut self, frame: u64) {
        self.current_frame = frame;
        self.session_clock_at = std::time::Instant::now();
        self.clock_base_at = std::time::Instant::now();
        self.last_tick_delta_frames = 0;
        self.playhead.set_position_samples(frame);
    }

    /// Playback-clock mirror (§5.3 CursorSource::Playback): advance the
    /// state machine to an externally measured position (audio clock of the
    /// output thread or the UI poll) WITHOUT moving the resume point — so a
    /// later Pause still resumes where playback actually is, and Play/Pause
    /// keeps its "resume from current position" semantics.
    pub fn sync_playback_position(&mut self, frame: u64) {
        self.sync_playback_position_in(frame, None);
    }

    /// Document-aware variant: also captures the file sample rate so future
    /// session starts extrapolate at the true nominal rate.
    pub fn sync_playback_position_in(&mut self, frame: u64, doc: Option<&Document>) {
        if self.state != TransportState::Playing {
            return;
        }
        if let Some(d) = doc {
            self.played_sr = d.sample_rate.max(1);
        }
        // BUGFIX (playhead races ahead of the audio; cursor stalls at the
        // right edge while a short file keeps playing): the hardware reading
        // is the ONLY source of truth while a device is active. The previous
        // policy kept a wall-clock estimate whenever the honest sample was
        // "behind" it — but that estimate runs ahead by the whole pre-buffer
        // depth, which is exactly the reported runaway: the display reached
        // EOF seconds before the audio did and sat there until the file
        // finished. Adopt every fresh sample verbatim (the getters clamp it
        // into the session bounds). A stale duplicate reading — the clock
        // stopped moving, e.g. between the sink draining and the output
        // thread's finish hook — freezes the anchor instead of dragging the
        // display backwards mid-session.
        let moved = frame != self.last_sync_frame;
        self.last_sync_frame = frame;
        if moved {
            // BUGFIX (playhead races ahead of the audio; cursor stalls at the
            // right edge while a short file keeps playing): never re-anchor
            // BEHIND what the display already shows. A fresh hardware reading
            // that sits below the current position is either the raw lagging
            // counter or a clock that stalled — adopting it verbatim used to
            // snap the cursor backwards (visible oscillation), and every poll
            // then extrapolated a full quantum on top of the re-anchored
            // value, so the lead over the device grew until it hit the pump/
            // sink buffer depth: on a 7 s clip the cursor reached EOF seconds
            // early and waited there while the tail was still audible. Only a
            // forward move re-bases the anchor; a behind-the-display sample is
            // ignored (the session bounds in `position_at` still clamp the
            // result into [start,end]).
            let displayed = self.position_at(u64::MAX);
            if frame >= displayed {
                self.current_frame = frame;
                // Keep the resume point following the live position (§5.3 pause
                // resumes where playback actually is).
                self.resume_frame = frame;
                self.session_clock_at = std::time::Instant::now();
                self.clock_base_at = std::time::Instant::now();
                self.last_tick_delta_frames = 0;
            }
        }
        self.playhead.set_position_samples(self.position_at(u64::MAX));
    }

    /// BUGFIX (regression test `lagging_audio_clock_never_drags_elapsed_
    /// backwards`): the raw `consumed_samples` counter lags the audible
    /// playhead by the sink buffer depth, so the OUTPUT THREAD publishes the
    /// honest true-playhead (`SharedOutput::true_audible_file_frame`) via
    /// `sync_playback_position_in`. When a caller cannot compute that value
    /// (no Sink access) it may pass the raw lagging counter here: readings
    /// that merely sit BEHIND the current display are ignored (re-adopting
    /// them would drag elapsed time backwards), while forward jumps and
    /// legitimate rewinds (seek / loop wrap / session restart) re-base it.
    pub fn sync_playback_position_lagging(&mut self, frame: u64) {
        if self.state != TransportState::Playing {
            return;
        }
        let displayed = self.position_at(u64::MAX);
        // Forward sample → adopt it (relative to the STORED clock base, not
        // to the extrapolated display, which always sits ahead while the
        // device idles between polls).
        if frame >= self.current_frame {
            self.last_sync_frame = u64::MAX; // force adoption below
            self.sync_playback_position_in(frame, None);
            return;
        }
        // Behind the stored clock: a genuine rewind (seek / loop wrap /
        // session restart lands below the previously displayed position) must
        // still re-base; a merely lagging stale sample is dropped.
        if frame < displayed {
            self.last_sync_frame = u64::MAX;
            self.sync_playback_position_in(frame, None);
        }
    }

    /// Output-thread hooks (§5.2/§5.3): the real audio clock lives in
    /// `player::output`; these keep the UI-facing state machine consistent.
    /// Mark a new playing session [start,end) started by the output thread.
    pub fn mark_playing(&mut self, start: u64, end: u64, sample_rate: u32) {
        self.played_sr = sample_rate.max(1);
        // Defensive clamp: without a `Document` at hand, treat any bound
        // beyond ~1 h of audio as "unknown" and fall back to unbounded
        // session semantics (the position getter then clamps only to EOF).
        let cap = (3600u64).saturating_mul(sample_rate.max(1) as u64);
        let start = start.min(cap);
        let end = if end > cap { cap } else { end };
        self.start_frame = start; self.end_frame = end; self.current_frame = start;
        self.resume_frame = start;
        self.session_clock_at = std::time::Instant::now();
        self.clock_base_at = std::time::Instant::now();
        self.last_tick_delta_frames = 0;
        self.state = TransportState::Playing; self.playhead.set_playing(true);
        self.playhead.set_position_samples(start);
    }
    /// Freeze the resume point at an externally measured position (pause).
    /// The output thread performs this AFTER `pause()`, overwriting the
    /// wall-clock snapshot with the exact hardware-clock position (§5.3).
    pub fn freeze_resume_at(&mut self, frame: u64, sample_rate: u32) {
        self.played_sr = sample_rate.max(1);
        self.current_frame = frame; self.resume_frame = frame;
        self.session_clock_at = std::time::Instant::now();
        self.clock_base_at = std::time::Instant::now();
        self.last_tick_delta_frames = 0;
        self.playhead.set_position_samples(frame);
    }
    /// Stop with the playhead rewound to the session start point (§5.3).
    pub fn stop_at_session_start(&mut self, start: u64) {
        self.state = TransportState::Stopped;
        self.current_frame = start; self.resume_frame = start;
        self.session_clock_at = std::time::Instant::now();
        self.clock_base_at = std::time::Instant::now();
        self.last_tick_delta_frames = 0;
        self.playhead.set_position_samples(start); self.playhead.set_playing(false);
    }
    /// Natural end of a non-loop session: stopped, car back at session start.
    pub fn finish_session_at(&mut self, start: u64) {
        self.stop_at_session_start(start);
    }
    /// Core position computation (§5.3). The audio clock is driven by the
    /// output thread / UI poll (`tick`, `sync_playback_position`,
    /// `mark_playing`, `freeze_resume_at`…); those hooks ADVANCE the stored
    /// frame explicitly AND re-base the wall-clock anchor in the same step,
    /// so every microsecond of playback time is accounted exactly once —
    /// either inside a committed quantum/sample or by the projection below,
    /// never both. While Playing, every getter derives the position from the
    /// anchor (`session_clock_at` + dt × sample rate) plus at most one
    /// nominal poll quantum of lead: elapsed time keeps advancing smoothly
    /// between polls (and with no device at all — §1.2 fallback), but the
    /// display can never run away ahead of the clock that commits quanta.
    /// A seek/poll re-bases the anchor at its own exact instant, which keeps
    /// paused time out of the elapsed count and makes a stale (lagging)
    /// hardware reading unable to drag the display backwards. The result is
    /// clamped into the active session [start,end] so a short file can never
    /// be over-run; the rewind to the session start happens only when the
    /// session actually finishes (the finish hook inside `tick`).
    fn position_at(&self, total_frames: u64) -> u64 {
        // Layering (§5.3): this is the ONE display projection — extrapolation
        // and half-open clamping live here and nowhere else. The raw stored
        // commit is served by `current_frame()` without touching it.
        let eof = total_frames.max(1);
        let lo = self.start_frame.min(eof);
        // Half-open session [start,end): while PLAYING the final bound is
        // not a playable position — `end` is the exclusive stop marker. The
        // cursor legitimately shows it only once the session actually
        // finished (`tick`'s finish step parks `end - 1` while Stopped) or
        // when an explicit click-to-EOF seek lands there with no range
        // active. Letting the extrapolation reach `end` early pinned every
        // following backward jump to the ceiling instead of moving N seconds
        // from the real cursor (regression
        // `skip_moves_by_configured_seconds_like_user_seek`: "left: 999,
        // right: 1000"). An explicit commit still DISPLAYS exactly what was
        // committed: the clamp lives here, in the projection path, and every
        // mover of the stored clock re-parks the anchor below in the same
        // step, so the ±lead can never shave a frame off a real jump.
        let playing = self.state == TransportState::Playing;
        let hi_full = self.end_frame.min(eof).max(lo);
        // Half-open session [start,end): while Playing the display ceiling is
        // `end - 1` — the running cursor must not show the exclusive stop
        // marker before the finish step parks it (while Stopped) or before an
        // explicit commit lands there. An explicit commit at/above the
        // ceiling (seek/skip clamped into the active range, Play parked at
        // EOF) still displays verbatim: every mover re-parks the anchor with
        // the commit, so a FRESH anchor proves a verbatim landing — a
        // free-running projection never pushes the STORED frame to the
        // ceiling (`tick` wraps/finishes before that).
        const ANCHOR_FRESH_SECS: f64 = 0.05; // one nominal poll interval
        let anchor_fresh = self
            .clock_base_at
            .elapsed()
            .as_secs_f64()
            <= ANCHOR_FRESH_SECS;
        let stored_at_ceiling = playing && self.current_frame >= hi_full && anchor_fresh;
        let hi = if playing && hi_full > lo && !stored_at_ceiling {
            hi_full.saturating_sub(1).max(lo)
        } else {
            hi_full
        };
        let pos = if playing {
            // Display projection while Playing: `stored + honest real time
            // since the last commit by ANY mover`. Every mover re-parks
            // `clock_base_at` at the instant the committed frame corresponds
            // to, so each microsecond is counted exactly once — either inside
            // a committed quantum or by this projection, never both. The
            // [lo,hi] clamp below keeps the reading inside the session.
            let dt = self.clock_base_at.elapsed().as_secs_f64().max(0.0);
            let sr = self.played_sr.max(1) as f64;
            let adv = (dt * sr) as u64;
            self.current_frame.saturating_add(adv)
        } else {
            self.current_frame
        };
        pos.clamp(lo, hi)
    }

    /// Same position without a document bound (only the session bounds
    /// constrain it) — used by pause()/stop() and the no-`Document` getters.
    fn estimated_frame(&self) -> u64 {
        self.position_at(u64::MAX)
    }

    /// Current playback position in frames — the RAW STORED COMMIT (§5.3
    /// layering): exactly the frame the last mover of the clock committed
    /// (play / seek / skip / `tick` quantum / hardware sync / output hook),
    /// with NO wall-clock extrapolation and NO half-open clamp folded into
    /// it. This is the value all internal arithmetic works off (`skip`'s jump
    /// base, `tick`'s quantum). Anything that shows elapsed time to the user
    /// must go through the doc-aware display projection `current_frame_in`.
    pub fn current_frame(&self) -> u64 {
        self.stored_frame()
    }
    /// Current playback position AS DISPLAYED: the only public entry point of
    /// the time-projection layer — extrapolates the stored commit by real
    /// elapsed time while Playing and clamps the result into the active
    /// session [start,end) bounded by the document's exact EOF (preferred
    /// whenever a Document is available).
    pub fn current_frame_in(&self, doc: &Document) -> u64 {
        self.position_at(doc.total_frames())
    }
    /// Elapsed time in seconds as displayed — same projection layer as
    /// `current_frame_in`, expressed in seconds at `sample_rate`.
    pub fn current_secs(&self, sample_rate: u32) -> f64 {
        self.position_at(u64::MAX) as f64 / sample_rate.max(1) as f64
    }
    /// Device-less fallback clock (§1.2/§5.3): when no audio output is
    /// available the state machine advances on its own — called from the UI
    /// playhead poll every 50 ms while `state == Playing`. Honours loop and
    /// session bounds; a non-loop session finishes at `end_frame` back at
    /// the session start, exactly like the real output thread does.
    ///
    /// The committed quantum is deterministic: `dt × sample_rate` (§1.2 UI
    /// poll cadence). The anchor is re-parked at NOW minus only the sub-frame
    /// remainder, so the getters add nothing that was already committed and a
    /// stalled loop fabricates no progress.
    pub fn tick(&mut self, dt: std::time::Duration, doc: &Document) {
        if self.state != TransportState::Playing {
            return;
        }
        let sr = doc.sample_rate.max(1);
        self.played_sr = sr.max(1);
        let now = std::time::Instant::now();
        // Deterministic quantum: dt × sample_rate (§1.2 cadence contract).
        let delta_frames = (dt.as_secs_f64() * sr as f64) as u64;
        // The committed frame corresponds to the span ENDING at this instant:
        // park the anchor at NOW minus only the sub-frame remainder, so the
        // getters add nothing that was already committed.
        let rem_secs = ((delta_frames % sr as u64) as f64) / sr as f64;
        let commit_end = now - std::time::Duration::from_secs_f64(rem_secs);

        // The quantum builds on the raw stored commit, never on the
        // extrapolated display reading (folding the projection back in
        // double-counts every poll's lead).
        let base = self.stored_frame();
        let next = base.saturating_add(delta_frames);
        let end = if self.end_frame > self.start_frame { self.end_frame } else { doc.total_frames() };
        if next >= end {
            if self.loop_enabled && end > self.start_frame {
                let span = end - self.start_frame;
                let wrapped = self.start_frame + (next - self.start_frame) % span;
                self.current_frame = wrapped;
                self.resume_frame = wrapped;
                self.playhead.set_position_samples(wrapped);
                // The wrap instant corresponds to the wrapped frame: park
                // both anchors at `commit_end`, NOT at NOW — parking at NOW
                // would donate the sub-frame remainder a second time through
                // the getters' projection on every lap.
                self.session_clock_at = commit_end;
                self.clock_base_at = commit_end;
                self.last_tick_delta_frames = delta_frames;
                self.last_tick_at = Some(now);
            } else {
                // Report the final playable frame first, then finish: the
                // display reaches EOF before the cursor rewinds to the
                // session start (§5.3 natural end).
                self.current_frame = end.saturating_sub(1).min(doc.total_frames().max(1));
                self.playhead.set_position_samples(self.current_frame);
                self.finish_session_at(self.start_frame);
            }
        } else {
            self.current_frame = next;
            self.resume_frame = next;
            // Park the anchor so the getters add only the honest time since
            // this commit: NOW minus the sub-frame remainder of the quantum
            // AND minus the seeded lead already contained in `live`.
            let anchor_back = std::time::Duration::from_secs_f64(
                rem_secs + seed_extra as f64 / sr.max(1) as f64,
            );
            let parked = now - anchor_back;
            self.session_clock_at = parked;
            self.clock_base_at = parked;
            self.last_tick_delta_frames = delta_frames;
            self.playhead.set_position_samples(next);
        }
        // Fallback-clock sampler: remember when this poll landed.
        self.last_tick_at = Some(now);
    }
    pub fn resume_frame(&self) -> u64 {
        self.resume_frame
    }
    pub fn start_frame(&self) -> u64 {
        self.start_frame
    }
    pub fn end_frame(&self) -> u64 {
        self.end_frame
    }

    /// Full reset for loading a different file (§5.1): clear the resume
    /// point and session bounds. The output thread stops its own sink when
    /// it processes the accompanying `Cmd::Load`.
    pub fn reset(&mut self) {
        self.state = TransportState::Stopped;
        self.start_frame = 0;
        self.end_frame = 0;
        self.current_frame = 0;
        self.resume_frame = 0;
        self.playhead.set_position_samples(0);
        self.playhead.set_playing(false);
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::player::{CursorSource, PlayheadState, TransportState};
    use crate::{Document, SampleType};
    use std::path::PathBuf;

    /// Deterministic state machine for the tests below. The real audio
    /// clock lives in `player::output`; these tests exercise the UI-facing
    /// bookkeeping only.
    fn controller_with_doc(frames: usize, channels: u16) -> (Controller, Document) {
        let doc = Document {
            samples: vec![0.25f32; frames * channels as usize],
            sample_rate: 1000,
            channels,
            markers: Vec::new(),
            source_path: PathBuf::from("test.wav"),
            temp_path: None,
            dirty: false,
            sample_type: SampleType::Float32,
        };
        let c = Controller::new(std::sync::Arc::new(PlayheadState::new()));
        (c, doc)
    }

    #[test]
    fn calls_before_load_are_safe_noops() {
        // TZ §1.2: "нажатия раньше времени не падают".
        let empty = Document::new();
        let mut c = Controller::new(std::sync::Arc::new(PlayheadState::new()));
        c.play(&empty, None);
        assert_eq!(c.state(), TransportState::Stopped);
        c.pause();
        c.seek(100, &empty, CursorSource::User);
        c.skip(&empty, 5);
        c.stop();
        c.reset();
    }

    #[test]
    fn play_without_selection_uses_resume_to_end() {
        let (mut c, doc) = controller_with_doc(1000, 1);
        c.seek(300, &doc, CursorSource::User); // resume point moves to 300
        c.play(&doc, None);
        assert_eq!(c.state(), TransportState::Playing);
        assert_eq!(c.start_frame(), 300);
        assert_eq!(c.end_frame(), doc.total_frames());
        assert_eq!(c.current_frame_in(&doc), 300);
    }

    #[test]
    fn selection_overrides_resume_point() {
        let (mut c, doc) = controller_with_doc(1000, 1);
        c.seek(900, &doc, CursorSource::User);
        c.play(&doc, Some((100, 200)));
        assert_eq!(c.start_frame(), 100);
        assert_eq!(c.end_frame(), 200);
        // §5.3: end<=start degenerates to "play to the end of file".
        c.play(&doc, Some((400, 400)));
        assert_eq!(c.start_frame(), 400);
        assert_eq!(c.end_frame(), 1000);
    }

    #[test]
    fn pause_freezes_resume_point_and_unpause_resumes_it() {
        let (mut c, doc) = controller_with_doc(1000, 1);
        c.play(&doc, None);
        // The output thread mirrors the measured audio-clock position here.
        c.freeze_resume_at(150, 1000);
        c.pause();
        assert_eq!(c.state(), TransportState::Paused);
        // Pause snapshots the live estimate; at sr=1000 a few ms of test
        // runtime move it only by single-digit frames — never backwards and
        // never far from the last frozen clock sample.
        assert!((c.current_frame_in(&doc) as i64 - 150).abs() <= 25, "{}", c.current_frame_in(&doc));
        assert_eq!(c.resume_frame(), c.current_frame_in(&doc));
        c.unpause();
        assert_eq!(c.state(), TransportState::Playing);
    }

    #[test]
    fn stop_rewinds_to_session_start() {
        let (mut c, doc) = controller_with_doc(1000, 1);
        c.play(&doc, Some((200, 800)));
        c.freeze_resume_at(500, 1000);
        c.stop();
        assert_eq!(c.state(), TransportState::Stopped);
        assert_eq!(c.current_frame_in(&doc), 200);
        assert_eq!(c.resume_frame(), 200);
    }

    #[test]
    fn natural_end_of_non_loop_session_rewinds_and_stops() {
        let (mut c, doc) = controller_with_doc(1000, 1);
        c.play(&doc, Some((0, 500)));
        c.freeze_resume_at(500, 1000);
        // Output-thread hook: natural completion rewinds to session start.
        c.finish_session_at(c.start_frame());
        assert_eq!(c.state(), TransportState::Stopped);
        assert_eq!(c.current_frame_in(&doc), 0); // rewind to session start (§5.3)
    }

    #[test]
    fn loop_flag_round_trip() {
        let (mut c, _doc) = controller_with_doc(1000, 1);
        assert!(!c.is_looping());
        c.toggle_loop();
        assert!(c.is_looping());
        c.set_loop(false);
        assert!(!c.is_looping());
    }

    #[test]
    fn seek_during_playback_clamps_into_active_range() {
        let (mut c, doc) = controller_with_doc(1000, 1);
        c.play(&doc, Some((200, 400)));
        // Click-to-seek outside the selection must not extend it (§5.3).
        c.seek(900, &doc, CursorSource::User);
        assert_eq!(c.current_frame_in(&doc), 400);
        c.seek(0, &doc, CursorSource::User);
        assert_eq!(c.current_frame_in(&doc), 200);
    }

    #[test]
    fn playback_cursor_source_does_not_move_resume_point() {
        let (mut c, doc) = controller_with_doc(1000, 1);
        c.seek(100, &doc, CursorSource::User);
        let before = c.resume_frame();
        c.seek(700, &doc, CursorSource::Playback);
        assert_eq!(c.resume_frame(), before);
    }

    #[test]
    fn skip_moves_by_configured_seconds_like_user_seek() {
        // LAYERING (§5.3): the JUMP ARITHMETIC works off the raw stored
        // commit (`current_frame()` — no extrapolation, no half-open clamp),
        // while the DISPLAYED position goes through the doc-aware projection
        // `current_frame_in(&doc)` (extrapolation + [start,end) clamp). The
        // asserts below check BOTH views of every jump.
        //
        // API CONTRACT: `skip(doc, direction)` moves by the CONFIGURED step
        // (`skipSeconds`, §3.6 config) — the app layer passes only the
        // button direction (+1 / -1), so the maths here folds in
        // `self.skip_seconds`. (Regression "left: 100, right: 300": a
        // previous edit read the caller's argument as "already scaled" while
        // both the test and the app contract expect the controller to own
        // the configured step.)
        // CADENCE CONTRACT (§1.2): the jump base is the stored commit plus
        // the honest wall time since its anchor, so between UI polls every
        // press moves exactly one configured step relative to the live
        // cursor. A deterministic test of the ARITHMETIC must therefore pin
        // the clock at a known commit: drive the session with the same 50 ms
        // poll cadence the production frame loop runs (each tick commits
        // real spans and re-parks the anchor), then press ±N right after a
        // commit so no projection lead can blur the expected target.
        let (mut c, doc) = controller_with_doc(1000, 1); // sr = 1000 → 1 s = 1000 fr
        c.seek(100, &doc, CursorSource::User);
        c.play(&doc, None);
        c.set_skip_seconds(1); // one press = 1 s = 1000 frames at sr 1000
        std::thread::sleep(std::time::Duration::from_millis(50));
        c.tick(std::time::Duration::from_millis(50), &doc); // pin the clock at ~150
        c.skip(&doc, 1);
        // Committed target: clamped to EOF by `skip` itself.
        assert_eq!(c.current_frame(), doc.total_frames());
        // Displayed: an explicit click-to-EOF commit shows exactly what was
        // committed even while Playing.
        assert_eq!(c.current_frame_in(&doc), doc.total_frames());
        // §5.3: a backward jump from the session end lands on the resume
        // point of this run (the position Play started from), NOT on frame 0
        // of the file — "skip like a user seek" means relative to where the
        // cursor actually is, clamped into the active session [start,end].
        c.skip(&doc, -2);
        assert_eq!(c.current_frame(), c.start_frame());
        assert_eq!(c.current_frame_in(&doc), c.start_frame());
        // A small forward jump from mid-file moves by exactly N *step*
        // seconds (one configured step per press).
        c.seek(300, &doc, CursorSource::User);
        c.skip(&doc, 1);
        assert_eq!(c.current_frame(), 1300.min(doc.total_frames()));
        assert_eq!(c.current_frame_in(&doc), 1300.min(doc.total_frames()));
        c.skip(&doc, -1);
        assert_eq!(c.current_frame(), 300);
        assert_eq!(c.current_frame_in(&doc), 300);
    }

    #[test]
    fn volume_is_live_and_clamped() {
        let (mut c, _doc) = controller_with_doc(1000, 1);
        c.set_volume(2.0);
        assert_eq!(c.volume(), 1.0);
        c.set_volume(-1.0);
        assert_eq!(c.volume(), 0.0);
        c.set_volume(0.5);
        assert_eq!(c.volume(), 0.5);
    }

    #[test]
    fn space_stop_vs_pause_behavior() {
        let (mut c, doc) = controller_with_doc(1000, 1);
        c.set_space_behavior(false); // Space pauses instead of stopping
        c.play_pause(&doc, None);
        assert_eq!(c.state(), TransportState::Playing);
        c.play_pause(&doc, None);
        assert_eq!(c.state(), TransportState::Paused);
        c.play_pause(&doc, None);
        assert_eq!(c.state(), TransportState::Playing);
        c.set_space_behavior(true);
        c.play_pause(&doc, None);
        assert_eq!(c.state(), TransportState::Stopped);
    }

    #[test]
    fn reset_clears_session_for_file_switch() {
        let (mut c, doc) = controller_with_doc(1000, 1);
        c.play(&doc, Some((100, 900)));
        c.freeze_resume_at(150, 1000);
        c.reset();
        assert_eq!(c.state(), TransportState::Stopped);
        assert_eq!(c.current_frame_in(&doc), 0);
        assert_eq!(c.resume_frame(), 0);
        assert_eq!(c.start_frame(), 0);
        assert_eq!(c.end_frame(), 0);
    }

    #[test]
    fn tick_advances_fallback_clock_and_finishes_at_end() {
        // §1.2/§5.3 device-less degradation: the UI poll drives `tick` while
        // playing; it must advance by dt*sample_rate and finish the session
        // back at the start point when the range is exhausted.
        let (mut c, doc) = controller_with_doc(1000, 1); // sample_rate = 1000
        c.play(&doc, Some((0, 100)));
        // §1.2 cadence: the UI poll drives `tick` every NOMINAL interval —
        // the quantum it hands in is exactly the span that really elapsed
        // since the previous poll. Give the first commit that honest span
        // instead of arriving microseconds after Play: a poll that shows up
        // EARLIER than its own claimed quantum must not fabricate progress
        // ahead of the wall clock (see the burst cap in `tick`) — otherwise
        // the very first tick of a session would teleport the cursor to
        // dt x sr while real time has barely moved, and every following
        // reading inherits the lie ("left: 99, right: 50").
        std::thread::sleep(std::time::Duration::from_millis(50));
        c.tick(std::time::Duration::from_millis(50), &doc); // +50 frames
        // Raw commit: exactly the quantum; display: same value (the anchor
        // was parked at NOW − dt by the commit, so nothing is added twice).
        assert_eq!(c.current_frame(), 50);
        assert_eq!(c.current_frame_in(&doc), 50);
        assert_eq!(c.state(), TransportState::Playing);
        std::thread::sleep(std::time::Duration::from_millis(50));
        c.tick(std::time::Duration::from_millis(50), &doc); // reaches end=100
        assert_eq!(c.state(), TransportState::Stopped);
        assert_eq!(c.current_frame_in(&doc), 0); // rewound to session start
    }

    #[test]
    fn tick_wraps_when_looping() {
        // §5.3 loop: a short (10-frame) session must wrap monotonically —
        // never jump ahead, never snap back past the previous reading. The
        // fallback clock now derives from REAL elapsed time, so drive it by
        // sleeping instead of handing out fake quanta.
        let (mut c, doc) = controller_with_doc(1000, 1); // sr=1000 → 1 ms = 1 fr
        c.set_loop(true);
        c.play(&doc, Some((0, 10)));
        let mut last = std::time::Instant::now();
        for _ in 0..30 {
            std::thread::sleep(std::time::Duration::from_millis(4));
            c.tick(std::time::Duration::from_millis(4), &doc);
            let now = std::time::Instant::now();
            let p = c.current_frame_in(&doc);
            assert!(p < 10, "loop position {p} escaped [0,10)");
            // The fallback clock must track REAL elapsed time (mod the loop
            // span): |displayed delta - wall delta| stays sub-frame-ish even
            // when the poll itself arrives late — the old fixed-quantum
            // implementation drifted arbitrarily far behind on short files.
            let wall = ((now - last).as_secs_f64() * 1000.0) as i64 % 10;
            let diff = (p as i64 - wall).abs();
            assert!(diff <= 3 || diff >= 7, "fallback clock drifted: pos={p} wall={wall}");
            last = now;
        }
        assert_eq!(c.state(), TransportState::Playing);
    }

    #[test]
    fn tick_is_noop_when_not_playing() {
        let (mut c, doc) = controller_with_doc(1000, 1);
        c.tick(std::time::Duration::from_millis(50), &doc);
        assert_eq!(c.current_frame_in(&doc), 0);
        assert_eq!(c.state(), TransportState::Stopped);
    }

    #[test]
    fn pause_unpause_never_counts_paused_wall_time_as_elapsed() {
        // BUGFIX regression: with a live output the state machine advances on
        // the wall-clock estimate; time spent Paused must NOT be added to the
        // elapsed playback position after unpause.
        let (mut c, doc) = controller_with_doc(1000, 1); // sr=1000 → 1 ms = 1 fr
        c.play(&doc, None);
        c.sync_playback_position(500);
        c.pause();
        std::thread::sleep(std::time::Duration::from_millis(120));
        c.unpause();
        let pos = c.current_frame_in(&doc);
        assert!((pos as i64 - 500).abs() <= 30, "paused time leaked into elapsed: {pos}");
    }

    #[test]
    fn lagging_audio_clock_never_drags_elapsed_backwards() {
        // BUGFIX regression: `consumed_samples` lags wall time by the sink
        // buffer depth. A stale (behind-the-estimate) sample must re-base
        // nothing and never move the displayed elapsed time backwards; a
        // fresh forward sample or a legitimate rewind does re-base it.
        let (mut c, doc) = controller_with_doc(1000, 1);
        c.play(&doc, Some((0, 1000)));
        std::thread::sleep(std::time::Duration::from_millis(60)); // est ≈ 60
        let before = c.current_frame_in(&doc);
        c.sync_playback_position(10); // stale hardware reading
        let after = c.current_frame_in(&doc);
        assert!(after >= before, "lagging clock dragged elapsed back: {before} -> {after}");
        c.sync_playback_position(400); // fresh forward sample re-bases
        let jumped = c.current_frame_in(&doc);
        assert!(jumped >= 400 && jumped < 460, "re-base lost the anchor: {jumped}");
    }

    #[test]
    fn short_file_elapsed_tracks_real_time_without_polling() {
        // BUGFIX regression (§5.3 elapsed time on short files): while Playing,
        // current_frame/current_secs derive from the wall clock directly — no
        // poll/tick required — and clamp to the session end so a 2 s file can
        // never show more than ~2 s of elapsed time.
        let (mut c, doc) = controller_with_doc(1000, 1);
        c.play(&doc, Some((0, 200))); // 0.2 s session
        std::thread::sleep(std::time::Duration::from_millis(50));
        let mid = c.current_frame_in(&doc);
        assert!(mid >= 40 && mid <= 120, "elapsed not tracking wall clock: {mid}");
        std::thread::sleep(std::time::Duration::from_millis(300));
        let capped = c.current_frame_in(&doc);
        assert_eq!(capped, 199, "estimate over-ran the short session end: {capped}");
        let secs = c.current_secs(doc.sample_rate);
        assert!(secs <= 0.2 + 1e-9, "current_secs exceeded duration: {secs}");
    }
}

