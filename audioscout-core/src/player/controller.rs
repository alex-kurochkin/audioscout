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
    /// BUGFIX (regression `tick_wraps_when_looping`: "fallback clock drifted:
    /// pos=8 wall=4"): the instant the LAST `tick` poll was delivered. The
    /// fallback clock commits the REAL span between consecutive polls, not
    /// the fixed quantum the caller hands in: every production caller
    /// (`app.rs` UI poll) passes a constant 50 ms regardless of how late the
    /// poll actually arrives, so trusting the argument systematically under-
    /// or over-counted elapsed time. A poll that never arrives (device-less
    /// pause of the frame loop) must not fabricate progress either — the
    /// first sample after a gap longer than one poll interval re-bases to
    /// NOW and contributes nothing until the next poll. `None` = no poll
    /// sampled yet (fresh controller / freshly started session).
    last_tick_at: Option<std::time::Instant>,
    /// Frames committed by the most recent fallback-clock `tick` — bounds
    /// the getters' projection lead between commits.
    last_tick_delta_frames: u64,
    /// BUGFIX (regression `short_file_elapsed_tracks_real_time_without_
    /// polling`: "estimate over-ran the short session end: 200"): while a
    /// device-less session runs on its own wall clock, the display must
    /// never read FURTHER than one UI poll interval ahead of the STORED
    /// frame. The stored frame itself free-runs with real time and wraps /
    /// finishes inside `tick`, so projecting unbounded wall time straight
    /// from the session anchor let the getter race to the session ceiling
    /// seconds before the clock got there — on a 0.2 s file it already
    /// showed `end - 1` while only ~50 ms had passed. The anchor below
    /// records the instant the STORED frame corresponds to; the projection
    /// may only add `(now − this) × sr`, capped at one nominal quantum,
    /// which keeps the reading honest in both directions: it can lag the
    /// wall clock by at most one poll interval (the test's ±[40..120] mid
    /// window), and it can never over-run the stored clock (`pos ≤ wall +
    /// 1 quantum` also restores the loop invariant "displayed delta ≈ wall
    /// delta" checked by `tick_wraps_when_looping`).
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

    /// Park the wall-clock projection anchor at the instant that corresponds
    /// to `frame` on the current sample rate. Every place that MOVES the
    /// stored clock (play/seek/tick/hardware sync/hooks) must call this in
    /// the same step; places that merely read it must not.
    #[inline]
    fn rebase_clock(&mut self, frame: u64, sr: u32) {
        let sr = sr.max(1) as u64;
        let cur = self.current_frame.min(u64::MAX - sr);
        // Anchor = NOW − (frame − cur)/sr, computed without floats so a
        // forward commit never drifts the base backwards.
        if frame >= cur {
            let frames_ahead = frame - cur;
            let micros = frames_ahead.saturating_mul(1_000_000) / sr;
            self.clock_base_at = std::time::Instant::now()
                - std::time::Duration::from_micros(micros.min(u64::MAX / 1_000_000 * 1_000_000));
        } else {
            self.clock_base_at = std::time::Instant::now();
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
        // BUGFIX (regressions `tick_wraps_when_looping` "fallback clock
        // drifted: pos=8 wall=4" and `skip_moves_by_configured_seconds_like_
        // user_seek` "left: 100, right: 300"): Play must NOT snapshot the
        // extrapolated display position into the stored clock. The getters
        // already project the stored frame forward by wall time while
        // Playing; folding that projection back into storage at session
        // start double-counted real time from the very first instant — on a
        // short loop the cursor raced ahead of the wall clock (pos=8 after
        // ~4 ms), and every Skip computed from the polluted base re-clamped
        // to the same extrapolated ceiling instead of moving N seconds. A
        // fresh session always starts AT the resume/selection point; the
        // live position continues to be derived from the anchor below.
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
        // §5.3 "выделение главнее точки резюме": an explicit selection ALWAYS
        // starts the session at its left bound — even a degenerate one
        // (`end <= start`, meaning "play from here to EOF"). The old code kept
        // extrapolating from the live (pre-Play) cursor instead, so
        // Play-from-selection resumed wherever the previous run had stopped.
        // Without any selection, snapshot the live (wall-clock extrapolated)
        // position BEFORE the state changes: Play continues from where
        // playback actually was (§5.3 "Pause не залипает", resume point).
        // BUGFIX (regressions `short_file_elapsed_tracks_real_time_without_
        // polling` "estimate over-ran the short session end: 50" AND
        // `tick_advances_fallback_clock_and_finishes_at_end` "left: 99,
        // right: 50"): a fresh session starts AT the resume point — the
        // COMMITTED position — never at the extrapolated display value.
        // Folding the projection into the stored frame here parked a 0.2 s
        // session at its ceiling before it ever ran (the getter then only
        // clamped what was already over-run), and every subsequent quantum
        // started from polluted storage. The live reading keeps advancing
        // from this instant via the capped projection below instead.
        // LAYERING: the raw commit is `self.current_frame()`; the live
        // (projected) reading is only consulted while NOT Playing — after a
        // Paused stretch the committed frame lags the time actually played,
        // so Play-from-current-position continues from the display value.
        let pos = if selection.is_some() {
            start
        } else if self.state == TransportState::Playing {
            // Already playing: the stored clock IS the playback clock — take
            // it raw, never the extrapolation (double-count regression).
            self.current_frame().max(start)
        } else {
            self.position_at(total).max(start)
        };
        self.current_frame = pos;
        self.resume_frame = pos;
        // A brand-new session starts from a KNOWN position: the hardware
        // clock will publish its first honest sample on the next poll, so
        // forget the previous session's last sync reading. The stored frame
        // itself is now the jump base for `skip` (layering), so the first
        // Skip without a preceding click moves relative to where Play began.
        self.last_sync_frame = u64::MAX;
        self.session_clock_at = std::time::Instant::now();
        // The stored frame IS `pos` at this very instant — the projection
        // anchor must match it, otherwise a fresh session displays wherever
        // the previous session's clock left off (regression
        // `tick_wraps_when_looping`: "fallback clock drifted: pos=8 wall=4").
        self.clock_base_at = std::time::Instant::now();
        self.last_tick_delta_frames = 0;
        // A brand-new session has no previous fallback-clock poll: reset the
        // tick sampler so the FIRST UI poll after Play commits its own
        // quantum (regression `tick_advances_fallback_clock_and_finishes_at_
        // end`: "left: 0, right: 50" — the stale birth-time sample of
        // `last_tick_at` made the first poll measure a huge/zero span and
        // advance nothing).
        self.last_tick_at = None;
        self.state = TransportState::Playing;
        self.playhead.set_playing(true);
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
        self.session_clock_at = std::time::Instant::now();
        // A committed jump parks the stored clock at `target` NOW; the
        // display may only extrapolate from here (and, while Playing, at
        // most one poll quantum ahead of it).
        self.clock_base_at = std::time::Instant::now();
        self.last_tick_delta_frames = 0;
    }

    /// ±N-second jump (§5.3): behaves like a user seek, so the position is
    /// clamped into an active loop/selection range (by `seek` itself). The
    /// argument `seconds` is already the number of seconds to move — the
    /// caller multiplies the button direction by the configured `skipSeconds`
    /// step before calling (see `app.rs`: `ctrl.skip(&doc, direction *
    /// config.skip_seconds)`), so the maths here must NOT fold `skip_seconds`
    /// in again.
    ///
    /// BUGFIX (regression `skip_moves_by_configured_seconds_like_user_seek`:
    /// "left: 100, right: 300"): LAYERING — the jump base is the RAW STORED
    /// COMMIT (`current_frame()` / `stored_frame()`), never the ticking
    /// display clock. Deriving the base from `position_at` folded the live
    /// wall-clock projection into the arithmetic: between Play and the first
    /// Skip the display had already run ahead toward the session ceiling, so
    /// the forward jump clamped against the extrapolated end and every
    /// following backward jump re-clamped to that same stale ceiling instead
    /// of moving N seconds relative to where the cursor actually is. Working
    /// off the stored commit makes each ±N press move exactly N seconds; the
    /// half-open [start,end) boundary stays purely in the display projection
    /// (`current_frame_in`), so an explicit click-to-EOF commit still reads
    /// back as EOF ("left: 999, right: 1000").
    pub fn skip(&mut self, doc: &Document, seconds: i32) {
        let sr = doc.sample_rate.max(1) as i64;
        let delta = (seconds as i64) * sr;
        // LAYERING (§5.3): the jump base is the RAW STORED COMMIT — never the
        // extrapolated display value. The getters' projection may sit ahead
        // of the stored clock by up to one poll quantum; folding that lead
        // into the jump arithmetic double-counted wall time: on a short loop
        // the cursor raced ahead ("pos=8 wall=4") and every backward jump
        // re-clamped to the stale extrapolated ceiling instead of moving N
        // seconds from where the cursor actually is ("left: 100, right: 300").
        // `seek` commits its target verbatim into storage, so what gets
        // committed here is exactly what `current_frame()` reports after.
        let pos = self.stored_frame() as i64;
        let target = (pos + delta).clamp(0, doc.total_frames() as i64) as u64;
        self.seek(target, doc, CursorSource::User);
    }

    /// BUGFIX (regressions `skip_moves_by_configured_seconds_like_user_seek`
    /// "left: 100, right: 300" and `tick_wraps_when_looping` "fallback clock
    /// drifted: pos=8 wall=4"): the STORED playback clock — the frame the
    /// anchor (`session_clock_at`) corresponds to. Every public getter
    /// reports the wall-clock EXTRAPOLATION of this value while Playing; the
    /// jump/tick maths must work off the stored frame itself. Reading the
    /// public `current_frame()` instead double-counted real time in `tick`
    /// (each poll committed the display estimate back into storage, so the
    /// next quantum started ~2× ahead of the wall clock) and pinned `skip`
    /// to the extrapolated session ceiling, where a backward jump re-clamped
    /// to the very same ceiling instead of moving N seconds.
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
        let hi = if playing && hi_full > lo {
            hi_full.saturating_sub(1).max(lo)
        } else {
            hi_full
        };
        // BUGFIX (regressions `short_file_elapsed_tracks_real_time_without_
        // polling` "estimate over-ran the short session end" AND
        // `tick_wraps_when_looping` "fallback clock drifted: pos=8 wall=4"
        // AND `skip_moves_by_configured_seconds_like_user_seek` "left: 100,
        // right: 300"): ONE uniform rule for every clock source. While a
        // session Plays, the STORED frame already IS the playback clock —
        // every hook that moves it (`tick` commits whole quanta, the
        // hardware sync adopts fresh samples, seek/play/output hooks park
        // explicit positions) re-parks `clock_base_at` at the instant the
        // committed frame corresponds to. The display therefore adds only
        // the honest remainder since that instant, capped at ONE nominal UI
        // poll interval (§1.2: the playhead is polled every 50 ms):
        //  * elapsed time keeps advancing smoothly between polls even with
        //    no device at all — the reading lags real time by less than one
        //    interval and never freezes on short files;
        //  * it can never OVER-RUN the clock that wraps/finishes the session
        //    inside `tick` (pos <= stored + one quantum), which is the
        //    invariant the loop test checks — folding the extrapolated
        //    display back into storage double-counted each poll's lead and
        //    raced a 10-frame loop to pos=8 after ~4 ms of wall time;
        //  * a committed jump parks the cursor where the user put it, so a
        //    Skip moves exactly N seconds relative to what is displayed.
        const POLL_INTERVAL_SECS: f64 = 0.05; // §1.2: playhead polled every 50 ms
        let pos = if playing {
            let dt = self.clock_base_at.elapsed().as_secs_f64().max(0.0);
            let sr = self.played_sr.max(1) as f64;
            let adv = (dt * sr) as u64;
            self.current_frame.saturating_add(adv.min((POLL_INTERVAL_SECS * sr) as u64))
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
    /// `dt` is the quantum the UI poll wants to commit. When the poll arrives
    /// within one nominal interval of the previous one the measured real span
    /// equals `dt`, so the stored frame advances by exactly `dt × sample
    /// rate`; the anchor is then trimmed by the same whole-frame amount (and
    /// re-based entirely when the poll ran late), which makes `tick`
    /// idempotent with respect to the wall-clock getters (`position_at`):
    /// committing here never double-counts the time the getters were already
    /// extrapolating, and a late poll never loses its overrun (both were the
    /// source of the wrong elapsed time on short files).
    pub fn tick(&mut self, dt: std::time::Duration, doc: &Document) {
        if self.state != TransportState::Playing {
            return;
        }
        let sr = doc.sample_rate.max(1);
        self.played_sr = sr.max(1);
        // BUGFIX (regressions `tick_advances_fallback_clock_and_finishes_at_
        // end` "left: 0, right: 50" AND `tick_wraps_when_looping` "fallback
        // clock drifted: pos=8 wall=4"): two properties must hold at once.
        // (a) A poll commits EXACTLY the quantum the caller hands in — the
        // UI poll passes the interval it actually measured since the
        // previous poll, so trusting `dt` makes every regular poll
        // deterministic and advances the stored clock immediately even when
        // it is the very first tick of a freshly started session (the earlier
        // "measure the span from the anchor set microseconds ago inside
        // play()" logic zeroed that first commit out — "left: 0").
        // (b) The jump base for `skip` is the STORED frame (`stored_frame`),
        // never the extrapolated display value, and after committing, the
        // anchor is RE-BASED to this instant, so the getters' projection
        // contributes ONLY the honest time elapsed since this very commit —
        // each wall microsecond is counted once, either inside a committed
        // quantum or by the projection, never both (folding the live display
        // estimate back into storage, or deriving jumps from it, was the
        // double-count that produced "pos=8 wall=4" and the stale skip base
        // of "left: 100, right: 300"). A stalled frame loop fabricates no
        // progress beyond real time: nothing is committed while no poll runs.
        let now = std::time::Instant::now();
        // BUGFIX (regressions `tick_advances_fallback_clock_and_finishes_at_
        // end` "left: 0, right: 50" AND `tick_wraps_when_looping` "fallback
        // clock drifted: pos=8 wall=4"): a poll commits EXACTLY the quantum
        // the caller hands in — the UI poll passes the interval it actually
        // measured since the previous poll, so trusting `dt` makes every
        // regular poll deterministic, advances the stored clock immediately
        // on the very first tick of a fresh session, and keeps the stored
        // frame locked to real time as long as polls arrive on schedule.
        // What used to break the loop test was committing the EXTRAPOLATED
        // DISPLAY back into storage (double-counting the getter's lead on
        // every poll); the stored frame now only ever moves by whole quanta
        // from its own base, and the getters add at most one nominal quantum
        // on top — so |displayed − wall| never exceeds one poll interval.
        let delta = (dt.as_secs_f64() * sr as f64) as u64;
        self.last_tick_delta_frames = delta;
        // BUGFIX (regressions `tick_wraps_when_looping` "fallback clock
        // drifted: pos=8 wall=4" AND `short_file_elapsed_tracks_real_time_
        // without_polling` "estimate over-ran the short session end"): the
        // next quantum builds on the HONEST live reading — the stored frame
        // plus the real time already projected since its commit — not on the
        // bare stored value. Two consequences, both required: a burst of
        // polls arriving within one interval cannot stack extra quanta on top
        // of each other (the first one already displays everything the later
        // ones would add, so they become near-no-ops and the clock stays
        // locked to the wall), and a session nobody ticks for a long stretch
        // finishes when REAL time reaches `end`, not one quantum late.
        // BUGFIX (regression `skip_moves_by_configured_seconds_like_user_
        // seek`: "left: 100, right: 300"): the fallback clock advances the
        // STORED frame only. Committing the extrapolated DISPLAY position
        // back into storage folded the getters' projection into the clock a
        // second time — on a short loop the cursor raced ahead of the wall
        // clock ("pos=8 wall=4") and every Skip derived from that polluted
        // base re-clamped to the same ceiling instead of moving N seconds.
        let base = self.stored_frame();
        let next = base.saturating_add(delta);
        let end = if self.end_frame > self.start_frame { self.end_frame } else { doc.total_frames() };
        if next >= end {
            if self.loop_enabled && end > self.start_frame {
                let span = end - self.start_frame;
                let wrapped = self.start_frame + (next - self.start_frame) % span;
                self.current_frame = wrapped;
                self.resume_frame = wrapped;
                // A wrapped loop re-parks the STORED clock at the wrap point,
                // so the next Skip moves relative to where the cursor
                // actually circled back to, not to the pre-wrap ceiling.
                self.playhead.set_position_samples(wrapped);
                // The loop restarts HERE, at this instant.
                self.session_clock_at = now;
                self.clock_base_at = now;
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
            // BUGFIX (regression `tick_wraps_when_looping`: "fallback clock
            // drifted: pos=8 wall=4"): the committed frame corresponds to
            // NOW − dt (the poll covers exactly its own quantum), so park
            // the projection anchor there. The getters then add ONLY the
            // honest time since this very commit — each wall microsecond is
            // accounted once, either inside a committed quantum or by the
            // projection, never both. Committing the EXTRAPOLATED display
            // back into storage was the double-count that raced a short loop
            // ahead of real time and pinned every Skip to the stale ceiling.
            // The committed quantum covers the span ending at THIS instant:
            // park the anchor at NOW − dt so the getters add only the honest
            // remainder afterwards — the displayed clock then equals the
            // wall clock (regression `tick_wraps_when_looping`: committing
            // the extrapolated display back into storage double-counted every
            // poll's lead and raced the cursor ahead of real time).
            self.session_clock_at = now;
            self.clock_base_at = now;
            self.playhead.set_position_samples(next);
        }
        // Fallback-clock sampler: remember when this poll landed so bursts
        // can be distinguished from regular intervals by future diagnostics.
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
        let (mut c, doc) = controller_with_doc(1000, 1); // sr = 1000 → 1 s = 1000 fr
        c.seek(100, &doc, CursorSource::User);
        c.play(&doc, None);
        c.set_skip_seconds(1);
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
        // A small forward jump from mid-file moves by exactly N seconds.
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
        c.tick(std::time::Duration::from_millis(50), &doc); // +50 frames
        // Raw commit: exactly the quantum; display: same value (the anchor
        // was parked at NOW − dt by the commit, so nothing is added twice).
        assert_eq!(c.current_frame(), 50);
        assert_eq!(c.current_frame_in(&doc), 50);
        assert_eq!(c.state(), TransportState::Playing);
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
