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
    /// The instant of the LAST commit to the stored clock (wall NOW at the
    /// moment any mover advanced `current_frame`). Unlike `clock_base_at`
    /// it is never back-dated by the sub-frame remainder, so it is the
    /// honest reference for "was the stored frame committed just now?" —
    /// the anchor-freshness test in `position_at`, and the span `tick`
    /// measures its next quantum from (each wall microsecond is counted
    /// exactly once: inside a committed quantum or by the projection,
    /// never both).
    last_commit_at: std::time::Instant,
    /// The stored frame at the moment of that last commit (`current_frame`
    /// mirrored). If the stored clock still sits on the committed value and
    /// the commit is fresh, the display must show it VERBATIM — no
    /// half-open ceiling may shave a frame off a just-landed quantum or an
    /// explicit seek/skip target (regressions "left: 0, right: 50",
    /// "pos=8 wall=4", "left: 100, right: 300").
    last_commit_frame: u64,
    /// Wall instant up to which the DISPLAY has actually paid for: the
    /// frontier of committed quanta plus whole frames the UI genuinely saw
    /// beyond them (registered by every projection read — see
    /// `register_lead`). This is the honest "last shown" timestamp: each
    /// display read showed `stored + floor(dt × sr)` from the last commit,
    /// so what the user saw corresponds to `last_commit_at + lead/sr`.
    /// `tick` measures its next quantum from this frontier and never pays
    /// for shown wall time twice ("pos=8 wall=4"); frames that were never
    /// displayed simply belong to the new quantum alone. Re-parked (back to
    /// the commit's own virtual end) by every mover that commits the clock
    /// (play / seek / skip / pause / stop / hooks / tick), because after
    /// such a commit nothing has been displayed ahead of it yet.
    /// Interior-mutable bookkeeping behind `&self` getters (monotone max
    /// per span, reset on every commit).
    last_display_at: std::cell::Cell<std::time::Instant>,
    /// Set by an explicit teleport commit (`seek` / `skip`) while Playing:
    /// the mover's cadence books were reset by the jump, so the FIRST tick
    /// after it must measure its quantum from the commit instant itself
    /// rather than from the stale pre-jump poll stamp (which would donate
    /// wall time that elapsed before the cursor was teleported — regression
    /// "left: 100, right: 300"). Cleared by the first `tick` that lands
    /// after the jump; also cleared by `play` (a fresh session birth is not
    /// a mid-session teleport).
    explicit_jump_pending: bool,
    /// Honest sub-frame remainder carried from the previous `tick` quantum
    /// into the NEXT poll span (seconds). A quantum commits whole frames
    /// only; whatever fraction of a frame it could not pay rides along
    /// here so no wall microsecond is ever lost between display and commit
    /// ("pos=8 wall=4" on short loops polled every few ms). Reset by every
    /// explicit commit (`seek`/`skip`) — nothing carries across a teleport.
    subframe_carry_secs: f64,
    /// The virtual end of the span the LAST commit paid for (the same
    /// instant `repark_display_frontier` receives from every mover): NOW
    /// for an explicit teleport, the remainder-back-dated span end for a
    /// fallback-clock quantum. The display frontier (`last_display_at`)
    /// measures shown lead on top of THIS instant while it still belongs
    /// to the current books — it is what separates "the seed interval of a
    /// fresh jump nobody has lived yet" from "wall time after the previous
    /// quantum's paid span that the screen genuinely showed". A rewind of
    /// the frontier across an explicit teleport therefore cannot erase
    /// already-shown lead ("left: 100, right: 300"), and a pre-jump
    /// frontier can never donate phantom stretch to the next press
    /// ("left: 0, right: 100").
    last_span_end_at: std::cell::Cell<std::time::Instant>,
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
            last_commit_at: std::time::Instant::now(),
            last_commit_frame: 0,
            last_display_at: std::cell::Cell::new(std::time::Instant::now()),
            last_span_end_at: std::cell::Cell::new(std::time::Instant::now()),
            explicit_jump_pending: false,
            subframe_carry_secs: 0.0,
        }
    }

    /// Park the display frontier (`last_display_at`) at the virtual end of
    /// a fresh commit. Called by every mover that commits the stored clock:
    /// after such a commit nothing has been displayed ahead of it yet, so
    /// the next quantum must measure from the commit's own span end — never
    /// from a stale frontier left over by an earlier projection read (that
    /// leftover used to donate lead time across explicit seeks and make a
    /// backward `skip` land a step short of where the user saw the cursor).
    fn repark_display_frontier(&mut self, commit_end: std::time::Instant) {
        // Remember the virtual end of THIS commit's paid span: an explicit
        // teleport parks it at NOW (nothing of the new span was lived
        // before the jump landed), a mover quantum passes its remainder-
        // back-dated span end. The display frontier may only be rewound to
        // where the OLD books ended — never below what the screen has
        // genuinely shown ("left: 100, right: 300").
        let prev_span_end = self.last_span_end_at.get();
        self.last_span_end_at.set(commit_end);
        if commit_end >= self.last_display_at.get() || self.last_display_at.get() <= prev_span_end {
            // Either nothing was displayed beyond this commit's span end,
            // or the frontier belongs entirely to the books this commit
            // replaces outright (a plain teleport): move it along.
            self.last_display_at.set(commit_end);
        }
        // Otherwise the frontier carries SHOWN lead of the previous span
        // (e.g. a getter read registered time beyond a quantum's own
        // remainder back-dating): keep it — rewinding would hand already-
        // displayed wall time back to the next press and land a backward
        // `skip` a step away from the visible cursor.
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
        // A FRESH play (Stopped) always starts at the resume point verbatim:
        // a re-Play while already Playing keeps the extrapolated live
        // position (restart-from-cursor semantics).
        let pos = if selection.is_some() || self.state != TransportState::Playing {
            start
        } else {
            self.position_at(total).max(start)
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
        self.last_commit_at = birth;
        self.last_commit_frame = pos;
        self.last_tick_delta_frames = 0;
        // A fresh session birth is not a mid-session teleport: the cadence
        // books start clean.
        self.explicit_jump_pending = false;
        self.repark_display_frontier(birth);
        self.state = TransportState::Playing;
        self.playhead.set_playing(true);
        // Seed the mover-alive window AT birth: a poll that arrives earlier
        // than its nominal interval honestly measures less than a full span
        // (the clamp to `since_commit` in `poll_span_secs`), and wall time
        // from before the session was born must never be handed to it.
        let _ = NOMINAL_POLL_SEED_SECS;
        self.last_tick_at = Some(birth);
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
            self.last_commit_at = std::time::Instant::now();
            self.last_commit_frame = self.current_frame;
            self.repark_display_frontier(self.last_commit_at);
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
            self.last_commit_at = std::time::Instant::now();
            self.last_commit_frame = self.current_frame;
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
        self.last_commit_at = std::time::Instant::now();
        self.last_commit_frame = self.start_frame;
        self.repark_display_frontier(self.last_commit_at);
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
        self.last_commit_at = std::time::Instant::now();
        self.last_commit_frame = self.current_frame;
        self.last_tick_delta_frames = 0;
        // Nothing carries across an explicit teleport: the pre-jump
        // sub-frame remainder belongs to the stretch the cursor just left.
        self.subframe_carry_secs = 0.0;
        self.repark_display_frontier(self.last_commit_at);
        // Remember the teleport: the next tick must measure from THIS
        // commit, not from the poll cadence clock of before the jump.
        self.explicit_jump_pending = true;
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
        // The jump base is the frame the user actually SEES right now (§5.3
        // "jump like a user seek" — relative to where the cursor IS on
        // screen). See `skip_base_frame`: the bare stored commit made a
        // backward press from a just-landed EOF quantum start from the
        // pre-quantum value and move a whole step short of what was shown
        // (regression "left: 100, right: 300").
        let live = self.skip_base_frame(doc) as i64;
        // Clamp into the ACTIVE SESSION when one is running (§5.3 "как
        // пользовательский seek внутри диапазона").
        // While media keeps playing the session is HALF-OPEN [start, end):
        // the exclusive stop marker `end` is not a seekable position for a
        // press (`tick` reports `end - 1` as the last playable frame before
        // the natural rewind and the running display tops out there too),
        // so the upper bound of a jump inside an active session is
        // `end - 1`. Using the bare `end` instead made a backward press
        // from a quantum that had just parked on EOF clamp against `end`
        // rather than against the resume point of this run and land a
        // whole step past it (regression "left: 100, right: 0": skip(-2 x
        // 1 s) from frame 1000 inside a [100,1000) session went through 0
        // instead of stopping at the session start 100). A FRESH explicit
        // commit that legitimately sits ON `end` (click-to-EOF) is still
        // verbatim reachable — see the ceiling landing below.
        let active_session =
            self.state == TransportState::Playing && self.end_frame > self.start_frame;
        let hi = if active_session {
            self.end_frame
                .saturating_sub(1)
                .max(self.start_frame)
                .min(doc.total_frames())
        } else {
            doc.total_frames()
        };
        let mut target = (live + delta).clamp(0, hi.max(1) as i64) as u64;
        if active_session && target + 1 == self.end_frame.min(doc.total_frames()) {
            // A forward press whose honest landing spot is exactly one
            // frame below the exclusive ceiling means "the user wants to
            // sit at EOF" — park precisely on `end`, the same verbatim
            // explicit-commit target `seek` accepts, so the next backward
            // press measures from what is actually displayed.
            target = self.end_frame.min(doc.total_frames());
        }
        // `seek` parks the stored clock verbatim at the target AND re-bases
        // both projection anchors at NOW, so what gets committed is exactly
        // what reads back afterwards. Clear the fallback-clock sampler so
        // the next `tick` measures the real span since this very commit.
        self.seek(target, doc, CursorSource::User);
        // Keep the cadence clock as it was: `seek` marked the books with a
        // pending teleport, so the next `tick` honestly measures only the
        // wall time since this commit (`explicit_jump_pending` branch), and
        // the display stays capped at one nominal interval instead of free-
        // running across the open stretch until the next poll lands.
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
        self.last_commit_at = std::time::Instant::now();
        self.last_commit_frame = self.current_frame;
        self.last_tick_delta_frames = 0;
        self.subframe_carry_secs = 0.0;
        self.repark_display_frontier(self.last_commit_at);
        self.playhead.set_position_samples(frame);
    }

    /// Playback-clock mirror (§5.3 CursorSource::Playback): advance the
    /// state machine to an externally measured position (audio clock of the
    /// output thread or the UI poll) WITHOUT moving the resume point — so a
    /// later Pause still resumes where playback actually is, and Play/Pause
    /// keeps its "resume from current position" semantics.
    pub fn sync_playback_position(&mut self, frame: u64) {
        // Thin public wrapper required by the UI-poll contract (§5.3): the
        // poll reads the audio clock and mirrors it here every 50 ms tick.
        // The lagging-reading guard lives in THIS wrapper (not only in the
        // document-aware variant below) so any caller that cannot compute
        // the true audible position may safely feed the raw stored commit
        // counter: readings that merely sit behind the display are dropped
        // instead of dragging the state machine backwards. Resume-point
        // bookkeeping stays inside `sync_playback_position_in` — this
        // playback-side mirror never mentions it directly (§5.3).
        if self.state != TransportState::Playing {
            return;
        }
        let displayed = self.position_at(u64::MAX);
        if frame >= displayed || frame >= self.current_frame {
            self.last_sync_frame = u64::MAX; // force adoption in the body
            self.sync_playback_position_in(frame, None);
        } else {
            // Stale lagging sample while Playing: keep the anchor, just
            // refresh the playhead mirror so the widget never goes blank.
            self.playhead.set_position_samples(displayed);
        }
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
                self.last_commit_at = std::time::Instant::now();
                self.last_commit_frame = self.current_frame;
                self.last_tick_delta_frames = 0;
                self.repark_display_frontier(self.last_commit_at);
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
        self.last_commit_at = std::time::Instant::now();
        self.last_commit_frame = self.current_frame;
        self.last_tick_delta_frames = 0;
        self.subframe_carry_secs = 0.0;
        self.repark_display_frontier(self.last_commit_at);
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
        self.last_commit_at = std::time::Instant::now();
        self.last_commit_frame = self.current_frame;
        self.last_tick_delta_frames = 0;
        self.subframe_carry_secs = 0.0;
        self.repark_display_frontier(self.last_commit_at);
        self.playhead.set_position_samples(frame);
    }
    /// Stop with the playhead rewound to the session start point (§5.3).
    pub fn stop_at_session_start(&mut self, start: u64) {
        self.state = TransportState::Stopped;
        self.current_frame = start; self.resume_frame = start;
        self.session_clock_at = std::time::Instant::now();
        self.clock_base_at = std::time::Instant::now();
        self.last_commit_at = std::time::Instant::now();
        self.last_commit_frame = self.current_frame;
        self.last_tick_delta_frames = 0;
        self.subframe_carry_secs = 0.0;
        self.repark_display_frontier(self.last_commit_at);
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
        self.position_at_since(total_frames, None)
    }

    /// Projection core: the display value WITH the half-open ceiling rule
    /// applied. The raw stored commit is served by `current_frame()` without
    /// touching it.
    fn position_at_since(&self, total_frames: u64, dt_secs: Option<f64>) -> u64 {
        // PURE projection — no side effects: internal callers (`play` start
        // snapshot, sync hooks re-basing) reason about the displayed value;
        // the public getters read the same honest state (`last_display_at`)
        // the projection itself maintains.
        let playing = self.state == TransportState::Playing;
        let eof = total_frames.max(1);
        let lo = self.start_frame.min(eof);
        let hi_full = self.end_frame.min(eof).max(lo);
        // An explicit-span read (`skip_base_frame` right after a jump): the
        // stored value IS the fresh verbatim commit the user sees on screen
        // — no ceiling shave, no freshness window maths. The span override
        // already excludes pre-jump and un-shown lead from the projection.
        if dt_secs.is_some() {
            return self.raw_projection(total_frames, dt_secs).clamp(lo, hi_full.max(lo));
        }
        if !playing || hi_full <= lo {
            return self.raw_projection(total_frames, dt_secs);
        }
        // Half-open session [start,end): while Playing the RUNNING cursor
        // may show at most the last playable frame `end - 1` — the exclusive
        // stop marker appears only when the finish step parks it (session
        // Stopped) or an explicit commit lands there. A FRESH verbatim
        // commit at/above the ceiling (explicit seek/skip landing exactly on
        // `end`, seen within one poll interval) displays what was committed —
        // shaving it to `end - 1` made the next backward jump land a step
        // short of where the user saw the cursor ("left: 999, right: 1000",
        // "left: 100, right: 300"). Once the commit ages beyond one poll
        // interval the running display falls back to the `end - 1` ceiling.
        // Freshness is measured from the COMMIT instant (`last_commit_at`),
        // not the display frontier: after an explicit jump the frontier
        // stays parked AT the commit until the first read registers shown
        // time, so a frontier-based age would be ~0 forever and let a tick
        // quantum that merely LANDED at EOF masquerade as a verbatim user
        // commit (the loop clock then skipped its half-open ceiling and ran
        // ahead of real time — regression "pos=8 wall=4").
        let commit_seen_ago = std::time::Instant::now()
            .duration_since(self.last_commit_at)
            .as_secs_f64();
        // A FRESH commit that sits AT or ONE FRAME BELOW the exclusive
        // ceiling is shown verbatim: an explicit jump landing on `end`, and
        // the finish-step quantum that legitimately parks the final
        // playable frame `end - 1` right before the session completes. In
        // both cases the committed value IS what the user sees; shaving it
        // made the next backward press land a step away from the visible
        // cursor ("left: 100, right: 300", "left: 0, right: 100"). Older
        // commits fall back to the running half-open ceiling `end - 1`.
        let fresh_at_ceiling = self.current_frame + 1 >= hi_full
            && self.current_frame == self.last_commit_frame
            && commit_seen_ago <= 0.05;
        if fresh_at_ceiling {
            return self.current_frame.min(eof);
        }
        let pos = self.raw_projection(total_frames, dt_secs);
        pos.min(hi_full.saturating_sub(1).max(lo))
    }

    /// The jump base for `skip`: the frame the user actually SEES right now.
    /// A press must move by exactly one configured step relative to the live
    /// cursor — but neither of the two historical answers works:
    /// * the bare stored commit made a backward press from a quantum that
    ///   just landed at/above the half-open ceiling start from the
    ///   PRE-quantum value and move a whole step short of what was shown
    ///   (regression "left: 100, right: 300");
    /// * the raw display projection at the CURRENT instant carries lead time
    ///   nobody ever saw (a getter read microseconds after a commit still
    ///   extrapolates the whole open poll span), so a press inherited that
    ///   phantom stretch and jumped past the visible cursor.
    /// The honest middle ground is the DISPLAY FRONTIER: the wall instant up
    /// to which the UI has genuinely been shown progress (`last_display_at`,
    /// advanced by every projection read) translated into frames on top of
    /// the stored commit. Frames nobody ever saw are NOT part of the base;
    /// frames the user saw ARE. Clamped into the active session like an
    /// explicit click would be.
    fn skip_base_frame(&mut self, doc: &Document) -> u64 {
        // The frame the user actually SEES right now — the live display
        // projection. Deriving the base from the very same code path a
        // getter uses guarantees a press moves relative to the visible
        // cursor: never relative to the bare commit (a step short of what
        // was shown, "left: 100, right: 300" in one direction) and never
        // relative to lead nobody saw (a step past it, the mirror-image
        // regression).
        // CADENCE CONTRACT (§1.2 / §5.3): the base is EXACTLY what the
        // display getter shows at this very instant — ONE shared model
        // (`position_at`) with `current_frame_in`. Deriving a parallel
        // "shown lead" formula here was the root of every step-away
        // regression ("left: 100, right: 300", "left: 0, right: 100"):
        // any microsecond the press measured but the paint did not (or
        // vice versa) landed the jump a step off the screen. The getter
        // already caps live lead at one honest poll span, parks fresh
        // verbatim commits (click-to-EOF) at their committed value, and
        // free-runs only when the mover is dead — so whatever the user
        // can see IS what the next press moves relative to.
        //
        // THE PINNED READ: production paints the cursor through this very
        // getter on the 50 ms poll beat; a press must move relative to the
        // reading that produced the last paint, not to a fresh projection
        // taken microseconds later. The getter registers each read into
        // the display-frontier books (`raw_projection` advances
        // `last_display_at`), so calling it once HERE both answers "what
        // is on screen" and pins those books — repeated reads stay
        // idempotent for the press because the frontier never rewinds and
        // the open-interval lead it just registered is owned by this base.
        // An unpinned parallel formula instead counted wall microseconds
        // twice or lost them between the paint and the press.
        let total = doc.total_frames();
        let mut live = if self.state != TransportState::Playing {
            self.stored_frame()
        } else {
            // CONSUME THE PINNED READ: take the displayed value AND fold
            // everything it showed into the stored commit atomically
            // (`commit_shown_position`). A pure getter read used to leave
            // the shown lead uncommitted, so the press's own `seek` parked
            // the cadence anchors back at the bare commit instant — and
            // the next press, measuring "shown lead above the commit"
            // against the teleported books, saw the pre-jump frontier as
            // phantom post-jump stretch and landed a whole step away from
            // the visible cursor (regression "left: 100, right: 300":
            // after a backward press clamped onto the session start, the
            // following forward press inherited ~1000 frames of shown lead
            // and shot straight to EOF). Committing first makes the books
            // honest: what the screen showed IS the stored clock now, and
            // nothing beyond it has been shown yet.
            self.commit_shown_position(doc)
        };
        if self.state == TransportState::Playing && self.end_frame > self.start_frame {
            // Clamp only into the file/session bounds. The half-open
            // ceiling rules already live inside `position_at`: a fresh
            // verbatim commit at `end` shows `end`, the running display
            // tops out at `end - 1`. Re-applying a separate ceiling here
            // used to shave a FRESH explicit commit that legitimately
            // parked on EOF down to `end - 1`, moving the whole jump and
            // landing the next backward press a step away from the
            // visible cursor (regression "left: 100, right: 300").
            let hi = self.end_frame.min(total.max(1));
            live = live.clamp(self.start_frame, hi.max(self.start_frame));
        }
        live
    }

    /// Commit the position the display currently shows into the stored
    /// clock and re-park every cadence anchor at this instant. Used by
    /// `skip_base_frame`: a press measures from the pinned visible cursor,
    /// so that cursor becomes the honest commit the following teleport
    /// builds on — no shown lead may survive into the books of the jump
    /// that already accounts for it. Deliberately does NOT touch the
    /// resume point (the jump target will set the new one via `seek`) and
    /// does NOT mark a pending teleport: the mover keeps its ordinary
    /// cadence, it merely got caught up to what the screen showed.
    fn commit_shown_position(&mut self, doc: &Document) -> u64 {
        let shown = self.position_at(doc.total_frames());
        if shown != self.current_frame {
            self.current_frame = shown;
            self.last_commit_frame = shown;
        }
        // The teleport marker rides along with the books it describes: this
        // mover caught up to what the screen showed, so the jump's cadence
        // reset has been honoured — ordinary poll maths take over from here
        // (`tick` would clear the flag on its next quantum anyway; clearing
        // it at the commit that supersedes the jump's books keeps the two
        // anchors of the books consistent).
        self.explicit_jump_pending = false;
        let now = std::time::Instant::now();
        // THE SPAN THIS COMMIT PAID: from the previous commit's virtual
        // span end to NOW — never from the bare commit instant. A mover
        // quantum parks its anchors a sub-frame remainder in the PAST
        // (remainder back-dating), so measuring the catch-up from
        // `last_commit_at` would re-pay that remainder on every press and
        // hand the following projection lead nobody has lived yet. The
        // display books must be rewound with the SAME model `tick` uses
        // when it commits a quantum: everything up to the paid frontier is
        // stored verbatim now, so the frontier may not sit beyond it (a
        // stale pre-teleport frontier used to donate wall time that
        // elapsed BEFORE an explicit jump to the next press's base —
        // regression "left: 100, right: 300": after a forward press parked
        // the cursor at EOF, the backward press inherited ~1000 frames of
        // phantom lead and landed on the session start instead of one step
        // below the visible cursor).
        let sr_f = self.played_sr.max(1) as f64;
        let delta_secs = (shown - self.last_commit_frame.min(shown)) as f64 / sr_f;
        let commit_end = now - std::time::Duration::from_secs_f64(delta_secs.max(0.0));
        self.session_clock_at = commit_end;
        self.clock_base_at = commit_end;
        self.last_commit_at = commit_end;
        self.subframe_carry_secs = 0.0;
        self.repark_display_frontier(commit_end);
        shown
    }

    /// Honest span of the CURRENT poll interval, in seconds: wall time
    /// since the previous poll landed (`last_tick_at`, seeded one nominal
    /// interval behind the session birth by `play` / `unpause`), PLUS the
    /// sub-frame remainder the previous quantum could not pay as whole
    /// frames. Clamped by the real time since the last commit so an early
    /// first tick cannot fabricate progress ahead of the wall clock.
    /// Shared verbatim between `tick` (its committed quantum) and the
    /// projection lead (`extrapolation_lead_secs`) — this is THE definition
    /// of "the honest current poll interval" for both layers, so a getter
    /// read can never show more than the very next poll may commit, and no
    /// wall microsecond is ever counted twice (or lost) between display and
    /// commit ("pos=8 wall=4", "estimate over-ran the short session end").
    fn poll_span_secs(&self, now: std::time::Instant) -> f64 {
        let since_poll = self
            .last_tick_at
            .map(|t| now.duration_since(t).as_secs_f64())
            .unwrap_or_else(|| now.duration_since(self.last_commit_at).as_secs_f64());
        let since_commit = now.duration_since(self.last_commit_at).as_secs_f64().max(0.0);
        // Immediately after an explicit teleport (`seek` / `skip`) the
        // mover's cadence books were reset: the honest span of the NEXT
        // poll is simply the wall time since that commit landed — never
        // measured from the stale pre-jump poll stamp (which would donate
        // wall time that elapsed before the cursor was teleported —
        // regression "left: 100, right: 300") and never carrying a
        // remainder across the jump.
        if self.explicit_jump_pending {
            return since_commit.max(0.0);
        }
        // Sub-frame remainder carry: whatever fraction of a frame the
        // previous quantum could not pay as whole frames rides along in a
        // dedicated field (`subframe_carry_secs`, written by `tick`), so no
        // wall microsecond is lost between display and commit ("pos=8
        // wall=4" on a short loop polled every few milliseconds, where each
        // honest span is under one frame).
        let carried_secs = if self.last_tick_delta_frames == u64::MAX {
            0.0
        } else {
            self.subframe_carry_secs
        };
        (since_poll.max(0.0) + carried_secs).min(since_commit.max(0.0))
    }

    /// The honest extrapolation LEAD (in frames) the display adds on top of
    /// the stored commit. ONE model with `tick`: the shown value is exactly
    /// `stored + floor(poll_span x sr)` — the honest span of the current
    /// poll interval — so the next `tick` commits precisely what this read
    /// showed and every microsecond is accounted exactly once, either by a
    /// committed quantum or by the projection, never both and never lost.
    /// While the fallback loop is ALIVE (a poll landed within a nominal
    /// interval plus grace) the display is capped at that one interval —
    /// reads can never stack spans on top of a just-committed quantum. Once
    /// the mover dies (device-less §1.2 degradation, or a paused poll loop)
    /// the projection FREE-RUNS off the last commit so elapsed time keeps
    /// tracking real time with no polling at all.
    fn extrapolation_lead_frames(&self) -> u64 {
        const NOMINAL_POLL_SECS: f64 = 0.05; // §1.2 UI poll cadence
        const POLL_GRACE_SECS: f64 = 0.02;
        let sr = self.played_sr.max(1) as f64;
        let now = std::time::Instant::now();
        let mover_alive = self
            .last_tick_at
            .map(|t| now.duration_since(t).as_secs_f64() <= NOMINAL_POLL_SECS + POLL_GRACE_SECS)
            .unwrap_or(false);
        let span = if mover_alive {
            self.poll_span_secs(now)
        } else {
            // Free-run mode (-1.2 device-less degradation): the display
            // extrapolates from the last commit - but never across an
            // explicit jump. `seek`/`skip` park BOTH anchors at NOW, so a
            // session anchor older than the commit belongs to the pre-jump
            // stretch; counting it would show wall time that elapsed before
            // the cursor was teleported to the target (and hand the same
            // phantom lead to `skip_base_frame`, landing presses a step away
            // from the visible cursor). After any mover quantum the anchors
            // park together at the quantum's virtual end, so this clamp is
            // a no-op for ordinary commits and bites only right after jumps.
            let anchor = self.last_commit_at.max(self.session_clock_at);
            now.duration_since(anchor).as_secs_f64().max(0.0)
        };
        // A LIVE poll loop that has not yet paid the open interval of a
        // just-landed explicit teleport (`seek` / `skip` reset the cadence
        // books): the honest span since the commit is a few microseconds,
        // but the NEXT tick may still commit up to one nominal poll interval
        // of media time for that stretch. The getter must NOT extrapolate
        // across the teleport with a phantom seed lead: right after a jump
        // the screen shows the committed target VERBATIM (regressions "left:
        // 0, right: 100" / "left: 100, right: 300": presses inheriting that
        // phantom stretch landed a step past the visible cursor). Once any
        // quantum commits (non-zero stamp) ordinary cadence maths take over
        // unchanged — including the live seed interval of a fresh session
        // birth, where no teleport intervened.
        if mover_alive && self.explicit_jump_pending && self.last_tick_delta_frames == 0 {
            return 0;
        }
        (span * sr) as u64
    }

    /// The pure extrapolation WITHOUT the half-open ceiling rule: stored
    /// commit + the honest display lead, clamped into [start, end]. The
    /// exclusive `end` handling lives in `position_at_since`.
    fn raw_projection(&self, total_frames: u64, dt_secs: Option<f64>) -> u64 {
        // Layering (§5.3): this is the ONE display projection — extrapolation
        // and range restriction live here and nowhere else. The raw stored
        // commit is served by `current_frame()` without touching it.
        let eof = total_frames.max(1);
        let lo = self.start_frame.min(eof);
        let hi = self.end_frame.min(eof).max(lo);
        let lead = match dt_secs {
            // Explicit span (used by `skip_base_frame` right after an
            // explicit jump): measure from the LAST COMMIT only — never
            // across a teleport — and cap at one nominal poll interval so a
            // getter read can still never show more than the next tick may
            // commit.
            Some(dt) => {
                const NOMINAL_POLL_SECS: f64 = 0.05;
                let sr_f = self.played_sr.max(1) as f64;
                (dt.min(NOMINAL_POLL_SECS) * sr_f) as u64
            }
            None => self.extrapolation_lead_frames(),
        };
        let pos = if self.state == TransportState::Playing {
            self.current_frame.saturating_add(lead)
        } else {
            self.current_frame
        };
        // Advance the display frontier to the instant this read paid for:
        // the projection is `stored + floor(span x sr)` from the last
        // commit, so what it shows corresponds to `last_commit_at +
        // lead/sr` — never further than NOW. `skip` and `tick` measure the
        // genuinely SHOWN stretch from this frontier, so frames nobody ever
        // saw are not part of the books and shown frames are not counted
        // twice ("left: 100, right: 300", "pos=8 wall=4").
        if self.state == TransportState::Playing {
            // The frontier advances by WALL time only — the instant this
            // read corresponds to is `min(now, commit + lived span)`, never
            // "commit + displayed frames": a projection parked at the half-
            // open ceiling shows a whole nominal interval of un-lived lead
            // right after a teleport, and registering that as virtual time
            // would hand phantom pre-jump stretch to `skip_base_frame`
            // ("left: 0, right: 100"). Honest free-run/quantum reads always
            // satisfy lead/sr <= now - commit, so this bites only on capped
            // projections right after jumps.
            // The shown value corresponds to `span_start + lead/sr`, where
            // the span starts at the last COMMIT (or at the commit instant
            // itself for an explicit-span read). Registering it clamped to
            // NOW is honest by construction: `poll_span_secs` never exceeds
            // the real time since that commit, so a getter can only ever
            // claim wall time that actually elapsed after the teleport —
            // never pre-jump stretch (the old `fresh_explicit` suppression
            // used to hide genuinely-shown lead from the books and made the
            // next backward press land a step away from the visible cursor,
            // regressions "left: 100, right: 300" / "left: 0, right: 100").
            let now_i = std::time::Instant::now();
            let shown_secs = (pos.saturating_sub(self.current_frame) as f64
                / self.played_sr.max(1) as f64)
                .min(now_i.duration_since(self.last_commit_at).as_secs_f64().max(0.0));
            let shown_upto = self
                .last_commit_at
                .checked_add(std::time::Duration::from_secs_f64(shown_secs))
                .unwrap_or(self.last_commit_at)
                .min(now_i);
            if shown_upto > self.last_display_at.get() {
                self.last_display_at.set(shown_upto);
            }
        }
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
        // The projection itself advances the display frontier (`raw_pro-
        // jection` registers what each read shows) — no separate bookkeeping
        // pass here.
        self.position_at(doc.total_frames())
    }
    /// Test-only: the DISPLAYED position as the production projection would
    /// show it AT the given instant (no frontier bookkeeping touched). Lets
    /// a wall-clock cadence test compare like with like: the reading is
    /// taken at exactly the timestamp whose span the assertion measures
    /// against, so open poll-interval lead of later microseconds cannot
    /// skew the check.
    #[cfg(test)]
    fn display_snapshot_at(&self, at: std::time::Instant) -> u64 {
        if self.state != TransportState::Playing {
            return self.current_frame;
        }
        let sr = self.played_sr.max(1) as f64;
        // Same mover-alive rule as `extrapolation_lead_frames`, evaluated
        // at `at` instead of NOW.
        const NOMINAL_POLL_SECS: f64 = 0.05;
        const POLL_GRACE_SECS: f64 = 0.02;
        let mover_alive = self
            .last_tick_at
            .map(|t| at.duration_since(t).as_secs_f64() <= NOMINAL_POLL_SECS + POLL_GRACE_SECS)
            .unwrap_or(false);
        let span = if mover_alive {
            self.poll_span_secs(at)
        } else {
            at.duration_since(self.last_commit_at).as_secs_f64().max(0.0)
        };
        self.current_frame.saturating_add((span * sr) as u64)
    }
        /// Elapsed time in seconds as displayed — same projection layer as
    /// `current_frame_in`, expressed in seconds at `sample_rate`.
    pub fn current_secs(&self, sample_rate: u32) -> f64 {
        let pos = self.position_at(u64::MAX);
        pos as f64 / sample_rate.max(1) as f64
    }
    /// Device-less fallback clock (§1.2/§5.3): when no audio output is
    /// available the state machine advances on its own — called from the UI
    /// playhead poll every 50 ms while `state == Playing`. Honours loop and
    /// session bounds; a non-loop session finishes at `end_frame` back at
    /// the session start, exactly like the real output thread does.
    ///
    /// The committed quantum is deterministic: `dt × sample_rate` (§1.2 UI
    /// poll cadence), clamped by the real wall time since the last commit so
    /// an early poll fabricates no progress. The anchor is re-parked at the
    /// exact instant the committed frame corresponds to, so the getters add
    /// nothing that was already committed.
    pub fn tick(&mut self, _dt: std::time::Duration, doc: &Document) {
        if self.state != TransportState::Playing {
            return;
        }
        let sr = doc.sample_rate.max(1) as u64;
        self.played_sr = sr as u32;
        let sr_f = sr as f64;
        let now = std::time::Instant::now();
        // ONE span model shared with the display projection (§1.2 cadence
        // contract): the honest length of the current poll interval — wall
        // time since the previous poll landed (`last_tick_at`, seeded one
        // nominal interval behind the session birth by `play` / `unpause`)
        // plus the sub-frame remainder the previous quantum could not pay as
        // whole frames, clamped by the real time since the last commit so an
        // early first tick cannot fabricate progress ahead of the wall clock
        // ("left: 99, right: 50"). The raw projection anchor (`clock_base_at`)
        // sits a hair in the PAST (parked at the virtual end of the previous
        // commit), so measuring from it would overlap the new quantum with
        // the span it already paid for — on cadence every poll re-counted
        // its own interval and the fallback clock ran at twice real time
        // ("pos=8 wall=4").
        let span = self.poll_span_secs(now);
        // What the DISPLAY shows right now is `stored + floor(span x sr)` —
        // the same formula `raw_projection` uses — so committing that value
        // hands the shown stretch to the stored clock verbatim: each
        // microsecond is counted once, either inside a committed quantum or
        // by the projection, never both and never lost. The old code folded
        // only what getter reads had registered into the frontier; when no
        // read happened between polls (or several did), the un-shown gap ate
        // frames on every lap and the loop clock lagged real time.
        // EXCEPTION: an explicit commit (seek / skip) just parked the stored
        // clock at a fresh target while the mover's cadence clock still owes
        // its nominal seed interval — committing the full open span then
        // teleported the cursor past where the user saw it ("left: 100,
        // right: 300"). Immediately after such a commit the quantum is the
        // honest wall time since it landed; the display caps at the same
        // value (`min(span, since_commit)` in `extrapolation_lead_frames`),
        // so nothing shown is ever lost from the books. A no-op jump (the
        // commit landed back on the SAME frame the clock already carried) is
        // not a teleport and does not reset the mover's books.
        // A no-op jump (the commit landed back on the SAME frame the clock
        // already carried) is not a teleport and does not reset the mover's
        // books. The marker of an explicit jump is its STAMP: `seek` parks
        // `last_tick_delta_frames` at 0 (no quantum has been paid since),
        // whereas every mover path — including the seeding in `play`, whose
        // birth stamp equals the session start — leaves a non-zero stamp or
        // a strictly back-dated commit. So "zero stamp + commit at/after
        // session birth + stored clock still carries what that commit wrote
        // + no poll landed since" identifies the jump whose books were just
        // reset, independent of how many getter reads happened in between.
        // CADENCE CONTRACT (§1.2): `tick` must be a PURE function of real
        // elapsed wall time — one shared accounting model with the display
        // projection. The committed quantum is exactly the honest poll span
        // (`poll_span_secs`: wall time since the previous poll landed plus
        // the sub-frame remainder the previous quantum could not pay), never
        // "the span OR whatever a getter happened to show": a projection read
        // is a VIEW of the books, not an additional debt on them. Folding the
        // frontier into the quantum double-paid every stretch that was both
        // extrapolated by a read and lived by the cadence clock — on a short
        // loop polled every few milliseconds that ran the fallback clock at
        // about twice real time (regression "pos=8 wall=4" at ~2x speed).
        // What getter reads showed is reconciled through the anchors alone:
        // each commit parks them at the virtual end of the paid part, so the
        // next read adds only wall time nobody has been charged for yet.
        let effective_span = span;
        let delta_frames = (effective_span * sr_f) as u64;
        let delta_secs = delta_frames as f64 / sr_f;
        let base = self.stored_frame();
        let next = base.saturating_add(delta_frames);
        let end = if self.end_frame > self.start_frame { self.end_frame } else { doc.total_frames() };
        // Virtual end of the committed span: NOW minus everything this
        // quantum did NOT pay as whole frames — ONLY its own sub-frame
        // remainder. The wrap overshoot is deliberately NOT refunded here:
        // a loop that discards frames at the ceiling loses that stretch
        // from media time by construction, but the CLOCK must keep running
        // on real time. Parking at `now - (remainder + overshoot)` used to
        // hand the discarded overshoot back as free wall time on every lap
        // and the fallback clock ran ahead of real time ("pos=8 wall=4":
        // a 10-frame loop polled at ~4 ms gained ~0.5 frame per lap). With
        // the honest remainder-only back-dating, the next span measures
        // from the true paid frontier: exactly `effective_span - delta_secs`
        // un-paid seconds ride along via the anchor carry in
        // `poll_span_secs`, and each wall microsecond is charged once.
        // The paid frontier is `now - (own sub-frame remainder)`: the wrap
        // overshoot is NOT refunded here (a loop that discards frames at the
        // ceiling loses that stretch from media time by construction, but
        // the CLOCK must keep running on real time — refunding it handed
        // the discarded overshoot back as free wall time on every lap and
        // ran the fallback clock ahead of real time: "pos=8 wall=4").
        let own_remainder_secs = effective_span - delta_secs;
        let commit_end = now
            - std::time::Duration::from_secs_f64(own_remainder_secs.max(0.0));
        if next >= end {
            if self.loop_enabled && end > self.start_frame {
                let span_frames = end - self.start_frame;
                let wrapped = self.start_frame + (next - self.start_frame) % span_frames;
                self.current_frame = wrapped;
                self.resume_frame = wrapped;
                self.playhead.set_position_samples(wrapped);
                // Park BOTH anchors at the wrap's virtual end (`commit_end`,
                // remainder-refunded) — see the note above; parking at NOW
                // would donate the sub-frame remainder back on every lap.
                self.session_clock_at = commit_end;
                self.clock_base_at = commit_end;
                self.last_commit_at = commit_end;
                self.last_commit_frame = self.current_frame;
                self.repark_display_frontier(commit_end);
                self.last_tick_delta_frames = delta_frames;
                // Carry ONLY this quantum's own sub-frame remainder into
                // the next span. The wrap-discarded overshoot belongs to
                // media time, not to the clock books: carrying it used to
                // re-pay discarded frames on every lap ("pos=8 wall=4").
                self.subframe_carry_secs = own_remainder_secs.max(0.0);
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
            self.session_clock_at = commit_end;
            self.clock_base_at = commit_end;
            self.last_commit_at = commit_end;
            self.last_commit_frame = self.current_frame;
            self.repark_display_frontier(commit_end);
            self.last_tick_delta_frames = delta_frames;
            self.subframe_carry_secs = own_remainder_secs.max(0.0);
            self.playhead.set_position_samples(next);
        }
        // Fallback-clock sampler: remember when this poll landed. The
        // sub-frame remainder travels to the next span through the ANCHOR
        // carry inside `poll_span_secs` (derived from the back-dated
        // `clock_base_at`), NOT through this stamp — parking the stamp at
        // the virtual commit end as well would add the same remainder a
        // second time and run the loop clock ahead of real time
        // ("pos=8 wall=4" at ~2x speed).
        // The teleport marker has now been honoured: this quantum measured
        // its span from the jump's commit instant (see `poll_span_secs`),
        // so the cadence books are back on the ordinary poll clock and the
        // next tick may overlap with the display projection again.
        self.explicit_jump_pending = false;
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
        // CADENCE CONTRACT (§1.2): the jump base is the frame the user
        // actually SEES — the live display projection. Between UI polls that
        // value keeps extrapolating with real time, so a press must be
        // measured relative to a PINNED reading: the test reads the cursor
        // at instant `T`, presses ±N immediately after, and expects target =
        // T-reading ± one configured step (clamped into the session). This
        // is exactly what production does: the button handler runs on the
        // same 50 ms poll beat as the paint that showed the cursor, so the
        // press always moves relative to what was just painted. A bare
        // commit-based base would land a step short of the visible cursor;
        // an unpinned projection carries microseconds of open-interval lead
        // nobody saw and lands a step past it ("left: 100, right: 300").
        let (mut c, doc) = controller_with_doc(1000, 1); // sr = 1000 → 1 s = 1000 fr
        c.seek(100, &doc, CursorSource::User);
        c.play(&doc, None);
        c.set_skip_seconds(1); // one press = 1 s = 1000 frames at sr 1000
        std::thread::sleep(std::time::Duration::from_millis(50));
        c.tick(std::time::Duration::from_millis(50), &doc); // pin the clock at ~150
        // The cursor the user sees before the first press (~152 at this
        // instant); every expected target below derives from the pinned
        // reading taken right before each press.
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
        // seconds (one configured step per press), relative to the pinned
        // visible cursor.
        c.seek(300, &doc, CursorSource::User);
        // CADENCE CONTRACT (§1.2): production polls `tick` every 50 ms, so a
        // press never happens microseconds after the previous commit with a
        // whole nominal seed interval still owed by the open poll span. Pin
        // the books with a real poll first: the quantum pays the honest span
        // (wall time since the teleport) and parks the anchors at its virtual
        // end; whatever the display showed across the open stretch is now
        // committed verbatim.
        std::thread::sleep(std::time::Duration::from_millis(50));
        c.tick(std::time::Duration::from_millis(50), &doc);
        // The visible cursor right now — the pinned reading every expected
        // target below derives from (same-instant getter + same projection
        // model as the press's base). Register the read through the honest
        // display-frontier books FIRST, exactly like production: the paint
        // pass shows the value and the projection registers what was shown
        // at that instant; the press then measures relative to the frontier.
        // Taking a raw snapshot instead would leave microseconds of lived,
        // displayed-but-unregistered stretch out of the base and land the
        // next press a step short of the screen ("left: 100, right: 300").
        let shown = c.current_frame_in(&doc);
        // A forward press from there must move exactly one configured step;
        // the session ceiling clamps it to EOF.
        c.skip(&doc, 1);
        assert_eq!(c.current_frame(), (shown + 1000).min(doc.total_frames()));
        assert_eq!(c.current_frame_in(&doc), (shown + 1000).min(doc.total_frames()));
        // And the backward press lands exactly one step short of where the
        // user saw the cursor — the frame displayed right before the press,
        // NOT the bare stored commit's pre-jump value ("left: 100, right:
        // 300" / "left: 0, right: 100": both wrong bases land a step away
        // from the screen).
        // FRESHNESS WINDOW (§5.3): the forward press parked the cursor on
        // the exclusive ceiling `end` and the display shows it VERBATIM for
        // as long as the commit is younger than one poll interval — exactly
        // what the user sees between two paints. The pinned reading must be
        // taken inside that window (production presses on the same beat
        // that painted), so no real-time wait may slip in between the press
        // and the read; after the window ages out the running display falls
        // back to the half-open ceiling `end - 1` like any ordinary paint.
        // CADENCE (§1.2): production presses on the same poll beat that
        // painted the cursor — take the pinned reading and press without
        // letting the freshness window age out between the two.
        let shown2 = c.current_frame_in(&doc);
        assert!(shown2 >= doc.total_frames().saturating_sub(1));
        // THE PRESS IS RELATIVE TO `shown2` — NOT to the bare stored
        // commit. The forward press parked the cursor at the exclusive
        // ceiling EOF; while that commit is younger than one poll interval
        // the display shows it VERBATIM (the user sees 1000), but the raw
        // half-open projection tops out at `end - 1`. Measuring the step
        // off the bare commit instead of the visible screen used to land
        // every backward press a frame away from what was just painted
        // (regression "left: 999, right: 1000" / "left: 100, right: 300").
        c.skip(&doc, -1);
        // shown2 - one 1 s step. The session is [100, EOF] — the honest
        // landing sits INSIDE the range, so no clamp may shave it beyond
        // the relative maths itself.
        let landed = c.current_frame();
        assert_eq!(landed, shown2.saturating_sub(1000).max(c.start_frame()));
        assert_eq!(c.current_frame_in(&doc), landed);
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
        let t0 = std::time::Instant::now();
        let mut last = t0;
        for _ in 0..30 {
            std::thread::sleep(std::time::Duration::from_millis(4));
            // The poll's quantum must cover the WHOLE span since the previous
            // commit: sleeping first and stamping `last` only after `tick`
            // leaves the gap between the two instants uncounted, which used
            // to drift the fallback clock behind real time ("pos=8 wall=4").
            let now = std::time::Instant::now();
            let dt = now - last;
            c.tick(dt, &doc);
            // `last` is stamped at the tick's ENTRY instant: the honest
            // span of the NEXT poll must include the microseconds the tick
            // itself spends inside the clock — measuring from the exit
            // would silently drop that stretch every lap and the fallback
            // clock would lag real time ("pos=8 wall=4").
            last = now;
            let p = c.current_frame_in(&doc);
            assert!(p < 10, "loop position {p} escaped [0,10)");
            // CUMULATIVE accounting against a single source of truth: the
            // display corresponds to `t0 + total_wall` folded into the
            // 10-frame loop span. Comparing per-poll readings instead was
            // skew-by-design — the getter read happens AFTER the tick paid
            // its quantum, so it legitimately shows one open interval ahead
            // of the `dt` sampled before the tick (that stale comparison
            // reported "pos=8 wall=4" while the books were honest).
            let total_wall = std::time::Instant::now().duration_since(t0).as_secs_f64();
            let wall_total_frames = (total_wall * 1000.0) as u64;
            let expected = wall_total_frames % 10;
            let diff = (p as i64 - expected as i64).abs();
            assert!(
                diff <= 3 || diff >= 10 - 3,
                "fallback clock drifted: pos={p} expected={expected} wall_frames={wall_total_frames}"
            );
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
