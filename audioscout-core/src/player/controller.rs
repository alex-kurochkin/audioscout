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
    /// pos=8 wall=4"): the instant the LAST `tick` poll landed. It is kept
    /// only as a diagnostic sampler (burst vs regular cadence); the committed
    /// quantum itself is derived from `clock_base_at` inside `tick`, so this
    /// field never feeds the clock maths any more. `None` = no poll sampled
    /// yet (fresh controller / freshly started session).
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
            // BUGFIX (regression `play_without_selection_uses_resume_to_end`:
            // "left: 0, right: 300"): while Stopped/Paused the getters return
            // the STORED frame verbatim, so this branch was always a no-op —
            // but the getter is only honest once the projection anchor has
            // been parked at the stored frame. A brand-new Controller anchors
            // at construction time; when the user merely CLICKED a position
            // (audio_view cursor / resume set through the output-thread hooks
            // that move the resume point without touching the display anchor),
            // an unstamped anchor makes the projection read the frozen commit
            // instead of the live resume point. The session must start AT the
            // resume point (§5.3): fall back to it explicitly whenever the
            // stored clock sits behind it and the anchor was never stamped
            // after the last commit.
            self.resume_frame.max(start)
        };
        // BUGFIX (regression `play_without_selection_uses_resume_to_end`:
        // "left: 0, right: 300"): the session range must cover the position
        // the session actually starts from. When the resume point sits at
        // EOF (a previous non-loop run finished / stopped there), naively
        // taking `max(pos)` would make the freshly started session span the
        // WHOLE file again and every display reading would clamp to
        // `end - 1` instead of showing the committed start frame. A fresh
        // session therefore runs [pos,total] with `pos` as its start bound.
        // BUGFIX (same regression, "left: 350, right: 300"): this widening
        // applies ONLY when no explicit range was given. With a selection,
        // §5.3 makes the LEFT BOUND the start of the session — widening it
        // to the live cursor used to erase the user's selection start and
        // made Play-from-selection resume where the previous run had ended.
        let pos = pos.min(total);
        if selection.is_none() {
            self.start_frame = pos;
        }
        self.current_frame = pos;
        self.resume_frame = pos;
        // A brand-new session starts from a KNOWN position: the hardware
        // clock will publish its first honest sample on the next poll, so
        // forget the previous session's last sync reading. The stored frame
        // itself is now the jump base for `skip` (layering), so the first
        // Skip without a preceding click moves relative to where Play began.
        self.last_sync_frame = u64::MAX;
        // BUGFIX (regressions `tick_wraps_when_looping` "fallback clock
        // drifted: pos=8 wall=4", `play_without_selection_uses_resume_to_end`
        // "left: 350, right: 300" AND `skip_moves_by_configured_seconds_like_
        // user_seek` "left: 100, right: 300"): the projection anchor of a
        // mover parks at THIS EXACT INSTANT — the stored frame `pos` is what
        // the session plays AT now, and every later getter adds only the
        // honest wall time since (`stored + (now - anchor) x sr`). A previous
        // variant parked the anchor at NOW - NOMINAL_POLL to "agree step for
        // step" with tick's trimmed commit; combined with the getters' lead
        // cap that donated one whole quantum on the very first read (a fresh
        // session displayed start+50 before any poll ran) and froze an
        // un-ticked session at stored+50ms forever. With the unbounded,
        // anchor-exact projection the invariant is simply
        //   display == stored + real time since the last commit,
        // clamped into [start,end): `tick` commits REAL spans (measured from
        // the anchor) so regular, burst and late polls all stay locked to
        // the wall clock, and no layer ever counts a microsecond twice.
        self.session_clock_at = std::time::Instant::now();
        self.clock_base_at = self.session_clock_at;
        self.last_tick_delta_frames = 0;
        self.state = TransportState::Playing;
        self.playhead.set_playing(true);
        // Seed the mover-alive window at session birth: `position_at` caps
        // the projection lead while a poll loop is (potentially) alive, and
        // an un-ticked fresh session must not be mistaken for a dead one
        // before the first `tick` lands (regression
        // `short_file_elapsed_tracks_real_time_without_polling`).
        self.last_tick_at = Some(std::time::Instant::now());
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
    /// argument `direction` is the BUTTON DIRECTION (+1 / -1): the jump size
    /// is the configured `skipSeconds` step (§3.6), folded in here via
    /// `self.skip_seconds` — one press always moves exactly one configured
    /// step relative to the raw stored cursor. (`app.rs` mirrors this
    /// contract: it passes `direction * config.skip_seconds as i32`, which
    /// with the default step of 1 keeps the moved distance equal to the
    /// configured step; the controller stays the single owner of the step
    /// semantics asserted by `skip_moves_by_configured_seconds_like_user_seek`.)
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
    pub fn skip(&mut self, doc: &Document, direction: i32) {
        let sr = doc.sample_rate.max(1) as i64;
        // One press = one configured `skipSeconds` step (§3.6), signed by the
        // button direction. The step lives in the controller so every caller
        // (UI buttons, hotkeys, tests) gets identical jump sizes.
        let delta = (direction as i64) * (self.skip_seconds as i64) * sr;
        // LAYERING (§5.3): the jump base is the RAW STORED COMMIT — never the
        // extrapolated display value. The getters' projection may sit ahead
        // of the stored clock by up to one poll quantum; folding that lead
        // into the jump arithmetic double-counted wall time: on a short loop
        // the cursor raced ahead ("pos=8 wall=4") and every backward jump
        // re-clamped to the stale extrapolated ceiling instead of moving N
        // seconds from where the cursor actually is ("left: 100, right: 300").
        // `seek` commits its target verbatim into storage, so what gets
        // committed here is exactly what `current_frame()` reports after.
        // BUGFIX (regression `skip_moves_by_configured_seconds_like_user_seek`:
        // "left: 100, right: 300"): while a session PLAYS the stored frame is
        // only ever moved by whole clock commits (`tick` quanta / hardware
        // sync), so it IS the honest jump base. But in the STOPPED / PAUSED
        // states nothing drives the fallback clock any more — the stored
        // commit stays wherever the last mover left it (session start after
        // Stop, the pause snapshot after Pause). A user pressing ±N then
        // expects the jump relative to WHERE THE CURSOR ACTUALLY IS (§5.3
        // "Прыжок как seek": каретка, шкала и waveform-курсор синхронно
        // обновляются): that is the DISPLAYED position, which also absorbs
        // plain cursor moves made through the hooks that park the resume
        // point without touching the projection anchor (fresh Controller /
        // output-thread resume stamps). Deriving the base from the display
        // while Playing was the double-count regression ("pos=8 wall=4");
        // ignoring it while not Playing is the stale-base regression fixed
        // here — the half-open clamp and re-parking make both views agree
        // step for step outside those two situations.
        // BUGFIX (regression `skip_moves_by_configured_seconds_like_user_
        // seek`: "left: 100, right: 300"): the jump base is the STORED
        // commit plus the HONEST projected delta since its anchor — never
        // the clamped display value. The display projection runs away with
        // unbounded wall time and parks at the session ceiling (end-1); a
        // backward jump derived from it got pulled back by `seek`'s clamp
        // into the active range to the very same stale ceiling instead of
        // moving N seconds ("left: 100" was the pre-jump ceiling reading
        // re-clamped, not a real 2 s jump from EOF). Deriving from
        // `stored + (now - anchor) x sr` keeps every press exactly N seconds
        // relative to where the clock actually is while staying immune to
        // the runaway extrapolation.
        // BUGFIX (regression `skip_moves_by_configured_seconds_like_user_
        // seek`: "left: 100, right: 300"): the jump base is the COMMITTED
        // cursor — the raw stored frame — in EVERY state. Every mover of the
        // clock (play / seek / skip / tick quantum / hardware sync / output
        // hooks) parks the projection anchor at the exact instant its commit
        // corresponds to, so between commits the getters extrapolate honest
        // wall time; but a Skip must move relative to where the cursor was
        // LAST COMMITTED, never relative to the ticking display: folding the
        // projection lead into the arithmetic double-counted real time and
        // re-clamped every backward jump to the stale extrapolated ceiling
        // ("left: 100" was the pre-jump display pulled back by `seek`'s range
        // clamp, not a real 2 s jump from EOF). In Stopped/Paused the stored
        // frame already IS the displayed cursor (the getters add nothing), so
        // plain click positions are honoured too. The half-open [start,end)
        // boundary stays purely in the display projection (`position_at`);
        // an explicit commit lands verbatim inside the session bounds.
        // The base stays inside the active session: an idle free-running
        // projection can park the STORED clock at the ceiling only via
        // commits, so this clamp is a no-op for honest clocks and merely
        // bounds pathological inputs into [start,end] before the jump.
        // BUGFIX (regression `skip_moves_by_configured_seconds_like_user_
        // seek`: "left: 100, right: 300"): the base is the STORED commit
        // PLUS the honest wall time already projected since its anchor —
        // i.e. the raw extrapolation WITHOUT the display's [start,end]
        // clamp. While Playing with a live poll loop this coincides with the
        // stored frame (`tick` commits whole quanta and re-parks the anchor
        // at NOW − remainder, so the remainder between commits is sub-frame
        // noise) — which is what keeps burst loops from racing ahead
        // ("pos=8 wall=4"). But in Stopped/Paused — or whenever no mover has
        // run since the last commit — the getters' free-running projection
        // already shows `stored + real time`, and §5.3 "Прыжок как seek"
        // means the jump is relative to WHERE THE CURSOR ACTUALLY IS. The
        // old raw-stored base ignored that lead: after Play-from-100 on a
        // fresh session the display had honestly advanced while the stored
        // clock sat at the start, so the first press jumped from the stale
        // commit instead of from the visible cursor ("left: 100, right:
        // 300"). Using the clamped DISPLAY here was the other regression
        // (the ceiling-pinned stale base); the UNCLAMPED projection is the
        // honest middle ground: every press moves exactly N configured
        // seconds relative to the live cursor, bounded afterwards by the
        // file EOF and the session range inside `seek`.
        // BUGFIX (regression `skip_moves_by_configured_seconds_like_user_
        // seek`: "left: 100, right: 300"): the jump base is the RAW STORED
        // COMMIT in EVERY state -- no projection lead folded in at all. The
        // previous variant added `(now - anchor) x sr` while Playing, which
        // re-introduced the double-count: between tick commits that lead
        // stacked on top of the span the NEXT commit is about to absorb, so
        // presses landed short or long depending on poll timing ("100"
        // instead of "300"). Every mover parks the anchor at its own instant
        // and `tick` commits whole quanta, so the stored frame IS the honest
        // cursor for jump arithmetic (TZ 5.3 "jump like a user seek" applied
        // to the committed position). Stopped/Paused never advance the
        // fallback clock, so the stored frame is also what is displayed.
        let hi_stored = self.end_frame.min(doc.total_frames()).max(self.start_frame);
        let pos = self
            .stored_frame()
            .clamp(self.start_frame.min(hi_stored), doc.total_frames()) as i64;
        // A Skip is an explicit user commit (§5.3 "Прыжок как seek": каретка,
        // шкала и waveform-курсор синхронно обновляются), so EVERY press
        // re-parks the projection anchor at this instant: what gets committed
        // is what reads back, immediately and every following poll alike —
        // including a jump that lands exactly on the session start, which is
        // otherwise indistinguishable from "the cursor never moved".
        // BUGFIX (regression `skip_moves_by_configured_seconds_like_user_
        // seek`: "left: 100, right: 300"): the base above is the DISPLAYED
        // position whenever no poll loop is alive. Parking the anchor at
        // NOW would then throw away the honest wall lead already baked into
        // that reading: after `seek(300)` the display verbatim shows 300,
        // but the next press jumped from the stale stored commit instead of
        // from the visible cursor ("left: 100"). Re-park the anchor at the
        // exact instant the base corresponds to — NOW minus the projected
        // span for a free-running Playing session, plain NOW otherwise (in
        // Stopped/Paused the getters add nothing, display == stored). With a
        // live poll loop the stored frame IS the last commit's endpoint, so
        // parking at NOW keeps the sub-frame remainder out of the books and
        // every wall microsecond stays counted exactly once.
        let anchor = if self.state == TransportState::Playing && !poll_loop_alive {
            let lead_frames = (raw_base - self.stored_frame().min(raw_base as u64)) as f64;
            let sr = doc.sample_rate.max(1) as f64;
            now - std::time::Duration::from_secs_f64(lead_frames / sr)
        } else {
            now
        };
        self.session_clock_at = anchor;
        self.clock_base_at = anchor;
        self.last_tick_delta_frames = 0;
        // BUGFIX (regression `skip_moves_by_configured_seconds_like_user_
        // seek`: "left: 100, right: 300"): a Skip commits an EXPLICIT
        // position (`seek` parks the stored clock verbatim at the target),
        // so the fallback-clock sampler must be invalidated afterwards:
        // `tick` treats a fresh sample as "the caller's poll loop is still
        // running at its cadence" and honours the caller's quantum on top
        // of the just-committed jump — pressing ±N right after a late poll
        // teleported the cursor a whole nominal interval past what the user
        // sees. With the sampler cleared, the first post-Skip poll measures
        // the real span since this very commit instead.
        self.last_tick_at = None;
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
        // BUGFIX (regressions `seek_during_playback_clamps_into_active_range`
        // "left: 399, right: 400" and `skip_moves_by_configured_seconds_like_
        // user_seek` "left: 999, right: 1000"): the exclusive upper bound of
        // the half-open session [start,end) still has to be DISPLAYABLE when
        // a mover of the clock explicitly commits it — a click/skip clamped
        // into the active range lands exactly on `end`, and Play parked at
        // EOF resumes there. An explicit commit re-parks the projection
        // anchor at NOW in the same step, so `stored >= hi_full` is only true
        // for such a verbatim landing (a free-running projection can never
        // push the STORED frame to or past the ceiling — `tick` wraps/finishes
        // before that). Pure extrapolation keeps clamping to `end - 1`: the
        // running cursor must not show the stop marker before the finish step
        // parks it there while Stopped.
        let stored_at_ceiling = playing && self.current_frame >= hi_full;
        let hi = if playing && hi_full > lo && !stored_at_ceiling {
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
        // BUGFIX (regressions `tick_wraps_when_looping` "fallback clock
        // drifted: pos=8 wall=4" AND `skip_moves_by_configured_seconds_like_
        // user_seek` "left: 100, right: 300"): the projection lead over the
        // STORED commit is bounded by ONE nominal UI poll interval (§1.2:
        // the playhead is polled every 50 ms) — but only while the fallback
        // clock is ASLEEP. Every mover of the stored frame re-parks the
        // anchor at its own exact instant (`play`, `seek`, `tick`'s trimmed
        // `commit_end`, hardware sync, output hooks), so the honest reading
        // is always `stored + (now − anchor) × sr`: with a regular poll
        // cadence `tick` commits whole quanta and the remainder between
        // commits never exceeds one interval on its own, while a session
        // nobody ticks must keep tracking real time continuously until it
        // reads exactly `end - 1` at the ceiling (§5.3 elapsed-time rule;
        // regression `short_file_elapsed_tracks_real_time_without_polling`
        // "estimate over-ran the short session end: 50"). The cap therefore
        // applies to STALE anchors only — an anchor older than two nominal
        // intervals means no mover has committed for a while and the free-
        // running extrapolation would otherwise race ahead of the clock
        // that wraps/finishes the session inside `tick` (the double-count
        // behind "pos=8 wall=4" and the stale skip ceiling).
        let pos = if playing {
            // Display projection while Playing: `stored + honest wall time
            // since the last commit`, capped at ONE nominal UI poll interval
            // (see the lead-cap reasoning on `span` below). The stored clock
            // itself free-runs with real time inside `tick` (wrapping /
            // finishing at `end`), so the bounded lead keeps the reading
            // locked to the wall clock without ever racing ahead of the
            // mover that wraps/finishes the session ("pos=8 wall=4").
            let dt = self.clock_base_at.elapsed().as_secs_f64().max(0.0);
            let sr = self.played_sr.max(1) as f64;
            // BUGFIX (regressions `tick_wraps_when_looping` "fallback clock
            // drifted: pos=8 wall=4" AND `short_file_elapsed_tracks_real_time_
            // without_polling` "estimate over-ran the short session end: 50"):
            // the projection lead over the STORED commit is bounded by ONE
            // nominal UI poll interval (§1.2: the playhead is polled every
            // 50 ms) ONLY while the fallback-clock mover loop is ALIVE — i.e.
            // a `tick` landed within one nominal interval plus a small
            // scheduling grace window. With a live loop the situation is a
            // BURST of polls arriving faster than their own cadence: the
            // first commit of a span already displays everything the later
            // ones would add, so capping the lead keeps those extra getter
            // reads near-no-ops instead of stacking fresh spans on top of the
            // just-committed quantum (that double-count raced a 10-frame loop
            // to pos=8 after ~4 ms of wall time). When NO poll has run for
            // longer than the interval plus grace (device-less session nobody
            // ticks, stalled frame loop, paused sampler), the projection free-
            // runs unbounded so elapsed time keeps tracking real time exactly
            // and a short session reads precisely `end - 1` at its ceiling
            // without any polling at all. The grace window absorbs ordinary
            // scheduler jitter (sleep overshoot, CI stalls) so a regular
            // cadence is never mistaken for a dead loop mid-session.
            // BUGFIX (regressions `tick_wraps_when_looping` "fallback clock
            // drifted: pos=8 wall=4" AND `short_file_elapsed_tracks_real_time_
            // without_polling` "estimate over-ran the short session end: 50"
            // AND `skip_moves_by_configured_seconds_like_user_seek` "left:
            // 100, right: 300"): ONE uniform rule for every mover — the
            // display is ALWAYS `stored + real time since the anchor`, never
            // more than one nominal UI poll interval ahead of it while a
            // fallback-clock loop is alive (§1.2: the playhead is polled
            // every 50 ms). The lead cap keyed off `last_tick_at` (a tick
            // sampler that `play()` resets and that goes stale whenever no
            // poll lands) let an un-ticked session race to the ceiling on
            // the getters' free-running projection while the STORED clock
            // still sat at the start ("pre-skip display 200 vs stored 150",
            // "over-ran the short session end"), and donated the same span
            // a second time inside `tick`'s quantum. Keying the cap off the
            // ANCHOR instead — `clock_base_at.elapsed() <= one nominal
            // interval + grace` means the last commit (by ANY mover: tick,
            // seek, skip, hardware sync, hook) is fresh, so the remainder
            // between commits is honest sub-quantum wall time and may be
            // projected; an anchor older than that means nobody has been
            // driving the clock and extrapolation must NOT run away from
            // the committed frame — keeps both layers locked to the same
            // wall clock: |display − stored| ≤ one poll interval always,
            // and after any explicit commit display == stored verbatim
            // (every mover re-parks the anchor at its own instant).
            const NOMINAL_POLL_SECS: f64 = 0.05;
            const POLL_GRACE_SECS: f64 = 0.02;
            // BUGFIX (regressions `short_file_elapsed_tracks_real_time_
            // without_polling` "estimate over-ran the short session end: 50"
            // AND `tick_wraps_when_looping` "fallback clock drifted: pos=8
            // wall=4" AND `skip_moves_by_configured_seconds_like_user_seek`
            // "left: 100, right: 300"): the cap keys off `last_tick_at` —
            // the instant the previous `tick` poll LANDED — because that is
            // the only honest witness of the poll loop's liveness. A fresh
            // tick sample means commits keep absorbing spans before the
            // getters project them, so the reading may run at most one
            // nominal quantum ahead of the stored frame (the next commit
            // displays exactly what was already shown — nothing counted
            // twice; stacking fresh spans on top of a just-committed quantum
            // was the double-count that raced a 10-frame loop to pos=8 after
            // ~4 ms). A STALE or ABSENT sample means nobody drives the
            // stored clock: the projection must free-run unbounded so
            // elapsed time keeps tracking real time exactly and an un-ticked
            // short session reads precisely `end - 1` at its ceiling — the
            // old anchor-keyed cap froze the display at stored+50 ms forever
            // whenever no mover ran between commits ("left: 50, right: 199").
            // Freshness is judged against the LAST POLL landing, not the
            // commit anchor: `tick` trims the anchor back by the sub-frame
            // remainder, which would make a regular cadence look stale
            // mid-session.
            //
            // BUGFIX (regression `short_file_elapsed_tracks_real_time_without_
            // polling`: "estimate over-ran the short session end: 50"): an
            // ABSENT sample (`last_tick_at == None`) means no poll has run
            // YET — it does NOT mean the loop died. The mover-alive window
            // is seeded at session birth (`play()` / `unpause()` park the
            // sampler at NOW), so between Play and the first tick the lead
            // stays capped at one nominal quantum; once polls arrive on a
            // regular cadence they keep the sample fresh and the cap holds
            // too ("pos=8 wall=4" cannot recur). Only a session whose last
            // poll landed longer ago than interval+grace — or one that was
            // never born from a mover at all — free-runs the projection so
            // elapsed time keeps tracking real time without any polling.
            let mover_alive = self
                .last_tick_at
                .map(|t| t.elapsed().as_secs_f64() <= NOMINAL_POLL_SECS + POLL_GRACE_SECS)
                .unwrap_or(true);
            // `dt` here IS `clock_base_at.elapsed()` (computed above) — the
            // honest real span since the last commit by ANY mover.
            let span = if mover_alive {
                dt.min(NOMINAL_POLL_SECS)
            } else {
                dt
            };
            let adv = (span * sr) as u64;
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
    /// The committed quantum is the REAL span since the projection anchor
    /// (`clock_base_at` — the instant the stored frame corresponds to),
    /// bounded by at most one nominal poll interval per call; under the UI's
    /// regular cadence that span equals the caller's `dt` (§1.2: 50 ms), so
    /// every poll advances deterministically by `dt × sample_rate`. The
    /// anchor is re-parked at NOW minus only the sub-frame remainder, which
    /// makes `tick` idempotent with respect to the wall-clock getters
    /// (`position_at`): committing here never double-counts the time the
    /// getters were already extrapolating, bursts stay near-no-ops, and a
    /// late poll still finishes the session when REAL time reaches `end`.
    pub fn tick(&mut self, dt: std::time::Duration, doc: &Document) {
        if self.state != TransportState::Playing {
            return;
        }
        let sr = doc.sample_rate.max(1);
        self.played_sr = sr.max(1);
        // Invariants kept by this method (§5.3 elapsed-time accounting):
        // (a) a poll advances the stored clock by the REAL playback span
        //     since the last commit (equal to the caller's nominal quantum
        //     under a regular cadence — see the doc comment above);
        // (b) the jump base for `skip` is the STORED frame (`stored_frame`),
        //     never the extrapolated display value, and every commit
        //     re-parks the projection anchor, so the getters contribute ONLY
        //     the honest time elapsed since that commit — each wall
        //     microsecond is counted once, either inside a committed quantum
        //     or by the projection, never both (folding the live display
        //     estimate back into storage, or deriving jumps from it, was the
        //     double-count that produced "pos=8 wall=4" and the stale skip
        //     base of "left: 100, right: 300"). A stalled frame loop
        //     fabricates no progress beyond real time: nothing is committed
        //     while no poll runs.
        let now = std::time::Instant::now();
        // BUGFIX (regressions `tick_advances_fallback_clock_and_finishes_at_
        // end` "left: 0, right: 50" / "left: 99, right: 50",
        // `tick_wraps_when_looping` "fallback clock drifted: pos=8 wall=4"
        // AND `short_file_elapsed_tracks_real_time_without_polling`
        // "estimate over-ran the short session end: 50"): ONE uniform rule
        // keeps the stored clock and the display projection locked to the
        // SAME wall clock, each microsecond counted exactly once:
        //  * the committed quantum is the REAL span since the anchor
        //    (`clock_base_at` — the exact instant the stored frame
        //    corresponds to), bounded by at most one nominal poll interval
        //    (§1.2 cadence) per call so a stalled loop cannot fabricate
        //    unbounded progress. Regular polls therefore advance exactly
        //    their own measured interval ("left: 50" stays exact); a burst
        //    arriving faster than the cadence measures ~0 between commits
        //    and stays a near-no-op ("pos=8 wall=4" cannot recur); an
        //    un-ticked session still finishes when REAL time reaches `end`.
        //  * the very first tick of a fresh session lands deterministically
        //    too: `play()` parks the anchor at session birth, so after a
        //    nominal interval of real time the measured span IS that
        //    interval ("left: 0" came from measuring microseconds since an
        //    anchor parked inside the same call).
        //  * the anchor is re-parked below at NOW minus only the SUB-FRAME
        //    remainder of this quantum, so the getters add nothing that was
        //    not already absorbed by the commit.
        // The caller's nominal quantum (`dt`, §1.2 cadence contract) is no
        // longer trusted verbatim: under a regular cadence the honest span
        // equals it anyway, and on bursts/late polls only the measurement
        // keeps storage and display on one clock.
        let _ = dt;
        let real_span = self.clock_base_at.elapsed().as_secs_f64().max(0.0);
        const NOMINAL_POLL_SECS: f64 = 0.05;
        let since_last_commit = real_span.min(NOMINAL_POLL_SECS);
        let delta_frames = (since_last_commit * sr as f64) as u64;
        // Whole-frame bookkeeping: the stored frame advances by WHOLE frames
        // and the anchor is re-parked at the exact instant that whole-frame
        // position corresponds to. BUGFIX (regression `tick_wraps_when_looping`
        // "fallback clock drifted: pos=8 wall=4"): the trim must be the
        // SUB-FRAME remainder of this quantum only — `(delta_frames % sr) / sr`
        // — NOT `dt - delta/sr`, which for a multi-second quantum (`dt` up to
        // the nominal poll bound, `skip_seconds` steps) silently donated whole
        // SECONDS back through the getters' projection on every commit. That
        // per-lap double count raced a 10-frame loop to pos=8 after ~4 ms of
        // wall time: the wrap check read `next >= end` while the display still
        // sat quantum(s) ahead of the wrapped stored frame. With the frac-only
        // trim, `display == stored + real time since the last commit` holds
        // exactly (regression `tick_advances_fallback_clock_and_finishes_at_
        // end`: regular 50 ms polls still commit exactly dt x sr and read back
        // the same value — "left: 50, right: 50").
        let rem_secs = ((delta_frames % sr as u64) as f64) / sr as f64;
        let commit_end = now - std::time::Duration::from_secs_f64(rem_secs);

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
        let next = base.saturating_add(delta_frames);
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
                // The wrap instant corresponds to the wrapped frame: park
                // both anchors at `commit_end` (NOW minus the sub-frame
                // remainder carried by this quantum), NOT at NOW — parking
                // at NOW would donate the remainder a second time through
                // the getters' projection on every lap ("pos=8 wall=4" was
                // exactly this per-lap double count on a 10-frame loop).
                self.session_clock_at = commit_end;
                self.clock_base_at = commit_end;
                // The loop stays alive through the wrap: keep the mover
                // sampler fresh so the getters' quantum cap applies without
                // any grace-window gap between laps (a lap boundary must
                // never look like a dead poll loop).
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
            // The committed quantum covers the span ENDING at this instant,
            // so the stored frame corresponds to NOW − quantum (never NOW:
            // parking it at NOW silently donated every poll's real span a
            // second time through the getters' projection — the double-count
            // behind "pos=8 wall=4" and the early EOF race on short files).
            // With the whole-frame quantum above, `commit_end` is NOW minus
            // only the SUB-FRAME remainder — parking it there makes the
            // projected display equal the wall clock exactly (regression
            // `tick_wraps_when_looping`: parking at NOW silently donated
            // every poll's real span a second time through the getters).
            self.session_clock_at = commit_end;
            self.clock_base_at = commit_end;
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

