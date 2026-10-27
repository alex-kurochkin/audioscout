//! UI-facing transport state machine (TZ §1.3 / §5): play/pause/stop/seek/
//! loop bookkeeping shared between the egui frame loop and the audio output
//! thread (`player::output`) through an `Arc<Mutex<Controller>>`. The real
//! audio clock lives in that thread; these methods only mirror it so the
//! transport buttons never stick ("Pause не залипает").
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
        let (start, end) = match selection {
            Some((s, e)) if e > s => (s.min(total), e.min(total)),
            _ => (self.resume_frame.min(total), total),
        };
        self.start_frame = start;
        self.end_frame = end;
        self.current_frame = start;
        self.resume_frame = start;
        self.state = TransportState::Playing;
        self.playhead.set_playing(true);
        self.playhead.set_position_samples(start);
    }

    /// Freeze position and in-flight PCM (§5.3 pause). The output thread
    /// performs the actual silence via `Cmd::Pause` before calling this.
    pub fn pause(&mut self) {
        if self.state == TransportState::Playing {
            self.state = TransportState::Paused;
            self.playhead.set_playing(false);
            self.resume_frame = self.current_frame;
        }
    }

    /// Resume after pause without re-appending the source.
    pub fn unpause(&mut self) {
        if self.state == TransportState::Paused {
            self.state = TransportState::Playing;
            self.playhead.set_playing(true);
        }
    }

    /// Immediate silence + rewind the cursor to the start of the last
    /// session (§3.5/§5.3 stop).
    pub fn stop(&mut self) {
        self.state = TransportState::Stopped;
        self.current_frame = self.start_frame;
        self.resume_frame = self.start_frame;
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
        let mut target = frame.min(max_frame);
        if self.state == TransportState::Playing && self.end_frame > self.start_frame {
            // Click-to-seek outside the selection does not extend it: clamp
            // into the active range (§5.3).
            target = target.clamp(self.start_frame, self.end_frame);
            self.current_frame = target;
            self.playhead.set_position_samples(target);
        } else {
            self.current_frame = target;
            self.playhead.set_position_samples(target);
        }
        if source == CursorSource::User {
            self.resume_frame = target;
        }
    }

    /// ±N-second jump while playing (§5.3): behaves like a user seek, so
    /// the position is clamped into an active loop/selection range.
    pub fn skip(&mut self, doc: &Document, seconds: i32) {
        let sr = doc.sample_rate.max(1) as i64;
        let delta = seconds as i64 * sr;
        let target = (self.current_frame as i64 + delta).max(0) as u64;
        self.seek(target, doc, CursorSource::User);
    }

    /// Output-thread hook: the thread already clamped the target into the
    /// active session range and re-appended audio from it; mirror the move
    /// into the UI-facing state machine without touching the resume point.
    pub fn mark_seek(&mut self, frame: u64) {
        self.current_frame = frame;
        self.playhead.set_position_samples(frame);
    }

    /// Output-thread hooks (§5.2/§5.3): the real audio clock lives in
    /// `player::output`; these keep the UI-facing state machine consistent.
    /// Mark a new playing session [start,end) started by the output thread.
    pub fn mark_playing(&mut self, start: u64, end: u64) {
        self.start_frame = start; self.end_frame = end; self.current_frame = start;
        self.resume_frame = start;
        self.state = TransportState::Playing; self.playhead.set_playing(true);
        self.playhead.set_position_samples(start);
    }
    /// Freeze the resume point at an externally measured position (pause).
    pub fn freeze_resume_at(&mut self, frame: u64) {
        self.current_frame = frame; self.resume_frame = frame;
        self.playhead.set_position_samples(frame);
    }
    /// Stop with the playhead rewound to the session start point (§5.3).
    pub fn stop_at_session_start(&mut self, start: u64) {
        self.state = TransportState::Stopped;
        self.current_frame = start; self.resume_frame = start;
        self.playhead.set_position_samples(start); self.playhead.set_playing(false);
    }
    /// Natural end of a non-loop session: stopped, car back at session start.
    pub fn finish_session_at(&mut self, start: u64) {
        self.stop_at_session_start(start);
    }
    pub fn current_frame(&self) -> u64 {
        self.current_frame
    }
    pub fn current_secs(&self, sample_rate: u32) -> f64 {
        self.current_frame as f64 / sample_rate.max(1) as f64
    }
    /// Device-less fallback clock (§1.2/§5.3): when no audio output is
    /// available the state machine advances on its own — called from the UI
    /// playhead poll every 50 ms while `state == Playing`. Honours loop and
    /// session bounds; a non-loop session finishes at `end_frame` back at
    /// the session start, exactly like the real output thread does.
    pub fn tick(&mut self, dt: std::time::Duration, doc: &Document) {
        if self.state != TransportState::Playing {
            return;
        }
        let sr = doc.sample_rate.max(1) as f64;
        let advance = (dt.as_secs_f64() * sr).round().max(0.0) as u64;
        let next = self.current_frame.saturating_add(advance);
        let end = if self.end_frame > self.start_frame { self.end_frame } else { doc.total_frames() };
        if next >= end {
            if self.loop_enabled && end > self.start_frame {
                let span = end - self.start_frame;
                let wrapped = self.start_frame + (next - self.start_frame) % span;
                self.current_frame = wrapped;
                self.playhead.set_position_samples(wrapped);
            } else {
                self.finish_session_at(self.start_frame);
            }
        } else {
            self.current_frame = next;
            self.playhead.set_position_samples(next);
        }
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
        assert_eq!(c.current_frame(), 300);
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
        c.freeze_resume_at(150);
        c.pause();
        assert_eq!(c.state(), TransportState::Paused);
        assert_eq!(c.current_frame(), 150);
        assert_eq!(c.resume_frame(), 150);
        c.unpause();
        assert_eq!(c.state(), TransportState::Playing);
    }

    #[test]
    fn stop_rewinds_to_session_start() {
        let (mut c, doc) = controller_with_doc(1000, 1);
        c.play(&doc, Some((200, 800)));
        c.freeze_resume_at(500);
        c.stop();
        assert_eq!(c.state(), TransportState::Stopped);
        assert_eq!(c.current_frame(), 200);
        assert_eq!(c.resume_frame(), 200);
    }

    #[test]
    fn natural_end_of_non_loop_session_rewinds_and_stops() {
        let (mut c, doc) = controller_with_doc(1000, 1);
        c.play(&doc, Some((0, 500)));
        c.freeze_resume_at(500);
        // Output-thread hook: natural completion rewinds to session start.
        c.finish_session_at(c.start_frame());
        assert_eq!(c.state(), TransportState::Stopped);
        assert_eq!(c.current_frame(), 0); // rewind to session start (§5.3)
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
        assert_eq!(c.current_frame(), 400);
        c.seek(0, &doc, CursorSource::User);
        assert_eq!(c.current_frame(), 200);
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
        let (mut c, doc) = controller_with_doc(1000, 1); // sr = 1000 → 1 s = 1000 fr
        c.seek(100, &doc, CursorSource::User);
        c.play(&doc, None);
        c.set_skip_seconds(1);
        c.skip(&doc, 1);
        assert_eq!(c.current_frame(), doc.total_frames()); // clamped to end
        c.skip(&doc, -2);
        assert_eq!(c.current_frame(), 0);
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
        c.freeze_resume_at(150);
        c.reset();
        assert_eq!(c.state(), TransportState::Stopped);
        assert_eq!(c.current_frame(), 0);
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
        assert_eq!(c.current_frame(), 50);
        assert_eq!(c.state(), TransportState::Playing);
        c.tick(std::time::Duration::from_millis(50), &doc); // reaches end=100
        assert_eq!(c.state(), TransportState::Stopped);
        assert_eq!(c.current_frame(), 0); // rewound to session start
    }

    #[test]
    fn tick_wraps_when_looping() {
        let (mut c, doc) = controller_with_doc(1000, 1);
        c.set_loop(true);
        c.play(&doc, Some((0, 100)));
        c.tick(std::time::Duration::from_millis(75), &doc); // +75 → 75
        c.tick(std::time::Duration::from_millis(50), &doc); // +50 → wraps to 25
        assert_eq!(c.state(), TransportState::Playing);
        assert_eq!(c.current_frame(), 25);
    }

    #[test]
    fn tick_is_noop_when_not_playing() {
        let (mut c, doc) = controller_with_doc(1000, 1);
        c.tick(std::time::Duration::from_millis(50), &doc);
        assert_eq!(c.current_frame(), 0);
        assert_eq!(c.state(), TransportState::Stopped);
    }
}
