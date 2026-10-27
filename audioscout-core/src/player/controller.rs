//! Transport controller (TZ §1.3 / §5): play/pause/stop/seek/loop on top of
//! the rodio output ([`AudioOutput`]). Owns the resume point and the
//! [`TransportState`] that the UI buttons derive from ("Pause не залипает").
//!
//! All public methods are safe to call before a document is loaded — they
//! degrade to no-ops (TZ §1.2: "нажатия раньше времени не падают").

use super::output::AudioOutput;
use super::{CursorSource, MeterData, PlayheadState, TransportState};
use crate::Document;
use std::sync::Arc;

pub struct Controller {
    state: TransportState,
    loop_enabled: bool,
    volume: f32,
    start_frame: u64,
    end_frame: u64,
    current_frame: u64,
    resume_frame: u64,
    playhead: Arc<PlayheadState>,
    output: AudioOutput,
    space_is_stop: bool,
    skip_seconds: u32,
}

impl Controller {
    pub fn new(playhead: Arc<PlayheadState>) -> Self {
        Self::with_output(playhead, AudioOutput::new())
    }

    /// Testable constructor: inject a pre-built output (e.g. the simulated
    /// one used by unit tests) instead of opening the default device.
    pub fn with_output(playhead: Arc<PlayheadState>, output: AudioOutput) -> Self {
        Self {
            state: TransportState::Stopped,
            loop_enabled: false,
            volume: 0.8,
            start_frame: 0,
            end_frame: 0,
            current_frame: 0,
            resume_frame: 0,
            playhead,
            output,
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
    /// Live volume change without restarting playback (§3.5).
    pub fn set_volume(&mut self, vol: f32) {
        self.volume = vol.clamp(0.0, 1.0);
        self.output.set_volume(self.volume);
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
        self.output.play_slice(&doc.samples, doc.channels, doc.sample_rate, start, end);
    }

    /// Freeze position and in-flight PCM (§5.3 pause).
    pub fn pause(&mut self) {
        if self.state == TransportState::Playing {
            self.state = TransportState::Paused;
            self.playhead.set_playing(false);
            self.sync_from_output(None);
            self.resume_frame = self.current_frame;
            self.output.pause();
        }
    }

    /// Resume after pause without re-appending the source.
    pub fn unpause(&mut self) {
        if self.state == TransportState::Paused {
            self.state = TransportState::Playing;
            self.playhead.set_playing(true);
            self.output.resume();
        }
    }

    /// Immediate silence + rewind the cursor to the start of the last
    /// session (§3.5/§5.3 stop).
    pub fn stop(&mut self) {
        self.output.stop();
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
            self.output.play_slice(&doc.samples, doc.channels, doc.sample_rate, target, self.end_frame);
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

    /// Pull the audio thread's counters into the controller state. Called
    /// from the 50 ms playhead poll (§1.2/§5.3). Returns true when a
    /// transport transition happened this tick.
    pub fn poll(&mut self, doc: &Document) -> bool {
        if self.state != TransportState::Playing {
            return false;
        }
        self.sync_from_output(Some(doc));
        if self.output.take_finished() {
            if self.loop_enabled {
                // Live retarget: re-append the same [start,end) slice
                // without recreating the stream — no glitch restart (§3.5).
                self.current_frame = self.start_frame;
                self.playhead.set_position_samples(self.current_frame);
                self.output.play_slice(&doc.samples, doc.channels, doc.sample_rate, self.start_frame, self.end_frame);
                return true;
            }
            // Natural end of a non-loop session: rewind to the session
            // start point (§5.3).
            self.state = TransportState::Stopped;
            self.current_frame = self.start_frame;
            self.resume_frame = self.start_frame;
            self.playhead.set_position_samples(self.current_frame);
            self.playhead.set_playing(false);
            return true;
        }
        false
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
        self.meter.reset();
    }
    /// Natural end of a non-loop session: stopped, car back at session start.
    pub fn finish_session_at(&mut self, start: u64) {
        self.stop_at_session_start(start);
    }
    pub fn current_frame(&self) -> u64 { self.current_frame }
    pub fn current_secs(&self, sample_rate: u32) -> f64 { self.current_frame as f64 / sample_rate as f64 }
    pub fn resume_frame(&self) -> u64 { self.resume_frame }
    pub fn reset(&mut self) {
        self.state = TransportState::Stopped; self.start_frame = 0; self.end_frame = 0;
        self.current_frame = 0; self.resume_frame = 0; self.playhead.set_position_samples(0);
        self.playhead.set_playing(false); self.meter.reset();
    }

    pub fn current_frame(&self) -> u64 {
        self.current_frame
    }
    pub fn current_secs(&self, sample_rate: u32) -> f64 {
        self.current_frame as f64 / sample_rate.max(1) as f64
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

    /// Full reset for loading a different file (§5.1): stop output, clear
    /// the resume point and session bounds.
    pub fn reset(&mut self) {
        self.output.stop();
        self.state = TransportState::Stopped;
        self.start_frame = 0;
        self.end_frame = 0;
        self.current_frame = 0;
        self.resume_frame = 0;
        self.playhead.set_position_samples(0);
        self.playhead.set_playing(false);
    }

    /// Consume the latest metering block from the audio path (§3.5.1).
    pub fn take_meter_data(&self) -> MeterData {
        self.output.take_meter_data()
    }

    #[cfg(test)]
    pub fn output_mut(&mut self) -> &mut AudioOutput {
        &mut self.output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::player::{CursorSource, PlayheadState, TransportState};
    use crate::{Document, SampleType};
    use std::path::PathBuf;

    /// Deterministic controller wired to the simulated output (no device).
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
        let c = Controller::with_output(std::sync::Arc::new(PlayheadState::new()), AudioOutput::new());
        (c, doc)
    }

    #[test]
    fn calls_before_load_are_safe_noops() {
        // TZ §1.2: "нажатия раньше времени не падают".
        let empty = Document::new();
        let mut c = Controller::with_output(std::sync::Arc::new(PlayheadState::new()), AudioOutput::new());
        c.play(&empty, None);
        assert_eq!(c.state(), TransportState::Stopped);
        c.pause();
        c.seek(100, &empty, CursorSource::User);
        c.skip(&empty, 5);
        assert!(!c.poll(&empty));
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
    fn pause_freezes_position_and_unpause_resumes_it() {
        let (mut c, doc) = controller_with_doc(1000, 1);
        c.play(&doc, None);
        c.output_mut().advance(150);
        c.pause();
        assert_eq!(c.state(), TransportState::Paused);
        let frozen = c.current_frame();
        assert_eq!(frozen, 150);
        // Frozen in-flight PCM does not advance while paused.
        assert_eq!(c.output_mut().advance(100), 0);
        assert_eq!(c.current_frame(), frozen);
        c.unpause();
        assert_eq!(c.state(), TransportState::Playing);
        c.output_mut().advance(50);
        c.poll(&doc);
        assert_eq!(c.current_frame(), 200);
    }

    #[test]
    fn stop_rewinds_to_session_start() {
        let (mut c, doc) = controller_with_doc(1000, 1);
        c.play(&doc, Some((200, 800)));
        c.output_mut().advance(300);
        c.poll(&doc);
        c.stop();
        assert_eq!(c.state(), TransportState::Stopped);
        assert_eq!(c.current_frame(), 200);
        assert_eq!(c.resume_frame(), 200);
    }

    #[test]
    fn natural_end_of_non_loop_session_rewinds_and_stops() {
        let (mut c, doc) = controller_with_doc(1000, 1);
        c.play(&doc, Some((0, 500)));
        assert!(c.output_mut().advance(500) > 0);
        assert!(c.poll(&doc));
        assert_eq!(c.state(), TransportState::Stopped);
        assert_eq!(c.current_frame(), 0); // rewind to session start (§5.3)
    }

    #[test]
    fn loop_restarts_slice_without_state_change() {
        let (mut c, doc) = controller_with_doc(1000, 1);
        c.set_loop(true);
        c.play(&doc, Some((100, 300)));
        c.output_mut().advance(200);
        assert!(c.poll(&doc)); // finished event consumed
        assert_eq!(c.state(), TransportState::Playing); // still playing
        assert_eq!(c.current_frame(), 100); // live retarget to loop start
        assert_eq!(c.start_frame(), 100);
        assert_eq!(c.end_frame(), 300);
        // The re-appended slice plays on.
        assert!(c.output_mut().advance(50) > 0);
    }

    #[test]
    fn seek_during_playback_clamps_into_active_range() {
        let (mut c, doc) = controller_with_doc(1000, 1);
        c.play(&doc, Some((200, 400)));
        c.output_mut().advance(50);
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
        c.set_volume(0.5); // no panic without an active sink
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
    fn meter_data_reports_peaks_and_rms_per_channel() {
        // Simulated stereo playback: L loud, R quiet.
        let frames = 4;
        let mut samples = Vec::new();
        for i in 0..frames {
            samples.push(if i % 2 == 0 { 0.8 } else { -0.6 }); // L
            samples.push(0.1); // R
        }
        let doc = Document {
            samples,
            sample_rate: 1000,
            channels: 2,
            markers: Vec::new(),
            source_path: PathBuf::from("m.wav"),
            temp_path: None,
            dirty: false,
            sample_type: SampleType::Float32,
        };
        let mut c = Controller::with_output(std::sync::Arc::new(PlayheadState::new()), AudioOutput::new());
        c.play(&doc, None);
        c.output_mut().advance(frames);
        let m = c.take_meter_data();
        assert!((m.peak_l - 0.8).abs() < 1e-6);
        assert!((m.peak_r - 0.1).abs() < 1e-6);
        assert!(m.rms_l > 0.0 && m.rms_l <= m.peak_l);
        // Snapshot-and-clear: a second read with no new audio is silent.
        let m2 = c.take_meter_data();
        assert_eq!(m2.peak_l, 0.0);
        assert_eq!(m2.rms_l, 0.0);
    }

    #[test]
    fn reset_clears_session_for_file_switch() {
        let (mut c, doc) = controller_with_doc(1000, 1);
        c.play(&doc, Some((100, 900)));
        c.output_mut().advance(50);
        c.reset();
        assert_eq!(c.state(), TransportState::Stopped);
        assert_eq!(c.current_frame(), 0);
        assert_eq!(c.resume_frame(), 0);
        assert_eq!(c.start_frame(), 0);
        assert_eq!(c.end_frame(), 0);
    }
}
