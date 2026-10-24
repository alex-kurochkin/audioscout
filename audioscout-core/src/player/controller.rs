use super::{CursorSource, MeterData, MeteringSource, PlayheadState, TransportState};
use crate::Document;
use std::sync::Arc;

pub struct Controller {
    state: TransportState, loop_enabled: bool, volume: f32,
    start_frame: u64, end_frame: u64, current_frame: u64, resume_frame: u64,
    playhead: Arc<PlayheadState>, meter: MeteringSource,
    space_is_stop: bool, skip_seconds: u32,
}

impl Controller {
    pub fn new(playhead: Arc<PlayheadState>) -> Self {
        Self {
            state: TransportState::Stopped, loop_enabled: false, volume: 0.8,
            start_frame: 0, end_frame: 0, current_frame: 0, resume_frame: 0,
            playhead, meter: MeteringSource::new(2), space_is_stop: true, skip_seconds: 5,
        }
    }
    pub fn state(&self) -> TransportState { self.state }
    pub fn is_looping(&self) -> bool { self.loop_enabled }
    pub fn set_loop(&mut self, enabled: bool) { self.loop_enabled = enabled; }
    pub fn toggle_loop(&mut self) { self.loop_enabled = !self.loop_enabled; }
    pub fn volume(&self) -> f32 { self.volume }
    pub fn set_volume(&mut self, vol: f32) { self.volume = vol.clamp(0.0, 1.0); }
    pub fn set_space_behavior(&mut self, is_stop: bool) { self.space_is_stop = is_stop; }
    pub fn set_skip_seconds(&mut self, secs: u32) { self.skip_seconds = secs.clamp(1, 60); }
    pub fn skip_seconds(&self) -> u32 { self.skip_seconds }
    pub fn play(&mut self, doc: &Document, selection: Option<(u64, u64)>) {
        let (start, end) = if let Some((s, e)) = selection { (s, e) } else { (self.resume_frame, doc.total_frames()) };
        self.start_frame = start; self.end_frame = end; self.current_frame = start;
        self.state = TransportState::Playing; self.playhead.set_playing(true); self.resume_frame = start;
    }
    pub fn pause(&mut self) {
        if self.state == TransportState::Playing {
            self.state = TransportState::Paused; self.playhead.set_playing(false); self.resume_frame = self.current_frame;
        }
    }
    pub fn stop(&mut self) {
        self.state = TransportState::Stopped; self.current_frame = self.resume_frame;
        self.playhead.set_position_samples(self.resume_frame); self.playhead.set_playing(false); self.meter.reset();
    }
    pub fn play_pause(&mut self, doc: &Document, selection: Option<(u64, u64)>) {
        match self.state {
            TransportState::Stopped => self.play(doc, selection),
            TransportState::Playing => { if self.space_is_stop { self.stop(); } else { self.pause(); } }
            TransportState::Paused => self.play(doc, selection),
        }
    }
    pub fn seek(&mut self, frame: u64, doc: &Document, source: CursorSource) {
        let max_frame = doc.total_frames();
        let target = frame.min(max_frame);
        if self.state == TransportState::Playing && self.end_frame > self.start_frame {
            self.current_frame = target.clamp(self.start_frame, self.end_frame);
        } else { self.current_frame = target; }
        self.playhead.set_position_samples(self.current_frame);
        if source == CursorSource::User { self.resume_frame = self.current_frame; }
    }
    pub fn skip(&mut self, doc: &Document, seconds: i32) {
        let sr = doc.sample_rate as i64;
        let skip_frames = (seconds as i64 * sr) as u64;
        let current = self.current_frame as i64;
        let target = (current + skip_frames as i64).max(0) as u64;
        self.seek(target, doc, CursorSource::User);
    }
    pub fn tick(&mut self, _doc: &Document) -> bool {
        if self.state != TransportState::Playing { return false; }
        self.current_frame += 1;
        if self.current_frame >= self.end_frame && self.end_frame > self.start_frame {
            if self.loop_enabled { self.current_frame = self.start_frame; }
            else { self.state = TransportState::Stopped; self.current_frame = self.resume_frame; self.playhead.set_playing(false); return true; }
        }
        self.playhead.set_position_samples(self.current_frame);
        false
    }
    pub fn current_frame(&self) -> u64 { self.current_frame }
    pub fn current_secs(&self, sample_rate: u32) -> f64 { self.current_frame as f64 / sample_rate as f64 }
    pub fn resume_frame(&self) -> u64 { self.resume_frame }
    pub fn reset(&mut self) {
        self.state = TransportState::Stopped; self.start_frame = 0; self.end_frame = 0;
        self.current_frame = 0; self.resume_frame = 0; self.playhead.set_position_samples(0);
        self.playhead.set_playing(false); self.meter.reset();
    }
    pub fn meter_data(&self) -> MeterData {
        let rms_l = if self.meter.sample_count > 0 { (self.meter.rms_sum_l / self.meter.sample_count as f64).sqrt() as f32 } else { 0.0 };
        let rms_r = if self.meter.sample_count > 0 { (self.meter.rms_sum_r / self.meter.sample_count as f64).sqrt() as f32 } else { 0.0 };
        MeterData { peak_l: self.meter.peak_l, peak_r: self.meter.peak_r, rms_l, rms_r }
    }
}
