pub mod controller;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportState { Stopped, Playing, Paused }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CursorSource { Playback, User }

pub struct PlayheadState {
    pub position_samples: AtomicU64,
    pub is_playing: AtomicBool,
}

impl PlayheadState {
    pub fn new() -> Self { Self { position_samples: AtomicU64::new(0), is_playing: AtomicBool::new(false) } }
    pub fn position_secs(&self, sample_rate: u32) -> f64 {
        self.position_samples.load(Ordering::Relaxed) as f64 / sample_rate as f64
    }
    pub fn set_position_samples(&self, pos: u64) { self.position_samples.store(pos, Ordering::Relaxed); }
    pub fn set_playing(&self, playing: bool) { self.is_playing.store(playing, Ordering::Relaxed); }
}
impl Default for PlayheadState { fn default() -> Self { Self::new() } }

#[derive(Debug, Clone, Copy, Default)]
pub struct MeterData { pub peak_l: f32, pub peak_r: f32, pub rms_l: f32, pub rms_r: f32 }

pub struct MeteringSource {
    pub peak_l: f32, pub peak_r: f32, pub rms_sum_l: f64, pub rms_sum_r: f64, pub sample_count: usize, channels: u16,
}

impl MeteringSource {
    pub fn new(channels: u16) -> Self {
        Self { peak_l: 0.0, peak_r: 0.0, rms_sum_l: 0.0, rms_sum_r: 0.0, sample_count: 0, channels }
    }
    pub fn process_samples(&mut self, samples: &[f32]) -> MeterData {
        let ch = self.channels.max(1) as usize;
        let mut i = 0;
        while i + ch <= samples.len() {
            let left = samples[i].abs();
            let right = if ch >= 2 { samples[i + 1].abs() } else { left };
            if left > self.peak_l { self.peak_l = left; }
            if right > self.peak_r { self.peak_r = right; }
            self.rms_sum_l += (samples[i] as f64).powi(2);
            self.rms_sum_r += (right as f64).powi(2);
            self.sample_count += 1;
            i += ch;
        }
        MeterData {
            peak_l: self.peak_l, peak_r: self.peak_r,
            rms_l: if self.sample_count > 0 { (self.rms_sum_l / self.sample_count as f64).sqrt() as f32 } else { 0.0 },
            rms_r: if self.sample_count > 0 { (self.rms_sum_r / self.sample_count as f64).sqrt() as f32 } else { 0.0 },
        }
    }
    pub fn reset(&mut self) { self.peak_l = 0.0; self.peak_r = 0.0; self.rms_sum_l = 0.0; self.rms_sum_r = 0.0; self.sample_count = 0; }
}
