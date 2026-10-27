#[derive(Debug, Clone, Copy)]
pub struct ViewRange { pub start_frame: u64, pub end_frame: u64 }

impl ViewRange {
    pub fn new(start: u64, end: u64) -> Self { Self { start_frame: start, end_frame: end } }
    pub fn len(&self) -> u64 { self.end_frame.saturating_sub(self.start_frame) }
    pub fn is_empty(&self) -> bool { self.start_frame >= self.end_frame }
    pub fn contains(&self, frame: u64) -> bool { frame >= self.start_frame && frame < self.end_frame }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Selection { pub start_frame: u64, pub end_frame: u64 }

impl Selection {
    pub fn new(start: u64, end: u64) -> Self {
        let (s, e) = if start <= end { (start, end) } else { (end, start) };
        Self { start_frame: s, end_frame: e }
    }
    pub fn is_empty(&self) -> bool { self.start_frame >= self.end_frame }
    pub fn len(&self) -> u64 { self.end_frame.saturating_sub(self.start_frame) }
    pub fn contains(&self, frame: u64) -> bool { frame >= self.start_frame && frame < self.end_frame }
}

/// Visualization style of the player's audio-view area.
///
/// Independent from the file-list thumbnail mode (`PreviewMode` in the crate
/// root): the spec (§3.1 / §3.5) requires two separate toggles — one on the
/// top toolbar for the table previews, one on the player panel for the big
/// visualization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioViewMode {
    Waveform,
    Spectrogram,
}

pub struct AudioView {
    pub view_range: ViewRange, pub total_frames: u64, pub cursor_frame: u64,
    pub selection: Option<Selection>, pub active_marker: Option<usize>, pub auto_scroll: bool,
    /// Current display mode of the visualization area (Waveform/Spectrogram).
    pub mode: AudioViewMode,
}

impl AudioView {
    pub fn new(total_frames: u64) -> Self {
        Self { view_range: ViewRange::new(0, total_frames), total_frames, cursor_frame: 0, selection: None, active_marker: None, auto_scroll: true, mode: AudioViewMode::Waveform }
    }
    /// Flip between the waveform and spectrogram visualizations.
    pub fn toggle_mode(&mut self) {
        self.mode = match self.mode {
            AudioViewMode::Waveform => AudioViewMode::Spectrogram,
            AudioViewMode::Spectrogram => AudioViewMode::Waveform,
        };
    }
    pub fn set_total_frames(&mut self, frames: u64) {
        self.total_frames = frames;
        if self.view_range.end_frame > frames { self.view_range = ViewRange::new(0, frames); }
        self.cursor_frame = self.cursor_frame.min(frames);
    }
    pub fn zoom_in(&mut self) { self.zoom_at(1.5, self.view_center()); }
    pub fn zoom_out(&mut self) { self.zoom_at(1.0 / 1.5, self.view_center()); }
    pub fn zoom_at(&mut self, factor: f64, anchor_frame: u64) {
        let current_len = self.view_range.len() as f64;
        let new_len = (current_len / factor).max(16.0).min(self.total_frames as f64);
        let anchor_ratio = if current_len > 0.0 {
            (anchor_frame.saturating_sub(self.view_range.start_frame) as f64 / current_len).clamp(0.0, 1.0)
        } else { 0.5 };
        let new_start = (anchor_frame as f64 - new_len * anchor_ratio).max(0.0) as u64;
        let new_end = (new_start as f64 + new_len).min(self.total_frames as f64) as u64;
        let new_start = new_end.saturating_sub(new_len as u64);
        self.view_range = ViewRange::new(new_start, new_end);
    }
    pub fn zoom_at_x(&mut self, factor: f64, x_ratio: f64) {
        let view_len = self.view_range.len() as f64;
        let anchor_frame = self.view_range.start_frame as f64 + view_len * x_ratio;
        self.zoom_at(factor, anchor_frame as u64);
    }
    pub fn zoom_reset(&mut self) { self.view_range = ViewRange::new(0, self.total_frames); }
    pub fn pan(&mut self, delta: i64) {
        let len = self.view_range.len() as i64;
        let new_start = (self.view_range.start_frame as i64 + delta).max(0).min(self.total_frames as i64 - len) as u64;
        self.view_range = ViewRange::new(new_start, new_start + len as u64);
    }
    pub fn scroll_horizontal(&mut self, fraction: f64) { let delta = (self.view_range.len() as f64 * fraction) as i64; self.pan(delta); }
    pub fn set_cursor(&mut self, frame: u64) { self.cursor_frame = frame.min(self.total_frames); }
    pub fn auto_scroll_to_cursor(&mut self) {
        if !self.auto_scroll { return; }
        if !self.view_range.contains(self.cursor_frame) {
            let len = self.view_range.len();
            let new_start = self.cursor_frame.saturating_sub(len / 4);
            let new_end = (new_start + len).min(self.total_frames);
            self.view_range = ViewRange::new(new_start.saturating_sub(len.saturating_sub(new_end - new_start)), new_end);
        }
    }
    pub fn set_selection(&mut self, start: u64, end: u64) { self.selection = Some(Selection::new(start, end)); }
    pub fn clear_selection(&mut self) { self.selection = None; }
    pub fn selection_range(&self) -> Option<(u64, u64)> { self.selection.map(|s| (s.start_frame, s.end_frame)) }
    pub fn frame_to_x(&self, frame: u64, widget_width: u32) -> f32 {
        let view_len = self.view_range.len() as f32;
        if view_len == 0.0 { return 0.0; }
        let ratio = (frame.saturating_sub(self.view_range.start_frame) as f32) / view_len;
        ratio * widget_width as f32
    }
    pub fn x_to_frame(&self, x: f32, widget_width: u32) -> u64 {
        if widget_width == 0 { return 0; }
        let ratio = (x / widget_width as f32).clamp(0.0, 1.0);
        let frame = self.view_range.start_frame as f64 + ratio as f64 * self.view_range.len() as f64;
        (frame as u64).min(self.total_frames)
    }
    pub fn is_sample_level(&self, widget_width: u32) -> bool {
        let view_len = self.view_range.len();
        view_len < 8192 && view_len <= widget_width as u64 * 40
    }
    fn view_center(&self) -> u64 { (self.view_range.start_frame + self.view_range.end_frame) / 2 }
}
