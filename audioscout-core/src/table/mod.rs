use crate::{ProbeStatus, SampleType};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileRow {
    pub path: PathBuf, pub name: String, pub duration_secs: f64, pub sample_rate: u32,
    pub sample_type: SampleType, pub channels: u16, pub size: u64, pub comment: Option<String>,
    pub date_created: u64, pub date_modified: u64, pub status: ProbeStatus, pub preview_uuid: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Column {
    Name, Preview, Duration, SampleRate, SampleType, Channels, Size, Comment, DateCreated, DateModified,
}

impl Column {
    pub fn key(&self) -> &'static str {
        match self { Column::Name => "name", Column::Preview => "preview", Column::Duration => "duration",
            Column::SampleRate => "sample_rate", Column::SampleType => "sample_type", Column::Channels => "channels",
            Column::Size => "size", Column::Comment => "comment", Column::DateCreated => "date_created", Column::DateModified => "date_modified" }
    }
    pub fn title(&self) -> &'static str {
        match self { Column::Name => "Name", Column::Preview => "Preview", Column::Duration => "Duration",
            Column::SampleRate => "Sample Rate", Column::SampleType => "Sample Type", Column::Channels => "Channels",
            Column::Size => "Size", Column::Comment => "Comment", Column::DateCreated => "Date Created", Column::DateModified => "Date Modified" }
    }
    pub fn default_width(&self) -> f32 {
        match self { Column::Name => 200.0, Column::Preview => 340.0, Column::Duration => 80.0,
            Column::SampleRate => 90.0, Column::SampleType => 90.0, Column::Channels => 70.0,
            Column::Size => 80.0, Column::Comment => 150.0, Column::DateCreated => 120.0, Column::DateModified => 120.0 }
    }
    pub fn sortable(&self) -> bool { !matches!(self, Column::Preview | Column::Comment) }
    pub fn from_key(key: &str) -> Option<Self> {
        match key { "name" => Some(Column::Name), "preview" => Some(Column::Preview), "duration" => Some(Column::Duration),
            "sample_rate" => Some(Column::SampleRate), "sample_type" => Some(Column::SampleType), "channels" => Some(Column::Channels),
            "size" => Some(Column::Size), "comment" => Some(Column::Comment), "date_created" => Some(Column::DateCreated),
            "date_modified" => Some(Column::DateModified), _ => None }
    }
    pub fn all() -> &'static [Column] {
        &[Column::Name, Column::Preview, Column::Duration, Column::SampleRate, Column::SampleType,
            Column::Channels, Column::Size, Column::Comment, Column::DateCreated, Column::DateModified]
    }
}

#[derive(Debug, Clone)]
pub struct ColumnLayout {
    pub visible: Vec<Column>, pub widths: std::collections::HashMap<String, f32>,
    pub sort_column: Option<Column>, pub sort_ascending: bool,
}

impl ColumnLayout {
    pub fn new() -> Self { Self { visible: Column::all().to_vec(), widths: std::collections::HashMap::new(), sort_column: None, sort_ascending: true } }

    /// Rebuild the layout from the persisted `ColumnConfig` (config.json).
    /// Used at app start so the Columns-dialog selection survives restarts.
    /// Rules: unknown keys are dropped, duplicates collapse to the first
    /// occurrence, and an empty/absent list falls back to all columns — the
    /// table must always render at least one column.
    pub fn from_config(cfg: &crate::config::ColumnConfig) -> Self {
        let mut layout = Self::new();
        if let Some(vis) = &cfg.visible {
            let mut cols: Vec<Column> = Vec::new();
            for key in vis {
                if let Some(c) = Column::from_key(key) {
                    if !cols.contains(&c) { cols.push(c); }
                }
            }
            if !cols.is_empty() { layout.visible = cols; }
        }
        layout.widths = cfg.widths.clone();
        layout
    }

    /// Snapshot of the current selection as column keys, in display order,
    /// ready to store into `ColumnConfig::visible`.
    pub fn visible_keys(&self) -> Vec<String> {
        self.visible.iter().map(|c| c.key().to_string()).collect()
    }

    pub fn width(&self, col: &Column) -> f32 { self.widths.get(col.key()).copied().unwrap_or(col.default_width()) }
    pub fn set_width(&mut self, col: &Column, width: f32) { self.widths.insert(col.key().to_string(), width); }
    pub fn toggle_visibility(&mut self, col: &Column) {
        if let Some(pos) = self.visible.iter().position(|c| c == col) {
            if self.visible.len() > 1 { self.visible.remove(pos); }
        } else { self.visible.push(*col); }
    }
    pub fn is_visible(&self, col: &Column) -> bool { self.visible.contains(col) }

    /// Move the column currently at index `from` so that it ends up at index
    /// `to` in the visible order (drag-and-drop reordering in the table
    /// header). Out-of-range indices and no-op moves are ignored.
    pub fn move_by_indices(&mut self, from: usize, to: usize) {
        if from >= self.visible.len() || to >= self.visible.len() || from == to {
            return;
        }
        let col = self.visible.remove(from);
        self.visible.insert(to, col);
    }

    /// Move column `col` before/after its new neighbour by key — convenience
    /// wrapper used by callers that track columns by identity rather than by
    /// live index (indices shift while items are dragged).
    pub fn move_column(&mut self, col: &Column, target: &Column, after: bool) {
        if col == target {
            return;
        }
        let Some(from) = self.visible.iter().position(|c| c == col) else { return };
        let Some(mut to) = self.visible.iter().position(|c| c == target) else { return };
        // Removing `from` shifts every later index down by one; compensate so
        // `to` still points at the same neighbour after the removal.
        if from < to { to -= 1; }
        self.move_by_indices(from, if after { to + 1 } else { to });
    }

    pub fn sort_rows(&self, rows: &mut [FileRow]) {
        if let Some(col) = self.sort_column {
            rows.sort_by(|a, b| {
                let cmp = match col {
                    Column::Name => a.name.to_lowercase().cmp(&b.name.to_lowercase()),
                    Column::Duration => a.duration_secs.partial_cmp(&b.duration_secs).unwrap_or(std::cmp::Ordering::Equal),
                    Column::SampleRate => a.sample_rate.cmp(&b.sample_rate),
                    Column::SampleType => format!("{}", a.sample_type).cmp(&format!("{}", b.sample_type)),
                    Column::Channels => a.channels.cmp(&b.channels),
                    Column::Size => a.size.cmp(&b.size),
                    Column::DateCreated => a.date_created.cmp(&b.date_created),
                    Column::DateModified => a.date_modified.cmp(&b.date_modified),
                    _ => std::cmp::Ordering::Equal,
                };
                if self.sort_ascending { cmp } else { cmp.reverse() }
            });
        }
    }
}
impl Default for ColumnLayout { fn default() -> Self { Self::new() } }
