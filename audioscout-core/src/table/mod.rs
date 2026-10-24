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
    pub fn width(&self, col: &Column) -> f32 { self.widths.get(col.key()).copied().unwrap_or(col.default_width()) }
    pub fn set_width(&mut self, col: &Column, width: f32) { self.widths.insert(col.key().to_string(), width); }
    pub fn toggle_visibility(&mut self, col: &Column) {
        if let Some(pos) = self.visible.iter().position(|c| c == col) {
            if self.visible.len() > 1 { self.visible.remove(pos); }
        } else { self.visible.push(*col); }
    }
    pub fn is_visible(&self, col: &Column) -> bool { self.visible.contains(col) }
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
