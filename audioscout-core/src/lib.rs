pub mod audio_io;
pub mod audio_view;
pub mod config;
pub mod db;
pub mod disk;
pub mod peaks_tree;
pub mod player;
pub mod preview_worker;
pub mod render;
pub mod scanner;
pub mod table;
pub mod undo;

use crossbeam_channel::{Receiver, Sender};
use std::path::PathBuf;

/// Preview rendering mode for the file-list thumbnails (and, per the spec,
/// an independent toggle from the audio-view visualization mode).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreviewMode {
    Waveform,
    Spectrogram,
}

impl PreviewMode {
    /// Key stored alongside previews in the DB (`previews.mode` column).
    pub fn as_str(self) -> &'static str {
        match self {
            PreviewMode::Waveform => "wave",
            PreviewMode::Spectrogram => "spectro",
        }
    }

    /// Deterministic suffix mixed into the preview UUID so waveform and
    /// spectrogram PNGs never overwrite each other in the cache directory.
    ///
    /// `v2` = previews now carry a per-channel 0 dBFS middle line; bumping the
    /// suffix forces already-cached (line-less) thumbnails to be re-rendered.
    pub fn uuid_suffix(self) -> &'static str {
        match self {
            PreviewMode::Waveform => "-v2",
            PreviewMode::Spectrogram => "-spectro-v2",
        }
    }

    /// Parse back from the stored string form (`as_str` / config values).
    pub fn from_str_name(s: &str) -> Self {
        match s { "spectro" => PreviewMode::Spectrogram, _ => PreviewMode::Waveform }
    }

    pub fn toggled(self) -> Self {
        match self {
            PreviewMode::Waveform => PreviewMode::Spectrogram,
            PreviewMode::Spectrogram => PreviewMode::Waveform,
        }
    }
}

/// Events sent from background threads to UI
#[derive(Debug, Clone)]
pub enum UiEvent {
    ScanDone {
        dir: PathBuf,
        rows: Vec<table::FileRow>,
        generation: u64,
    },
    PreviewReady {
        path: PathBuf,
        uuid: String,
        generation: u64,
    },
    PlayheadUpdate {
        position_secs: f64,
    },
    PlaybackFinished,
    VolumeMeterUpdate {
        peak_l: f32,
        peak_r: f32,
        rms_l: f32,
        rms_r: f32,
    },
    DisksChanged {
        disks: Vec<disk::DiskInfo>,
    },
}

/// Application-wide event channel
pub struct EventChannel {
    pub sender: Sender<UiEvent>,
    pub receiver: Receiver<UiEvent>,
}

impl EventChannel {
    pub fn new() -> Self {
        let (sender, receiver) = crossbeam_channel::unbounded();
        Self { sender, receiver }
    }
}

impl Default for EventChannel {
    fn default() -> Self {
        Self::new()
    }
}

/// Sample type enumeration for audio files
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SampleType {
    UInt8,
    Int16,
    Int24,
    Int32,
    Float32,
    Float64,
    ALaw,
    ULaw,
    Compressed,
}

impl std::fmt::Display for SampleType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SampleType::UInt8 => write!(f, "UInt8"),
            SampleType::Int16 => write!(f, "Int16"),
            SampleType::Int24 => write!(f, "Int24"),
            SampleType::Int32 => write!(f, "Int32"),
            SampleType::Float32 => write!(f, "Float32"),
            SampleType::Float64 => write!(f, "Float64"),
            SampleType::ALaw => write!(f, "ALaw"),
            SampleType::ULaw => write!(f, "ULaw"),
            SampleType::Compressed => write!(f, "Compressed"),
        }
    }
}

/// Probe status for audio files
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ProbeStatus {
    Ok,
    Broken,
    Unavailable,
}

/// Audio file metadata from probing
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AudioMeta {
    pub duration_secs: f64,
    pub sample_rate: u32,
    pub channels: u16,
    pub sample_type: SampleType,
    pub comment: Option<String>,
    pub status: ProbeStatus,
}

/// Marker (label) on the timeline
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Marker {
    pub name: String,
    pub time_samples: u64,
    pub time_sec: f64,
}

/// Document state - loaded audio file
#[derive(Debug, Clone)]
pub struct Document {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
    pub markers: Vec<Marker>,
    pub source_path: PathBuf,
    pub temp_path: Option<PathBuf>,
    pub dirty: bool,
    pub sample_type: SampleType,
}

impl Document {
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
            sample_rate: 44100,
            channels: 1,
            markers: Vec::new(),
            source_path: PathBuf::new(),
            temp_path: None,
            dirty: false,
            sample_type: SampleType::Float32,
        }
    }

    pub fn total_frames(&self) -> u64 {
        if self.channels == 0 {
            0
        } else {
            (self.samples.len() as u64) / (self.channels as u64)
        }
    }

    pub fn duration_secs(&self) -> f64 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.total_frames() as f64 / self.sample_rate as f64
        }
    }

    pub fn is_stereo(&self) -> bool {
        self.channels == 2
    }
}

impl Default for Document {
    fn default() -> Self {
        Self::new()
    }
}

/// Supported audio file extensions
pub const AUDIO_EXTENSIONS: &[&str] = &[
    "wav", "aiff", "aif", "caf", "flac", "mp3", "ogg", "oga", "m4a", "aac", "wma", "opus", "wv",
    "au", "mp2",
];

/// Editable file extensions (PCM read/write)
pub const EDITABLE_EXTENSIONS: &[&str] = &["wav", "aiff", "aif", "caf"];

pub fn is_audio_file(path: &std::path::Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| AUDIO_EXTENSIONS.contains(&e.to_lowercase().as_str()))
        .unwrap_or(false)
}

pub fn is_editable_file(path: &std::path::Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| EDITABLE_EXTENSIONS.contains(&e.to_lowercase().as_str()))
        .unwrap_or(false)
}
