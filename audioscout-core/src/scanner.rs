use crate::{table::FileRow, is_audio_file};
use crossbeam_channel::Sender;
use crate::UiEvent;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use walkdir::WalkDir;

pub struct Scanner { cancel: Arc<AtomicBool>, generation: Arc<AtomicU64> }

impl Scanner {
    pub fn new() -> Self { Self { cancel: Arc::new(AtomicBool::new(false)), generation: Arc::new(AtomicU64::new(0)) } }
    pub fn scan(&self, dir: PathBuf, sender: Sender<UiEvent>) {
        self.cancel.store(true, Ordering::SeqCst);
        let cancel = Arc::new(AtomicBool::new(false));
        self.cancel.store(false, Ordering::SeqCst);
        let cancel_clone = cancel.clone();
        let gen = self.generation.fetch_add(1, Ordering::SeqCst) + 1;
        let sender_clone = sender.clone();
        let dir_clone = dir.clone();
        std::thread::spawn(move || {
            let mut rows = Vec::new();
            for entry in WalkDir::new(&dir_clone).max_depth(1).min_depth(1).into_iter().filter_map(|e| e.ok()) {
                if cancel_clone.load(Ordering::Relaxed) { return; }
                let path = entry.path();
                if !path.is_file() || !is_audio_file(path) { continue; }
                let meta = crate::audio_io::probe_file(path);
                let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
                let file_size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                let metadata = entry.metadata().ok();
                let mtime = metadata.as_ref().and_then(|m| m.modified().ok())
                    .map(|t| t.duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs()).unwrap_or(0);
                // Try to get birth time (created time), fallback to mtime if not available
                let ctime = metadata.as_ref().and_then(|m| m.created().ok())
                    .map(|t| t.duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs())
                    .unwrap_or(mtime);
                // Resolve the preview *synchronously* during the scan: if the
                // PNG is already cached on disk (deterministic filename from
                // path/mtime/size/channels) the row ships with its UUID and
                // shows up instantly in the table. Only genuinely missing
                // previews fall through to the background worker. This removes
                // the multi-second "⏳ delay" after re-opening a directory
                // whose thumbnails are already generated. We deliberately do
                // NOT touch SQLite here: the scanner thread must never block
                // behind writes made by concurrent preview render threads
                // (that used to stall the whole ScanDone event). The DB rows
                // are written by the render threads themselves and back-filled
                // lazily.
                let preview_uuid = if meta.status == crate::ProbeStatus::Ok {
                    crate::preview_worker::cached_preview_png(path, meta.channels)
                } else {
                    None
                };
                rows.push(FileRow {
                    path: path.to_path_buf(), name: file_name, duration_secs: meta.duration_secs,
                    sample_rate: meta.sample_rate, sample_type: meta.sample_type, channels: meta.channels,
                    size: file_size, comment: meta.comment, date_created: ctime, date_modified: mtime,
                    status: meta.status, preview_uuid,
                });
            }
            rows.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
            if !cancel_clone.load(Ordering::Relaxed) {
                let _ = sender_clone.send(UiEvent::ScanDone { dir: dir_clone, rows, generation: gen });
            }
        });
    }
    pub fn cancel(&self) { self.cancel.store(true, Ordering::SeqCst); }
}
impl Default for Scanner { fn default() -> Self { Self::new() } }
