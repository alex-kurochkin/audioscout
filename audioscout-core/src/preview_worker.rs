use crate::config::PreviewMode;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};
use crossbeam_channel::Sender;
use crate::UiEvent;

/// Preview rendering job
#[derive(Debug, Clone)]
pub struct PreviewJob {
    pub path: PathBuf,
    pub mode: PreviewMode,
    pub generation: u64,
}

/// Background worker for rendering preview images
pub struct PreviewWorker {
    queue: Arc<(Mutex<QueueInner>, Condvar)>,
    quit: Arc<Mutex<bool>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

struct QueueInner {
    jobs: VecDeque<PreviewJob>,
    pending_paths: std::collections::HashSet<PathBuf>,
}

impl PreviewWorker {
    pub fn new(sender: Sender<UiEvent>) -> Self {
        let queue = Arc::new((
            Mutex::new(QueueInner {
                jobs: VecDeque::new(),
                pending_paths: std::collections::HashSet::new(),
            }),
            Condvar::new(),
        ));
        let quit = Arc::new(Mutex::new(false));

        let queue_clone = queue.clone();
        let quit_clone = quit.clone();
        let sender_clone = sender.clone();

        let handle = std::thread::spawn(move || {
            loop {
                let job = {
                    let (lock, cvar) = &*queue_clone;
                    let mut inner = lock.lock().unwrap();

                    while inner.jobs.is_empty() {
                        if *quit_clone.lock().unwrap() {
                            return;
                        }
                        inner = cvar.wait(inner).unwrap();
                    }

                    if *quit_clone.lock().unwrap() {
                        return;
                    }

                    inner.jobs.pop_front()
                };

                if let Some(job) = job {
                    let config_dir = crate::config::config_dir();
                    let db_path = config_dir.join("app.db");

                    // Check if preview already exists in database
                    let existing_uuid = if let Ok(db) = crate::db::Database::open(&db_path) {
                        if let Ok(metadata) = std::fs::metadata(&job.path) {
                            let size = metadata.len();
                            let mtime = metadata.modified()
                                .ok()
                                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                                .map(|d| d.as_secs())
                                .unwrap_or(0);
                            let mode_str = match job.mode {
                                PreviewMode::Wave => "wave",
                                PreviewMode::Spectrogram => "spectrogram",
                            };
                            db.load_preview(&job.path, size, mtime, mode_str)
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    let uuid = if let Some(existing) = existing_uuid {
                        // Preview already exists in database, check if file exists
                        let previews_dir = config_dir.join("previews");
                        let png_path = previews_dir.join(format!("{}.png", existing));
                        if png_path.exists() {
                            existing
                        } else {
                            // File missing, regenerate
                            match render_preview(&job.path, job.mode) {
                                Ok(new_uuid) => new_uuid,
                                Err(_) => {
                                    let (lock, _) = &*queue_clone;
                                    if let Ok(mut inner) = lock.lock() {
                                        inner.pending_paths.remove(&job.path);
                                    }
                                    continue;
                                }
                            }
                        }
                    } else {
                        // Generate new preview
                        match render_preview(&job.path, job.mode) {
                            Ok(new_uuid) => new_uuid,
                            Err(_) => {
                                let (lock, _) = &*queue_clone;
                                if let Ok(mut inner) = lock.lock() {
                                    inner.pending_paths.remove(&job.path);
                                }
                                continue;
                            }
                        }
                    };

                    // Save to database
                    if let Ok(db) = crate::db::Database::open(&db_path) {
                        if let Ok(metadata) = std::fs::metadata(&job.path) {
                            let size = metadata.len();
                            let mtime = metadata.modified()
                                .ok()
                                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                                .map(|d| d.as_secs())
                                .unwrap_or(0);
                            let mode_str = match job.mode {
                                PreviewMode::Wave => "wave",
                                PreviewMode::Spectrogram => "spectrogram",
                            };
                            let _ = db.save_preview(&job.path, size, mtime, mode_str, &uuid);
                        }
                    }

                    // Send event to UI
                    let _ = sender_clone.send(UiEvent::PreviewReady {
                        path: job.path.clone(),
                        uuid,
                        mode: job.mode,
                        generation: job.generation,
                    });

                    // Remove from pending
                    let (lock, _) = &*queue_clone;
                    if let Ok(mut inner) = lock.lock() {
                        inner.pending_paths.remove(&job.path);
                    }
                }
            }
        });

        Self {
            queue,
            quit,
            handle: Some(handle),
        }
    }

    pub fn submit(&self, job: PreviewJob) {
        let (lock, cvar) = &*self.queue;
        let mut inner = lock.lock().unwrap();

        if inner.pending_paths.contains(&job.path) {
            return;
        }

        inner.pending_paths.insert(job.path.clone());
        inner.jobs.push_back(job);
        cvar.notify_one();
    }

    pub fn clear_queue(&self) {
        let (lock, _) = &*self.queue;
        let mut inner = lock.lock().unwrap();
        inner.jobs.clear();
        inner.pending_paths.clear();
    }

    pub fn shutdown(&mut self) {
        *self.quit.lock().unwrap() = true;
        let (_, cvar) = &*self.queue;
        cvar.notify_all();

        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for PreviewWorker {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Generate deterministic UUID based on file path, mtime, size, and mode
fn generate_preview_uuid(path: &PathBuf, mode: PreviewMode) -> Result<String, String> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let metadata = std::fs::metadata(path)
        .map_err(|e| format!("Failed to get metadata: {}", e))?;

    let mtime = metadata.modified()
        .map_err(|e| format!("Failed to get mtime: {}", e))?
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| format!("Failed to convert mtime: {}", e))?
        .as_secs();

    let size = metadata.len();
    let mode_str = match mode {
        PreviewMode::Wave => "wave",
        PreviewMode::Spectrogram => "spectrogram",
    };

    let mut hasher = DefaultHasher::new();
    path.hash(&mut hasher);
    mtime.hash(&mut hasher);
    size.hash(&mut hasher);
    mode_str.hash(&mut hasher);
    let hash = hasher.finish();

    // Use first 16 bytes of hash as UUID
    let uuid_bytes = hash.to_le_bytes();
    let uuid = uuid::Uuid::from_bytes([
        uuid_bytes[0], uuid_bytes[1], uuid_bytes[2], uuid_bytes[3],
        uuid_bytes[4], uuid_bytes[5], uuid_bytes[6], uuid_bytes[7],
        uuid_bytes[0], uuid_bytes[1], uuid_bytes[2], uuid_bytes[3],
        uuid_bytes[4], uuid_bytes[5], uuid_bytes[6], uuid_bytes[7],
    ]);

    Ok(uuid.to_string())
}

/// Render a preview image (waveform or spectrogram)
/// Uses downsampling to show the entire file content
/// Returns existing UUID if preview already exists
fn render_preview(path: &PathBuf, mode: PreviewMode) -> Result<String, String> {
    // Generate deterministic UUID
    let uuid = generate_preview_uuid(path, mode)?;

    // Check if preview already exists
    let config_dir = crate::config::config_dir();
    let previews_dir = config_dir.join("previews");
    let _ = std::fs::create_dir_all(&previews_dir);
    let png_path = previews_dir.join(format!("{}.png", uuid));

    if png_path.exists() {
        // Preview already exists, return existing UUID
        return Ok(uuid);
    }

    // Load audio data
    let (samples, sample_rate, channels, _) = crate::audio_io::load_audio(path)
        .map_err(|e| format!("Load failed: {}", e))?;

    let width = 340u32;
    let height = 20u32;

    let img = match mode {
        PreviewMode::Wave => {
            crate::render::waveform::render_waveform(&samples, sample_rate, channels, width, height)
        }
        PreviewMode::Spectrogram => {
            crate::render::spectrogram::render_spectrogram(&samples, sample_rate, channels, width, height)
        }
    };

    // Save as PNG
    img.save(&png_path).map_err(|e| format!("Save failed: {}", e))?;

    Ok(uuid)
}

/// Clean up orphaned preview files
pub fn cleanup_previews() -> Result<usize, String> {
    let config_dir = crate::config::config_dir();
    let previews_dir = config_dir.join("previews");

    if !previews_dir.exists() {
        return Ok(0);
    }

    let db_path = config_dir.join("app.db");
    let db = crate::db::Database::open(&db_path)
        .map_err(|e| format!("Failed to open database: {}", e))?;

    // Get all valid UUIDs from database
    let valid_uuids = db.get_all_preview_uuids()
        .map_err(|e| format!("Failed to get preview UUIDs: {}", e))?;

    let mut removed_count = 0;

    // Iterate through all PNG files in previews directory
    if let Ok(entries) = std::fs::read_dir(&previews_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("png") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    if !valid_uuids.contains(&stem.to_string()) {
                        // This preview is not referenced in database, delete it
                        if std::fs::remove_file(&path).is_ok() {
                            removed_count += 1;
                        }
                    }
                }
            }
        }
    }

    Ok(removed_count)
}
