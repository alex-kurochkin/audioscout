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

                    // Wait for job or quit signal
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
                    // Render preview
                    let result = render_preview(&job.path, job.mode);

                    if let Ok(uuid) = result {
                        // Save to disk
                        let config_dir = crate::config::config_dir();
                        let previews_dir = config_dir.join("previews");
                        let _ = std::fs::create_dir_all(&previews_dir);
                        let _png_path = previews_dir.join(format!("{}.png", uuid));

                        // Store in database (simplified)
                        let _ = sender_clone.send(UiEvent::PreviewReady {
                            path: job.path.clone(),
                            uuid,
                            mode: job.mode,
                            generation: job.generation,
                        });
                    }

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

    /// Submit a preview job
    pub fn submit(&self, job: PreviewJob) {
        let (lock, cvar) = &*self.queue;
        let mut inner = lock.lock().unwrap();

        // Dedup: skip if already pending for this path
        if inner.pending_paths.contains(&job.path) {
            return;
        }

        inner.pending_paths.insert(job.path.clone());
        inner.jobs.push_back(job);
        cvar.notify_one();
    }

    /// Clear all pending jobs
    pub fn clear_queue(&self) {
        let (lock, _) = &*self.queue;
        let mut inner = lock.lock().unwrap();
        inner.jobs.clear();
        inner.pending_paths.clear();
    }

    /// Shutdown the worker
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

/// Render a preview image (waveform or spectrogram)
/// Uses downsampling to show the entire file content
fn render_preview(path: &PathBuf, mode: PreviewMode) -> Result<String, String> {
    let uuid = uuid::Uuid::new_v4().to_string();

    // Load audio data
    let (samples, sample_rate, channels, _) = crate::audio_io::load_audio(path)
        .map_err(|e| format!("Load failed: {}", e))?;

    let width = 340u32;
    let height = 20u32;

    let img = match mode {
        PreviewMode::Wave => {
            // Render entire file with downsampling
            crate::render::waveform::render_waveform(&samples, sample_rate, channels, width, height)
        }
        PreviewMode::Spectrogram => {
            // Render entire file with downsampling
            crate::render::spectrogram::render_spectrogram(&samples, sample_rate, channels, width, height)
        }
    };

    // Save as PNG
    let config_dir = crate::config::config_dir();
    let previews_dir = config_dir.join("previews");
    let _ = std::fs::create_dir_all(&previews_dir);
    let _png_path = previews_dir.join(format!("{}.png", uuid));

    img.save(&_png_path).map_err(|e| format!("Save failed: {}", e))?;

    Ok(uuid)
}
