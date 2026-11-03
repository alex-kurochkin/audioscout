use std::collections::{HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use crossbeam_channel::Sender;
use uuid::Uuid;

use crate::{PreviewMode, UiEvent};
use crate::db::Database;

/// Preview rendering job
#[derive(Debug, Clone)]
pub struct PreviewJob {
    pub path: PathBuf,
    pub generation: u64,
    /// Thumbnail style to render. The worker renders it into a PNG keyed by
    /// a mode-specific UUID, so wave and spectrogram caches coexist.
    pub mode: PreviewMode,
}

/// Upper bound on simultaneously running preview renders. Decoding is
/// CPU/IO-heavy; going wider starves the UI thread and the disk, which is
/// exactly what made previous versions feel *slower*.
const MAX_CONCURRENT_REVIEWS: usize = 3;

/// Hard cap on the number of jobs kept in the queue. When a directory has
/// thousands of files, only a handful of rows are visible — everything past
/// the cap is dropped (and its pending flag released) so fresh scans never
/// sit behind an hours-long backlog of thumbnails nobody can see.
const MAX_QUEUE_LEN: usize = 256;

/// Byte budget for decoding a single preview thumbnail.
///
/// Previews must cover the **whole file**, so this is intentionally an
/// effectively-unlimited budget: no truncation, no silent padding of the tail.
/// The waveform path reaches full coverage cheaply anyway — compressed files
/// are seek-sampled (O(target × window) bytes regardless of duration) and PCM
/// files are streamed in a single pass — while the safety caps inside
/// `audio_io::envelope` still stop endless/corrupt streams.
pub const PREVIEW_DECODE_BUDGET: u64 = crate::audio_io::envelope::UNLIMITED_BUDGET;

/// Background worker for rendering preview images.
///
/// Queueing model (fixed after the "even slower" regression):
/// - Each accepted job runs on its own short-lived thread, so one slow decode
///   can no longer serialise (and stall) the whole preview pipeline.
/// - `clear_queue` discards *queued* jobs but never forgets paths that are
///   actively rendering — dropping those used to leave the path stuck in
///   `pending_paths` forever, so every later scan silently skipped it and it
///   showed ⏳ indefinitely.
/// - Finished previews travel through a lock-free channel and are forwarded
///   to the UI by a dedicated delivery thread.
pub struct PreviewWorker {
    inner: Arc<WorkerInner>,
}

struct WorkerInner {
    /// Job queue guarded by mutex + condvar.
    queue: Mutex<VecDeque<PreviewJob>>,
    cvar: Condvar,
    /// Paths with a job either queued or actively rendering. Dedup guard.
    pending: Mutex<HashSet<PathBuf>>,
    /// Number of currently running render threads.
    active: AtomicUsize,
    quit: AtomicBool,
    /// Rendered results waiting to be forwarded to the UI.
    ready_tx: Sender<(PathBuf, String, u64)>,
    ready_rx: crossbeam_channel::Receiver<(PathBuf, String, u64)>,
}

impl PreviewWorker {
    pub fn new(sender: Sender<UiEvent>) -> Self {
        let (ready_tx, ready_rx) = crossbeam_channel::unbounded();
        let inner = Arc::new(WorkerInner {
            queue: Mutex::new(VecDeque::new()),
            cvar: Condvar::new(),
            pending: Mutex::new(HashSet::new()),
            active: AtomicUsize::new(0),
            quit: AtomicBool::new(false),
            ready_tx,
            ready_rx,
        });

        let db_path = crate::config::config_dir().join("app.db");

        // Delivery thread: forwards finished previews to the UI as soon as
        // they are ready (independent of the scheduler loop).
        {
            let inner = inner.clone();
            std::thread::spawn(move || {
                for (path, uuid, generation) in inner.ready_rx.iter() {
                    if sender
                        .send(UiEvent::PreviewReady {
                            path,
                            uuid,
                            generation,
                        })
                        .is_err()
                    {
                        break;
                    }
                }
            });
        }

        // Scheduler thread: pops jobs and spawns bounded render threads.
        {
            let inner = inner.clone();
            std::thread::spawn(move || scheduler_loop(inner, &db_path));
        }

        Self { inner }
    }

    /// Submit a preview job.
    ///
    /// Returns `true` when the job was enqueued, `false` when it was skipped
    /// as a duplicate (a job or render for this path is already in flight) or
    /// because the queue is saturated. Callers must use this flag to keep
    /// their own bookkeeping in sync: if a job is dropped here, no
    /// [`UiEvent::PreviewReady`] will ever arrive for it, and treating it as
    /// "in flight" anyway would leave the path stuck in a "generating" state
    /// forever.
    pub fn submit(&self, job: PreviewJob) -> bool {
        if self.inner.quit.load(Ordering::SeqCst) {
            return false;
        }
        {
            let mut pending = self.inner.pending.lock().unwrap();
            if pending.contains(&job.path) {
                return false;
            }
            pending.insert(job.path.clone());
        }
        {
            let mut queue = self.inner.queue.lock().unwrap();
            if queue.len() >= MAX_QUEUE_LEN {
                // Queue overflow: drop this job and release its pending flag
                // so a later scan can retry it.
                drop(queue);
                self.inner.pending.lock().unwrap().remove(&job.path);
                return false;
            }
            queue.push_back(job);
        }
        self.inner.cvar.notify_one();
        true
    }

    /// Clear all *queued* jobs (e.g. after switching directories).
    ///
    /// Jobs already handed to a render thread keep running and still deliver
    /// their result — their paths stay in `pending` until the render thread
    /// finishes, so a re-scan cannot double-submit them and they can never
    /// get permanently stuck.
    pub fn clear_queue(&self) {
        let drained: Vec<PreviewJob> = self.inner.queue.lock().unwrap().drain(..).collect();
        if !drained.is_empty() {
            let mut pending = self.inner.pending.lock().unwrap();
            for job in drained {
                pending.remove(&job.path);
            }
        }
    }

    /// Drop every trace of `path` from the pipeline: its queued job (if any)
    /// and its dedup flag. Used by the per-file context-menu "Refresh
    /// preview": before purging a file's cache the app forgets the path here,
    /// so (a) a queued job can never render into the PNG being deleted and
    /// (b) the follow-up regeneration job is not rejected as a duplicate.
    /// An *actively rendering* job cannot be cancelled; its stale result is
    /// discarded on the app side via the bumped scan generation.
    pub fn forget_path(&self, path: &Path) {
        let mut queue = self.inner.queue.lock().unwrap();
        queue.retain(|job| job.path != path);
        drop(queue);
        self.inner.pending.lock().unwrap().remove(path);
    }

    /// Stop accepting work and wait for running renders to finish.
    pub fn shutdown(&mut self) {
        self.inner.quit.store(true, Ordering::SeqCst);
        self.inner.cvar.notify_all();
        self.clear_queue();

        // Wait for in-flight renders (each one is individually bounded by
        // the decode budget, so this terminates quickly).
        while self.inner.active.load(Ordering::SeqCst) > 0 {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }
}

impl Drop for PreviewWorker {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn scheduler_loop(inner: Arc<WorkerInner>, db_path: &Path) {
    loop {
        // Take one job, honouring the concurrency cap.
        let job = {
            let mut queue = inner.queue.lock().unwrap();
            loop {
                if inner.quit.load(Ordering::SeqCst) {
                    return;
                }
                if inner.active.load(Ordering::SeqCst) < MAX_CONCURRENT_REVIEWS {
                    if let Some(job) = queue.pop_front() {
                        break job;
                    }
                }
                // Wait for new work / slot free-up / quit. The timeout also
                // lets us notice state changes even if a notify raced.
                let (q, _res) = inner
                    .cvar
                    .wait_timeout(queue, std::time::Duration::from_millis(100))
                    .unwrap();
                queue = q;
            }
        };

        inner.active.fetch_add(1, Ordering::SeqCst);
        let inner2 = inner.clone();
        let db_path = db_path.to_path_buf();
        std::thread::Builder::new()
            .stack_size(2 << 20)
            .spawn(move || {
                // Always release the slot, even on panic.
                let _guard = SlotGuard(inner2.clone());
                process_job(job, &inner2, &db_path);
            })
            .expect("spawn preview render thread");
    }
}

/// RAII: decrements the active-render counter when the render thread ends
/// (normally or via panic unwind) and wakes the scheduler.
struct SlotGuard(Arc<WorkerInner>);
impl Drop for SlotGuard {
    fn drop(&mut self) {
        self.0.active.fetch_sub(1, Ordering::SeqCst);
        self.0.cvar.notify_all();
    }
}

/// Per-job execution: cache check → render → persist → deliver.
fn process_job(job: PreviewJob, inner: &Arc<WorkerInner>, db_path: &Path) {
    // One SQLite connection per render thread (the old code opened two
    // connections *per job*, which dominated throughput on large folders).
    let db = crate::db::Database::open(db_path).ok();

    let result = (|| -> Result<String, String> {
        let meta = std::fs::metadata(&job.path).map_err(|e| e.to_string())?;
        let size = meta.len();
        let mtime = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);

        // Cache hit: DB row + PNG on disk → zero decoding work.
        if let Some(db) = &db {
            if let Some(existing) = db.load_preview(&job.path, size, mtime, job.mode.as_str()) {
                let png = preview_png_path(&existing);
                if png.exists() {
                    return Ok(existing);
                }
            }
        }

        let uuid = render_preview(&job.path, job.mode)?;

        if let Some(db) = &db {
            let _ = db.save_preview(&job.path, size, mtime, job.mode.as_str(), &uuid);
        }
        Ok(uuid)
    })();

    match result {
        Ok(uuid) => {
            let _ = inner
                .ready_tx
                .try_send((job.path.clone(), uuid, job.generation));
        }
        Err(e) => {
            tracing::debug!("Preview failed for {}: {}", job.path.display(), e);
        }
    }

    // Release the dedup flag last, after delivery, so a concurrent re-scan
    // can never observe "not pending" before the result was sent.
    inner.pending.lock().unwrap().remove(&job.path);
}

/// Generate deterministic UUID based on file path, mtime, size, and channels.
///
/// `mode_suffix` distinguishes caches for different preview styles (wave vs
/// spectrogram) so their PNGs never overwrite each other.
fn generate_preview_uuid(path: &Path, channels: u16, mode_suffix: &str) -> Result<String, String> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let metadata = std::fs::metadata(path).map_err(|e| format!("Failed to get meta {}", e))?;

    let mtime = metadata
        .modified()
        .map_err(|e| format!("Failed to get mtime: {}", e))?
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| format!("Failed to convert mtime: {}", e))?
        .as_secs();

    let size = metadata.len();

    let mut hasher = DefaultHasher::new();
    path.hash(&mut hasher);
    mtime.hash(&mut hasher);
    size.hash(&mut hasher);
    channels.hash(&mut hasher); // Include channels in hash
    mode_suffix.hash(&mut hasher); // …and the preview mode
    let hash = hasher.finish();

    // Use first 8 bytes of hash to build a stable UUID string
    let uuid_bytes = hash.to_le_bytes();
    let uuid = Uuid::from_bytes([
        uuid_bytes[0], uuid_bytes[1], uuid_bytes[2], uuid_bytes[3], uuid_bytes[4], uuid_bytes[5],
        uuid_bytes[6], uuid_bytes[7], uuid_bytes[0], uuid_bytes[1], uuid_bytes[2], uuid_bytes[3],
        uuid_bytes[4], uuid_bytes[5], uuid_bytes[6], uuid_bytes[7],
    ]);

    Ok(uuid.to_string())
}

fn preview_png_path(uuid: &str) -> PathBuf {
    crate::config::config_dir()
        .join("previews")
        .join(format!("{}.png", uuid))
}

/// Does a cached preview PNG with this UUID exist in the storage directory?
/// Used by the app to re-validate row UUIDs after a background purge — the
/// filesystem (not SQLite) is the source of truth for thumbnails.
pub fn preview_png_exists(uuid: &str) -> bool {
    preview_png_path(uuid).is_file()
}

/// Delete preview cache entries (DB rows + PNG files on disk) for the audio
/// files directly inside `dir` — non-recursive, matching exactly what the
/// file list shows for that directory. Returns the number of DB rows removed.
/// Used by the toolbar "Refresh preview" button.
///
/// Takes a live [`Database`] handle and opens an *independent* connection to
/// the same file internally (`db.reopen()`): this purge runs on a background
/// thread while the UI thread still owns its handle, and sharing one raw
/// SQLite connection across threads is not valid usage — the old code cloned
/// the handle into the worker, so the DELETEs were unreliable/invisible and
/// every cache entry survived the refresh.
///
/// BUGFIX (previews were "refreshed" but the identical images came back): the
/// previous implementation collected the PNG names from the *live* directory
/// listing via `load_preview(path, size, mtime, …)` and deleted rows with a
/// raw textual prefix match in SQL. Both steps silently matched nothing in
/// real usage:
/// - stored path strings are whatever spelling each writer used (`./dir/x`,
///   `dir//x`, symlinked absolute forms), so neither the `substr = dir || '/'`
///   DELETE nor the exact-text `load_preview` lookups hit any row;
/// - every miss left both the DB row and its deterministic PNG intact, so the
///   very next render call short-circuited on the still-existing cache and the
///   old thumbnails were re-displayed — "nothing was deleted".
///
/// The purge is now driven entirely by the **DB contents**:
/// 1. Snapshot `(path, uuid)` rows scoped to `dir` (component-wise parent
///    comparison, tolerant of path spellings).
/// 2. Derive each PNG location two ways and delete both if present:
///    a. the UUID recorded in the row (covers previews written under a
///       different rendering style or older format), and
///    b. the deterministic hash of the *exact stored path string* + current
///       file metadata + channel count, for both mode suffixes — this catches
///       PNGs whose rows never made it into the DB (the scanner treats the
///       filesystem as the source of truth and back-fills lazily).
/// 3. Delete the rows themselves.
///
/// BUGFIX #2 (refresh still showed the *old* images): the UI used to run
/// this purge **synchronously on the egui thread**. While it was busy
/// probing audio files and deleting PNGs, every [`UiEvent::PreviewReady`]
/// the still-running render threads sent into the channel was consumed by
/// the very next event loop with the *pre-purge* UUIDs (the events were
/// queued before the PNGs/rows went away) and re-loaded from the texture
/// cache — so the stale thumbnails were "unloaded and loaded again" and
/// nothing appeared deleted. The purge is now spawned onto its own thread
/// (see `AudioScoutApp::refresh_current_dir_previews`); the app keeps
/// pumping repaints until it reports completion via [`UiEvent::PreviewsPurged`],
/// and only then re-submits generation jobs — which can no longer race the
/// deletion.
pub fn delete_previews_for_dir_with_db(db: &Database, dir: &Path) -> std::io::Result<usize> {
    let db = db.reopen().map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    // Snapshot BEFORE deleting: the uuid column is the only pointer to the
    // PNG filenames that were registered in the DB.
    let rows = db
        .list_previews_in_dir_any_spelling(dir)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    let mut png_files: Vec<PathBuf> = Vec::new();
    for (path_str, uuid) in &rows {
        png_files.push(preview_png_path(uuid));

        // Also unlink the deterministic-name PNGs for this exact stored path
        // spelling (both styles), computed from live metadata when available.
        let p = Path::new(path_str);
        if let Ok(meta) = std::fs::metadata(p) {
            if meta.is_file() {
                let channels = crate::audio_io::probe_file_header(p).channels.max(1);
                for mode in [PreviewMode::Waveform, PreviewMode::Spectrogram] {
                    if let Ok(u) = generate_preview_uuid(p, channels, mode.uuid_suffix()) {
                        png_files.push(preview_png_path(&u));
                    }
                }
            }
        }
    }

    // Rows belonging to files that vanished from disk (or were saved under a
    // spelling the snapshot missed) must not survive either — this deletes by
    // normalised component scope over the whole table.
    let removed_rows = db
        .delete_all_previews_in_dir(dir)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    // PNGs of *all* rows scoped to the directory, including ones whose DB row
    // was missing (deterministic name from the live listing covers those).
    for path in crate::db::Database::top_level_files(dir) {
        let canon = path.clone();
        if let Ok(meta) = std::fs::metadata(&canon) {
            if !meta.is_file() {
                continue;
            }
            let channels = crate::audio_io::probe_file_header(&canon).channels.max(1);
            for mode in [PreviewMode::Waveform, PreviewMode::Spectrogram] {
                if let Ok(u) = generate_preview_uuid(&canon, channels, mode.uuid_suffix()) {
                    png_files.push(preview_png_path(&u));
                }
            }
        }
    }

    let mut removed_files = 0usize;
    for png in png_files {
        if std::fs::remove_file(&png).is_ok() {
            removed_files += 1;
        }
    }

    tracing::info!(
        "Refresh preview: purged {} DB rows and {} PNG files for {}",
        removed_rows,
        removed_files,
        dir.display()
    );
    Ok(removed_rows)
}

/// Same as [`delete_previews_for_dir_with_db`] but takes ownership of the
/// handle — used by the background purge thread spawned from the UI, which
/// opens its own connection via `Database::reopen` and moves it here.
pub fn delete_previews_for_dir_with_db_owned(db: Database, dir: &Path) -> std::io::Result<usize> {
    delete_previews_for_dir_with_db_no_reopen(&db, dir)
}

/// Delete **every** cached preview PNG that belongs to one audio file,
/// regardless of how (or whether) it was registered in SQLite. Returns the
/// list of unlinked files.
///
/// BUGFIX (context-menu "Refresh preview" deleted nothing): the per-file
/// purge used to derive the PNG filenames exclusively from DB rows plus a
/// re-hash of *live* metadata (`probe_file_header`, whose channel count can
/// differ from what the original render recorded). Any cache entry whose row
/// was missing or stale resolved to a filename that does not exist on disk,
/// so zero files were unlinked. Because the scanner treats the **filesystem
/// as the source of truth** for thumbnails (`cached_preview_png` only checks
/// that `<uuid>.png` exists), the untouched PNG made the very next render
/// short-circuit and hand back the identical image — "the preview is unloaded
/// from memory and the same file is loaded back; nothing is deleted".
///
/// This helper closes that hole deterministically: every candidate UUID is
/// recomputed for each spelling of the path stored in the DB **and** the live
/// listing, across *both* preview modes and channel counts 1..=8 (the small
/// set of values a header probe could ever have produced), and every existing
/// PNG behind those names is removed before any regeneration starts. It runs
/// on the purge's background thread (file I/O + header probes off the UI
/// thread), immediately before the DB half of the purge.
pub fn purge_preview_storage_for_file(file: &Path) -> Vec<PathBuf> {
    let db_path = crate::config::config_dir().join("app.db");
    purge_preview_storage_for_file_with_db(
        crate::db::Database::open(&db_path).ok().as_ref(),
        file,
    )
}

/// Body of [`purge_preview_storage_for_file`] with an optional live DB handle
/// used to enumerate the alternative path spellings recorded in `previews`.
pub(crate) fn purge_preview_storage_for_file_with_db(
    db: Option<&Database>,
    file: &Path,
) -> Vec<PathBuf> {
    // Collect every textual spelling this file is known under: the live path,
    // its canonical form (symlinks / `..` resolved — a render may have hashed
    // that) plus whatever strings previous writers stored in the DB
    // (`./dir/x.wav`, `dir//x.wav`, …). The deterministic PNG name hashes the
    // *exact* string, so all spellings must be tried to find the real cached
    // images.
    let mut spellings: Vec<PathBuf> = Vec::new();
    let mut add_spelling = |p: PathBuf| {
        if !spellings.contains(&p) {
            spellings.push(p);
        }
    };
    add_spelling(file.to_path_buf());
    if let Ok(canon) = std::fs::canonicalize(file) {
        add_spelling(canon);
    }
    if let Some(db) = db {
        if let Ok(rows) = db.list_previews_for_file_any_spelling(file) {
            for (p, _uuid) in rows {
                add_spelling(PathBuf::from(p));
            }
        }
    }

    let mut candidates: Vec<String> = Vec::new();
    for mode in [PreviewMode::Waveform, PreviewMode::Spectrogram] {
        for sp in &spellings {
            // Hashing needs the file's mtime/size; every spelling of an
            // existing file resolves through the same inode.
            if std::fs::metadata(sp).is_err() {
                continue;
            }
            for ch in 1u16..=8 {
                if let Ok(u) = generate_preview_uuid(sp, ch, mode.uuid_suffix()) {
                    candidates.push(u);
                }
            }
        }
    }

    let mut removed = Vec::new();
    for uuid in candidates {
        let png = preview_png_path(&uuid);
        if png.is_file() && std::fs::remove_file(&png).is_ok() {
            removed.push(png);
        }
    }
    removed
}

/// Purge the preview cache of a **single audio file** (context-menu
/// "Refresh preview"): every DB row belonging to `file` (any path spelling,
/// any preview style) plus every PNG behind those rows and the deterministic
/// names computed from live metadata. Mirrors the directory-wide purge scoped
/// down to one file; the same rules about running on a background thread with
/// an independently opened connection apply (see
/// [`delete_previews_for_dir_with_db`]).
pub fn delete_preview_for_file_with_db_owned(db: Database, file: &Path) -> std::io::Result<usize> {
    // Storage FIRST: unlink every PNG that could belong to this file before
    // touching the DB (`purge_preview_storage_for_file` explains why a
    // DB-row-driven filename list misses the real cache files). The DB handle
    // is used only to enumerate alternative path spellings here; the rows are
    // deleted afterwards, so no render job can observe "row gone, PNG still
    // cached" and short-circuit onto the stale image.
    let removed_files = purge_preview_storage_for_file_with_db(Some(&db), file).len();

    // Snapshot BEFORE deleting (kept for logging/diagnostics of what the rows
    // pointed at).
    let rows = db
        .list_previews_for_file_any_spelling(file)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    // Rows whose UUID names a PNG the deterministic sweep above did not cover
    // (e.g. previews written by an older format version) — unlink those too.
    let mut extra_files = 0usize;
    for (_path_str, uuid) in &rows {
        let png = preview_png_path(uuid);
        if png.is_file() && std::fs::remove_file(&png).is_ok() {
            extra_files += 1;
        }
    }

    // Remove the rows themselves (all modes / all spellings of this file).
    let removed_rows = db
        .delete_all_previews_for_file(file)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    tracing::info!(
        "Refresh preview: purged {} DB rows and {} PNG files for {}",
        removed_rows,
        removed_files + extra_files,
        file.display()
    );
    Ok(removed_rows)
}

/// Convenience wrapper that opens the app database itself. Prefer
/// [`delete_previews_for_dir_with_db`] when a live handle already exists —
/// it guarantees the purge works on the *same* database file the app uses
/// (which honours `AUDIOSCOUT_CONFIG_DIR` and any relocated config).
pub fn delete_previews_for_dir(dir: &Path) -> std::io::Result<usize> {
    let db_path = crate::config::config_dir().join("app.db");
    let db = crate::db::Database::open(&db_path)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    delete_previews_for_dir_with_db_no_reopen(&db, dir)
}

/// Same body as [`delete_previews_for_dir_with_db`] without the extra
/// connection (used when the caller already handed us a dedicated handle).
fn delete_previews_for_dir_with_db_no_reopen(db: &Database, dir: &Path) -> std::io::Result<usize> {
    let rows = db
        .list_previews_in_dir_any_spelling(dir)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    let mut png_files: Vec<PathBuf> = Vec::new();
    for (path_str, uuid) in &rows {
        png_files.push(preview_png_path(uuid));
        let p = Path::new(path_str);
        if let Ok(meta) = std::fs::metadata(p) {
            if meta.is_file() {
                let channels = crate::audio_io::probe_file_header(p).channels.max(1);
                for mode in [PreviewMode::Waveform, PreviewMode::Spectrogram] {
                    if let Ok(u) = generate_preview_uuid(p, channels, mode.uuid_suffix()) {
                        png_files.push(preview_png_path(&u));
                    }
                }
            }
        }
    }

    let removed_rows = db
        .delete_all_previews_in_dir(dir)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    for path in crate::db::Database::top_level_files(dir) {
        if let Ok(meta) = std::fs::metadata(&path) {
            if !meta.is_file() {
                continue;
            }
            let channels = crate::audio_io::probe_file_header(&path).channels.max(1);
            for mode in [PreviewMode::Waveform, PreviewMode::Spectrogram] {
                if let Ok(u) = generate_preview_uuid(&path, channels, mode.uuid_suffix()) {
                    png_files.push(preview_png_path(&u));
                }
            }
        }
    }

    let mut removed_files = 0usize;
    for png in png_files {
        if std::fs::remove_file(&png).is_ok() {
            removed_files += 1;
        }
    }

    tracing::info!(
        "Refresh preview: purged {} DB rows and {} PNG files for {}",
        removed_rows,
        removed_files,
        dir.display()
    );
    Ok(removed_rows)
}

/// Fast, *synchronous*, lock-free cache lookup used by the scanner: returns
/// the UUID of an already-generated preview if its PNG is sitting in the
/// previews directory.
///
/// The filename is deterministic (hash of path + mtime + size + channel
/// count), so a pure `stat()` answers "is this thumbnail already rendered?"
/// without touching SQLite or decoding any audio. Keeping the DB out of the
/// scan path matters: while render threads are busy writing rows, a scanning
/// thread issuing queries on a second connection could block behind those
/// writes and delay the entire `ScanDone` event — which is exactly what made
/// previews appear in the UI long after the PNGs showed up on disk.
pub fn cached_preview_png(path: &Path, channels: u16, mode: PreviewMode) -> Option<String> {
    let channels = if channels > 0 { channels } else { 1 };
    let uuid = generate_preview_uuid(path, channels, mode.uuid_suffix()).ok()?;
    if preview_png_path(&uuid).exists() {
        Some(uuid)
    } else {
        None
    }
}

/// Render a preview image (waveform or spectrogram) for the file list.
///
/// Waveform visual model taken from the `waveform-preview` example (per-track
/// min/max peak envelope, one column per pixel, mono → 1 track,
/// stereo/multichannel → first 2 tracks). Unlike the example — which loads the
/// *entire* signal into memory before building its peaks tree — the envelope
/// here is folded while streaming through the decoder and covers the **whole
/// file**: compressed formats are seek-sampled across the full timeline
/// (O(target × window) work regardless of duration) and PCM is streamed in a
/// single pass. [`PREVIEW_DECODE_BUDGET`] is unlimited by design so no tail is
/// ever truncated to silence.
///
/// Spectrogram mode decodes the full file and runs a windowed FFT over the
/// samples, averaging them into one column per pixel (`render::spectrogram`).
///
/// Returns the existing UUID when the preview PNG is already cached.
pub fn render_preview(path: &Path, mode: PreviewMode) -> Result<String, String> {
    const PREVIEW_WIDTH: u32 = 340;
    // Envelope resolution: ~4 points per horizontal pixel is plenty for a
    // 340-px thumbnail and keeps the reduce pass trivially cheap.
    const ENVELOPE_FRAMES: usize = (PREVIEW_WIDTH as usize) * 4;

    // Channel count straight from the file header (cheap probe) — picks the
    // layout and stabilises the UUID without decoding anything.
    let meta = crate::audio_io::probe_file_header(path);
    let channels = if meta.channels > 0 { meta.channels } else { 1 };

    let uuid = generate_preview_uuid(path, channels, mode.uuid_suffix())?;

    let previews_dir = crate::config::config_dir().join("previews");
    let _ = std::fs::create_dir_all(&previews_dir);
    let png_path = previews_dir.join(format!("{}.png", uuid));

    if png_path.exists() {
        return Ok(uuid);
    }

    // Use taller preview for stereo/multichannel (2 tracks need more space)
    let height = if channels == 1 { 20u32 } else { 40u32 };

    let img = match mode {
        PreviewMode::Waveform => {
            let (env, _sample_rate) = crate::audio_io::envelope::load_audio_envelope(
                path,
                ENVELOPE_FRAMES,
                PREVIEW_DECODE_BUDGET,
            )
            .map_err(|e| format!("Load failed: {}", e))?;

            crate::render::waveform::render_envelope(
                &env,
                PREVIEW_WIDTH,
                height,
                crate::render::waveform::Style::Dark,
            )
        }
        PreviewMode::Spectrogram => {
            // The FFT analysis needs real samples, not a min/max envelope.
            // Decode the *whole* file so the picture spans the full track;
            // `render_spectrogram_preview` then averages the signal into one
            // column per pixel (a fixed hop would only ever show the first
            // `width × hop` frames — i.e. just the beginning).
            let max_frames = usize::MAX / 8;
            let (samples, _sr, ch) =
                load_audio_bounded(path, max_frames)
                    .map_err(|e| format!("Load failed: {}", e))?;
            crate::render::spectrogram::render_spectrogram_preview(
                &samples,
                if ch > 0 { ch } else { channels },
                PREVIEW_WIDTH,
                height,
            )
        }
    };

    // Save as PNG
    img.save(&png_path).map_err(|e| format!("Save failed: {}", e))?;

    Ok(uuid)
}

/// Decode at most `max_frames` of audio (interleaved f32) for spectrogram
/// previews, reusing the standard loader and truncating afterwards. Keeping
/// the cap modest bounds both decode time and memory for long/large files.
fn load_audio_bounded(path: &Path, max_frames: usize) -> crate::audio_io::Result<(Vec<f32>, u32, u16)> {
    let (samples, sr, ch, st) = crate::audio_io::load_audio(path)?;
    let ch = (ch.max(1)) as usize;
    let keep_frames = max_frames.min(samples.len() / ch);
    let mut samples = samples;
    samples.truncate(keep_frames * ch);
    let _ = st;
    Ok((samples, sr, ch as u16))
}

/// Clean up orphaned preview files
pub fn cleanup_previews() -> Result<usize, String> {
    let config_dir = crate::config::config_dir();
    let previews_dir = config_dir.join("previews");

    if !previews_dir.exists() {
        return Ok(0);
    }

    let db_path = config_dir.join("app.db");
    let db =
        crate::db::Database::open(&db_path).map_err(|e| format!("Failed to open database: {}", e))?;

    // Get all valid UUIDs from database
    let valid_uuids = db
        .get_all_preview_uuids()
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

#[cfg(test)]
mod tests {
    use super::*;

    /// BUGFIX regression (context-menu "Refresh preview" deleted nothing):
    /// the per-file purge used to resolve PNG names only from DB rows plus a
    /// re-hash of *live* header metadata. When the row was missing/stale the
    /// sweep matched zero files, the PNG survived, and the scanner (which
    /// treats the storage dir as the source of truth) immediately handed the
    /// identical image back — "previews are unloaded and the same ones load
    /// again". The deterministic sweep must unlink the cached PNG even with
    /// an empty database, for both modes and any plausible channel count.
    #[test]
    fn purge_preview_storage_removes_png_without_db_rows() {
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("AUDIOSCOUT_CONFIG_DIR", tmp.path());

        // A real audio file on disk (the hash needs its mtime/size).
        let audio_dir = tempfile::tempdir().unwrap();
        let file = audio_dir.path().join("tone.wav");
        std::fs::write(&file, b"RIFF fake wav contents").unwrap();

        // Seed cached PNGs exactly the way the render pipeline names them:
        // deterministic hash of (path spelling, mtime, size, channels, mode).
        // Spellings used here: the live path, its canonical form and a `.`-
        // prefixed variant — all resolve to existing files, which is the rule
        // the purge sweep follows.
        let canon = std::fs::canonicalize(&file).unwrap();
        let odd_spelling = {
            let mut p = audio_dir.path().to_path_buf();
            p.push(".");
            p.push("tone.wav");
            p
        };
        let mut seeded = Vec::new();
        for sp in [&file, &odd_spelling, &canon] {
            for ch in [1u16, 2] {
                for mode in [PreviewMode::Waveform, PreviewMode::Spectrogram] {
                    if let Ok(uuid) = generate_preview_uuid(sp, ch, mode.uuid_suffix()) {
                        let png = preview_png_path(&uuid);
                        std::fs::create_dir_all(png.parent().unwrap()).unwrap();
                        std::fs::write(&png, b"png").unwrap();
                        seeded.push(png);
                    }
                }
            }
        }
        assert!(!seeded.is_empty());
        // An unrelated preview that must survive the per-file purge.
        let other = preview_png_path("00000000-0000-0000-0000-000000000000");
        std::fs::write(&other, b"png").unwrap();

        // No DB involvement at all: empty/nonexistent app.db.
        let removed = purge_preview_storage_for_file(&file);
        assert_eq!(
            removed.len(),
            seeded.len(),
            "every cached PNG of this file must be unlinked regardless of SQLite"
        );
        for png in &seeded {
            assert!(!png.exists(), "stale preview {} survived the purge", png.display());
        }
        assert!(other.exists(), "previews of other files must not be touched");

        std::env::remove_var("AUDIOSCOUT_CONFIG_DIR");
    }
}
