use audioscout_core::config::ConfigStore;
use audioscout_core::player::controller::Controller;
use audioscout_core::player::output::RodioOutput;
use audioscout_core::player::{MeterData, PlayheadState, TransportState};
use audioscout_core::scanner::Scanner;
use audioscout_core::table::{Column, ColumnLayout, FileRow};
use audioscout_core::audio_view::{AudioView, AudioViewMode};
use audioscout_core::undo::Stack as UndoStack;
use audioscout_core::db::Database;
use audioscout_core::audio_io::PcmFile;
use audioscout_core::disk::DiskProvider;
use audioscout_core::preview_worker::{PreviewWorker, PreviewJob};
use audioscout_core::{Document, EventChannel, Marker, PreviewMode, UiEvent, is_audio_file, is_editable_file};
use crossbeam_channel::Receiver;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use image::GenericImageView;

/// Main application state
pub struct AudioScoutApp {
    config_store: ConfigStore,
    db: Database,
    event_rx: Receiver<UiEvent>,
    event_tx: crossbeam_channel::Sender<UiEvent>,
    scanner: Scanner,
    preview_worker: PreviewWorker,
    current_dir: Option<PathBuf>,
    file_rows: Vec<FileRow>,
    scan_generation: u64,
    column_layout: ColumnLayout,
    selected_row: Option<usize>,
    filter_text: String,
    document: Document,
    active_file: Option<PathBuf>,
    player: Arc<Mutex<Controller>>,
    /// Real audio output (rodio/cpal, tz §5.2). `None` when no audio device
    /// could be opened — transport stays usable for browsing/editing.
    output: Option<RodioOutput>,
    /// Volume-meter display state (§3.5.1): live values + peak-hold timer.
    meter_live: MeterData,
    meter_hold: MeterData,
    meter_hold_until: f64,
    /// Wall-clock of the previous UI frame — drives the ~300 ms meter
    /// decay after pause/stop (§3.5.1 "При паузе/стопе шкалы гаснут до нуля").
    last_frame_time: f64,
    audio_view: AudioView,
    undo_stack: UndoStack,
    show_settings: bool,
    show_columns_dialog: bool,
    status_message: String,
    normalize_dbfs: f32,
    expanded_dirs: HashMap<PathBuf, Vec<PathBuf>>,
    disks: Vec<audioscout_core::disk::DiskInfo>,
    /// Paths with a preview job currently accepted by the worker (waiting for
    /// `UiEvent::PreviewReady`). Cleared when the result arrives or the scan
    /// generation changes.
    preview_inflight: HashSet<PathBuf>,
    /// Paths whose submit was rejected (duplicate / queue full); retried by
    /// `retry_missing_previews` while previews are pending.
    preview_retry: HashSet<PathBuf>,
    /// Active thumbnail style for the file list (Waveform / Spectrogram).
    /// Shared with the scanner (`Scanner::preview_mode_handle`) so cached
    /// lookups during a scan use the same mode as the submitted jobs.
    preview_mode: Arc<Mutex<PreviewMode>>,
    /// Per-mode texture caches. Keyed by UUID (which already embeds a
    /// mode-specific suffix), kept separate per mode so switching back and
    /// forth does not force re-uploading textures.
    preview_cache_wave: HashMap<String, egui::TextureHandle>,
    preview_cache_spectro: HashMap<String, egui::TextureHandle>,
    // Cached peaks tree for fast waveform rendering
    // cached_channels: Vec of (channel_data, peaks_tree) for each visible channel (1 for mono, 2 for stereo/multichannel)
    cached_channels: Vec<(Vec<f32>, audioscout_core::peaks_tree::Peaks)>,
    /// Texture of the live spectrogram image in the audio view. Re-rendered
    /// (at widget resolution) whenever the view range / document / panel size
    /// changes while `AudioView::mode` is Spectrogram — see
    /// `refresh_view_spectrogram`.
    view_spectro_texture: Option<egui::TextureHandle>,
    /// Key identifying the last rendered spectrogram image:
    /// (view start, view end, width, height, document generation). When it
    /// differs from the current state the picture is recomputed.
    view_spectro_key: Option<(u64, u64, u32, u32, u64)>,
    /// Bumped on every event that invalidates the spectrogram image (file
    /// load, edits/undo/redo, mode switch) so a re-render is triggered even
    /// when the numeric view key stays the same.
    document_generation: u64,
    /// 50 ms playhead poll bookkeeping (§1.2/§5.3): last time the UI-side
    /// fallback poll read the transport position while playing (the output
    /// thread is the primary source of PlayheadUpdate events; this throttle
    /// only drives the cursor when no real audio device is available).
    last_playhead_poll: std::time::Instant,
    /// BUGFIX (playback freezes ~0.5 s after start): set once per detected
    /// audio-stream stall so the status-bar message is shown a single time
    /// and re-armed when playback resumes (`poll_playhead` clears it on
    /// every non-stalled tick).
    stall_notified: bool,
    /// BUGFIX (cursor jumps ahead then snaps back while playing): the real
    /// audio clock (`consumed_samples`) lags behind wall time by the size of
    /// the sink/pump buffer (~150 ms) plus one 50 ms poll quantum. If the UI
    /// paints that raw lagging value, every poll tick moves the cursor in
    /// discrete steps and any transient over-forward position (e.g. a
    /// PlayheadUpdate event racing ahead of the hardware clock) must be
    /// "corrected" backwards on the next tick — the visible jump-ahead /
    /// snap-back oscillation, which grows as the buffered lead accumulates
    /// and is most noticeable on short files. Instead the UI extrapolates
    /// the clock forward between polls at exactly the nominal playback rate
    /// (never past the session end), so motion is smooth and monotonic; the
    /// small extrapolation error (< one poll quantum) is absorbed gradually
    /// because the estimate is always kept >= the true clock.
    playhead_est_frame: u64,
    /// Wall-clock instant that corresponds to `playhead_est_frame`.
    playhead_est_at: std::time::Instant,
    /// Session bounds [start,end) currently being played — used to clamp the
    /// extrapolated cursor so it can never run past the end of a short file.
    playhead_session: Option<(u64, u64)>,
    /// BUGFIX (cursor races ahead of the audio and stalls at the window edge
    /// while the file finishes playing): the free-running wall-clock estimate
    /// was previously allowed to lead the true hardware clock by an unbounded
    /// amount — every poll re-anchored the estimate at `max(estimate, clock)`
    /// instead of pulling it back toward the clock, so the visible lead only
    /// grew (it equals the whole pump/sink buffer depth plus event jitter once
    /// the pump has fed the last chunk). On a short file (e.g. 7 s) the cursor
    /// reached the right edge of the player window seconds before the audio
    /// actually finished and stood there while playback caught up. Now the
    /// estimate may lead the true clock by at most one poll quantum (~50 ms):
    /// each poll blends the raw clock back into the anchor whenever the lead
    /// exceeds that cap, which keeps motion smooth AND time-accurate.
    playhead_lead_cap_frames: u64,
    /// BUGFIX (cursor races ahead of the audio and stalls at the right edge
    /// while a short file finishes playing): hard ceiling for the visible
    /// cursor — the furthest FILE frame that has provably been handed to the
    /// Sink (`RodioOutput::playhead_ceiling_frame`, fed by the pump thread).
    /// Wall-clock extrapolation alone cannot know how much audio is actually
    /// queued ahead of the device; without this clamp the estimate drifts up
    /// to the full pump lead (~0.75 s+) past the audible position, reaches
    /// EOF and the window edge early, and stands there while the sound keeps
    /// playing — very visible on short files (e.g. 7 s). Cached here so the
    /// field can be invalidated on every session transition (Play / Seek /
    /// Stop) exactly like `playhead_est_frame`; the output thread republishes
    /// the honest value from the pump on its first append after a transition.
    playhead_fed_ceiling: Option<u64>,
}

impl AudioScoutApp {
    /// Expand a directory and all its parent directories in the tree
    fn expand_directory_tree(expanded_dirs: &mut HashMap<PathBuf, Vec<PathBuf>>, dir: &Path) {
        // Collect all ancestors from dir up to root
        let mut current = dir.to_path_buf();
        let mut ancestors = Vec::new();

        while current.parent().is_some() {
            ancestors.push(current.clone());
            if let Some(parent) = current.parent() {
                current = parent.to_path_buf();
            } else {
                break;
            }
        }

        // Reverse to process from root down to dir
        ancestors.reverse();

        // Expand each ancestor
        for ancestor in ancestors {
            if !expanded_dirs.contains_key(&ancestor) {
                if let Ok(entries) = std::fs::read_dir(&ancestor) {
                    let mut subdirs: Vec<PathBuf> = entries
                        .filter_map(|e| e.ok())
                        .filter(|e| e.path().is_dir())
                        .map(|e| e.path())
                        .collect();
                    subdirs.sort_by(|a, b| {
                        a.file_name()
                            .and_then(|n| n.to_str())
                            .map(|s| s.to_lowercase())
                            .cmp(&b.file_name().and_then(|n| n.to_str()).map(|s| s.to_lowercase()))
                    });
                    expanded_dirs.insert(ancestor, subdirs);
                }
            }
        }
    }

    pub fn new(
        _cc: &eframe::CreationContext<'_>,
        config_dir: PathBuf,
        start_dir: Option<PathBuf>,
    ) -> Self {
        let config_store = ConfigStore::new(&config_dir);
        let db_path = config_dir.join("app.db");
        let db = Database::open(&db_path).expect("Failed to open database");
        let channel = EventChannel::new();
        let event_tx = channel.sender.clone();
        let event_rx = channel.receiver;
        let player = Arc::new(Mutex::new(Controller::new(Arc::new(PlayheadState::new()))));
        // Single application-wide OutputStream, created once at startup and
        // kept alive for the whole session (tz §5.2). Without an audio device
        // RodioOutput::new returns None and playback is silently unavailable.
        let output = RodioOutput::new(player.clone(), event_tx.clone());
        // DEBUG (playback diagnostics): a None here means the audio device
        // never opened — Play can then only run the state machine, no sound.
        match &output {
            Some(_) => tracing::info!("[app] audio output initialized OK"),
            None => tracing::error!("[app] NO AUDIO OUTPUT (RodioOutput::new returned None) — playback will be silent; check '[output]' error lines above and libasound2/cpal device availability"),
        }
        let undo_dir = config_dir.join("undo");
        let undo_stack = UndoStack::new(undo_dir, config_store.config().undo_depth, config_store.config().undo_disk_limit_mb);
        let preview_worker = PreviewWorker::new(event_tx.clone());
        let scanner = Scanner::new();
        // Grab the shared preview-mode handle *before* `scanner` is moved
        // into the struct below (Rust evaluates struct fields in order, so
        // borrowing `scanner` after it has been moved would fail to compile).
        let preview_mode = scanner.preview_mode_handle();

        let mut app = Self {
            config_store, db, event_rx, event_tx,
            scanner, preview_worker,
            current_dir: None, file_rows: Vec::new(), scan_generation: 0,
            column_layout: ColumnLayout::new(), selected_row: None, filter_text: String::new(),
            document: Document::new(), active_file: None, player, output,
            meter_live: MeterData::default(), meter_hold: MeterData::default(),
            meter_hold_until: 0.0, last_frame_time: 0.0,
            audio_view: AudioView::new(0), undo_stack,
            preview_mode,
            preview_cache_wave: HashMap::new(),
            preview_cache_spectro: HashMap::new(),
            preview_inflight: HashSet::new(), preview_retry: HashSet::new(),
            show_settings: false, show_columns_dialog: false, status_message: String::new(),
            normalize_dbfs: -1.0, expanded_dirs: HashMap::new(),
            disks: Vec::new(),
            cached_channels: Vec::new(),
            view_spectro_texture: None,
            view_spectro_key: None,
            document_generation: 0,
            last_playhead_poll: std::time::Instant::now(),
            stall_notified: false,
            playhead_est_frame: 0,
            playhead_est_at: std::time::Instant::now(),
            playhead_session: None,
            // One 50 ms poll quantum at 192 kHz — the absolute worst case;
            // re-derived from the document's sample rate on every poll.
            playhead_lead_cap_frames: 9600,
            // No session engaged yet — the pump publishes the real ceiling
            // after its first append (see `poll_playhead`).
            playhead_fed_ceiling: None,
        };

        // Restore the two independent Waveform/Spectrogram toggles from the
        // config (§3.1 table previews, §3.5 player visualization).
        let saved_preview_mode = PreviewMode::from_str_name(&app.config_store.config().preview_mode);
        *app.preview_mode.lock().unwrap() = saved_preview_mode;
        app.scanner.set_preview_mode(saved_preview_mode);
        app.audio_view.mode = match app.config_store.config().audio_view_mode.as_str() {
            "spectro" => AudioViewMode::Spectrogram,
            _ => AudioViewMode::Waveform,
        };

        if let Some(dir_str) = app.db.load_session("lastDir") {
            let dir = PathBuf::from(dir_str);
            if dir.exists() {
                app.current_dir = Some(dir.clone());
                app.scanner.scan(dir.clone(), app.event_tx.clone());

                // Auto-expand the last active directory and all its parents
                Self::expand_directory_tree(&mut app.expanded_dirs, &dir);
            }
        } else if let Some(dir) = start_dir {
            app.current_dir = Some(dir.clone());
            app.scanner.scan(dir.clone(), app.event_tx.clone());
            Self::expand_directory_tree(&mut app.expanded_dirs, &dir);
        } else {
            // Fallback: expand home directory if no lastDir
            if let Some(home) = dirs_next::home_dir() {
                Self::expand_directory_tree(&mut app.expanded_dirs, &home);
            }
        }

        // Start background thread for disk auto-refresh
        let disk_event_tx = app.event_tx.clone();
        std::thread::spawn(move || {
            let mut provider = audioscout_core::disk::SysinfoDiskProvider::new();
            loop {
                provider.refresh();
                let disks = provider.list_disks();
                if disk_event_tx.send(UiEvent::DisksChanged { disks }).is_err() {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_secs(5));
            }
        });

        // Clean up orphaned preview files on startup
        if let Ok(removed) = audioscout_core::preview_worker::cleanup_previews() {
            if removed > 0 {
                tracing::info!("Cleaned up {} orphaned preview files", removed);
            }
        }

        app
    }

    /// Top-panel Waveform/Spectrogram toggle for the *table thumbnails*
    /// (§3.1). Flips the shared mode, re-scans the current directory so rows
    /// pick up cached PNGs of the other style immediately, and repumps the
    /// texture cache for whatever is missing.
    fn toggle_preview_mode(&mut self) {
        let new_mode = self.preview_mode.lock().unwrap().toggled();
        *self.preview_mode.lock().unwrap() = new_mode;
        self.scanner.set_preview_mode(new_mode);
        self.config_store.config_mut().preview_mode = new_mode.as_str().to_string();
        let _ = self.config_store.save();
        if let Some(dir) = self.current_dir.clone() {
            // Fresh scan: rows arrive with per-mode cached UUIDs; missing
            // ones fall through to `request_previews` via ScanDone.
            self.scanner.scan(dir, self.event_tx.clone());
        } else {
            self.request_previews();
        }
    }

    /// Player-panel Waveform/Spectrogram toggle for the *visualization area*
    /// (§3.5). Independent from the table thumbnail mode.
    fn toggle_audio_view_mode(&mut self) {
        self.audio_view.toggle_mode();
        let name = match self.audio_view.mode {
            AudioViewMode::Spectrogram => "spectro",
            AudioViewMode::Waveform => "wave",
        };
        self.config_store.config_mut().audio_view_mode = name.to_string();
        let _ = self.config_store.save();
    }

    fn request_previews(&mut self) {
        let generation = self.scan_generation;
        let mode = *self.preview_mode.lock().unwrap();

        // Clear queue to cancel any pending previews from previous directory
        self.preview_worker.clear_queue();

        for row in &self.file_rows {
            // Skip if preview already exists (scanner resolved it from cache,
            // or a previous batch delivered it)
            if row.preview_uuid.is_some() {
                continue;
            }

            // Submit preview job. If the worker drops the job (duplicate
            // already rendering, or saturated queue), remember the path so we
            // retry on the next scan instead of leaving ⏳ forever.
            if !self.preview_worker.submit(PreviewJob {
                path: row.path.clone(),
                generation,
                mode,
            }) {
                self.preview_retry.insert(row.path.clone());
            } else {
                self.preview_inflight.insert(row.path.clone());
            }
        }
    }

    /// Background preview generation is event-driven: without an active
    /// repaint loop, `PreviewReady` events could sit unread in the channel
    /// long after the PNGs appeared on disk. While any thumbnail is still
    /// missing, keep the UI pumping at a modest rate so results show up as
    /// soon as they arrive.
    fn previews_pending(&self) -> bool {
        self.file_rows.iter().any(|r| r.preview_uuid.is_none())
    }

    /// Re-submit rows that are neither cached nor currently in flight
    /// (e.g. jobs previously rejected because the queue was full).
    fn retry_missing_previews(&mut self) {
        let generation = self.scan_generation;
        let mode = *self.preview_mode.lock().unwrap();
        let missing: Vec<PathBuf> = self
            .file_rows
            .iter()
            .filter(|r| r.preview_uuid.is_none())
            .filter(|r| !self.preview_inflight.contains(&r.path))
            .map(|r| r.path.clone())
            .collect();

        for path in missing {
            if self
                .preview_worker
                .submit(PreviewJob { path: path.clone(), generation, mode })
            {
                self.preview_inflight.insert(path);
            }
        }
    }

    fn process_events(&mut self, ctx: &egui::Context) {
        while let Ok(event) = self.event_rx.try_recv() {
            match event {
                UiEvent::ScanDone { dir, rows, generation } => {
                    if Some(&dir) == self.current_dir.as_ref() && generation >= self.scan_generation {
                        self.scan_generation = generation;
                        self.file_rows = rows;
                        self.column_layout.sort_rows(&mut self.file_rows);
                        // Previous-generation bookkeeping is stale now.
                        self.preview_inflight.clear();
                        self.preview_retry.clear();
                        if let Some(file_str) = self.db.load_session("lastFile") {
                            let file_path = PathBuf::from(file_str);
                            if let Some(idx) = self.file_rows.iter().position(|r| r.path == file_path) {
                                self.selected_row = Some(idx);
                            }
                        }

                        // Request previews for all files
                        self.request_previews();
                        // New rows may need thumbnails → start the pump.
                        ctx.request_repaint();
                    }
                }
                UiEvent::PreviewReady { path, uuid, generation } => {
                    // Only update if this is the current generation
                    if generation == self.scan_generation {
                        self.preview_inflight.remove(&path);
                        self.preview_retry.remove(&path);
                        if let Some(row) = self.file_rows.iter_mut().find(|r| r.path == path) {
                            row.preview_uuid = Some(uuid);
                        }
                        // Repaint so the freshly delivered thumbnail becomes
                        // visible immediately instead of on the next input.
                        ctx.request_repaint();
                    }
                }
                UiEvent::PlayheadUpdate { position_secs } => {
                    let frame = (position_secs * self.document.sample_rate as f64) as u64;
                    // The output thread owns the audio clock while playing —
                    // mirror it into the UI state machine as a playback-side
                    // move (never a user seek, so the resume point stays put;
                    // tz §5.3 CursorSource::Playback).
                    if self.output.is_some() && self.player_state() == TransportState::Playing {
                        let mut ctrl = self.player.lock().unwrap_or_else(|p| p.into_inner());
                        ctrl.freeze_resume_at(frame, self.document.sample_rate);
                        drop(ctrl);
                        // BUGFIX (cursor jump-ahead / snap-back): this event
                        // carries the raw hardware clock, which LAGS wall time
                        // by the sink buffer depth. Feeding it straight into
                        // `set_cursor` made the painted cursor jump ahead
                        // (poll tick extrapolation) and then snap back here
                        // (stale true clock) every cycle. During live playback
                        // the poll in `poll_playhead` is the single source of
                        // truth for the visible cursor; this path only keeps
                        // the controller's resume point fresh.
                        if self.playhead_session.is_none() {
                            self.audio_view.set_cursor(frame);
                            self.audio_view.auto_scroll_to_cursor();
                        }
                    } else {
                        self.audio_view.set_cursor(frame);
                        self.audio_view.auto_scroll_to_cursor();
                    }
                }
                UiEvent::PlaybackFinished => { self.status_message = "Playback finished".to_string(); }
                UiEvent::VolumeMeterUpdate { peak_l, peak_r, rms_l, rms_r } => {
                    // Volume meter (§3.5.1): instantaneous scale with optional
                    // peak-hold; RMS scale is always live.
                    let cfg = self.config_store.config();
                    let now = ctx.input(|i| i.time);
                    self.meter_live = MeterData { peak_l, peak_r, rms_l, rms_r };
                    let hold = cfg.meter_hold_enabled;
                    let secs = cfg.meter_hold_seconds.max(0.2).min(5.0);
                    if hold {
                        if peak_l >= self.meter_hold.peak_l || now >= self.meter_hold_until {
                            self.meter_hold.peak_l = peak_l;
                        }
                        if peak_r >= self.meter_hold.peak_r || now >= self.meter_hold_until {
                            self.meter_hold.peak_r = peak_r;
                        }
                        if peak_l.max(peak_r) > 0.0 { self.meter_hold_until = now + secs; }
                    } else {
                        self.meter_hold = self.meter_live;
                        self.meter_hold_until = now;
                    }
                }
                UiEvent::DisksChanged { disks } => {
                    self.disks = disks;
                }
            }
        }
    }

    fn load_file(&mut self, path: &Path) {
        // All supported audio formats load into the player (§3.4: playback must
        // work for non-editable files too). "Editable" (PCM read/write) only
        // gates destructive edit operations (mute/normalize/save), never
        // loading or playback — see `apply_mute`/`apply_normalize`.
        if !is_audio_file(path) {
            self.status_message = "Unsupported file format".to_string();
            return;
        }
        self.transport_stop();
        self.player.lock().unwrap_or_else(|p| p.into_inner()).reset();
        match audioscout_core::audio_io::load_audio(path) {
            Ok((samples, sr, ch, st)) => {
                // DEBUG (playback diagnostics): frames==0 here means the file
                // decoded to an empty buffer — Play would compute end==start
                // and drop the append.
                tracing::info!("[load] OK {path:?} samples={} sr={sr} ch={ch} frames={}", samples.len(), samples.len() / (ch.max(1) as usize));
                // BUGFIX (sticky "not editable" message): a successful load
                // must clear any stale status from a previously rejected file.
                self.status_message.clear();
                self.document = Document {
                    samples, sample_rate: sr, channels: ch, markers: Vec::new(),
                    source_path: path.to_path_buf(), temp_path: None, dirty: false, sample_type: st,
                };
                if let Ok(meta) = std::fs::metadata(path) {
                    let size = meta.len();
                    let mtime = meta.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_secs()).unwrap_or(0);
                    if let Ok(markers) = self.db.load_markers(path, size, mtime) { self.document.markers = markers; }
                }
                self.audio_view = AudioView::new(self.document.total_frames());
                self.active_file = Some(path.to_path_buf());
                // Hand the fresh PCM buffer to the output thread (§5.1: stop
                // of previous output + resume-point reset happen there).
                if let Some(out) = &self.output {
                    out.load_document(Arc::new(self.document.clone()));
                }
                self.undo_stack.clear();
                let _ = self.db.save_session("lastFile", &path.to_string_lossy());
                let _ = self.db.save_session("lastPos", "0");

                // Build peaks tree cache for fast waveform rendering
                self.rebuild_peaks_cache();
            }
            Err(e) => { tracing::error!("[load] FAILED {path:?}: {e}"); self.status_message = format!("Load failed: {}", e); }
        }
    }

    /// Rebuild peaks tree cache after loading a new file or after edits
    fn rebuild_peaks_cache(&mut self) {
        let ch = self.document.channels as usize;
        let total_frames = self.document.total_frames() as usize;

        // Determine number of visible channels: 1 for mono, 2 for stereo/multichannel
        let visible_channels = if ch == 1 { 1 } else { 2 };

        self.cached_channels.clear();

        for channel_idx in 0..visible_channels {
            // Extract channel data from interleaved samples
            let channel_data: Vec<f32> = (0..total_frames)
                .map(|frame| {
                    let sample_idx = frame * ch + channel_idx;
                    if sample_idx < self.document.samples.len() {
                        self.document.samples[sample_idx]
                    } else {
                        0.0
                    }
                })
                .collect();

            // Build peaks tree for this channel
            let peaks = audioscout_core::peaks_tree::Peaks::build(&channel_data);
            self.cached_channels.push((channel_data, peaks));
        }

        // Any structural change to the document invalidates the live
        // spectrogram image; force a re-render on the next audio-view frame.
        self.document_generation += 1;
    }

    /// Rebuild the interleaved PCM slice covering the current view range and
    /// render it as a full-resolution spectrogram texture (audio-view mode
    /// switch, §3.6). Cheap no-op when nothing relevant changed since the
    /// last render — the hot `ui()` path stays idempotent.
    fn refresh_view_spectrogram(&mut self, ui: &egui::Ui, rect: egui::Rect) {
        let width = (rect.width().max(1.0)) as u32;
        let height = (rect.height().max(1.0)) as u32;
        let start = self.audio_view.view_range.start_frame;
        let end = self.audio_view.view_range.end_frame.min(self.document.total_frames());
        let key = (start, end, width, height, self.document_generation);
        if self.view_spectro_key == Some(key) && self.view_spectro_texture.is_some() {
            return;
        }

        let ch = self.document.channels.max(1) as usize;
        let a = (start as usize * ch).min(self.document.samples.len());
        let b = (end as usize * ch).min(self.document.samples.len());
        let rgba = audioscout_core::render::spectrogram::render_view_spectrogram(
            &self.document.samples[a..b],
            self.document.channels,
            width,
            height,
        );
        let color_image = egui::ColorImage::from_rgba_unmultiplied(
            [rgba.width() as usize, rgba.height() as usize],
            rgba.as_flat_samples().as_slice(),
        );
        self.view_spectro_texture = Some(ui.ctx().load_texture(
            "view_spectrogram",
            color_image,
            egui::TextureOptions::LINEAR,
        ));
        self.view_spectro_key = Some(key);
    }

    fn save_file(&mut self) {
        if !self.document.dirty { return; }
        let path = self.document.source_path.clone();
        if path.as_os_str().is_empty() { return; }
        match audioscout_core::audio_io::wav::WavFile::save(&path, &self.document.samples, self.document.sample_rate, self.document.channels, self.document.sample_type) {
            Ok(()) => {
                self.document.dirty = false;
                self.status_message = "File saved".to_string();
                if let Ok(meta) = std::fs::metadata(&path) {
                    let size = meta.len();
                    let mtime = meta.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_secs()).unwrap_or(0);
                    let _ = self.db.save_markers(&path, size, mtime, &self.document.markers);
                }
            }
            Err(e) => { self.status_message = format!("Save failed: {}", e); }
        }
    }

    fn mute_selection(&mut self) {
        if self.active_file.is_none() { self.status_message = "No active file".to_string(); return; }
        if !is_editable_file(&self.document.source_path) { self.status_message = "Not editable".to_string(); return; }
        let (start, end) = if let Some(sel) = self.audio_view.selection_range() { sel } else { (0, self.document.total_frames()) };
        let ch = self.document.channels as usize;
        let start_idx = start as usize * ch;
        let end_idx = (end as usize * ch).min(self.document.samples.len());
        let dump_path = self.undo_stack.new_dump_path();
        let original: Vec<f32> = self.document.samples[start_idx..end_idx].to_vec();
        if self.undo_stack.save_dump(&dump_path, &original).is_err() { self.status_message = "Undo dump failed".to_string(); return; }
        for s in &mut self.document.samples[start_idx..end_idx] { *s = 0.0; }
        self.undo_stack.push(audioscout_core::undo::UndoEntry {
            op: audioscout_core::undo::UndoOp::Mute { dump_path, start_frame: start, end_frame: end },
            file_path: self.document.source_path.clone(),
        });
        self.document.dirty = true;
        self.status_message = format!("Muted frames {}..{}", start, end);

        // Rebuild peaks cache after edit
        self.rebuild_peaks_cache();
    }

    fn normalize_selection(&mut self) {
        if self.active_file.is_none() { self.status_message = "No active file".to_string(); return; }
        let (start, end) = if let Some(sel) = self.audio_view.selection_range() { sel } else { (0, self.document.total_frames()) };
        let ch = self.document.channels as usize;
        let start_idx = start as usize * ch;
        let end_idx = (end as usize * ch).min(self.document.samples.len());
        let peak: f32 = self.document.samples[start_idx..end_idx].iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        if peak == 0.0 { self.status_message = "Peak is zero".to_string(); return; }
        let target = 10.0f64.powf(self.normalize_dbfs as f64 / 20.0) as f32;
        let gain = target / peak;
        let dump_path = self.undo_stack.new_dump_path();
        let original: Vec<f32> = self.document.samples[start_idx..end_idx].to_vec();
        if self.undo_stack.save_dump(&dump_path, &original).is_err() { return; }
        for s in &mut self.document.samples[start_idx..end_idx] { *s *= gain; }
        self.undo_stack.push(audioscout_core::undo::UndoEntry {
            op: audioscout_core::undo::UndoOp::Normalize { dump_path, start_frame: start, end_frame: end, dbfs: self.normalize_dbfs as f64 },
            file_path: self.document.source_path.clone(),
        });
        self.document.dirty = true;
        self.status_message = format!("Normalized to {} dBFS", self.normalize_dbfs);

        // Rebuild peaks cache after edit
        self.rebuild_peaks_cache();
    }

    fn undo(&mut self) {
        if let Some(entry) = self.undo_stack.pop_undo() {
            match &entry.op {
                audioscout_core::undo::UndoOp::Mute { dump_path, start_frame, end_frame }
                | audioscout_core::undo::UndoOp::Normalize { dump_path, start_frame, end_frame, .. } => {
                    if let Ok(original) = self.undo_stack.load_dump(dump_path) {
                        let ch = self.document.channels as usize;
                        let start_idx = *start_frame as usize * ch;
                        let end_idx = (*end_frame as usize * ch).min(self.document.samples.len());
                        let copy_len = (end_idx - start_idx).min(original.len());
                        self.document.samples[start_idx..start_idx + copy_len].copy_from_slice(&original[..copy_len]);
                    }
                }
                _ => {}
            }
            self.document.dirty = true;
            self.status_message = "Undo".to_string();

            // Rebuild peaks cache after undo
            self.rebuild_peaks_cache();
        }
    }

    fn redo(&mut self) {
        if let Some(entry) = self.undo_stack.pop_redo() {
            match &entry.op {
                audioscout_core::undo::UndoOp::Mute { start_frame, end_frame, .. } => {
                    let ch = self.document.channels as usize;
                    let start_idx = *start_frame as usize * ch;
                    let end_idx = (*end_frame as usize * ch).min(self.document.samples.len());
                    for s in &mut self.document.samples[start_idx..end_idx] { *s = 0.0; }
                }
                audioscout_core::undo::UndoOp::Normalize { start_frame, end_frame, dbfs, .. } => {
                    let ch = self.document.channels as usize;
                    let start_idx = *start_frame as usize * ch;
                    let end_idx = (*end_frame as usize * ch).min(self.document.samples.len());
                    let peak: f32 = self.document.samples[start_idx..end_idx].iter().map(|s| s.abs()).fold(0.0f32, f32::max);
                    if peak > 0.0 {
                        let target = 10.0f64.powf(*dbfs / 20.0) as f32;
                        let gain = target / peak;
                        for s in &mut self.document.samples[start_idx..end_idx] { *s *= gain; }
                    }
                }
                _ => {}
            }
            self.document.dirty = true;
            self.status_message = "Redo".to_string();

            // Rebuild peaks cache after redo
            self.rebuild_peaks_cache();
        }
    }

    fn add_marker(&mut self) {
        let frame = self.audio_view.cursor_frame;
        let time_sec = frame as f64 / self.document.sample_rate as f64;
        let name = format!("M{}", self.document.markers.len() + 1);
        self.document.markers.push(Marker { name, time_samples: frame, time_sec });
        self.document.markers.sort_by_key(|m| m.time_samples);
        self.status_message = format!("Marker added at {:.3}s", time_sec);
    }

    fn delete_active_marker(&mut self) {
        if let Some(idx) = self.audio_view.active_marker {
            if idx < self.document.markers.len() { self.document.markers.remove(idx); self.audio_view.active_marker = None; self.status_message = "Marker deleted".to_string(); }
        }
    }

    /// Transport play/pause routed through the real audio output when one is
    /// available; without a device it degrades to state-machine-only mode so
    /// the UI never hangs on missing audio (§1.2 "раньше времени не падают").
    fn transport_play_pause(&mut self) {
        let sel = self.audio_view.selection_range();
        let state = self.player_state();
        if state == TransportState::Playing {
            // Pause: freeze position + in-flight PCM (§5.3). Without a real
            // output device the state machine alone carries the semantics.
            tracing::debug!("[play] pause pressed (state={state:?})");
            if let Some(out) = &self.output { out.pause(); }
            // BUGFIX (cursor jump-ahead / snap-back): stop extrapolating while
            // paused; freeze the estimate at the true hardware clock so a
            // later resume re-arms it from a consistent point.
            self.playhead_session = None;
            let mut ctrl = self.player.lock().unwrap_or_else(|p| p.into_inner());
            ctrl.pause();
        } else {
            // Play: selection wins over the resume point (§3.5/§5.3).
            let start = sel.map(|(a, _)| a).unwrap_or_else(|| {
                self.player.lock().unwrap_or_else(|p| p.into_inner()).resume_frame()
            });
            // BUGFIX (play never starts): with no selection the session must
            // run to EOF. Passing 0 here made Cmd::Play compute `e > s` as
            // false for every start==0 file and silently drop the append —
            // the button toggled but nothing played.
            let end = sel.map(|(_, b)| b).unwrap_or(self.document.total_frames());
            // DEBUG (playback diagnostics): the full decision context at the
            // exact moment the user presses Play/Space. If playback does not
            // start, one of these values names the culprit:
            //   output=None          → audio device failed to open at startup
            //                          (see "[output] ... FAILED" lines),
            //   total_frames=0       → the document is empty/not loaded,
            //   e <= s               → degenerate range, dropped by apply_play.
            tracing::info!(
                "[play] pressed state={state:?} sel={sel:?} start={start} end={end} \
                 total_frames={} sr={} ch={} output={} path={:?}",
                self.document.total_frames(),
                self.document.sample_rate,
                self.document.channels,
                self.output.is_some(),
                self.active_file,
            );
            if let Some(out) = &self.output {
                out.play(start, end);
            }
            // BUGFIX (cursor jump-ahead / snap-back): (re)arm the UI-side
            // extrapolation exactly at the session start. The hardware clock
            // will read ~start for the next ~150 ms while the sink buffer
            // drains; without this re-arming the stale estimate from the
            // previous run would be blended in and the cursor would visibly
            // jump back to `start` after having shown a later position.
            self.playhead_est_frame = start;
            self.playhead_est_at = std::time::Instant::now();
            // BUGFIX (cursor races ahead of the audio and stalls at the right
            // edge while a short file finishes): drop the cached fed-ceiling —
            // the new session has nothing queued yet, and the output thread
            // republishes it as soon as the pump appends its first chunk.
            self.playhead_fed_ceiling = None;
            self.playhead_session = if end > start { Some((start, end)) } else { Some((start, self.document.total_frames())) };
            let mut ctrl = self.player.lock().unwrap_or_else(|p| p.into_inner());
            if self.output.is_some() {
                // With a live output the state machine must not advance on its
                // own clock — the real playhead arrives via PlayheadUpdate.
                ctrl.mark_playing(start, if end > start { end } else { self.document.total_frames() }, self.document.sample_rate);
            } else {
                ctrl.play(&self.document, sel);
            }
        }
    }

    fn transport_stop(&mut self) {
        if let Some(out) = &self.output { out.stop(); }
        // BUGFIX (cursor jump-ahead / snap-back): Stop rewinds the hardware
        // clock to the session start; drop any live extrapolation so the next
        // Play re-arms it from a consistent point.
        self.playhead_session = None;
        // BUGFIX (cursor races ahead of the audio): no queued samples remain
        // above the rewound clock — invalidate the cached fed-ceiling too.
        self.playhead_fed_ceiling = None;
        let mut ctrl = self.player.lock().unwrap_or_else(|p| p.into_inner());
        ctrl.stop();
    }

    fn transport_seek(&mut self, frame: u64, source: audioscout_core::player::CursorSource) {
        if let Some(out) = &self.output { out.seek(frame, source); }
        // BUGFIX (cursor jump-ahead / snap-back): a seek teleports the true
        // clock; move the extrapolation anchor with it immediately, otherwise
        // the cursor keeps sliding away from the new position at nominal rate
        // until the next poll drags it back.
        self.playhead_est_frame = frame;
        self.playhead_est_at = std::time::Instant::now();
        // BUGFIX (cursor races ahead of the audio): a seek teleports the true
        // clock AND discards everything queued above the target; rewind the
        // cached ceiling with it so the extrapolation cannot show frames that
        // are no longer (or not yet) audible. The output thread republishes
        // the real value from the pump on its first append after the seek.
        self.playhead_fed_ceiling = Some(frame);
        let mut ctrl = self.player.lock().unwrap_or_else(|p| p.into_inner());
        ctrl.seek(frame, &self.document, source);
    }

    fn transport_toggle_loop(&mut self) {
        let enabled = {
            let ctrl = self.player.lock().unwrap_or_else(|p| p.into_inner());
            !ctrl.is_looping()
        };
        if let Some(out) = &self.output { out.set_loop(enabled); }
        let mut ctrl = self.player.lock().unwrap_or_else(|p| p.into_inner());
        ctrl.set_loop(enabled);
    }

    fn transport_set_volume(&mut self, vol: f32) {
        if let Some(out) = &self.output { out.set_volume(vol); }
        let mut ctrl = self.player.lock().unwrap_or_else(|p| p.into_inner());
        ctrl.set_volume(vol);
    }

    fn player_state(&self) -> TransportState {
        self.player.lock().unwrap_or_else(|p| p.into_inner()).state()
    }

    /// UI-side playhead poll (§1.2/§5.3, 50 ms cadence driven by the repaint
    /// loop in `update`). The rodio output thread already emits
    /// `PlayheadUpdate` events from the real audio clock; this fallback keeps
    /// the cursor moving when no audio device is available (the state
    /// machine advances on its own clock via `Controller::tick`) and simply
    /// refreshes the cursor from the controller otherwise. `last_playhead_poll`
    /// throttles the work to one poll per 50 ms tick.
    fn poll_playhead(&mut self, ctx: &egui::Context) {
        if self.player_state() != TransportState::Playing {
            return;
        }
        let now = std::time::Instant::now();
        if now.duration_since(self.last_playhead_poll) < std::time::Duration::from_millis(50) {
            return;
        }
        self.last_playhead_poll = now;
        if self.output.is_some() {
            // BUGFIX (playback freezes ~0.5 s after start, no sound): the
            // audio stream can be killed by PulseAudio/ALSA while the app
            // still thinks it is Playing — then neither the cursor nor the
            // meter ever move again. `poll_stall` watches the real sample
            // clock and pauses the transport on a >1 s freeze; surface that
            // to the user once per stall instead of letting the UI hang in a
            // fake playing state.
            let stalled = self
                .output
                .as_mut()
                .map(|out| out.poll_stall())
                .unwrap_or(false);
            if stalled {
                self.stall_notified = true;
                // BUGFIX (regression "controller thread is gone"): a stall
                // reported while the transport still claims Playing must also
                // stop the state machine — otherwise poll_playhead returns at
                // its very first `player_state() != Playing` guard and never
                // runs again, leaving the UI frozen in a fake playing state.
                // `poll_stall` already paused the controller on clock-freeze;
                // do it here too so every stall path (including the dead-
                // controller one) leaves the app in Paused and Play can
                // restart the session.
                {
                    let mut ctrl = self.player.lock().unwrap_or_else(|p| p.into_inner());
                    if ctrl.state() == TransportState::Playing {
                        ctrl.pause();
                    }
                }
                self.status_message = "Playback stalled: audio stream was dropped \
                                       by the system (PulseAudio/ALSA). Press Play \
                                       to retry."
                    .to_string();
                ctx.request_repaint();
                return;
            }
            if self.stall_notified && self.player_state() == TransportState::Playing {
                // A fresh play attempt after a stall — clear the warning.
                self.stall_notified = false;
            }
            // BUGFIX (Play/Pause restarts from 0): the state machine's
            // current_frame must follow the real audio clock, otherwise
            // Pause freezes resume_frame at a stale value and every repeated
            // Play re-appends the slice from the session start. Read the
            // clock directly from SharedOutput every poll tick (tz §5.3: пул
            // позиции ±50 мс) — PlayheadUpdate events stay as an additional
            // fresher update path.
            let shared = self.output.as_ref().expect("output present").shared.clone();
            let ch = self.document.channels.max(1);
            let true_frame = shared.current_frame(ch);
            // BUGFIX (cursor jump-ahead / snap-back): paint an extrapolated
            // position instead of the raw hardware clock. The clock lags wall
            // time by the sink buffer (~150 ms) plus up to one 50 ms poll
            // quantum; drawing it directly makes the cursor crawl in coarse
            // steps and any transient forward reading gets snapped back on
            // the next tick — oscillation that accumulates with playback and
            // is very visible on short files. Estimate = max(true clock,
            // anchor advanced at the nominal rate), clamped to the session
            // end so a short file can never be over-run by the estimate.
            //
            // BUGFIX (cursor races ahead of the audio and stalls at the right
            // edge while the file finishes): `max(est, true_frame)` alone let
            // the estimate lead the real clock without bound — every poll
            // re-anchored at the already-inflated estimate, so the lead grew
            // to the full pump/sink buffer depth (the pump keeps ~1 s of
            // samples queued ahead of the device). On a 7 s clip the cursor
            // reached EOF (and the window edge) visibly early and froze there
            // while the audio kept playing. The estimate may now lead the
            // true clock by at most one poll quantum: when it drifts further
            // ahead, the anchor is pulled back to `true_clock + cap`, which
            // tracks the device within ~50 ms while staying smooth and
            // monotonic (never snaps backwards mid-session).
            let sr = self.document.sample_rate.max(1) as f64;
            let elapsed = self.playhead_est_at.elapsed().as_secs_f64().clamp(0.0, 0.25);
            let est = self
                .playhead_est_frame
                .saturating_add((elapsed * sr) as u64)
                .max(true_frame);
            self.playhead_lead_cap_frames = (0.05 * sr) as u64 + 1;
            let lead_ceiling = true_frame.saturating_add(self.playhead_lead_cap_frames);
            // BUGFIX (cursor races ahead of the audio and stalls at the right
            // edge while a short file finishes): the pump thread publishes the
            // furthest FILE frame it has handed to the Sink
            // (`playhead_ceiling_frame`). The visible cursor may never display
            // frames above that ceiling — they provably have not reached the
            // device queue yet, so no wall-clock extrapolation is allowed to
            // paint them. Combined with the one-quantum lead cap this pins the
            // cursor to the audible position: on a 7 s clip it now reaches the
            // right edge exactly when the sound does, instead of seconds early.
            // `None` (no session engaged / fresh Load-Pause-Stop reset) means
            // "nothing queued yet": fall back to the true clock so the cursor
            // cannot race ahead during the pre-buffer window.
            self.playhead_fed_ceiling = self
                .output
                .as_ref()
                .and_then(|out| out.playhead_ceiling_frame());
            let fed_ceiling = self.playhead_fed_ceiling.unwrap_or(true_frame);
            let frame = est.min(lead_ceiling).min(fed_ceiling);
            let frame = match self.playhead_session {
                Some((_s, e)) => frame.min(e.saturating_sub(1)),
                None => frame,
            };
            self.playhead_est_frame = frame;
            self.playhead_est_at = std::time::Instant::now();
            {
                let mut ctrl = self.player.lock().unwrap_or_else(|p| p.into_inner());
                ctrl.sync_playback_position_in(true_frame, Some(&self.document));
            }
            self.audio_view.set_cursor(frame);
            self.audio_view.auto_scroll_to_cursor();
            // Keep the 50 ms poll cadence alive even when egui would idle the
            // window (no input events during hands-free playback).
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
            return;
        }
        // Device-less degradation: advance the state machine's simulated
        // clock and mirror it into the waveform cursor.
        let frame = {
            let mut ctrl = self.player.lock().unwrap_or_else(|p| p.into_inner());
            ctrl.tick(std::time::Duration::from_millis(50), &self.document);
            ctrl.current_frame()
        };
        self.audio_view.set_cursor(frame);
        self.audio_view.auto_scroll_to_cursor();
    }

    fn player_current_secs(&self) -> f64 {
        let ctrl = self.player.lock().unwrap_or_else(|p| p.into_inner());
        ctrl.current_secs(self.document.sample_rate)
    }

    /// Two vertical scales per channel — instantaneous peak (+hold marker)
    /// and RMS — drawn at the right side of the player window (§3.5.1).
    fn draw_volume_meter(&self, ui: &egui::Ui, rect: egui::Rect) {
        let painter = ui.painter();
        let ch = self.document.channels.max(1);
        let vis_channels = if ch == 1 { 1usize } else { 2usize };
        let bar_w = ((rect.width() - 8.0 * vis_channels as f32) / vis_channels as f32 * 0.5).max(6.0);
        let gap = 8.0f32;
        let mut x = rect.min.x;
        for c in 0..vis_channels {
            let (live_peak, live_rms, hold_peak) = if c == 0 {
                (self.meter_live.peak_l, self.meter_live.rms_l, self.meter_hold.peak_l)
            } else {
                (self.meter_live.peak_r, self.meter_live.rms_r, self.meter_hold.peak_r)
            };
            let h = rect.height();
            let base = rect.max.y;
            // background troughs
            for dx in [0.0f32, bar_w + 2.0] {
                painter.rect_filled(
                    egui::Rect::from_min_max(egui::pos2(x + dx, base - h), egui::pos2(x + dx + bar_w, base)),
                    2.0,
                    egui::Color32::from_gray(40),
                );
            }
            let peak_y = base - h * live_peak.clamp(0.0, 1.0);
            let rms_y = base - h * live_rms.clamp(0.0, 1.0);
            // instantaneous scale (green→red by level)
            let peak_col = if live_peak > 0.95 { egui::Color32::RED } else { egui::Color32::from_rgb(80, 220, 80) };
            painter.rect_filled(egui::Rect::from_min_max(egui::pos2(x, peak_y), egui::pos2(x + bar_w, base)), 1.0, peak_col);
            // RMS scale (blue)
            painter.rect_filled(egui::Rect::from_min_max(egui::pos2(x + bar_w + 2.0, rms_y), egui::pos2(x + 2.0 * bar_w + 2.0, base)), 1.0, egui::Color32::LIGHT_BLUE);
            // peak-hold marker line
            if self.config_store.config().meter_hold_enabled {
                let hy = base - h * hold_peak.clamp(0.0, 1.0);
                // egui 0.29: `line_segment` takes [Pos2; 2] and a single
                // `impl Into<PathStroke>` — width+color go in as one tuple.
                painter.line_segment(
                    [egui::pos2(x, hy), egui::pos2(x + bar_w, hy)],
                    (1.5_f32, egui::Color32::YELLOW),
                );
            }
            if vis_channels == 2 {
                painter.text(egui::pos2(x + bar_w, base + 0.0 - 2.0), egui::Align2::CENTER_BOTTOM, if c == 0 { "L" } else { "R" },
                    egui::FontId::proportional(9.0), ui.visuals().text_color());
            }
            x += 2.0 * bar_w + gap + 2.0;
        }
    }

    fn navigate_file(&mut self, direction: i32) {
        if self.file_rows.is_empty() { return; }
        let current_idx = self.selected_row.unwrap_or(0) as i32;
        let new_idx = if direction > 0 { (current_idx + 1).min(self.file_rows.len() as i32 - 1) } else { (current_idx - 1).max(0) };
        if new_idx >= 0 && (new_idx as usize) < self.file_rows.len() {
            self.selected_row = Some(new_idx as usize);
            let path = self.file_rows[new_idx as usize].path.clone();
            self.load_file(&path);
        }
    }

    fn render_folder_node(
        &mut self,
        ui: &mut egui::Ui,
        path: &Path,
        name: &str,
        depth: usize,
        dir_to_select: &mut Option<PathBuf>,
        dir_to_expand: &mut Option<PathBuf>,
        dir_to_collapse: &mut Option<PathBuf>,
    ) {
        let is_current = self.current_dir.as_ref() == Some(&path.to_path_buf());
        let is_expanded = self.expanded_dirs.contains_key(path);

        let text_color = if is_current {
            egui::Color32::YELLOW
        } else {
            ui.visuals().text_color()
        };

        // Create a button-like label for the folder
        let icon = if is_expanded { "▼" } else { "▶" };
        let label = format!("{} {} {}", icon, if is_expanded { "📂" } else { "📁" }, name);

        let response = ui.add(
            egui::Button::new(egui::RichText::new(&label).color(text_color))
                .fill(if is_current { egui::Color32::from_rgba_premultiplied(50, 50, 50, 100) } else { egui::Color32::TRANSPARENT })
                .stroke(egui::Stroke::NONE)
        );

        // Change cursor to hand on hover
        if response.hovered() {
            ui.output_mut(|o| o.cursor_icon = egui::CursorIcon::PointingHand);
        }

        // Handle clicks
        if response.clicked() {
            // Single click - select the directory
            *dir_to_select = Some(path.to_path_buf());
            // Also toggle expand/collapse
            if is_expanded {
                *dir_to_collapse = Some(path.to_path_buf());
            } else {
                *dir_to_expand = Some(path.to_path_buf());
            }
        }

        // Render children if expanded
        if is_expanded {
            ui.indent(egui::Id::new(path), |ui| {
                if let Some(children) = self.expanded_dirs.get(path).cloned() {
                    for child in children {
                        let child_name = child.file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("")
                            .to_string();

                        // Skip hidden directories
                        if child_name.starts_with('.') {
                            continue;
                        }

                        // Recursive call for child
                        self.render_folder_node(
                            ui,
                            &child,
                            &child_name,
                            depth + 1,
                            dir_to_select,
                            dir_to_expand,
                            dir_to_collapse,
                        );
                    }
                }
            });
        }
    }

    fn handle_keyboard(&mut self, ctx: &egui::Context) {
        let config = self.config_store.config().clone();
        ctx.input(|i| {
            for event in &i.events {
                if let egui::Event::Key { key, pressed: true, modifiers, .. } = event {
                    for (action, binding) in &config.hotkeys {
                        let key_match = match binding.key.as_str() {
                            "Space" => *key == egui::Key::Space,
                            "Delete" => *key == egui::Key::Delete,
                            "Escape" => *key == egui::Key::Escape,
                            "ArrowUp" => *key == egui::Key::ArrowUp,
                            "ArrowDown" => *key == egui::Key::ArrowDown,
                            other => other.chars().next().and_then(|_| egui::Key::from_name(other)).map(|k| *key == k).unwrap_or(false),
                        };
                        let mod_match = binding.ctrl == modifiers.ctrl && binding.shift == modifiers.shift && binding.alt == modifiers.alt;
                        if key_match && mod_match {
                            match action.as_str() {
                                "playpause" => self.transport_play_pause(),
                                "mute" => self.mute_selection(),
                                "normalize" => self.normalize_selection(),
                                "undo" => self.undo(),
                                "redo" => self.redo(),
                                "addMarker" => self.add_marker(),
                                "deleteMarker" => self.delete_active_marker(),
                                "clearSelection" => self.audio_view.clear_selection(),
                                "zoomIn" => self.audio_view.zoom_in(),
                                "zoomOut" => self.audio_view.zoom_out(),
                                "zoomReset" => self.audio_view.zoom_reset(),
                                "prev" => self.navigate_file(-1),
                                "next" => self.navigate_file(1),
                                "gotoStart" => { self.audio_view.set_cursor(0); self.transport_seek(0, audioscout_core::player::CursorSource::User); }
                                _ => {}
                            }
                        }
                    }
                    if *key == egui::Key::ArrowLeft || *key == egui::Key::ArrowRight {
                        let direction = if *key == egui::Key::ArrowRight { 1 } else { -1 };
                        if self.player_state() == TransportState::Playing {
                            let target = {
                                let mut ctrl = self.player.lock().unwrap_or_else(|p| p.into_inner());
                                ctrl.skip(&self.document, direction * config.skip_seconds as i32);
                                ctrl.current_frame()
                            };
                            self.transport_seek(target, audioscout_core::player::CursorSource::User);
                            let frame = target;
                            self.audio_view.set_cursor(frame);
                        } else {
                            let step = if self.audio_view.view_range.len() > 100 { self.audio_view.view_range.len() / 100 } else { 1 };
                            let new_pos = if direction > 0 { self.audio_view.cursor_frame.saturating_add(step) } else { self.audio_view.cursor_frame.saturating_sub(step) };
                            self.audio_view.set_cursor(new_pos);
                        }
                    }
                }
            }
        });
    }
}

impl eframe::App for AudioScoutApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Meter decay when nothing new arrives (pause/stop): exponential-ish
        // fade to zero over ~300 ms (§3.5.1). While playing, fresh
        // VolumeMeterUpdate events overwrite these values every poll tick.
        {
            let now = ctx.input(|i| i.time);
            let dt = (now - self.last_frame_time).clamp(0.0, 0.1) as f32;
            self.last_frame_time = now;
            if self.player_state() != TransportState::Playing {
                const DECAY: f32 = 10.0; // ~1/e per 100 ms → silent in ~300 ms
                let k = (1.0 - DECAY * dt).max(0.0);
                self.meter_live.peak_l *= k; self.meter_live.peak_r *= k;
                self.meter_live.rms_l *= k; self.meter_live.rms_r *= k;
                self.meter_hold.peak_l *= k; self.meter_hold.peak_r *= k;
            }
        }
        self.process_events(ctx);
        self.handle_keyboard(ctx);
        // 50 ms playhead poll (§1.2/§5.3): with a live output the position
        // arrives via PlayheadUpdate events; without one this fallback tick
        // advances the state machine's clock so the cursor still moves.
        self.poll_playhead(ctx);
        if self.player_state() == TransportState::Playing { ctx.request_repaint_after(std::time::Duration::from_millis(50)); }

        // Preview pump: while any thumbnail is still missing, keep draining
        // events and repainting on a fixed cadence. Without this, egui's
        // on-demand redraw could leave finished previews invisible (and their
        // `PreviewReady` events unread in the channel) until the user moved
        // the mouse — the "PNGs exist on disk but appear late in the UI" bug.
        if self.previews_pending() {
            self.retry_missing_previews();
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }

        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Filter:");
                ui.text_edit_singleline(&mut self.filter_text);
                if ui.button("✕").clicked() { self.filter_text.clear(); }
                ui.separator();
                if ui.button("Columns").clicked() { self.show_columns_dialog = true; }
                ui.separator();
                // Table thumbnail style toggle (§3.1): the code to switch
                // modes existed but had no control — this button exposes the
                // current state and flips it atomically for the whole set.
                let preview_spectro = *self.preview_mode.lock().unwrap() == PreviewMode::Spectrogram;
                let preview_btn = egui::SelectableLabel::new(
                    preview_spectro,
                    if preview_spectro { "Spectrogram" } else { "Waveform" },
                );
                if ui.add(preview_btn).on_hover_text("Toggle table previews between waveform and spectrogram").clicked() {
                    self.toggle_preview_mode();
                }
                // NOTE: the second Waveform/Spectrogram toggle that used to sit
                // here (visualization-area mode) was removed by request — the
                // audio-view display mode has exactly ONE control now, the one
                // on the player panel below.
                ui.separator();
                if ui.button("Settings").clicked() { self.show_settings = true; }
            });
        });

        // Declare paths_to_load BEFORE CentralPanel to avoid borrow conflicts
        let mut paths_to_load: Vec<PathBuf> = Vec::new();

        // Left panel - Folder tree
        // Collect actions to perform after the closure (avoid borrow conflicts)
        let mut dir_to_select: Option<PathBuf> = None;
        let mut dir_to_expand: Option<PathBuf> = None;
        let mut dir_to_collapse: Option<PathBuf> = None;

        egui::SidePanel::left("folder_tree")
            .default_width(200.0)
            .resizable(true)
            .show(ctx, |ui| {
                ui.heading("Folders");
                ui.separator();

                egui::ScrollArea::vertical().show(ui, |ui| {
                    // Show Home directory
                    if let Some(home) = dirs_next::home_dir() {
                        self.render_folder_node(
                            ui,
                            &home,
                            "Home",
                            0,
                            &mut dir_to_select,
                            &mut dir_to_expand,
                            &mut dir_to_collapse,
                        );
                    }

                    // Show root directory
                    let root = PathBuf::from("/");
                    self.render_folder_node(
                        ui,
                        &root,
                        "/",
                        0,
                        &mut dir_to_select,
                        &mut dir_to_expand,
                        &mut dir_to_collapse,
                    );

                    // Show connected disks (excluding Home and /)
                    let home = dirs_next::home_dir();
                    let disks_to_render: Vec<(PathBuf, String)> = self.disks.iter()
                        .filter(|disk| {
                            // Skip if it's Home or root (already shown)
                            if home.as_ref() == Some(&disk.mount_point) {
                                return false;
                            }
                            if disk.mount_point == PathBuf::from("/") {
                                return false;
                            }
                            true
                        })
                        .map(|disk| (disk.mount_point.clone(), disk.name.clone()))
                        .collect();

                    for (mount_point, name) in disks_to_render {
                        self.render_folder_node(
                            ui,
                            &mount_point,
                            &name,
                            0,
                            &mut dir_to_select,
                            &mut dir_to_expand,
                            &mut dir_to_collapse,
                        );
                    }
                });
            });

        // Apply actions after closure
        if let Some(dir) = dir_to_select {
            self.current_dir = Some(dir.clone());
            self.scanner.scan(dir, self.event_tx.clone());
        }
        if let Some(dir) = dir_to_expand {
            if let Ok(entries) = std::fs::read_dir(&dir) {
                let mut subdirs: Vec<PathBuf> = entries
                    .filter_map(|e| e.ok())
                    .filter(|e| e.path().is_dir())
                    .map(|e| e.path())
                    .collect();
                subdirs.sort_by(|a, b| {
                    a.file_name()
                        .and_then(|n| n.to_str())
                        .map(|s| s.to_lowercase())
                        .cmp(&b.file_name().and_then(|n| n.to_str()).map(|s| s.to_lowercase()))
                });
                self.expanded_dirs.insert(dir, subdirs);
            }
        }
        if let Some(dir) = dir_to_collapse {
            self.expanded_dirs.remove(&dir);
        }

        // Bottom stack — declaration order matters: in egui the FIRST declared
        // bottom panel docks to the very bottom edge of the window and each
        // subsequent one is placed above the previous. So this declaration
        // order (audio_view → player → actions → status) yields the visual
        // stack, top → bottom: status bar, file actions (Mute/Apply…),
        // player controls (Play/Stop…), visualization window. The transport
        // panel therefore sits *above* the file-content view, directly
        // under the file-actions panel (bug #1). All four panels are declared
        // before the CentralPanel below, so the central area's
        // `available_height()` already accounts for their real, measured
        // sizes — the visualization window is resizable, its height changes
        // at runtime, and the file list shrinks/grows with it instead of
        // rendering underneath the player windows (bug #2).

        // Audio view panel (visualization area) — declared FIRST among the
        // bottom panels so it docks to the very bottom edge of the window;
        // the player controls are placed *above* it, and the file-actions
        // panel above those (bug #1: Play/Stop must sit under the
        // file-actions panel and above the audio-content view).
        egui::TopBottomPanel::bottom("audio_view").default_height(200.0).resizable(true).show(ctx, |ui| {
            let (rect, response) = ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
            ui.painter().rect_filled(rect, 0.0, egui::Color32::from_rgb(25, 25, 30));

            if !self.document.samples.is_empty() {
                let num_tracks = self.cached_channels.len();
                // BUGFIX (cursor drawn behind the waveform): the playhead line
                // used to be painted BEFORE the spectrogram image / waveform
                // columns, so it ended up underneath them and was only visible
                // through gaps. Track where overlays must start: everything
                // below is drawn on top of the audio content (§3.5 — курсор
                // поверх дорожек).
                let overlay_start_y = rect.top();

                // Spectrogram mode: draw the pre-rendered widget-resolution
                // image first; markers/selection/cursor overlay on top of it.
                if self.audio_view.mode == AudioViewMode::Spectrogram {
                    self.refresh_view_spectrogram(ui, rect);
                    if let Some(tex) = &self.view_spectro_texture {
                        ui.put(rect, egui::Image::new(tex).fit_to_exact_size(rect.size()));
                    }
                }

                // The per-channel 0 dBFS middle lines are NOT drawn here any
                // more: in waveform mode they are painted at the end of each
                // track's loop iteration below (on top of that track's wave
                // columns), and in spectrogram mode `render_view_spectrogram`
                // bakes them into the image. Drawing a single line only for
                // mono here used to leave stereo/multichannel tracks without
                // a zero reference.

                // Markers
                for (idx, marker) in self.document.markers.iter().enumerate() {
                    let mx = self.audio_view.frame_to_x(marker.time_samples, rect.width() as u32);
                    let screen_x = rect.left() + mx;
                    if screen_x >= rect.left() && screen_x <= rect.right() {
                        let color = if self.audio_view.active_marker == Some(idx) { egui::Color32::YELLOW } else { egui::Color32::from_rgb(100, 180, 255) };
                        let mut y = overlay_start_y;
                        while y < rect.bottom() {
                            let seg_end = (y + 4.0).min(rect.bottom());
                            ui.painter().line_segment([egui::pos2(screen_x, y), egui::pos2(screen_x, seg_end)], egui::Stroke::new(1.0_f32, color));
                            y += 8.0;
                        }
                        ui.painter().text(egui::pos2(screen_x + 2.0, overlay_start_y + 2.0), egui::Align2::LEFT_TOP, &marker.name, egui::FontId::proportional(10.0), color);
                    }
                }

                // Selection
                if let Some(sel) = self.audio_view.selection_range() {
                    let x1 = rect.left() + self.audio_view.frame_to_x(sel.0, rect.width() as u32);
                    let x2 = rect.left() + self.audio_view.frame_to_x(sel.1, rect.width() as u32);
                    let sel_rect = egui::Rect::from_min_max(egui::pos2(x1.max(rect.left()), overlay_start_y), egui::pos2(x2.min(rect.right()), rect.bottom()));
                    ui.painter().rect_filled(sel_rect, 0.0, egui::Color32::from_rgba_premultiplied(80, 120, 255, 60));
                }

                // Waveform - using cached Peaks tree for fast rendering
                // Render 1 track for mono, 2 tracks for stereo/multichannel
                let start_frame = self.audio_view.view_range.start_frame;
                let end_frame = self.audio_view.view_range.end_frame;
                let view_frames = end_frame.saturating_sub(start_frame);

                if view_frames > 0 && num_tracks > 0 && self.audio_view.mode == AudioViewMode::Waveform {
                    let track_height = rect.height() / num_tracks as f32;
                    let pixels = rect.width() as u32;
                    let step = view_frames as f64 / pixels as f64;
                    let start = start_frame as f64;

                    for (track_idx, (channel_data, peaks)) in self.cached_channels.iter().enumerate() {
                        let track_top = rect.top() + track_idx as f32 * track_height;
                        let track_mid_y = track_top + track_height / 2.0;

                        // Draw separator line between tracks (except before first)
                        if track_idx > 0 {
                            ui.painter().line_segment(
                                [egui::pos2(rect.left(), track_top), egui::pos2(rect.right(), track_top)],
                                egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(60, 60, 65))
                            );
                        }

                        // Get min/max for each pixel column using cached peaks tree
                        let columns = peaks.spans(channel_data, start, step, pixels as usize);

                        let half_h = track_height / 2.0;
                        for (px, &(min_val, max_val)) in columns.iter().enumerate() {
                            let x = rect.left() + px as f32;
                            let y_top = track_mid_y - max_val * half_h;
                            let y_bot = track_mid_y - min_val * half_h;
                            ui.painter().line_segment(
                                [egui::pos2(x, y_top), egui::pos2(x, y_bot)],
                                egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(50, 200, 80))
                            );
                        }

                        // Per-channel middle line at 0 dBFS (amplitude 0),
                        // painted AFTER this track's wave columns so it stays
                        // visible on top of dense signal — one line for every
                        // channel (mono: 1, stereo/multichannel: 2), matching
                        // the table-preview renderer.
                        ui.painter().line_segment(
                            [egui::pos2(rect.left(), track_mid_y.round()), egui::pos2(rect.right(), track_mid_y.round())],
                            egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(148, 163, 184))
                        );
                    }
                }

                // Cursor — BUGFIX (отрисовку надо переместить наверх, сейчас
                // курсор находится за waveform): the playhead is painted LAST
                // inside this panel, i.e. on top of the spectrogram image,
                // the waveform columns and the selection highlight, so it is
                // always fully visible while tracking playback (§3.5).
                let cursor_x = self.audio_view.frame_to_x(self.audio_view.cursor_frame, rect.width() as u32);
                let cursor_screen_x = rect.left() + cursor_x;
                if cursor_screen_x >= rect.left() && cursor_screen_x <= rect.right() {
                    ui.painter().line_segment([egui::pos2(cursor_screen_x, overlay_start_y), egui::pos2(cursor_screen_x, rect.bottom())], egui::Stroke::new(1.5_f32, egui::Color32::RED));
                    // Small "handle" triangle at the very TOP of the view —
                    // the user asked for the cursor to sit above everything;
                    // the marker makes it readable even over dense waveforms.
                    ui.painter().add(egui::Shape::convex_polygon(
                        vec![
                            egui::pos2(cursor_screen_x - 4.0, overlay_start_y),
                            egui::pos2(cursor_screen_x + 4.0, overlay_start_y),
                            egui::pos2(cursor_screen_x, overlay_start_y + 6.0),
                        ],
                        // Color32 already satisfies `Into<Color32>`; the `.into()` call made
                        // the target type ambiguous (E0283) since Color32 converts into Hsva,
                        // Rgba and others too.
                        egui::Color32::RED,
                        egui::Stroke::NONE,
                    ));
                }
            }

            if response.clicked() {
                if let Some(pos) = response.interact_pointer_pos() {
                    let x_ratio = (pos.x - rect.left()) / rect.width();
                    let frame = self.audio_view.x_to_frame(x_ratio * rect.width(), rect.width() as u32);
                    self.audio_view.set_cursor(frame);
                    self.transport_seek(frame, audioscout_core::player::CursorSource::User);
                }
            }

            ui.input(|i| {
                let scroll = i.smooth_scroll_delta;
                if i.modifiers.ctrl {
                    if scroll.y > 0.0 { if let Some(pos) = i.pointer.hover_pos() { let x_ratio = ((pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0); self.audio_view.zoom_at_x(1.25, x_ratio as f64); } }
                    else if scroll.y < 0.0 { if let Some(pos) = i.pointer.hover_pos() { let x_ratio = ((pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0); self.audio_view.zoom_at_x(1.0 / 1.25, x_ratio as f64); } }
                } else if scroll.y != 0.0 { self.audio_view.scroll_horizontal(-scroll.y as f64 * 0.01); }
            });
        });

        // Player panel (transport controls) — declared after the audio view,
        // i.e. docked *above* it and directly under the file-actions panel
        // (bug #1: Play/Stop must sit above the audio-content view).
        egui::TopBottomPanel::bottom("player").default_height(80.0).show(ctx, |ui| {
            // §3.5.1: transport controls on the left, the volume meter is
            // docked to the right edge of the same window (controls are laid
            // out inside the reduced rect so nothing overlaps).
            let panel_rect = ui.max_rect();
            let meter_w = 90.0f32;
            let meter_rect = egui::Rect::from_min_max(
                egui::pos2(panel_rect.right() - meter_w, panel_rect.min.y + 4.0),
                egui::pos2(panel_rect.right() - 4.0, panel_rect.max.y - 4.0),
            );
            // Reserve the right strip for the meter; everything else lays
            // out in a child UI confined to the remaining area.
            ui.allocate_ui(egui::vec2(meter_rect.min.x - 6.0 - panel_rect.min.x, panel_rect.height()), |ui| {
            ui.horizontal(|ui| {
                if ui.button("|<").clicked() { self.navigate_file(-1); }
                let play_text = match self.player_state() { TransportState::Playing => "⏸", _ => "▶" };
                if ui.button(play_text).clicked() { self.transport_play_pause(); }
                if ui.button("⏹").clicked() { self.transport_stop(); }
                if ui.button(">|").clicked() { self.navigate_file(1); }
                let (looping, playing_now) = {
                    let ctrl = self.player.lock().unwrap_or_else(|p| p.into_inner());
                    (ctrl.is_looping(), ctrl.state() == TransportState::Playing)
                };
                let loop_color = if looping && playing_now { egui::Color32::YELLOW } else if looping { egui::Color32::GREEN } else { ui.visuals().text_color() };
                if ui.colored_label(loop_color, "🔁").clicked() { self.transport_toggle_loop(); }
                ui.separator();
                if ui.button("+").clicked() { self.audio_view.zoom_in(); }
                if ui.button("−").clicked() { self.audio_view.zoom_out(); }
                ui.separator();
                ui.label("Vol:");
                let mut vol = self.player.lock().unwrap_or_else(|p| p.into_inner()).volume() * 100.0;
                if ui.add(egui::Slider::new(&mut vol, 0.0..=100.0)).changed() { self.transport_set_volume(vol / 100.0); }
                ui.separator();
                let save_color = if self.document.dirty { egui::Color32::YELLOW } else { ui.visuals().text_color() };
                let save_label = if self.document.dirty { "Save*" } else { "Save" };
                if ui.colored_label(save_color, save_label).clicked() { self.save_file(); }
                ui.separator();
                // Player-panel display-mode toggle (§3.5): reflects and flips
                // the live `AudioView::mode` state (persisted in config.json).
                let view_spectro = self.audio_view.mode == AudioViewMode::Spectrogram;
                let view_btn = egui::SelectableLabel::new(
                    view_spectro,
                    if view_spectro { "Spectrogram" } else { "Waveform" },
                );
                if ui.add(view_btn).on_hover_text("Toggle the visualization between waveform and spectrogram").clicked() {
                    self.toggle_audio_view_mode();
                }
            });
            ui.horizontal(|ui| {
                let duration = self.document.duration_secs();
                let current_pos = self.player_current_secs();
                let time_str = if duration < 60.0 { format!("{:.1}s / {:.1}s", current_pos, duration) } else { format!("{} / {}", format_time_hms(current_pos), format_time_hms(duration)) };
                ui.label(time_str);
                if duration > 0.0 { let progress = (current_pos / duration) as f32; ui.add(egui::ProgressBar::new(progress).desired_width(ui.available_width())); }
            });
            });
            self.draw_volume_meter(ui, meter_rect);
        });

        // File-actions panel (Mute / Normalize / Apply …) — declared after
        // the player, i.e. docked *above* it: this is the panel the transport
        // controls must sit under (bug #1).
        egui::TopBottomPanel::bottom("actions").default_height(40.0).show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Mute").clicked() { self.mute_selection(); }
                ui.label("Normalize:");
                ui.add(egui::DragValue::new(&mut self.normalize_dbfs).range(-60.0..=0.0).suffix(" dBFS"));
                if ui.button("Apply").clicked() { self.normalize_selection(); }
                ui.separator();
                if ui.button("Delete all markers").clicked() { self.document.markers.clear(); }
                if let Some(path) = &self.active_file {
                    ui.separator();
                    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                    ui.strong(name);
                }
            });
        });

        // Status bar — declared LAST among the bottom panels, which docks it
        // highest in the stack, right under the file-list area (§13).
        egui::TopBottomPanel::bottom("status").default_height(20.0).show(ctx, |ui| {
            ui.label(&self.status_message);
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(format!("{} files", self.file_rows.len()));
                if let Some(dir) = &self.current_dir { ui.label(dir.display().to_string()); }
            });
            ui.separator();

            let visible_cols = self.column_layout.visible.clone();

            // --- Table layout ------------------------------------------------------
            // BUGFIX (column headers): the header row used to be emitted
            // *after* the ScrollArea, i.e. at the BOTTOM of the files window;
            // later it became a SECOND, separate `Grid` ("file_table_header")
            // above the list. Two independent grids never share a real cell
            // mesh: egui derives column offsets from each grid's own
            // allocated rect, so any context difference (e.g. the vertical
            // scrollbar eating width inside the body's ScrollArea only) made
            // the header columns drift against the file-row columns. The
            // header is now the FIRST ROW of the very same Grid that holds
            // every data row — titles and cells share one grid by
            // construction (see "ONE grid" below).
            //
            // The table's own vertical ScrollArea is sized exactly to the
            // remaining space (`auto_shrink([false, false])` + `max_height`)
            // so the scrollbar spans the real list region.
            //
            // BUGFIX (window height): the file-list height must account for
            // the whole bottom stack — status bar + file actions + player
            // controls + the (user-resizable) visualization window. All four
            // bottom panels are declared *before* this CentralPanel, so egui
            // has already reserved their real, measured space: here
            // `ui.available_height()` is exactly what is left between the
            // current cursor position and the top edge of the bottom stack
            // (panel margins/inner spacing are subtracted by egui itself
            // when the scroll area allocates its rect). The buggy version
            // ALSO subtracted a hand-made estimate of the same bottom stack
            // (`bottom_panels_height`, computed with hardcoded constants in
            // the actions panel), i.e. it double-counted the reserved space:
            // the file window came out far shorter than it should be, and
            // every drift of that estimate forced an extra re-layout frame.
            // Now the available height is used directly, so resizing the
            // visualization panel still shrinks/grows the list one-to-one.
            let list_max_height = ui.available_height().max(60.0);

            // Borrow the pieces the row loop needs *before* opening the
            // table closure, so `self` is not captured mutably inside.
            {
                let column_layout = &self.column_layout;
                // Per-mode texture caches: pick the one matching the active
                // thumbnail style so switching modes never re-uploads or mixes
                // wave/spectrogram textures.
                let preview_cache: &mut HashMap<String, egui::TextureHandle> = match *self.preview_mode.lock().unwrap() {
                    PreviewMode::Waveform => &mut self.preview_cache_wave,
                    PreviewMode::Spectrogram => &mut self.preview_cache_spectro,
                };
                let selected_row = &mut self.selected_row;
                let filter_text = self.filter_text.to_lowercase();
                // REFACTOR (table grid): the header used to be emitted *after*
                // the ScrollArea (i.e. under the list), then it was moved on
                // top but rendered as a SECOND, separate `Grid`
                // ("file_table_header") while the data rows lived in another
                // one ("file_table"/"file_table_body"). Two independent grids
                // never share a real cell mesh: their column offsets come
                // from separately allocated rects, so any difference in the
                // surrounding context (e.g. the vertical scrollbar eating
                // width inside the body's grid only, or differing outer
                // margins) shifted the whole header against the body — the
                // header cells were not part of the same grid as the row
                // cells, and the title columns did not line up with the
                // file-row columns.
                // Now there is exactly ONE `Grid` ("file_table") inside the
                // ScrollArea: the header is its FIRST ROW, followed by all
                // data rows. Header cells and row cells are laid out by the
                // very same grid pass over the very same rect, so every
                // column title shares its column geometry with the cells
                // underneath it by construction. Per-column widths
                // (`column_widths`) and the shared cell painter (`draw_cell`)
                // additionally pin each cell to its configured column width.
                let column_widths: Vec<f32> = visible_cols
                    .iter()
                    .map(|c| column_layout.width(c))
                    .collect();

                // Shared cell painter: draws one table cell into `cell_ui`
                // constrained to the exact column width. Used by BOTH the
                // header row and the data rows of the single table grid, so
                // geometry is identical. Returns the child Ui's min_rect (in
                // parent coordinates), which doubles as the cell's
                // click-response rect.
                fn draw_cell(
                    cell_ui: &mut egui::Ui,
                    width: f32,
                    height: f32,
                    contents: impl FnOnce(&mut egui::Ui),
                ) -> egui::Rect {
                    let inner = cell_ui.allocate_ui_with_layout(
                        egui::vec2(width, height),
                        egui::Layout::left_to_right(egui::Align::Center),
                        |ui| {
                            contents(ui);
                            ui.min_rect()
                        },
                    );
                    inner.inner
                }

                // Sort indicator state for the header row, read *before* the
                // immutable borrows used inside the grid closure below.
                let sort_column = self.column_layout.sort_column;
                let sort_ascending = self.column_layout.sort_ascending;

                // --- The files table: ONE grid whose FIRST ROW is the header
                // and all following rows are data rows (scrollable). Because
                // both live in the same Grid over the same allocated rect,
                // every column title shares its exact column geometry with
                // the file-row cells underneath it. ---
                // `file_rows` is borrowed here — no other code mutates
                // `self.file_rows`/`self.column_layout` while the closure is
                // open; a clicked column title only records the request, and
                // the actual re-sort happens after the panel closure.
                let file_rows = &self.file_rows;
                // Clicks on sortable titles are recorded here and applied
                // after the whole CentralPanel closure (the layout is only
                // borrowed immutably inside).
                let mut sort_request: Option<Column> = None;

                egui::ScrollArea::vertical()
                    .id_salt("file_list")
                    .auto_shrink([false, false])
                    .max_height(list_max_height)
                    .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysVisible)
                    .show(ui, |ui| {
                        // Collected inside the scroll closure, appended to
                        // `paths_to_load` after the closure returns.
                        let mut local_paths_to_load: Vec<PathBuf> = Vec::new();

                        let grid = egui::Grid::new("file_table")
                            .num_columns(visible_cols.len())
                            .spacing([10.0, 4.0])
                            .striped(true);

                        grid.show(ui, |ui| {
                            // --- Header row: the FIRST row of the very same
                            // grid that holds every data row below. Each
                            // title cell is painted through the shared
                            // `draw_cell` helper with the same per-column
                            // width (`column_widths[ci]`) and the same grid
                            // spacing as the data cells, so it sits exactly
                            // in its own column of the common mesh. ---
                            for (ci, col) in visible_cols.iter().enumerate() {
                                let width = column_widths[ci];
                                let label = col.title();
                                let indicator = if sort_column == Some(*col) {
                                    if sort_ascending { " ▲" } else { " ▼" }
                                } else {
                                    ""
                                };
                                let sortable = col.sortable();
                                let col_copy = *col;
                                draw_cell(ui, width, 24.0, |ui| {
                                    let response = if sortable {
                                        ui.add_sized(
                                            [width, 24.0],
                                            egui::Button::new(format!("{}{}", label, indicator)),
                                        )
                                    } else {
                                        ui.add_sized(
                                            [width, 24.0],
                                            egui::Label::new(egui::RichText::new(label).strong()),
                                        )
                                    };
                                    if sortable && response.clicked() {
                                        sort_request = Some(col_copy);
                                    }
                                });
                            }
                            ui.end_row();

                            // --- Data rows ---
                            for (idx, row) in file_rows.iter().enumerate() {
                                if !filter_text.is_empty() && !row.name.to_lowercase().contains(&filter_text) {
                                    continue;
                                }

                                let is_selected = *selected_row == Some(idx);
                                let is_broken = row.status == audioscout_core::ProbeStatus::Broken;
                                let is_unavailable = row.status == audioscout_core::ProbeStatus::Unavailable;
                                let text_color = if is_broken {
                                    egui::Color32::from_rgb(255, 100, 100)
                                } else if is_selected {
                                    egui::Color32::from_rgb(100, 200, 255)
                                } else {
                                    ui.visuals().text_color()
                                };

                                // Track if this row was clicked
                                let mut row_clicked = false;
                                let mut row_double_clicked = false;

                                for (ci, col) in visible_cols.iter().enumerate() {
                                    let width = column_widths[ci];
                                    // Preview column needs more height for stereo (40px) + margins
                                    let cell_height = if *col == Column::Preview { 44.0 } else { 20.0 };

                                    let cell_rect = draw_cell(ui, width, cell_height, |ui| {
                                        if *col == Column::Preview {
                                            // Display preview image if available
                                            if let Some(uuid) = &row.preview_uuid {
                                                // Check cache first
                                                if let Some(texture) = preview_cache.get(uuid) {
                                                    // Use actual texture size
                                                    let size = texture.size();
                                                    ui.add(egui::Image::new(texture).fit_to_exact_size(egui::vec2(size[0] as f32, size[1] as f32)));
                                                } else {
                                                    // Load from disk and cache
                                                    let config_dir = audioscout_core::config::config_dir();
                                                    let previews_dir = config_dir.join("previews");
                                                    let png_path = previews_dir.join(format!("{}.png", uuid));

                                                    if png_path.exists() {
                                                        if let Ok(image_data) = std::fs::read(&png_path) {
                                                            if let Ok(image) = image::load_from_memory(&image_data) {
                                                                // Get actual image dimensions
                                                                let (img_width, img_height) = image.dimensions();
                                                                let rgba = image.to_rgba8();
                                                                let pixels = rgba.as_flat_samples();
                                                                let color_image = egui::ColorImage::from_rgba_unmultiplied(
                                                                    [img_width as usize, img_height as usize],
                                                                    pixels.as_slice(),
                                                                );
                                                                let texture = ui.ctx().load_texture(
                                                                    format!("preview_{}", uuid),
                                                                    color_image,
                                                                    egui::TextureOptions::LINEAR,
                                                                );
                                                                // Cache the texture
                                                                preview_cache.insert(uuid.clone(), texture.clone());
                                                                ui.add(egui::Image::new(&texture).fit_to_exact_size(egui::vec2(img_width as f32, img_height as f32)));
                                                            } else {
                                                                ui.label("⚠️");
                                                            }
                                                        } else {
                                                            ui.label("⚠️");
                                                        }
                                                    } else {
                                                        ui.label("⏳");
                                                    }
                                                }
                                            } else {
                                                ui.label("⏳");
                                            }
                                        } else {
                                            let text = match col {
                                                Column::Name => row.name.clone(),
                                                Column::Duration => format_duration(row.duration_secs),
                                                Column::SampleRate => format!("{} Hz", row.sample_rate),
                                                Column::SampleType => format!("{}", row.sample_type),
                                                Column::Channels => match row.channels {
                                                    1 => "Mono".to_string(),
                                                    2 => "Stereo".to_string(),
                                                    n => format!("{}ch", n),
                                                },
                                                Column::Size => format_size(row.size),
                                                Column::Comment => row.comment.clone().unwrap_or_default(),
                                                Column::DateCreated => format_timestamp(row.date_created),
                                                Column::DateModified => format_timestamp(row.date_modified),
                                                _ => String::new(),
                                            };

                                            if is_unavailable && *col == Column::Name {
                                                ui.colored_label(
                                                    egui::Color32::from_rgb(180, 180, 100),
                                                    format!("{} [unavailable]", text),
                                                );
                                            } else {
                                                ui.colored_label(text_color, text);
                                            }
                                        }
                                    });

                                    // Check if any cell in this row was
                                    // clicked: interact over the whole cell
                                    // rect so clicks between widgets still
                                    // select the row (same as before, when
                                    // every cell carried its own response).
                                    let cell_id = ui.id().with(("file_cell", idx, ci));
                                    let cell_response = ui.interact(cell_rect, cell_id, egui::Sense::click());

                                    // Check if any cell in this row was clicked
                                    if cell_response.clicked() {
                                        row_clicked = true;
                                    }
                                    if cell_response.double_clicked() {
                                        row_double_clicked = true;
                                    }
                                }

                                // Handle row selection
                                if row_clicked {
                                    *selected_row = Some(idx);
                                }

                                // Handle double-click to load file
                                if row_double_clicked && !is_broken && !is_unavailable {
                                    local_paths_to_load.push(row.path.clone());
                                }

                                ui.end_row();
                            }
                        });

                        // Hand the collected double-click paths over to the
                        // outer `paths_to_load` (loaded after CentralPanel).
                        paths_to_load.append(&mut local_paths_to_load);
                    });

                // A column title clicked in the header row: apply the sort
                // AFTER the whole panel closure has finished (the grid only
                // borrows `column_layout` immutably inside its closure, so a
                // click just records the request here).
                if let Some(col) = sort_request {
                    if self.column_layout.sort_column == Some(col) {
                        self.column_layout.sort_ascending = !self.column_layout.sort_ascending;
                    } else {
                        self.column_layout.sort_column = Some(col);
                        self.column_layout.sort_ascending = true;
                    }
                    self.column_layout.sort_rows(&mut self.file_rows);
                }
            }
        });

        // Load files AFTER the CentralPanel closure (no borrow conflict)
        for path in paths_to_load {
            self.load_file(&path);
        }









        // Settings dialog
        if self.show_settings {
            // Collect button actions locally first: `Window::open(&mut
            // self.show_settings)` borrows `*self` for the whole call, so the
            // closure cannot also hold `&mut self` (E0500). Copying the plain
            // fields out beforehand keeps the closure on disjoint fields only.
            #[derive(PartialEq)]
            enum SettingsAction {
                None,
                DeleteMarkers,
                DeletePreviews,
                CleanOrphans,
            }

            let mut skip_seconds_changed: Option<u32> = None;
            let mut action = SettingsAction::None;

            // Read-only copies of the settings values (plain `Copy` fields).
            // Reading them through `&self.*` inside the closure would conflict
            // with the `&mut self.show_settings` borrow passed to `open()`.
            let current_skip_seconds = self.config_store.config().skip_seconds;

            egui::Window::new("Settings").open(&mut self.show_settings).show(ctx, |ui| {
                ui.label("AudioScout Settings");
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("Skip seconds:");
                    let mut skip = current_skip_seconds;
                    if ui.add(egui::DragValue::new(&mut skip).range(1..=60)).changed() {
                        skip_seconds_changed = Some(skip);
                    }
                });
                ui.separator();
                if ui.button("Delete all markers").clicked() { action = SettingsAction::DeleteMarkers; }
                if ui.button("Delete all previews").clicked() { action = SettingsAction::DeletePreviews; }
                if ui.button("Clean orphaned previews").clicked() { action = SettingsAction::CleanOrphans; }
            });

            if let Some(skip) = skip_seconds_changed {
                self.config_store.config_mut().skip_seconds = skip;
                self.player.lock().unwrap_or_else(|p| p.into_inner()).set_skip_seconds(skip);
                let _ = self.config_store.save();
            }

            match action {
                SettingsAction::None => {}
                SettingsAction::DeleteMarkers => {
                    let _ = self.db.delete_all_markers();
                    self.status_message = "All markers deleted".to_string();
                }
                SettingsAction::DeletePreviews => {
                    let _ = self.db.delete_all_previews();
                    // Also delete preview files from disk
                    let config_dir = audioscout_core::config::config_dir();
                    let previews_dir = config_dir.join("previews");
                    if previews_dir.exists() {
                        let _ = std::fs::remove_dir_all(&previews_dir);
                    }
                    // Clear in-memory texture caches (both modes)
                    self.preview_cache_wave.clear();
                    self.preview_cache_spectro.clear();
                    // Drop the stale UUIDs from the visible rows: they point at
                    // PNGs that no longer exist on disk. Without this,
                    // `request_previews()` skips every row (it only submits
                    // jobs for rows with `preview_uuid == None`) and the
                    // preview pump thinks nothing is pending — so regeneration
                    // never starts after deletion.
                    for row in &mut self.file_rows {
                        row.preview_uuid = None;
                    }
                    // Cancel any in-flight jobs rendering the deleted previews
                    // and reset bookkeeping so fresh jobs are submitted.
                    self.preview_worker.clear_queue();
                    self.preview_inflight.clear();
                    self.preview_retry.clear();
                    // Kick off regeneration for the current directory.
                    self.request_previews();
                    ctx.request_repaint();
                    self.status_message = "All previews deleted, regenerating…".to_string();
                }
                SettingsAction::CleanOrphans => {
                    match audioscout_core::preview_worker::cleanup_previews() {
                        Ok(count) => self.status_message = format!("Cleaned {} orphaned previews", count),
                        Err(e) => self.status_message = format!("Cleanup failed: {}", e),
                    }
                }
            }
        }

        if self.show_columns_dialog {
            egui::Window::new("Columns").open(&mut self.show_columns_dialog).show(ctx, |ui| {
                ui.label("Visible columns:");
                for col in Column::all() { let mut visible = self.column_layout.is_visible(col); if ui.checkbox(&mut visible, col.title()).changed() { self.column_layout.toggle_visibility(col); } }
            });
        }

        if let Some(dir) = &self.current_dir { let _ = self.db.save_session("lastDir", &dir.to_string_lossy()); }
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        // Shutdown preview worker to stop background thread
        self.preview_worker.shutdown();

        // Clear preview caches to free GPU memory
        self.preview_cache_wave.clear();
        self.preview_cache_spectro.clear();

        if let Some(dir) = &self.current_dir { let _ = self.db.save_session("lastDir", &dir.to_string_lossy()); }
        if let Some(file) = &self.active_file {
            let _ = self.db.save_session("lastFile", &file.to_string_lossy());
            let pos = self.player_current_secs();
            let _ = self.db.save_session("lastPos", &pos.to_string());
        }
        let _ = self.config_store.save();
    }
}

fn format_duration(secs: f64) -> String { if secs < 60.0 { format!("{:.1}s", secs) } else { let mins = (secs / 60.0) as u32; let s = (secs % 60.0) as u32; format!("{}:{:02}", mins, s) } }
fn format_size(bytes: u64) -> String { if bytes < 1024 { format!("{} B", bytes) } else if bytes < 1024 * 1024 { format!("{:.1} KB", bytes as f64 / 1024.0) } else { format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0)) } }
fn format_timestamp(ts: u64) -> String {
    if ts == 0 { return String::new(); }
    use chrono::{DateTime, Local};
    let datetime = DateTime::from_timestamp(ts as i64, 0)
        .map(|dt| dt.with_timezone(&Local))
        .unwrap_or_else(|| Local::now());
    // Format: DD-MM-YYYY HH:MM:SS (date with leading zeros, time with leading zeros)
    datetime.format("%d-%m-%Y %H:%M:%S").to_string()
}
fn format_time_hms(secs: f64) -> String { let total_secs = secs as u64; let h = total_secs / 3600; let m = (total_secs % 3600) / 60; let s = total_secs % 60; if h > 0 { format!("{}:{:02}:{:02}", h, m, s) } else { format!("{}:{:02}", m, s) } }
