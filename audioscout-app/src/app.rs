use audioscout_core::config::{ConfigStore, PreviewMode};
use audioscout_core::player::controller::Controller;
use audioscout_core::player::{PlayheadState, TransportState};
use audioscout_core::scanner::Scanner;
use audioscout_core::table::{Column, ColumnLayout, FileRow};
use audioscout_core::audio_view::AudioView;
use audioscout_core::undo::Stack as UndoStack;
use audioscout_core::db::Database;
use audioscout_core::audio_io::PcmFile;
use audioscout_core::disk::DiskProvider;
use audioscout_core::{Document, EventChannel, Marker, UiEvent, is_editable_file};
use crossbeam_channel::Receiver;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Main application state
pub struct AudioScoutApp {
    config_store: ConfigStore,
    db: Database,
    event_rx: Receiver<UiEvent>,
    event_tx: crossbeam_channel::Sender<UiEvent>,
    scanner: Scanner,
    current_dir: Option<PathBuf>,
    file_rows: Vec<FileRow>,
    scan_generation: u64,
    column_layout: ColumnLayout,
    selected_row: Option<usize>,
    filter_text: String,
    document: Document,
    active_file: Option<PathBuf>,
    player: Controller,
    audio_view: AudioView,
    undo_stack: UndoStack,
    table_preview_mode: PreviewMode,
    show_settings: bool,
    show_columns_dialog: bool,
    status_message: String,
    normalize_dbfs: f32,
    expanded_dirs: HashMap<PathBuf, Vec<PathBuf>>,
    disks: Vec<audioscout_core::disk::DiskInfo>,
}

impl AudioScoutApp {
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
        let player = Controller::new(Arc::new(PlayheadState::new()));
        let undo_dir = config_dir.join("undo");
        let undo_stack = UndoStack::new(undo_dir, config_store.config().undo_depth, config_store.config().undo_disk_limit_mb);

        let mut app = Self {
            config_store, db, event_rx, event_tx,
            scanner: Scanner::new(), current_dir: None, file_rows: Vec::new(), scan_generation: 0,
            column_layout: ColumnLayout::new(), selected_row: None, filter_text: String::new(),
            document: Document::new(), active_file: None, player,
            audio_view: AudioView::new(0), undo_stack,
            table_preview_mode: PreviewMode::Wave,
            show_settings: false, show_columns_dialog: false, status_message: String::new(),
            normalize_dbfs: -1.0, expanded_dirs: HashMap::new(),
            disks: Vec::new(),
        };

        if let Some(dir_str) = app.db.load_session("lastDir") {
            let dir = PathBuf::from(dir_str);
            if dir.exists() { app.current_dir = Some(dir.clone()); app.scanner.scan(dir, app.event_tx.clone()); }
        }
        if let Some(dir) = start_dir { app.current_dir = Some(dir.clone()); app.scanner.scan(dir, app.event_tx.clone()); }

        // Auto-expand home directory on startup
        if let Some(home) = dirs_next::home_dir() {
            if let Ok(entries) = std::fs::read_dir(&home) {
                let mut subdirs: Vec<PathBuf> = entries
                    .filter_map(|e| e.ok())
                    .filter(|e| e.path().is_dir())
                    .map(|e| e.path())
                    .collect();
                subdirs.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
                app.expanded_dirs.insert(home, subdirs);
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

        app
    }

    fn process_events(&mut self) {
        while let Ok(event) = self.event_rx.try_recv() {
            match event {
                UiEvent::ScanDone { dir, rows, generation } => {
                    if Some(&dir) == self.current_dir.as_ref() && generation >= self.scan_generation {
                        self.scan_generation = generation;
                        self.file_rows = rows;
                        self.column_layout.sort_rows(&mut self.file_rows);
                        if let Some(file_str) = self.db.load_session("lastFile") {
                            let file_path = PathBuf::from(file_str);
                            if let Some(idx) = self.file_rows.iter().position(|r| r.path == file_path) {
                                self.selected_row = Some(idx);
                            }
                        }
                    }
                }
                UiEvent::PreviewReady { path, uuid, .. } => {
                    if let Some(row) = self.file_rows.iter_mut().find(|r| r.path == path) { row.preview_uuid = Some(uuid); }
                }
                UiEvent::PlayheadUpdate { position_secs } => {
                    let frame = (position_secs * self.document.sample_rate as f64) as u64;
                    self.audio_view.set_cursor(frame);
                    self.audio_view.auto_scroll_to_cursor();
                }
                UiEvent::PlaybackFinished => { self.status_message = "Playback finished".to_string(); }
                UiEvent::VolumeMeterUpdate { .. } => {}
                UiEvent::DisksChanged { disks } => {
                    self.disks = disks;
                }
            }
        }
    }

    fn load_file(&mut self, path: &Path) {
        if !is_editable_file(path) { self.status_message = "File format not editable".to_string(); return; }
        self.player.stop();
        self.player.reset();
        match audioscout_core::audio_io::load_audio(path) {
            Ok((samples, sr, ch, st)) => {
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
                self.undo_stack.clear();
                let _ = self.db.save_session("lastFile", &path.to_string_lossy());
                let _ = self.db.save_session("lastPos", "0");
            }
            Err(e) => { self.status_message = format!("Load failed: {}", e); }
        }
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
                                "playpause" => { let sel = self.audio_view.selection_range(); self.player.play_pause(&self.document, sel); }
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
                                "gotoStart" => { self.audio_view.set_cursor(0); self.player.seek(0, &self.document, audioscout_core::player::CursorSource::User); }
                                _ => {}
                            }
                        }
                    }
                    if *key == egui::Key::ArrowLeft || *key == egui::Key::ArrowRight {
                        let direction = if *key == egui::Key::ArrowRight { 1 } else { -1 };
                        if self.player.state() == TransportState::Playing {
                            self.player.skip(&self.document, direction * config.skip_seconds as i32);
                            let frame = self.player.current_frame();
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
        self.process_events();
        self.handle_keyboard(ctx);
        if self.player.state() == TransportState::Playing { ctx.request_repaint_after(std::time::Duration::from_millis(50)); }

        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Filter:");
                ui.text_edit_singleline(&mut self.filter_text);
                if ui.button("✕").clicked() { self.filter_text.clear(); }
                ui.separator();
                if ui.button("Columns").clicked() { self.show_columns_dialog = true; }
                let mode_text = match self.table_preview_mode { PreviewMode::Wave => "Waveform", PreviewMode::Spectrogram => "Spectrogram" };
                if ui.button(mode_text).clicked() { self.table_preview_mode = match self.table_preview_mode { PreviewMode::Wave => PreviewMode::Spectrogram, PreviewMode::Spectrogram => PreviewMode::Wave }; }
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
                subdirs.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
                self.expanded_dirs.insert(dir, subdirs);
            }
        }
        if let Some(dir) = dir_to_collapse {
            self.expanded_dirs.remove(&dir);
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(format!("{} files", self.file_rows.len()));
                if let Some(dir) = &self.current_dir { ui.label(dir.display().to_string()); }
            });
            ui.separator();

            let visible_cols = self.column_layout.visible.clone();
            egui::ScrollArea::vertical().show(ui, |ui| {
                egui::Grid::new("file_table")
                    .num_columns(visible_cols.len())
                    .spacing([10.0, 4.0])
                    .striped(true)
                    .show(ui, |ui| {
                        // Header row
                        for col in &visible_cols {
                            let label = col.title();
                            let sort_indicator = if self.column_layout.sort_column == Some(*col) {
                                if self.column_layout.sort_ascending { " ▲" } else { " ▼" }
                            } else { "" };

                            let response = if col.sortable() {
                                ui.button(format!("{}{}", label, sort_indicator))
                            } else {
                                ui.strong(label)
                            };

                            if col.sortable() && response.clicked() {
                                if self.column_layout.sort_column == Some(*col) {
                                    self.column_layout.sort_ascending = !self.column_layout.sort_ascending;
                                } else {
                                    self.column_layout.sort_column = Some(*col);
                                    self.column_layout.sort_ascending = true;
                                }
                                self.column_layout.sort_rows(&mut self.file_rows);
                            }
                        }
                        ui.end_row();

                        // File rows
                        for (idx, row) in self.file_rows.iter().enumerate() {
                            if !self.filter_text.is_empty() && !row.name.to_lowercase().contains(&self.filter_text.to_lowercase()) {
                                continue;
                            }

                            let is_selected = self.selected_row == Some(idx);
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

                            for col in &visible_cols {
                                let cell_response = ui.allocate_ui(
                                    egui::vec2(self.column_layout.width(col), 20.0),
                                    |ui| {
                                        let text = match col {
                                            Column::Name => row.name.clone(),
                                            Column::Preview => String::new(),
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
                                        };

                                        if is_unavailable && *col == Column::Name {
                                            ui.colored_label(
                                                egui::Color32::from_rgb(180, 180, 100),
                                                format!("{} [unavailable]", text),
                                            )
                                        } else if *col == Column::Preview {
                                            // Placeholder for preview
                                            ui.label("🎵")
                                        } else {
                                            ui.colored_label(text_color, text)
                                        }
                                    }
                                );

                                // Check if any cell in this row was clicked
                                if cell_response.response.clicked() {
                                    row_clicked = true;
                                }
                                if cell_response.response.double_clicked() {
                                    row_double_clicked = true;
                                }
                            }

                            // Handle row selection
                            if row_clicked {
                                self.selected_row = Some(idx);
                            }

                            // Handle double-click to load file
                            if row_double_clicked && !is_broken && !is_unavailable {
                                paths_to_load.push(row.path.clone());
                            }

                            ui.end_row();
                        }
                    });
            });
        });

        // Load files AFTER the CentralPanel closure (no borrow conflict)
        for path in paths_to_load {
            self.load_file(&path);
        }

        // Status bar (bottom-most)
        egui::TopBottomPanel::bottom("status").default_height(20.0).show(ctx, |ui| {
            ui.label(&self.status_message);
        });

        // Player panel (above status, below audio_view)
        egui::TopBottomPanel::bottom("player").default_height(80.0).show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("|<").clicked() { self.navigate_file(-1); }
                let play_text = match self.player.state() { TransportState::Playing => "⏸", _ => "▶" };
                if ui.button(play_text).clicked() { let sel = self.audio_view.selection_range(); self.player.play_pause(&self.document, sel); }
                if ui.button("⏹").clicked() { self.player.stop(); }
                if ui.button(">|").clicked() { self.navigate_file(1); }
                let loop_color = if self.player.is_looping() && self.player.state() == TransportState::Playing { egui::Color32::YELLOW } else if self.player.is_looping() { egui::Color32::GREEN } else { ui.visuals().text_color() };
                if ui.colored_label(loop_color, "🔁").clicked() { self.player.toggle_loop(); }
                ui.separator();
                if ui.button("+").clicked() { self.audio_view.zoom_in(); }
                if ui.button("−").clicked() { self.audio_view.zoom_out(); }
                ui.separator();
                ui.label("Vol:");
                let mut vol = self.player.volume() * 100.0;
                if ui.add(egui::Slider::new(&mut vol, 0.0..=100.0)).changed() { self.player.set_volume(vol / 100.0); }
                ui.separator();
                let save_color = if self.document.dirty { egui::Color32::YELLOW } else { ui.visuals().text_color() };
                let save_label = if self.document.dirty { "Save*" } else { "Save" };
                if ui.colored_label(save_color, save_label).clicked() { self.save_file(); }
            });
            ui.horizontal(|ui| {
                let duration = self.document.duration_secs();
                let current_pos = self.player.current_secs(self.document.sample_rate);
                let time_str = if duration < 60.0 { format!("{:.1}s / {:.1}s", current_pos, duration) } else { format!("{} / {}", format_time_hms(current_pos), format_time_hms(duration)) };
                ui.label(time_str);
                if duration > 0.0 { let progress = (current_pos / duration) as f32; ui.add(egui::ProgressBar::new(progress).desired_width(ui.available_width())); }
            });
        });

        // Audio view panel (above player)
        egui::TopBottomPanel::bottom("audio_view").default_height(200.0).resizable(true).show(ctx, |ui| {
            let (rect, response) = ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
            ui.painter().rect_filled(rect, 0.0, egui::Color32::from_rgb(25, 25, 30));

            if !self.document.samples.is_empty() {
                let mid_y = rect.center().y;
                ui.painter().line_segment([egui::pos2(rect.left(), mid_y), egui::pos2(rect.right(), mid_y)], egui::Stroke::new(1.0, egui::Color32::from_rgb(60, 60, 65)));

                // Cursor
                let cursor_x = self.audio_view.frame_to_x(self.audio_view.cursor_frame, rect.width() as u32);
                let cursor_screen_x = rect.left() + cursor_x;
                if cursor_screen_x >= rect.left() && cursor_screen_x <= rect.right() {
                    ui.painter().line_segment([egui::pos2(cursor_screen_x, rect.top()), egui::pos2(cursor_screen_x, rect.bottom())], egui::Stroke::new(1.5, egui::Color32::RED));
                }

                // Markers
                for (idx, marker) in self.document.markers.iter().enumerate() {
                    let mx = self.audio_view.frame_to_x(marker.time_samples, rect.width() as u32);
                    let screen_x = rect.left() + mx;
                    if screen_x >= rect.left() && screen_x <= rect.right() {
                        let color = if self.audio_view.active_marker == Some(idx) { egui::Color32::YELLOW } else { egui::Color32::from_rgb(100, 180, 255) };
                        let mut y = rect.top();
                        while y < rect.bottom() {
                            let seg_end = (y + 4.0).min(rect.bottom());
                            ui.painter().line_segment([egui::pos2(screen_x, y), egui::pos2(screen_x, seg_end)], egui::Stroke::new(1.0, color));
                            y += 8.0;
                        }
                        ui.painter().text(egui::pos2(screen_x + 2.0, rect.top() + 2.0), egui::Align2::LEFT_TOP, &marker.name, egui::FontId::proportional(10.0), color);
                    }
                }

                // Selection
                if let Some(sel) = self.audio_view.selection_range() {
                    let x1 = rect.left() + self.audio_view.frame_to_x(sel.0, rect.width() as u32);
                    let x2 = rect.left() + self.audio_view.frame_to_x(sel.1, rect.width() as u32);
                    let sel_rect = egui::Rect::from_min_max(egui::pos2(x1.max(rect.left()), rect.top()), egui::pos2(x2.min(rect.right()), rect.bottom()));
                    ui.painter().rect_filled(sel_rect, 0.0, egui::Color32::from_rgba_premultiplied(80, 120, 255, 60));
                }

                // Waveform
                let ch = self.document.channels as usize;
                let start_frame = self.audio_view.view_range.start_frame;
                let end_frame = self.audio_view.view_range.end_frame;
                let view_frames = end_frame.saturating_sub(start_frame);
                if view_frames > 0 {
                    let pixels = rect.width() as u32;
                    let frames_per_pixel = (view_frames as f64 / pixels as f64).max(1.0);
                    for px in 0..pixels {
                        let frame = start_frame + (px as f64 * frames_per_pixel) as u64;
                        let frame_end = (start_frame as f64 + (px as f64 + 1.0) * frames_per_pixel).min(end_frame as f64) as u64;
                        let mut min_val: f32 = 0.0;
                        let mut max_val: f32 = 0.0;
                        for f in frame..frame_end {
                            let idx = f as usize * ch;
                            if idx < self.document.samples.len() {
                                let v = self.document.samples[idx];
                                min_val = min_val.min(v);
                                max_val = max_val.max(v);
                            }
                        }
                        let x = rect.left() + px as f32;
                        let half_h = rect.height() / 2.0;
                        let y_top = mid_y - max_val * half_h;
                        let y_bot = mid_y - min_val * half_h;
                        ui.painter().line_segment([egui::pos2(x, y_top), egui::pos2(x, y_bot)], egui::Stroke::new(1.0, egui::Color32::from_rgb(50, 200, 80)));
                    }
                }
            }

            if response.clicked() {
                if let Some(pos) = response.interact_pointer_pos() {
                    let x_ratio = (pos.x - rect.left()) / rect.width();
                    let frame = self.audio_view.x_to_frame(x_ratio * rect.width(), rect.width() as u32);
                    self.audio_view.set_cursor(frame);
                    self.player.seek(frame, &self.document, audioscout_core::player::CursorSource::User);
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

        // Actions panel
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

        // Settings dialog
        if self.show_settings {
            egui::Window::new("Settings").open(&mut self.show_settings).show(ctx, |ui| {
                ui.label("AudioScout Settings");
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("Skip seconds:");
                    let mut skip = self.config_store.config().skip_seconds;
                    if ui.add(egui::DragValue::new(&mut skip).range(1..=60)).changed() { self.config_store.config_mut().skip_seconds = skip; self.player.set_skip_seconds(skip); let _ = self.config_store.save(); }
                });
                ui.separator();
                if ui.button("Delete all markers").clicked() { let _ = self.db.delete_all_markers(); self.status_message = "All markers deleted".to_string(); }
                if ui.button("Delete all previews").clicked() { let _ = self.db.delete_all_previews(); self.status_message = "All previews deleted".to_string(); }
            });
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
        if let Some(dir) = &self.current_dir { let _ = self.db.save_session("lastDir", &dir.to_string_lossy()); }
        if let Some(file) = &self.active_file {
            let _ = self.db.save_session("lastFile", &file.to_string_lossy());
            let pos = self.player.current_secs(self.document.sample_rate);
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
