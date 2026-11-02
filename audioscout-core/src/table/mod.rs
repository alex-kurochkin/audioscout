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
    /// Order of ALL known columns (visible + hidden), persisted through
    /// `columns/full_order`. Hidden columns keep their slot here so a later
    /// re-show lands exactly where the column sat before it was hidden —
    /// even across app restarts and header drags. Empty means "derive from
    /// the current display order" (see `full_order_keys`).
    pub full_order: Vec<Column>,
}

/// Parse a persisted key list into validated, de-duplicated columns
/// (unknown keys dropped, first occurrence wins).
fn sanitize_keys(keys: &[String]) -> Vec<Column> {
    let mut cols: Vec<Column> = Vec::new();
    for key in keys {
        if let Some(c) = Column::from_key(key) {
            if !cols.contains(&c) { cols.push(c); }
        }
    }
    cols
}

impl ColumnLayout {
    pub fn new() -> Self { Self { visible: Column::all().to_vec(), widths: std::collections::HashMap::new(), sort_column: None, sort_ascending: true, full_order: Vec::new() } }

    /// Rebuild the layout from the persisted `ColumnConfig` (config.json).
    /// Used at app start so BOTH the Columns-dialog selection AND the
    /// header-drag ordering survive restarts.
    ///
    /// Storage model (tz §3.3): `columns/order` holds the order of the
    /// *visible* columns (the live display order), `columns/full_order` the
    /// order of every known column, `columns/visible` the visibility set.
    /// The final display order is derived by filtering `full_order` through
    /// the visibility set, with any visible-but-unlisted columns appended —
    /// so a stale or absent `order` can never drop or duplicate a column.
    /// Rules: unknown keys are dropped, duplicates collapse to the first
    /// occurrence, and an empty/absent list falls back to all columns — the
    /// table must always render at least one column.
    pub fn from_config(cfg: &crate::config::ColumnConfig) -> Self {
        let mut layout = Self::new();
        // Sanitised visibility set (legacy configs may lack `visible`).
        let visible_set: Option<Vec<Column>> = cfg.visible.as_ref().map(|vis| {
            let mut cols: Vec<Column> = Vec::new();
            for key in vis {
                if let Some(c) = Column::from_key(key) {
                    if !cols.contains(&c) { cols.push(c); }
                }
            }
            cols
        });
        if let Some(set) = &visible_set {
            if !set.is_empty() { layout.visible = set.clone(); }
        }
        // Display order comes from `full_order` restricted to the visible
        // set; when it is unusable fall back to `order`, then to the
        // historical meaning of `visible` (which doubled as the display
        // order in older builds).
        let full = sanitize_keys(&cfg.full_order);
        let order = sanitize_keys(&cfg.order);
        // `derived` filters a persisted order through the visibility set; it
        // takes the visible set explicitly so the closure borrows nothing
        // from `layout` (the caller reassigns `layout.visible` afterwards).
        let derived = |base: &[Column], vis_set: &[Column]| -> Vec<Column> {
            let mut cols: Vec<Column> = base
                .iter()
                .filter(|c| vis_set.contains(c))
                .cloned()
                .collect();
            // Anything visible but missing from the base order goes last,
            // keeping its relative order from the visible set.
            for c in vis_set {
                if !cols.contains(c) { cols.push(*c); }
            }
            cols
        };
        let vis_snapshot = layout.visible.clone();
        let ordered = if !full.is_empty() {
            derived(&full, &vis_snapshot)
        } else if !order.is_empty() {
            derived(&order, &vis_snapshot)
        } else if let Some(v) = visible_set.filter(|v| !v.is_empty()) {
            // Legacy fallback: only `visible` existed and stored the order.
            v.into_iter().filter(|c| vis_snapshot.contains(c)).collect()
        } else {
            vis_snapshot.clone()
        };
        if !ordered.is_empty() { layout.visible = ordered; }
        // Keep the FULL arrangement too — hidden columns remember where they
        // were parked, so re-showing one on this restored layout lands it in
        // its pre-hide slot instead of falling back to canonical order.
        if !full.is_empty() {
            layout.full_order = full;
        } else if !order.is_empty() {
            // No full arrangement persisted: the visible order alone is the
            // best approximation we have.
            layout.full_order = derived(&order, &vis_snapshot);
        }
        layout.widths = cfg.widths.clone();
        layout
    }

    /// Snapshot of the current selection as column keys, in display order,
    /// ready to store into `ColumnConfig::visible` and `ColumnConfig::order`.
    pub fn visible_keys(&self) -> Vec<String> {
        self.visible.iter().map(|c| c.key().to_string()).collect()
    }

    /// Order of ALL known columns (`Column::all`) consistent with the current
    /// display arrangement: the visible columns appear in exactly their live
    /// display order, and every hidden column is parked right after its
    /// nearest visible predecessor (canonical order breaks ties).
    /// `from_config` filters this list through the visibility set, so the
    /// visible slots alone determine the restored display order on the next
    /// launch — and a column re-shown later lands right where it used to sit.
    pub fn full_order_keys(&self) -> Vec<String> {
        let all = Column::all();
        // If we already carry an explicit full arrangement (restored from the
        // config), keep it — but re-park any hidden column whose canonical
        // predecessor/successor is no longer where it used to be, so a later
        // re-show lands next to its current visible neighbour. Simplest
        // consistent rule: rebuild whenever the stored order disagrees with
        // the live display order of the visible subset.
        if !self.full_order.is_empty() {
            let derived_visible: Vec<Column> = self
                .full_order
                .iter()
                .filter(|c| self.visible.contains(c))
                .cloned()
                .collect();
            if derived_visible == self.visible && self.full_order.len() == all.len() {
                return self.full_order.iter().map(|c| c.key().to_string()).collect();
            }
        }
        let canon_pos = |c: &Column| all.iter().position(|x| x == c).unwrap_or(0);

        // Walk the visible columns in DISPLAY order, emitting each one
        // followed by the hidden columns whose canonical home sits between
        // it and the next visible column (in canonical order). A dragged
        // arrangement therefore round-trips losslessly: filtering the result
        // through the visibility set reproduces `visible` exactly.
        let mut out: Vec<String> = Vec::with_capacity(all.len());
        let mut emitted: Vec<Column> = Vec::with_capacity(all.len());
        for vi in 0..self.visible.len() {
            let vis = &self.visible[vi];
            out.push(vis.key().to_string());
            emitted.push(*vis);
            // Park the still-hidden columns that `toggle_visibility` would
            // re-insert right after this column: every hidden column whose
            // LAST visible canonical predecessor is exactly `vis`. This
            // mirrors the live insertion rule here, so hide → store →
            // restore → re-show lands the column back at this very slot on
            // the next launch — independent of where a user drag moved the
            // visible neighbours around.
            let idx = canon_pos(vis);
            for hi_idx in (idx + 1)..all.len() {
                let h = all[hi_idx];
                if self.visible.contains(&h) || emitted.contains(&h) {
                    continue;
                }
                // Nearest visible canonical predecessor of `h`.
                let pred_visible = all[..hi_idx]
                    .iter()
                    .rev()
                    .find(|c| self.visible.contains(c));
                if pred_visible == Some(vis) {
                    out.push(h.key().to_string());
                    emitted.push(h);
                }
            }
        }
        // Anything still unclaimed (hidden columns outside every gap above)
        // goes to the very end, keeping canonical relative order so
        // hide/show cycles stay stable.
        for c in all.iter().filter(|c| !emitted.contains(c)) {
            out.push(c.key().to_string());
        }
        out
    }

    /// Materialise the explicit full arrangement (visible + hidden columns)
    /// from the current live display order. Afterwards `full_order` filtered
    /// through `visible` reproduces the display order exactly, and every
    /// still-hidden column keeps a stable slot next to its visible
    /// neighbours — which is what makes hide → store → restore → re-show land
    /// the column back where it sat before it was hidden. No-op when an
    /// arrangement consistent with the live order already exists.
    fn ensure_full_order(&mut self) {
        let derived_visible: Vec<Column> = self
            .full_order
            .iter()
            .filter(|c| self.visible.contains(c))
            .cloned()
            .collect();
        if !self.full_order.is_empty()
            && derived_visible == self.visible
            && self.full_order.len() == Column::all().len()
        {
            return;
        }
        self.full_order = self
            .full_order_keys()
            .iter()
            .filter_map(|k| Column::from_key(k))
            .collect();
    }

    pub fn width(&self, col: &Column) -> f32 { self.widths.get(col.key()).copied().unwrap_or(col.default_width()) }
    pub fn set_width(&mut self, col: &Column, width: f32) { self.widths.insert(col.key().to_string(), width); }

    /// Write the current layout (visibility + display order + full order +
    /// widths) into a `ColumnConfig` snapshot — the single place that keeps
    /// all four fields consistent (tz §3.3). Callers persist the result via
    /// `ConfigStore::save`.
    pub fn store_into(&self, cfg: &mut crate::config::ColumnConfig) {
        let vis = self.visible_keys();
        cfg.visible = Some(vis.clone());
        cfg.order = vis;
        // Persist the FULL arrangement (with hidden columns parked in their
        // stable slots), not just the visible order — otherwise a hide →
        // restart → re-show cycle loses the pre-hide slot of the column.
        cfg.full_order = self.full_order_keys();
        cfg.widths = self.widths.clone();
    }

    pub fn toggle_visibility(&mut self, col: &Column) {
        if let Some(pos) = self.visible.iter().position(|c| c == col) {
            // Materialise the full arrangement BEFORE removing the column
            // from `visible`, so the hidden column keeps its live display
            // slot — a later re-show puts it back exactly there (see below).
            self.ensure_full_order();
            if self.visible.len() > 1 { self.visible.remove(pos); }
        } else {
            // Re-show rule (in priority order):
            // 1. If the layout carries an explicit FULL arrangement that has
            //    a slot for this column (restored from `columns/full_order`,
            //    or written by the hide itself), insert it right after the
            //    LAST visible column preceding that slot in the full
            //    arrangement — i.e. exactly where it sat before it was
            //    hidden, even across restarts and drags. Counting only over
            //    `full_order[..fp]` (not via a search through the whole list)
            //    is what keeps the slot correct when a visible neighbour sits
            //    AFTER the parked column in the full arrangement (e.g. Size
            //    dragged to the front: full = [size, name, preview, duration,
            //    …] → re-shown Duration must land right after Name, not at
            //    the end).
            // 2. Otherwise land next to its canonical predecessor when that
            //    neighbour is visible; failing that, append at the end.
            let full_slot = self
                .full_order
                .iter()
                .position(|c| c == col)
                .filter(|_| {
                    // Only trust the stored arrangement while it agrees with
                    // the live display order of the visible subset.
                    self.full_order
                        .iter()
                        .filter(|c| self.visible.contains(c))
                        .cloned()
                        .collect::<Vec<_>>()
                        == self.visible
                });
            let insert_at = match full_slot {
                Some(fp) => {
                    // The column keeps the slot it had before it was hidden:
                    // in the stored full arrangement all visible columns
                    // preceding it form exactly the live display prefix, so
                    // re-inserting right after the last of them reproduces
                    // its pre-hide position (and works even when a drag moved
                    // some following visible column into the parked column's
                    // canonical gap).
                    let before_count = self.full_order[..fp]
                        .iter()
                        .filter(|c| self.visible.contains(c))
                        .count();
                    before_count
                }
                None => {
                    let idx = Column::all().iter().position(|c| c == col).unwrap_or(0);
                    Column::all()[..idx]
                        .iter()
                        .rev()
                        .find(|c| self.visible.contains(c))
                        .and_then(|c| self.visible.iter().position(|v| v == c))
                        .map_or(self.visible.len(), |p| p + 1)
                }
            };
            self.visible.insert(insert_at, *col);
        }
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
        // Keep an explicitly carried full arrangement in sync with the new
        // display order (drag-and-drop reorder): rebuild it so the moved
        // column takes its live slot and every hidden column stays parked
        // next to its current visible neighbours. Without this the stale
        // arrangement would win on the next store/restore cycle and undo
        // the drag across a restart.
        if !self.full_order.is_empty() {
            self.ensure_full_order();
        }
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
