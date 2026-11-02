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
    pub fn new() -> Self { Self { visible: Column::all().to_vec(), widths: std::collections::HashMap::new(), sort_column: None, sort_ascending: true } }

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
        let derived = |base: &[Column]| -> Vec<Column> {
            let mut cols: Vec<Column> = base
                .iter()
                .filter(|c| layout.visible.contains(c))
                .cloned()
                .collect();
            // Anything visible but missing from the base order goes last,
            // keeping its relative order from `layout.visible`.
            for c in &layout.visible {
                if !cols.contains(c) { cols.push(*c); }
            }
            cols
        };
        let ordered = if !full.is_empty() {
            derived(&full)
        } else if !order.is_empty() {
            derived(&order)
        } else if let Some(v) = visible_set.filter(|v| !v.is_empty()) {
            // Legacy fallback: only `visible` existed and stored the order.
            v.into_iter().filter(|c| layout.visible.contains(c)).collect()
        } else {
            layout.visible.clone()
        };
        if !ordered.is_empty() { layout.visible = ordered; }
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
        let canon_pos = |c: &Column| all.iter().position(|x| x == c).unwrap_or(0);

        // Walk the visible columns in DISPLAY order, emitting each one
        // followed by the hidden columns whose canonical home sits between
        // it and the next visible column (in canonical order). A dragged
        // arrangement therefore round-trips losslessly: filtering the result
        // through the visibility set reproduces `visible` exactly.
        let mut out: Vec<String> = Vec::with_capacity(all.len());
        let mut emitted: Vec<bool> = vec![false; all.len()];
        for vi in 0..self.visible.len() {
            let vis = &self.visible[vi];
            out.push(vis.key().to_string());
            emitted[canon_pos(vis)] = true;
            // Hidden neighbours that belong "after" this visible column:
            // everything sitting strictly between it and the NEXT *visible*
            // column in CANONICAL terms — i.e. hidden columns not claimed
            // earlier. NOTE: the window bounds are the canonical positions of
            // the neighbouring VISIBLE columns themselves (`vi`/`vi+1`), not
            // of their display-order predecessors/successors — with a dragged
            // layout those may sit far away on the canonical scale and would
            // claim hidden columns into the wrong gap (e.g. hiding Duration
            // from [Size, Name, …] must park it after Name, its canonical
            // neighbour, so re-showing lands it right where it used to sit).
            let lo = canon_pos(vis);
            let hi = self
                .visible
                .get(vi + 1)
                .map(|c| canon_pos(c))
                .unwrap_or(all.len());
            for h in all.iter().filter(|c| !self.visible.contains(c)) {
                let hp = canon_pos(h);
                if !emitted[hp] && hp > lo && hp < hi {
                    out.push(h.key().to_string());
                    emitted[hp] = true;
                }
            }
        }
        // Hidden columns outside every gap above (e.g. canonically before
        // the first visible column) go to the very front, keeping their
        // canonical relative order so hide/show cycles stay stable.
        let leftovers: Vec<String> = all
            .iter()
            .filter(|c| !emitted[canon_pos(c)])
            .map(|c| c.key().to_string())
            .collect();
        let mut final_out: Vec<String> = Vec::with_capacity(all.len());
        final_out.extend(leftovers);
        final_out.extend(out);
        final_out
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
        cfg.full_order = self.full_order_keys();
        cfg.widths = self.widths.clone();
    }

    pub fn toggle_visibility(&mut self, col: &Column) {
        if let Some(pos) = self.visible.iter().position(|c| c == col) {
            if self.visible.len() > 1 { self.visible.remove(pos); }
        } else {
            // Re-shown column lands next to its canonical predecessor when
            // that neighbour is visible — otherwise it goes to the end. This
            // keeps the display order stable across hide/show cycles and is
            // persisted through `store_into`.
            let idx = Column::all().iter().position(|c| c == col).unwrap_or(0);
            let insert_at = Column::all()[..idx]
                .iter()
                .rev()
                .find(|c| self.visible.contains(c))
                .and_then(|c| self.visible.iter().position(|v| v == c))
                .map_or(self.visible.len(), |p| p + 1);
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
