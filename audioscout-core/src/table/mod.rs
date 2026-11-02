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
    /// display arrangement: every visible column keeps its slot from the
    /// last-known full order (so a dragged layout is stored exactly), hidden
    /// columns fill the remaining slots near their old canonical neighbours.
    /// `from_config` filters this list through the visibility set, so the
    /// visible slots alone determine the restored display order on the next
    /// launch — and a column re-shown later lands right where it used to sit.
    pub fn full_order_keys(&self) -> Vec<String> {
        self.full_order_keys_against(&Column::all().iter().map(|c| c.key().to_string()).collect::<Vec<_>>())
    }

    /// Slot-interleave helper behind [`full_order_keys`](Self::full_order_keys):
    /// `anchor` is the previous full order (canonical on a fresh layout).
    /// Visible columns keep their anchor slots when they still fit the new
    /// display order, otherwise they take the earliest free slots; hidden
    /// columns return to their anchor gaps. The result is always a
    /// permutation of all known columns.
    fn full_order_keys_against(&self, anchor: &[String]) -> Vec<String> {
        let all = Column::all();
        let n = all.len();
        // Canonical fallback for columns missing from a stale anchor list.
        let canon_pos = |c: &Column| all.iter().position(|x| x == c).unwrap_or(0);
        let anchor_pos = |key: &str, c: &Column| -> usize {
            anchor.iter().position(|k| k == key).unwrap_or(canon_pos(c))
        };

        // Pass 1: assign each visible column a target slot inside the
        // anchor ordering. Start at its anchor slot; if the previous visible
        // column already occupies a later-or-equal slot (the user dragged
        // past it), slide forward until the placement stays monotone.
        let mut placed_slots: Vec<usize> = Vec::with_capacity(self.visible.len());
        let mut out: Vec<Option<Column>> = vec![None; n];
        for col in &self.visible {
            let mut slot = anchor_pos(col.key(), col);
            while slot < n && (out[slot].is_some() || placed_slots.last().map_or(false, |p| slot <= *p)) {
                slot += 1;
            }
            if slot >= n {
                slot = (0..n).find(|s| out[*s].is_none()).unwrap_or(n - 1);
            }
            out[slot] = Some(*col);
            placed_slots.push(slot);
        }

        // Pass 2: hidden columns back into the remaining gaps, closest to
        // their own anchor position first — stable across hide/show cycles.
        let mut hidden: Vec<Column> = all.iter().cloned().filter(|c| !self.visible.contains(c)).collect();
        while !hidden.is_empty() {
            let mut best: Option<(usize, usize)> = None; // (gap index, hidden index)
            for (hi, h) in hidden.iter().enumerate() {
                let want = anchor_pos(h.key(), h);
                let gap = (want..n).find(|g| out[*g].is_none())
                    .or_else(|| (0..n).find(|g| out[*g].is_none()))
                    .unwrap_or(0);
                if best.as_ref().map_or(true, |(bg, _)| gap < *bg) {
                    best = Some((gap, hi));
                }
            }
            if let Some((gap, hi)) = best {
                out[gap] = Some(hidden.remove(hi));
            }
        }

        out.into_iter().flatten().map(|c| c.key().to_string()).collect()
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
