use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Canonical list of rebindable actions shown in the Settings→Hotkeys table.
/// `hotkey_defaults()` and the runtime matcher must stay in sync with this.
pub const HOTKEY_ACTIONS: &[&str] = &[
    "playpause", "prev", "next", "gotoStart", "mute", "normalize",
    "undo", "redo", "addMarker", "deleteMarker", "clearSelection",
    "zoomIn", "zoomOut", "zoomReset",
];

/// Human-readable labels for the Hotkeys table (Settings dialog).
pub fn hotkey_label(action: &str) -> &'static str {
    // The Settings-window mock-up shows the raw action ids (camelCase, e.g.
    // "addMarker", "clearSelection") in the Action column of the Hotkeys
    // table, so we display the id itself.
    match action {
        "playpause" => "playPause",
        "prev" => "prevFile",
        "next" => "nextFile",
        "gotoStart" => "goToStart",
        "mute" => "muteSelection",
        "normalize" => "normalizeSelection",
        "undo" => "undo",
        "redo" => "redo",
        "addMarker" => "addMarker",
        "deleteMarker" => "deleteMarker",
        "clearSelection" => "clearSelection",
        "zoomIn" => "zoomIn",
        "zoomOut" => "zoomOut",
        "zoomReset" => "zoomReset",
        _ => "Unknown",
    }
}

/// Default shortcut strings per tz §9 (hotkey model): Space toggles
/// playback, Ctrl+Z/Ctrl+Shift+Z is undo/redo, Ctrl+arrows zoom, etc.
/// Used by Settings→Hotkeys "Reset all to defaults".  The map is ordered
/// exactly like `HOTKEY_ACTIONS` — serde_json keeps insertion order, so the
/// persisted config.json and the Settings table list actions in the tz order.
pub fn hotkey_defaults() -> HashMap<String, HotkeyBinding> {
    // (action id, key, ctrl, shift, alt) — tz §9.
    const DEFAULTS: [(&str, &str, bool, bool, bool); 14] = [
        ("playpause", "Space", false, false, false),   // play/pause audio
        ("prev", "P", false, false, false),            // previous file
        ("next", "N", false, false, false),            // next file
        ("gotoStart", "W", false, false, false),       // rewind to start
        ("mute", "M", true, false, false),             // mute selection
        ("normalize", "N", true, false, false),        // normalize selection
        ("undo", "Z", true, false, false),             // undo
        ("redo", "Z", true, true, false),              // redo
        ("addMarker", "M", false, false, true),        // add marker
        ("deleteMarker", "Delete", false, false, false), // delete marker
        ("clearSelection", "Escape", false, false, false), // clear selection
        ("zoomIn", "ArrowUp", true, false, false),     // zoom in
        ("zoomOut", "ArrowDown", true, false, false),  // zoom out
        ("zoomReset", "0", true, false, false),        // reset zoom
    ];
    DEFAULTS
        .iter()
        .map(|(id, key, ctrl, shift, alt)| {
            (
                (*id).to_string(),
                HotkeyBinding {
                    key: (*key).to_string(),
                    ctrl: *ctrl,
                    shift: *shift,
                    alt: *alt,
                },
            )
        })
        .collect()
}

/// Keys egui can name back (`Key::from_name` round-trip) — used to validate
/// a captured shortcut before rebinding it.
pub fn is_rebindable_key(key: &str) -> bool {
    matches!(key, "Space" | "Delete" | "Escape" | "ArrowUp" | "ArrowDown" | "ArrowLeft" | "ArrowRight" | "Home" | "End" | "Insert" | "Backspace" | "Enter" | "Tab")
        || (key.len() == 1 && key.chars().next().map(|c| c.is_ascii_alphanumeric()).unwrap_or(false))
        || (key.starts_with('F') && key[1..].parse::<u8>().map(|n| (1..=24).contains(&n)).unwrap_or(false))
}

/// Pretty label for a binding in the Hotkeys table ("Ctrl+Shift+Z").
/// Key names are title-cased (`Escape` -> `Esc`, `ArrowLeft` -> `Left`),
/// matching the compact shortcuts shown in the Settings mock-up.
pub fn format_hotkey(b: &HotkeyBinding) -> String {
    let mut s = String::new();
    if b.ctrl { s.push_str("Ctrl+"); }
    if b.shift { s.push_str("Shift+"); }
    if b.alt { s.push_str("Alt+"); }
    s.push_str(&pretty_key_name(&b.key));
    s
}

/// Compact display name for an egui `Key` debug string.
pub fn pretty_key_name(key: &str) -> String {
    match key {
        "Escape" => "Esc".to_string(),
        "ArrowLeft" => "Left".to_string(),
        "ArrowRight" => "Right".to_string(),
        "ArrowUp" => "Up".to_string(),
        "ArrowDown" => "Down".to_string(),
        other => other.to_string(),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub volume: f32, pub space_behavior_stop: bool, pub undo_depth: usize, pub undo_disk_limit_mb: usize,
    pub undo_dir: Option<String>, pub preview_prefetch: usize, pub skip_seconds: u32,
    pub meter_hold_enabled: bool,
    /// Peak-hold release delay in milliseconds (Settings→Behavior and
    /// appearance, "Volumeter peak hold delay").  Once the held peak stops
    /// being beaten for this long it resets to the live level.  Default 200.
    #[serde(default = "default_meter_hold_ms")]
    pub meter_hold_ms: u32,
    /// Volume-meter "red" threshold in dB (e.g. -0.1).  Tunable in
    /// Settings→Behavior and appearance; currently only persisted to
    /// config.json (the meter still uses its built-in red cutoff).
    #[serde(default = "default_meter_red_db")]
    pub meter_red_db: f32,
    pub columns: ColumnConfig,
    pub hotkeys: HashMap<String, HotkeyBinding>,
    /// Table thumbnail style (Waveform/Spectrogram toggle on the top panel).
    /// Serialized as "wave"/"spectro"; old configs without this key default
    /// to waveform via `deserialize_with` below.
    #[serde(default = "default_preview_mode", deserialize_with = "de_preview_mode", serialize_with = "ser_preview_mode")]
    pub preview_mode: String,
    /// Player visualization style (Waveform/Spectrogram toggle on the player
    /// panel). Independent from `preview_mode` (§3.5).
    #[serde(default = "default_preview_mode", deserialize_with = "de_preview_mode", serialize_with = "ser_preview_mode")]
    pub audio_view_mode: String,
    #[serde(skip)] pub version: u64,
}

fn default_preview_mode() -> String { "wave".into() }

/// Default volume-meter red threshold: -0.1 dB (Settings→Behavior and
/// appearance, "Volumeter red value").  Stored as a negative dB level; the
/// UI shows its absolute magnitude ("0.1" + "dB" suffix).
fn default_meter_red_db() -> f32 { -0.1 }

/// Default peak-hold release delay: 200 ms (Settings→Behavior and
/// appearance, "Volumeter peak hold delay").
fn default_meter_hold_ms() -> u32 { 200 }

// The mode values live in `audioscout_core::PreviewMode` (string forms
// "wave"/"spectro"). Config must not depend on egui-side types, so they are
// stored as plain strings and validated here.
fn de_preview_mode<'de, D>(d: D) -> Result<String, D::Error>
where D: serde::Deserializer<'de> {
    use serde::Deserialize;
    let s = String::deserialize(d)?;
    Ok(match s.as_str() { "spectro" => "spectro", _ => "wave" }.to_string())
}
fn ser_preview_mode<S>(v: &String, s: S) -> Result<S::Ok, S::Error>
where S: serde::Serializer {
    s.serialize_str(v)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnConfig { pub order: Vec<String>, pub full_order: Vec<String>, pub widths: HashMap<String, f32> }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotkeyBinding { pub key: String, pub ctrl: bool, pub shift: bool, pub alt: bool }

impl Default for AppConfig {
    fn default() -> Self {
        // Single source of truth for the tz §9 defaults — shared with
        // Settings→Hotkeys "Reset all to defaults".
        let hotkeys = hotkey_defaults();
        Self {
            volume: 0.8,
            space_behavior_stop: true, undo_depth: 20, undo_disk_limit_mb: 2048, undo_dir: None,
            preview_prefetch: 30, skip_seconds: 5, meter_hold_enabled: false, meter_hold_ms: default_meter_hold_ms(),
            meter_red_db: default_meter_red_db(),
            columns: ColumnConfig {
                order: vec!["name","preview","duration","sample_rate","sample_type","channels","size","comment","date_created","date_modified"].into_iter().map(String::from).collect(),
                full_order: vec!["name","preview","duration","sample_rate","sample_type","channels","size","comment","date_created","date_modified"].into_iter().map(String::from).collect(),
                widths: HashMap::new(),
            },
            hotkeys,
            preview_mode: "wave".into(),
            audio_view_mode: "wave".into(),
            version: 0,
        }
    }
}

pub struct ConfigStore { config: AppConfig, config_path: PathBuf }

impl ConfigStore {
    pub fn new(config_dir: &Path) -> Self {
        let config_path = config_dir.join("config.json");
        let config = if config_path.exists() {
            // Merge over `Default`: a config written by an older build simply
            // lacks the newer keys — instead of discarding *all* user settings
            // because one field is missing, patch the defaults with whatever
            // the file actually contains.
            std::fs::read_to_string(&config_path)
                .ok()
                .and_then(|s| {
                    serde_json::from_str::<serde_json::Value>(&s).ok().map(|v| {
                        let mut base = serde_json::to_value(AppConfig::default()).expect("serializable");
                        if let (Some(base_obj), Some(v_obj)) = (base.as_object_mut(), v.as_object()) {
                            for (k, val) in v_obj { base_obj.insert(k.clone(), val.clone()); }
                        }
                        serde_json::from_value::<AppConfig>(base).unwrap_or_default()
                    })
                })
                .unwrap_or_default()
        } else { AppConfig::default() };
        Self { config, config_path }
    }
    pub fn config(&self) -> &AppConfig { &self.config }
    pub fn config_mut(&mut self) -> &mut AppConfig { self.config.version += 1; &mut self.config }
    pub fn save(&self) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(&self.config).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let tmp_path = self.config_path.with_extension("json.tmp");
        std::fs::write(&tmp_path, &json)?;
        std::fs::rename(&tmp_path, &self.config_path)?;
        Ok(())
    }
    pub fn version(&self) -> u64 { self.config.version }

    /// Delete every stored marker (DB rows + per-file marker sidecars).
    /// Used by Settings→Storage maintenance "Delete all markers".
    pub fn delete_all_markers(db: &crate::db::Database, config_dir: &Path) -> std::io::Result<usize> {
        let n = db
            .delete_all_markers()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        // Marker sidecars are stored in the DB (tz §7); nothing on disk to
        // purge beyond the rows themselves.
        let _ = config_dir;
        Ok(n)
    }

    /// Delete all cached preview images (DB rows + PNG files on disk).
    /// Used by Settings→Storage maintenance "Delete all markers previews".
    /// Returns the number of DB rows removed.
    pub fn delete_all_previews(db: &crate::db::Database, config_dir: &Path) -> std::io::Result<usize> {
        let n = db
            .delete_all_previews()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let previews_dir = config_dir.join("previews");
        if previews_dir.exists() { std::fs::remove_dir_all(&previews_dir)?; }
        Ok(n)
    }

    /// Remove preview files/rows whose source audio no longer exists or has
    /// changed (tz §12 "Scan markers"/orphan cleanup). Thin wrapper kept so
    /// both callers share one entry point.
    pub fn clean_orphaned_previews() -> std::io::Result<usize> {
        crate::preview_worker::cleanup_previews()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}

/// Default undo-data directory shown in Settings→Undo history when the user
/// has not overridden `AppConfig::undo_dir`: `~/.config/audioscout/undo/`.
pub fn default_undo_dir() -> String {
    format!("{}/undo/", config_dir().display())
}

pub fn config_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("AUDIOSCOUT_CONFIG_DIR") { PathBuf::from(dir) }
    else if let Some(home) = dirs_next::home_dir() { home.join(".config").join("audioscout") }
    else { PathBuf::from(".config/audioscout") }
}
