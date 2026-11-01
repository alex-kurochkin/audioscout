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
    match action {
        "playpause" => "Play / Pause",
        "prev" => "Previous file",
        "next" => "Next file",
        "gotoStart" => "Go to start",
        "mute" => "Mute selection",
        "normalize" => "Normalize selection",
        "undo" => "Undo",
        "redo" => "Redo",
        "addMarker" => "Add marker",
        "deleteMarker" => "Delete marker",
        "clearSelection" => "Clear selection",
        "zoomIn" => "Zoom in",
        "zoomOut" => "Zoom out",
        "zoomReset" => "Zoom reset",
        _ => "Unknown",
    }
}

/// Default shortcut strings per tz §9 (hotkey model): Space toggles
/// playback, Ctrl+Z/Ctrl+Shift+Z is undo/redo, Ctrl+arrows zoom, etc.
/// Used by Settings→Hotkeys "Reset all to defaults".
pub fn hotkey_defaults() -> HashMap<String, HotkeyBinding> {
    let mut hotkeys = HashMap::new();
    hotkeys.insert("mute".into(), HotkeyBinding { key: "M".into(), ctrl: true, shift: false, alt: false });
    hotkeys.insert("normalize".into(), HotkeyBinding { key: "N".into(), ctrl: true, shift: false, alt: false });
    hotkeys.insert("prev".into(), HotkeyBinding { key: "P".into(), ctrl: false, shift: false, alt: false });
    hotkeys.insert("next".into(), HotkeyBinding { key: "N".into(), ctrl: false, shift: false, alt: false });
    hotkeys.insert("playpause".into(), HotkeyBinding { key: "Space".into(), ctrl: false, shift: false, alt: false });
    hotkeys.insert("gotoStart".into(), HotkeyBinding { key: "W".into(), ctrl: false, shift: false, alt: false });
    hotkeys.insert("undo".into(), HotkeyBinding { key: "Z".into(), ctrl: true, shift: false, alt: false });
    hotkeys.insert("redo".into(), HotkeyBinding { key: "Z".into(), ctrl: true, shift: true, alt: false });
    hotkeys.insert("zoomIn".into(), HotkeyBinding { key: "ArrowUp".into(), ctrl: true, shift: false, alt: false });
    hotkeys.insert("zoomOut".into(), HotkeyBinding { key: "ArrowDown".into(), ctrl: true, shift: false, alt: false });
    hotkeys.insert("zoomReset".into(), HotkeyBinding { key: "0".into(), ctrl: true, shift: false, alt: false });
    hotkeys.insert("addMarker".into(), HotkeyBinding { key: "M".into(), ctrl: false, shift: false, alt: true });
    hotkeys.insert("deleteMarker".into(), HotkeyBinding { key: "Delete".into(), ctrl: false, shift: false, alt: false });
    hotkeys.insert("clearSelection".into(), HotkeyBinding { key: "Escape".into(), ctrl: false, shift: false, alt: false });
    hotkeys
}

/// Keys egui can name back (`Key::from_name` round-trip) — used to validate
/// a captured shortcut before rebinding it.
pub fn is_rebindable_key(key: &str) -> bool {
    matches!(key, "Space" | "Delete" | "Escape" | "ArrowUp" | "ArrowDown" | "ArrowLeft" | "ArrowRight" | "Home" | "End" | "Insert" | "Backspace" | "Enter" | "Tab")
        || (key.len() == 1 && key.chars().next().map(|c| c.is_ascii_alphanumeric()).unwrap_or(false))
        || (key.starts_with('F') && key[1..].parse::<u8>().map(|n| (1..=24).contains(&n)).unwrap_or(false))
}

/// Pretty label for a binding in the Hotkeys table ("Ctrl+Shift+Z").
pub fn format_hotkey(b: &HotkeyBinding) -> String {
    let mut s = String::new();
    if b.ctrl { s.push_str("Ctrl+"); }
    if b.shift { s.push_str("Shift+"); }
    if b.alt { s.push_str("Alt+"); }
    s.push_str(&b.key);
    s
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub volume: f32, pub space_behavior_stop: bool, pub undo_depth: usize, pub undo_disk_limit_mb: usize,
    pub undo_dir: Option<String>, pub preview_prefetch: usize, pub skip_seconds: u32,
    pub meter_hold_enabled: bool, pub meter_hold_seconds: f64, pub columns: ColumnConfig,
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
            preview_prefetch: 30, skip_seconds: 5, meter_hold_enabled: false, meter_hold_seconds: 1.0,
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
        let n = db.delete_all_markers().map_err(std::io::Error::new(std::io::ErrorKind::Other,)?;
        // Marker sidecars are stored in the DB (tz §7); nothing on disk to
        // purge beyond the rows themselves.
        let _ = config_dir;
        Ok(n)
    }

    /// Delete all cached preview images (DB rows + PNG files on disk).
    /// Used by Settings→Storage maintenance "Delete all markers previews".
    /// Returns the number of DB rows removed.
    pub fn delete_all_previews(db: &crate::db::Database, config_dir: &Path) -> std::io::Result<usize> {
        let n = db.delete_all_previews().map_err(std::io::Error::new(std::io::ErrorKind::Other,)?;
        let previews_dir = config_dir.join("previews");
        if previews_dir.exists() { std::fs::remove_dir_all(&previews_dir)?; }
        Ok(n)
    }

    /// Remove preview files/rows whose source audio no longer exists or has
    /// changed (tz §12 "Scan markers"/orphan cleanup). Thin wrapper kept so
    /// both callers share one entry point.
    pub fn clean_orphaned_previews() -> std::io::Result<usize> {
        crate::preview_worker::cleanup_previews()
    }
}

pub fn config_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("AUDIOSCOUT_CONFIG_DIR") { PathBuf::from(dir) }
    else if let Some(home) = dirs_next::home_dir() { home.join(".config").join("audioscout") }
    else { PathBuf::from(".config/audioscout") }
}
