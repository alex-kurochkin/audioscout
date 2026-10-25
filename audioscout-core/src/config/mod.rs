use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub volume: f32, pub space_behavior_stop: bool, pub undo_depth: usize, pub undo_disk_limit_mb: usize,
    pub undo_dir: Option<String>, pub preview_prefetch: usize, pub skip_seconds: u32,
    pub meter_hold_enabled: bool, pub meter_hold_seconds: f64, pub columns: ColumnConfig,
    pub hotkeys: HashMap<String, HotkeyBinding>, #[serde(skip)] pub version: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnConfig { pub order: Vec<String>, pub full_order: Vec<String>, pub widths: HashMap<String, f32> }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotkeyBinding { pub key: String, pub ctrl: bool, pub shift: bool, pub alt: bool }

impl Default for AppConfig {
    fn default() -> Self {
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
        Self {
            volume: 0.8,
            space_behavior_stop: true, undo_depth: 20, undo_disk_limit_mb: 2048, undo_dir: None,
            preview_prefetch: 30, skip_seconds: 5, meter_hold_enabled: false, meter_hold_seconds: 1.0,
            columns: ColumnConfig {
                order: vec!["name","preview","duration","sample_rate","sample_type","channels","size","comment","date_created","date_modified"].into_iter().map(String::from).collect(),
                full_order: vec!["name","preview","duration","sample_rate","sample_type","channels","size","comment","date_created","date_modified"].into_iter().map(String::from).collect(),
                widths: HashMap::new(),
            },
            hotkeys, version: 0,
        }
    }
}

pub struct ConfigStore { config: AppConfig, config_path: PathBuf }

impl ConfigStore {
    pub fn new(config_dir: &Path) -> Self {
        let config_path = config_dir.join("config.json");
        let config = if config_path.exists() {
            std::fs::read_to_string(&config_path).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default()
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
}

pub fn config_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("AUDIOSCOUT_CONFIG_DIR") { PathBuf::from(dir) }
    else if let Some(home) = dirs_next::home_dir() { home.join(".config").join("audioscout") }
    else { PathBuf::from(".config/audioscout") }
}
