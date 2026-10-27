//! Regression tests for the bottom-panel layout bugs (player above the
//! visualization window, correct file-list height, toggle buttons present).

use std::path::Path;

fn app_source() -> String {
    let candidates = [
        Path::new("../audioscout-app/src/app.rs"),
        Path::new("audioscout-app/src/app.rs"),
    ];
    for c in &candidates {
        if c.exists() {
            return std::fs::read_to_string(c).unwrap();
        }
    }
    panic!("app.rs not found");
}

/// Extract the `id` of every TopBottomPanel declaration, in source order.
fn bottom_panel_ids(src: &str) -> Vec<String> {
    let mut ids = Vec::new();
    for line in src.lines() {
        if let Some(pos) = line.find("TopBottomPanel::bottom(\"") {
            let rest = &line[pos + "TopBottomPanel::bottom(\"".len()..];
            if let Some(end) = rest.find('"') {
                ids.push(rest[..end].to_string());
            }
        }
    }
    ids
}

#[test]
fn player_panel_declared_above_audio_view() {
    // In egui the FIRST declared bottom panel docks to the very bottom edge,
    // so declaration order player → audio_view → actions → status yields the
    // visual stack (top→bottom): status bar, file actions (Mute/Apply…),
    // player controls (Play/Stop…), visualization window. The transport
    // panel sits *above* the audio-content view and directly under the
    // file-actions panel — exactly what bug #1 asked for.
    let ids = bottom_panel_ids(&app_source());
    assert_eq!(ids, vec!["audio_view", "player", "actions", "status"],
        "unexpected bottom panel order {:?}", ids);
}

#[test]
fn all_bottom_panels_declared_before_central_panel() {
    // The file list height comes from `available_height()` inside the
    // CentralPanel; that only accounts for the bottom windows (whose height
    // can change via resizing) if they are reserved *before* it is shown.
    let src = app_source();
    let central = src.find("CentralPanel::default().show").expect("central panel");
    for name in ["\"status\"", "\"player\"", "\"audio_view\"", "\"actions\""] {
        let pos = src
            .find(&format!("TopBottomPanel::bottom({name}"))
            .unwrap_or_else(|| panic!("panel {name} missing"));
        assert!(pos < central, "panel {name} must be declared before CentralPanel");
    }
}

#[test]
fn toggle_buttons_present() {
    let src = app_source();
    // Top toolbar: table-preview waveform/spectrogram toggle.
    assert!(src.contains("self.toggle_preview_mode()"), "toolbar preview toggle missing");
    // Player panel: visualization-area toggle.
    assert!(src.contains("self.toggle_audio_view_mode()"), "player view toggle missing");
}
