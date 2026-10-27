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
    // Visualization-area toggle (bug #3): the switch code existed but its
    // state was unknown — now `AudioView::mode` is a real, persisted state
    // flipped through `toggle_audio_view_mode()` from a toolbar button.
    assert!(src.contains("self.toggle_audio_view_mode()"), "view-mode toggle missing");

    // There are exactly two view-mode toggle handlers in the source: the
    // definition (`fn toggle_audio_view_mode`) plus one call site per button.
    // One button sits on the top toolbar (bug #3 asks for the toggle there),
    // the other on the player panel (§3.5 "toggle Waveform/Spectrogram").
    let mut toggle_uses = src
        .match_indices("self.toggle_audio_view_mode()")
        .map(|(i, _)| i);
    let toolbar_toggle_at = toggle_uses.next().expect("toolbar view toggle missing");
    let player_toggle_at = toggle_uses.next().expect("player-panel view toggle missing");
    assert!(
        toggle_uses.next().is_none(),
        "unexpected extra view-toggle call sites"
    );

    let toolbar_at = src.find("TopBottomPanel::top(\"toolbar\")").expect("toolbar panel");
    let side_at = src.find("SidePanel::left").expect("folder tree panel");
    assert!(
        toolbar_at < toolbar_toggle_at && toolbar_toggle_at < side_at,
        "the first Waveform/Spectrogram toggle must be rendered on the top toolbar panel"
    );

    let player_at = src
        .find("TopBottomPanel::bottom(\"player\")")
        .expect("player panel");
    let actions_at = src
        .find("TopBottomPanel::bottom(\"actions\")")
        .expect("actions panel");
    assert!(
        player_at < player_toggle_at && player_toggle_at < actions_at,
        "the second Waveform/Spectrogram toggle must be rendered on the player panel"
    );

    // Both toggles must reflect their live state via SelectableLabel so the
    // current mode is visible (bug #3: "state of the switch code unknown").
    let selectable_count = src.matches("egui::SelectableLabel::new(").count();
    assert!(
        selectable_count >= 3,
        "expected at least three stateful toggle buttons (table preview + \
         toolbar/player view toggles), found {}",
        selectable_count
    );

    // Core side: the mode is real state with a flip helper.
    let av = std::fs::read_to_string(
        Path::new("../audioscout-core/src/audio_view.rs")
            .exists()
            .then(|| Path::new("../audioscout-core/src/audio_view.rs"))
            .unwrap_or(Path::new("audioscout-core/src/audio_view.rs")),
    )
    .unwrap();
    assert!(av.contains("pub enum AudioViewMode"), "AudioViewMode enum missing");
    assert!(av.contains("pub fn toggle_mode"), "AudioView::toggle_mode missing");
}

/// Regression test for the *window height* bug: the file window (the table's
/// vertical ScrollArea inside the CentralPanel) must be sized from egui's own
/// `available_height()` only. All four bottom panels — status bar, file
/// actions, player controls and the resizable visualization window — are
/// declared before the CentralPanel, so their real, measured space is already
/// reserved by the time the central area computes its remaining height. The
/// old code additionally subtracted a hand-made estimate of the very same
/// stack (`bottom_panels_height`, built from hardcoded 40 px / 20 px
/// constants), double-counting the reservation and making the file window far
/// shorter than it should be (and unstable: every drift of the estimate
/// triggered another repaint).
#[test]
fn file_list_height_uses_only_eguis_available_height() {
    // Comment-insensitive view of the source: explanatory BUGFIX comments may
    // mention the removed identifiers, only real code references matter.
    let code: String = app_source()
        .lines()
        .map(|l| {
            let t = l.trim_start();
            if t.starts_with("//") {
                ""
            } else {
                match l.find("//") {
                    Some(i) => &l[..i],
                    None => l,
                }
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    let src = app_source();

    // The stale cross-frame estimate and its ad-hoc measurement are gone.
    assert!(
        !code.contains("bottom_panels_height"),
        "the file-list height must not rely on a hand-made bottom-stack estimate"
    );
    assert!(
        !code.contains("measured_stack") && !code.contains("stack_below_this_panel"),
        "manual bottom-stack measurement duplicates egui's own reservation"
    );

    // The list height is derived directly from `available_height()`.
    assert!(
        src.contains("let list_max_height = ui.available_height().max(60.0);"),
        "file-list height must be exactly the space egui has left above the bottom stack"
    );

    // …and that value is what clamps the table's scroll area.
    assert!(
        src.contains(".max_height(list_max_height)"),
        "the file-list ScrollArea must use the computed height"
    );
}

/// Regression test for the files-window grid bug: the column-header cells
/// used to live in a SECOND `egui::Grid` ("file_table_header") separate from
/// the data rows' grid ("file_table_body"). Two independent grids never share
/// a real cell mesh — their column offsets come from separately allocated
/// rects (e.g. the vertical scrollbar eats width only inside the scroll area
/// that hosts the body grid), so the header drifted against the body rows.
/// The fix: exactly ONE grid ("file_table") whose FIRST ROW contains the
/// header cells, followed by all data rows — titles and cells are then cells
/// of one and the same grid by construction.
#[test]
fn file_table_has_one_grid_with_header_as_first_row() {
    // Comment-insensitive view of the source again: BUGFIX comments may
    // mention the removed grid ids, only real code references matter.
    let code: String = app_source()
        .lines()
        .map(|l| {
            let t = l.trim_start();
            if t.starts_with("//") {
                ""
            } else {
                match l.find("//") {
                    Some(i) => &l[..i],
                    None => l,
                }
            }
        })
        .collect::<Vec<_>>()
        .join("\n");

    // The two-grid layout is gone…
    assert!(
        !code.contains("file_table_header") && !code.contains("file_table_body"),
        "header and body must not be rendered as two separate grids"
    );

    // …and there is exactly one table grid in the files window.
    let grid_id = "egui::Grid::new(\"file_table\")";
    let occurrences = code.matches(grid_id).count();
    assert_eq!(
        occurrences, 1,
        "the files window must build exactly one egui Grid named \"file_table\""
    );

    // Inside that single grid's closure: the header row is emitted first
    // (its end_row precedes the data-row loop), and the data rows close the
    // same grid.show call.
    let grid_pos = code.find(grid_id).unwrap();
    let after_grid = &code[grid_pos..];
    let header_end_row = after_grid
        .find("ui.end_row();")
        .expect("header row must end with ui.end_row() before the data rows");
    let data_loop = after_grid
        .find("for (idx, row) in file_rows.iter().enumerate()")
        .expect("data rows must be rendered in the same grid closure");
    assert!(
        header_end_row < data_loop,
        "the header cells must be the FIRST ROW of the single file_table grid"
    );
    let grid_close = after_grid
        .rfind("});")
        .expect("grid.show closure must be closed");
    assert!(
        grid_close > data_loop,
        "the data rows must live inside the same grid.show closure as the header"
    );
}
