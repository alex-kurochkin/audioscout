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
/// used to live in a SECOND `egui::Grid` ("file_table_header") whose column
/// geometry was computed independently of the data-rows' grid, so the header
/// drifted against the body rows. The fix binds every header cell to the
/// file-row cell underneath it: both grids use the SAME per-column widths
/// (`column_widths`), the SAME horizontal spacing and the SAME cell painter
/// (`draw_cell`), so their columns are identical by construction — each
/// title sits exactly above its column of file rows (and stays visible while
/// the body scrolls).
#[test]
fn file_table_header_is_bound_to_the_row_columns() {
    // Comment-insensitive view of the source again: comments may mention
    // removed constructs, only real code references matter.
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

    // The old "one shared mesh" experiment (header as first row of the body
    // grid inside the ScrollArea) is gone: the header now lives in its own
    // fixed grid ABOVE the scroll area…
    let header_grid_id = "egui::Grid::new(\"file_table_header\")";
    let body_grid_id = "egui::Grid::new(\"file_table\")";
    assert_eq!(
        code.matches(header_grid_id).count(),
        1,
        "the header must be rendered as one dedicated grid"
    );
    assert_eq!(
        code.matches(body_grid_id).count(),
        1,
        "the data rows must be rendered as one dedicated grid"
    );
    let header_pos = code.find(header_grid_id).unwrap();
    let body_pos = code.find(body_grid_id).unwrap();
    let scroll_pos = code
        .find(".id_salt(\"file_list\")")
        .expect("the body must live in the file-list ScrollArea");
    assert!(
        header_pos < scroll_pos && scroll_pos < body_pos,
        "the header grid must sit above the ScrollArea that hosts the data grid"
    );

    // …but the two grids share ONE set of per-column widths, so every header
    // cell is locked to the column of file rows beneath it.
    assert!(
        code.contains("let column_widths: Vec<f32>"),
        "a single shared list of column widths must drive the table layout"
    );
    let width_uses = code.matches("column_widths[ci]").count();
    assert!(
        width_uses >= 2,
        "both the header cells and the data cells must take their width from \
         the same column_widths vector (found {width_uses} uses)"
    );

    // Same horizontal spacing on both grids → identical column offsets.
    assert!(
        code.contains(".spacing([10.0, 4.0])"),
        "the table grids must use the shared spacing"
    );
    let spacing_uses = code.matches(".spacing([10.0, 4.0])").count();
    assert_eq!(
        spacing_uses, 2,
        "header grid and data grid must declare the same spacing"
    );

    // And the same cell painter for header and data cells.
    let painter_uses = code.matches("draw_cell(ui, width,").count();
    assert!(
        painter_uses >= 2,
        "header and data cells must be painted through the same draw_cell helper \
         (found {painter_uses} uses)"
    );

    // No leftover second body grid id anywhere.
    assert!(
        !code.contains("file_table_body"),
        "there must be no separate \"file_table_body\" grid anymore"
    );
}

/// Regression test for the duplicated Waveform/Spectrogram toggle: the
/// visualization-area mode had TWO controls (one on the toolbar, one on the
/// player panel). The user asked to remove the second one; exactly one
/// audio-view toggle must remain in the whole UI.
#[test]
fn audio_view_mode_has_a_single_toggle_button() {
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

    assert_eq!(
        code.matches("let view_btn = egui::SelectableLabel::new(").count(),
        1,
        "the waveform/spectrogram visualization toggle must exist exactly once"
    );
    assert_eq!(
        code.matches("self.toggle_audio_view_mode()").count(),
        1,
        "only one control may flip the audio-view display mode"
    );
    // The table-thumbnail toggle (preview_btn) is a different control and
    // must still be present.
    assert_eq!(
        code.matches("let preview_btn = egui::SelectableLabel::new(").count(),
        1,
        "the table-preview waveform/spectrogram toggle must remain"
    );
}
