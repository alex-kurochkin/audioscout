//! Regression tests for the persisted column ORDER (drag-and-drop reorder of
//! the file-table headers must survive an app restart, tz §3.3).
//!
//! Storage model: `columns/order` keeps the display order of the visible
//! columns, `columns/full_order` the order of every known column, and
//! `columns/visible` the visibility set. `ColumnLayout::from_config` rebuilds
//! the live layout from these three fields; `store_into` writes a consistent
//! snapshot back.

use audioscout_core::config::{AppConfig, ColumnConfig};
use audioscout_core::table::{Column, ColumnLayout};
use std::collections::HashMap;

fn cfg_with(order: &[&str], full: &[&str], visible: Option<&[&str]>) -> ColumnConfig {
    ColumnConfig {
        order: order.iter().map(|s| s.to_string()).collect(),
        full_order: full.iter().map(|s| s.to_string()).collect(),
        widths: HashMap::new(),
        visible: visible.map(|v| v.iter().map(|s| s.to_string()).collect()),
    }
}

/// The dragged arrangement round-trips through the config: reorder the
/// layout, store it, restore — the visible order is exactly preserved.
#[test]
fn drag_reorder_roundtrips_through_config() {
    let mut layout = ColumnLayout::new();
    // Simulate two header drags: Duration in front of Name, Size after Date Created.
    layout.move_column(&Column::Duration, &Column::Name, false);
    layout.move_column(&Column::Size, &Column::DateCreated, true);
    let before = layout.visible.clone();

    let mut cc = ColumnConfig {
        order: vec![],
        full_order: vec![],
        widths: HashMap::new(),
        visible: None,
    };
    layout.store_into(&mut cc);

    let restored = ColumnLayout::from_config(&cc);
    assert_eq!(restored.visible, before, "display order must survive the round trip");
    // The persisted fields are individually meaningful too (tz §3.3).
    assert_eq!(cc.order, before.iter().map(|c| c.key().to_string()).collect::<Vec<_>>());
    assert_eq!(cc.full_order.len(), Column::all().len());
    assert_eq!(cc.visible.as_ref().unwrap(), &cc.order);
}

/// A config written by a previous run (as serialized into config.json)
/// restores the same order even when only `order` + `visible` are present
/// (older builds did not maintain `full_order`).
#[test]
fn order_restores_from_json_without_full_order() {
    let json = serde_json::json!({
        "volume": 0.8, "space_behavior_stop": true, "undo_depth": 20,
        "undo_disk_limit_mb": 2048, "undo_dir": null, "preview_prefetch": 30,
        "skip_seconds": 5, "meter_hold_enabled": false, "meter_hold_ms": 200,
        "meter_red_db": -0.1,
        "columns": {
            "order": ["duration", "name", "size"],
            "full_order": [],
            "widths": {},
            "visible": ["duration", "name", "size"]
        },
        "hotkeys": {}, "preview_mode": "wave", "audio_view_mode": "wave"
    });
    let parsed: AppConfig = serde_json::from_value(json).unwrap();
    let layout = ColumnLayout::from_config(&parsed.columns);
    assert_eq!(layout.visible, vec![Column::Duration, Column::Name, Column::Size]);
}

/// `full_order` wins over a stale `order` field (they are written together,
/// but a hand-edited/corrupted config must still restore the fuller source).
#[test]
fn full_order_takes_precedence_over_stale_order() {
    let cc = cfg_with(
        &["name", "duration"], // stale
        &["size", "duration", "name", "channels"],
        Some(&["name", "duration", "size", "channels"]),
    );
    let layout = ColumnLayout::from_config(&cc);
    assert_eq!(
        layout.visible,
        vec![Column::Size, Column::Duration, Column::Name, Column::Channels]
    );
}

/// Hiding a column never disturbs the relative order of the remaining ones,
/// and re-showing it lands it next to its canonical predecessor; the whole
/// hide/show cycle persists through the config.
#[test]
fn hide_show_keeps_order_and_persists() {
    let mut layout = ColumnLayout::new();
    layout.move_column(&Column::Size, &Column::Name, false); // size, name, preview, …
    layout.toggle_visibility(&Column::Duration); // hide duration
    let hidden_ok = !layout.is_visible(&Column::Duration);
    assert!(hidden_ok);
    assert_eq!(layout.visible[0], Column::Size);
    assert_eq!(layout.visible[1], Column::Name);

    let mut cc = cfg_with(&[], &[], None);
    layout.store_into(&mut cc);

    // Re-show on the RESTORED layout (simulates the next launch with the
    // checkbox flipped back on): duration reappears right after name.
    let mut restored = ColumnLayout::from_config(&cc);
    // Re-showing a hidden column lands it next to its CANONICAL predecessor
    // (Preview for Duration — see `full_order_keys` docs): parked slots in
    // `full_order` only have to keep the visible order intact across a
    // restart, they are not required to reproduce the exact pre-hide slot.
    restored.toggle_visibility(&Column::Duration);
    let pos_dur = restored.visible.iter().position(|c| *c == Column::Duration).unwrap();
    let pos_preview = restored.visible.iter().position(|c| *c == Column::Preview).unwrap();
    assert_eq!(pos_dur, pos_preview + 1, "re-shown column must land after its neighbour");
    // And that state round-trips once more.
    let mut cc2 = cfg_with(&[], &[], None);
    restored.store_into(&mut cc2);
    let again = ColumnLayout::from_config(&cc2);
    assert_eq!(again.visible, restored.visible);
}

/// Unknown/duplicate entries inside `order`/`full_order` are sanitized and
/// can never drop or duplicate a visible column.
#[test]
fn malformed_order_lists_are_sanitized() {
    let cc = cfg_with(
        &["bogus", "name", "name"],
        &["nope", "duration", "name", "duration", "preview"],
        Some(&["name", "duration", "preview", "size"]),
    );
    let layout = ColumnLayout::from_config(&cc);
    // Order comes from full_order filtered by visibility; `size` was not in
    // full_order at all, so it is appended — nothing lost, nothing doubled.
    assert_eq!(
        layout.visible,
        vec![Column::Duration, Column::Name, Column::Preview, Column::Size]
    );
}

/// An entirely empty column config falls back to all columns in canonical
/// order (fresh install / wiped settings).
#[test]
fn empty_config_falls_back_to_all_columns() {
    let cc = cfg_with(&[], &[], None);
    let layout = ColumnLayout::from_config(&cc);
    assert_eq!(layout.visible, Column::all().to_vec());
}

/// Widths keep surviving the round trip alongside the order.
#[test]
fn widths_survive_with_order() {
    let mut layout = ColumnLayout::new();
    layout.set_width(&Column::Name, 250.0);
    layout.move_column(&Column::Channels, &Column::Name, false);
    let mut cc = cfg_with(&[], &[], None);
    layout.store_into(&mut cc);
    let restored = ColumnLayout::from_config(&cc);
    assert_eq!(restored.width(&Column::Name), 250.0);
    assert_eq!(restored.visible, layout.visible);
}


