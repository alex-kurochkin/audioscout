#[test]
fn visible_columns_roundtrip() {
    use audioscout_core::config::{AppConfig, ColumnConfig};
    use audioscout_core::table::{Column, ColumnLayout};
    use std::collections::HashMap;
    // 1. Round trip: layout -> keys -> config -> layout
    let mut layout = ColumnLayout::new();
    layout.toggle_visibility(&Column::Size);
    layout.toggle_visibility(&Column::Comment);
    let keys = layout.visible_keys();
    assert!(!keys.contains(&"size".to_string()));
    let cfg = ColumnConfig { order: vec![], full_order: vec![], widths: HashMap::new(), visible: Some(keys.clone()) };
    let restored = ColumnLayout::from_config(&cfg);
    assert_eq!(restored.visible, layout.visible);
    // 2. Old config without `visible` -> all columns (backward compat via serde default)
    let json = serde_json::json!({"volume":0.8,"space_behavior_stop":true,"undo_depth":20,"undo_disk_limit_mb":2048,"undo_dir":null,"preview_prefetch":30,"skip_seconds":5,"meter_hold_enabled":false,"meter_hold_ms":200,"meter_red_db":-0.1,"columns":{"order":[],"full_order":[],"widths":{}},"hotkeys":{},"preview_mode":"wave","audio_view_mode":"wave"});
    let parsed: AppConfig = serde_json::from_value(json).unwrap();
    assert!(parsed.columns.visible.is_none());
    let l2 = ColumnLayout::from_config(&parsed.columns);
    assert_eq!(l2.visible.len(), 10);
    // 3. Unknown/dup keys are sanitized
    let cfg3 = ColumnConfig { order: vec![], full_order: vec![], widths: HashMap::new(), visible: Some(vec!["name".into(),"bogus".into(),"name".into(),"duration".into()]) };
    let l3 = ColumnLayout::from_config(&cfg3);
    assert_eq!(l3.visible, vec![Column::Name, Column::Duration]);
}
