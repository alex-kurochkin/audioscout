use audioscout_core::table::{Column, ColumnLayout};
use audioscout_core::config::ColumnConfig;
use std::collections::HashMap;

fn cfg_with(order: &[&str], full: &[&str], visible: Option<&[&str]>) -> ColumnConfig {
    ColumnConfig {
        order: order.iter().map(|s| s.to_string()).collect(),
        full_order: full.iter().map(|s| s.to_string()).collect(),
        widths: HashMap::new(),
        visible: visible.map(|v| v.iter().map(|s| s.to_string()).collect()),
    }
}

#[test]
fn dbg_flow() {
    let mut layout = ColumnLayout::new();
    layout.move_column(&Column::Size, &Column::Name, false);
    layout.toggle_visibility(&Column::Duration); // hide
    let mut cc = cfg_with(&[], &[], None);
    layout.store_into(&mut cc);
    let mut restored = ColumnLayout::from_config(&cc);
    println!("restored.visible before reshow: {:?}", restored.visible);
    println!("restored.full_order: {:?}", restored.full_order);
    restored.toggle_visibility(&Column::Duration); // re-show
    println!("restored.visible after reshow: {:?}", restored.visible);
    let pos_dur = restored.visible.iter().position(|c| *c == Column::Duration).unwrap();
    let pos_name = restored.visible.iter().position(|c| *c == Column::Name).unwrap();
    println!("pos_dur={} pos_name={}", pos_dur, pos_name);
    assert_eq!(pos_dur, pos_name + 1);
}
