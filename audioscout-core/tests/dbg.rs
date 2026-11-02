use audioscout_core::config::ColumnConfig;
use audioscout_core::table::{Column, ColumnLayout};
use std::collections::HashMap;

#[test]
fn dbg_full_order() {
    let mut layout = ColumnLayout::new();
    layout.move_column(&Column::Size, &Column::Name, false);
    layout.toggle_visibility(&Column::Duration);
    eprintln!("visible: {:?}", layout.visible.iter().map(|c| c.key()).collect::<Vec<_>>());
    eprintln!("full_field: {:?}", layout.full_order.iter().map(|c| c.key()).collect::<Vec<_>>());
    eprintln!("full_keys:  {:?}", layout.full_order_keys());
    let mut cc = ColumnConfig { order: vec![], full_order: vec![], widths: HashMap::new(), visible: None };
    layout.store_into(&mut cc);
    eprintln!("cc.full: {:?}", cc.full_order);
    let restored = ColumnLayout::from_config(&cc);
    eprintln!("restored visible: {:?}", restored.visible.iter().map(|c| c.key()).collect::<Vec<_>>());
    eprintln!("restored full_field: {:?}", restored.full_order.iter().map(|c| c.key()).collect::<Vec<_>>());
    let mut r2 = restored.clone();
    r2.toggle_visibility(&Column::Duration);
    eprintln!("after reshow:     {:?}", r2.visible.iter().map(|c| c.key()).collect::<Vec<_>>());
}
