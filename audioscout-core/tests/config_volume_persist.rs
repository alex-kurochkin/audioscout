//! Regression test for volume persistence (bug: changing the volume in the
//! transport slider never reached config.json, so every launch reset to the
//! default gain). The write-on-change flow is: mutate `AppConfig::volume`
//! through `ConfigStore::config_mut`, `save()` atomically into config.json,
//! then a fresh `ConfigStore::new` over the same directory must restore it.

use audioscout_core::config::ConfigStore;

#[test]
fn volume_change_survives_store_reload() {
    let dir = std::env::temp_dir().join(format!(
        "audioscout-volume-persist-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create temp config dir");

    // First "session": change the volume and persist immediately.
    {
        let mut store = ConfigStore::new(&dir);
        assert_eq!(store.config().volume, 0.8, "default gain per tz §3.5");
        store.config_mut().volume = 0.42;
        store.save().expect("atomic config save");
    }

    // Second "launch": the value must come back from config.json.
    let reloaded = ConfigStore::new(&dir);
    assert!(
        (reloaded.config().volume - 0.42).abs() < f32::EPSILON,
        "volume must survive the restart, got {}",
        reloaded.config().volume
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// Saving unrelated fields must not clobber the persisted volume (guards the
/// merge-over-defaults loader: older configs lacking keys are patched, not
/// discarded).
#[test]
fn partial_config_update_keeps_volume() {
    let dir = std::env::temp_dir().join(format!(
        "audioscout-volume-partial-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    {
        let mut store = ConfigStore::new(&dir);
        store.config_mut().volume = 0.25;
        store.config_mut().skip_seconds = 9;
        store.save().unwrap();
        // Touch only an unrelated setting, as e.g. the Settings dialog does.
        store.config_mut().preview_prefetch = 12;
        store.save().unwrap();
    }
    let reloaded = ConfigStore::new(&dir);
    assert!((reloaded.config().volume - 0.25).abs() < f32::EPSILON);
    assert_eq!(reloaded.config().skip_seconds, 9);
    assert_eq!(reloaded.config().preview_prefetch, 12);
    std::fs::remove_dir_all(&dir).ok();
}
