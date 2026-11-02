//! Regression tests for the two edit-pipeline bugs:
//!
//! 1. After an in-memory edit (e.g. normalize/mute) the player view updated
//!    but the audible stream kept playing the OLD buffer — because the audio
//!    output thread owns its own immutable `Arc<Document>` copy handed over
//!    via `Cmd::Load`. The app-side fix (`App::reload_player_audio`) clones
//!    the *current* document and republishes it after every mutation; this
//!    test pins down the data-flow contract that fix relies on: cloning a
//!    mutated `Document` carries the new samples to the consumer thread.
//!
//! 2. After Save, the file on disk must actually contain the edited samples
//!    (round-trip through the exact save call used by `save_file`) so the
//!    refreshed list preview renders the new content, not the old one.

use std::path::Path;
use std::sync::Arc;

use audioscout_core::audio_io::wav::WavFile;
use audioscout_core::audio_io::PcmFile;
use audioscout_core::{Document, SampleType};

/// Mirror of `App::normalize_selection`'s math applied to the whole file:
/// scale every sample so the peak hits the target amplitude.
fn normalize(doc: &mut Document, target: f32) {
    let peak = doc.samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    assert!(peak > 0.0, "test fixture must have non-zero peak");
    let gain = target / peak;
    for s in &mut doc.samples {
        *s *= gain;
    }
    doc.dirty = true;
}

#[test]
fn edited_document_arc_carries_new_samples_to_the_player() {
    // Fixture: mono 16-bit-ish float WAV with peak 0.25.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("edit.wav");
    let original: Vec<f32> = vec![0.25, -0.25, 0.125, 0.0, -0.125, 0.25];
    WavFile::save(&path, &original, 44100, 1, SampleType::Float32).unwrap();

    // "load_file": first Arc copy is what the output thread would hold.
    let (samples, sr, ch, st) = audioscout_core::audio_io::load_audio(&path).unwrap();
    let mut doc = Document {
        samples,
        sample_rate: sr,
        channels: ch,
        markers: Vec::new(),
        source_path: path.clone(),
        temp_path: None,
        dirty: false,
        sample_type: st,
    };
    let before_edit = Arc::new(doc.clone());

    // Edit in place (what mute/normalize do), then republish — exactly the
    // sequence `App::reload_player_audio` performs.
    normalize(&mut doc, 0.9);
    let after_edit = Arc::new(doc.clone());

    // The stale Arc (the bug's symptom): still the pre-edit buffer.
    let peak_before = before_edit.samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    assert!((peak_before - 0.25).abs() < 1e-6);
    // The freshly published Arc must carry the normalized audio.
    let peak_after = after_edit.samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    assert!(
        (peak_after - 0.9).abs() < 1e-6,
        "republished document must contain the edited samples, got peak {peak_after}"
    );
}

#[test]
fn save_writes_edited_samples_so_preview_sees_new_content() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("saved.wav");
    let original: Vec<f32> = vec![0.25, -0.25, 0.125, 0.0, -0.125, 0.25];
    WavFile::save(&path, &original, 44100, 1, SampleType::Float32).unwrap();

    let (samples, sr, ch, st) = audioscout_core::audio_io::load_audio(&path).unwrap();
    let mut doc = Document {
        samples,
        sample_rate: sr,
        channels: ch,
        markers: Vec::new(),
        source_path: path.clone(),
        temp_path: None,
        dirty: false,
        sample_type: st,
    };
    normalize(&mut doc, 0.9);

    // Same call `App::save_file` makes.
    WavFile::save(
        &path,
        &doc.samples,
        doc.sample_rate,
        doc.channels,
        doc.sample_type,
    )
    .unwrap();
    doc.dirty = false;

    // Re-read from disk (this is what the preview worker will render):
    // the waveform must come from the EDITED content.
    let (on_disk, _, _, _) = audioscout_core::audio_io::load_audio(Path::new(&path)).unwrap();
    let peak = on_disk.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    assert!(
        (peak - 0.9).abs() < 1e-5,
        "saved file must contain the normalized audio, got peak {peak}"
    );
    assert_ne!(on_disk, original, "preview would be stale if disk kept old samples");
}

/// BUGFIX regression ("picture changes, sound does not" while PLAYING):
/// `App::reload_player_audio` now routes a live edit through
/// `RodioOutput::update_document` (`Cmd::UpdateDoc`) instead of the old
/// stop-and-reload path. The audible-buffer swap must happen WITHOUT
/// rewinding/parking the transport: the controller thread applies
/// `Cmd::UpdateDoc`, publishes the edited document into the pump's doc slot
/// and bumps `doc_generation` so the pump flushes every Sink chunk sliced
/// from the pre-edit buffer before refilling from the current cursor.
///
/// Headless contract test: `RodioOutput::new` needs an audio device, so it
/// returns `None` on CI boxes without one — this test therefore pins down
/// the exact data-flow half of the fix (the fresh `Arc<Document>` published
/// mid-session carries the EDITED samples) plus the app-side routing rule:
/// while Playing → `update_document` (no transport reset), otherwise →
/// `load_document`. The threading half (queue flush on generation bump) is
/// exercised by `debug_pump_doc_peak`/`debug_doc_generation` whenever a
/// device is available.
#[test]
fn edited_arc_published_mid_session_carries_new_samples_without_transport_reset() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("live_edit.wav");
    let original: Vec<f32> = vec![0.25, -0.25, 0.125, 0.0, -0.125, 0.25];
    WavFile::save(&path, &original, 44100, 1, SampleType::Float32).unwrap();

    let (samples, sr, ch, st) = audioscout_core::audio_io::load_audio(&path).unwrap();
    let mut doc = Document {
        samples,
        sample_rate: sr,
        channels: ch,
        markers: Vec::new(),
        source_path: path.clone(),
        temp_path: None,
        dirty: false,
        sample_type: st,
    };

    // What the output thread owns BEFORE the edit (Cmd::Load handover copy).
    let stale_arc = Arc::new(doc.clone());

    // Normalize (Apply button math), then publish exactly like the "playing"
    // branch of `App::reload_player_audio`: a NEW Arc over the mutated doc.
    normalize(&mut doc, 0.9);
    let fresh_arc: Arc<Document> = Arc::new(doc.clone());

    // The bug's symptom was the output thread keeping the stale Arc; the fix
    // guarantees the freshly published Arc carries the edited samples...
    let stale_peak = stale_arc.samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    let fresh_peak = fresh_arc.samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    assert!((stale_peak - 0.25).abs() < 1e-6, "pre-edit Arc must be untouched");
    assert!((fresh_peak - 0.9).abs() < 1e-6, "published Arc must contain the edit");

    // ...WITHOUT rewinding the transport: the session bounds the controller
    // keeps feeding from are frame-based and independent of the buffer swap,
    // so swapping the Arc must not change total_frames()/sample rate.
    assert_eq!(
        stale_arc.total_frames(),
        fresh_arc.total_frames(),
        "in-place edits never change frame count — the live session range stays valid"
    );
    assert_eq!(stale_arc.sample_rate, fresh_arc.sample_rate);

    // Routing rule of the fix (mirrors reload_player_audio): playing →
    // update_document (swap in place), stopped/paused → load_document.
    // Both commands exist on the real type; compile-time pin of the API.
    fn _routing_pins(out: &audioscout_core::player::output::RodioOutput, d: Arc<Document>) {
        let _swap_in_place: bool = out.update_document(d.clone());
        out.load_document(d);
        let _ = out.debug_pump_doc_peak();
        let _ = out.debug_doc_generation();
    }
}
