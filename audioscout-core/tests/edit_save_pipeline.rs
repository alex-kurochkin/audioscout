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
