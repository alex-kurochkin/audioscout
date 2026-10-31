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
/// drifted against the body rows. The fix puts the header back into ONE
/// shared grid: "file_table" is the only table grid, it lives inside the
/// file-list ScrollArea, and the header cells are emitted as its FIRST ROW
/// (before the data-row loop) — so titles and row cells are laid out by the
/// very same grid pass over the very same rect and share every column's
/// geometry by construction. Per-column widths (`column_widths`) and the
/// shared cell painter (`draw_cell`) pin each cell to its configured width.
#[test]
fn file_table_header_shares_one_grid_with_the_rows() {
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

    // There must be exactly ONE table grid; the separate header grid and the
    // old body-grid id are gone.
    let header_grid_id = "egui::Grid::new(\"file_table_header\")";
    let body_grid_id = "egui::Grid::new(\"file_table\")";
    assert_eq!(
        code.matches(header_grid_id).count(),
        0,
        "the header must NOT be rendered as a separate second grid anymore"
    );
    assert_eq!(
        code.matches(body_grid_id).count(),
        1,
        "there must be exactly one \"file_table\" grid holding header and rows"
    );
    assert!(
        !code.contains("file_table_body"),
        "there must be no separate \"file_table_body\" grid anymore"
    );

    // That single grid lives INSIDE the file-list ScrollArea…
    let scroll_pos = code
        .find(".id_salt(\"file_list\")")
        .expect("the table must live in the file-list ScrollArea");
    let grid_pos = code.find(body_grid_id).unwrap();
    assert!(
        scroll_pos < grid_pos,
        "the shared table grid must be hosted by the ScrollArea"
    );

    // …and the header row is emitted BEFORE the data-row loop, i.e. it is
    // the FIRST ROW of that same grid.
    let header_loop_pos = code
        .find("for (ci, col) in visible_cols.iter().enumerate()")
        .expect("header cells must iterate over the visible columns");
    let data_loop_pos = code
        .find("for (idx, row) in file_rows.iter().enumerate()")
        .expect("data rows must iterate over file_rows");
    assert!(
        grid_pos < header_loop_pos && header_loop_pos < data_loop_pos,
        "the header cells must be laid out inside the shared grid, ahead of \
         the data rows (one grid, header first)"
    );
    // The header row is terminated with end_row() before the data loop starts.
    let between = &code[header_loop_pos..data_loop_pos];
    assert!(
        between.contains("ui.end_row()"),
        "the header row must end with ui.end_row() before the first data row"
    );

    // Header and data cells take their width from ONE shared vector of
    // per-column widths and are painted through the same helper.
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
    let painter_uses = code.matches("draw_cell(ui, width,").count();
    assert!(
        painter_uses >= 2,
        "header and data cells must be painted through the same draw_cell helper \
         (found {painter_uses} uses)"
    );
}

/// Regression guard for the Waveform/Spectrogram toggle pair: the
/// visualization-area mode is flipped from exactly TWO controls — one on
/// the top toolbar, one on the player panel (§3.5) — both driving the same
/// live `AudioView::mode` state through identically-named widgets. The
/// table-thumbnail toggle (preview_btn) is a separate control and must
/// remain untouched.
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
        2,
        "the waveform/spectrogram visualization toggle must exist exactly twice \
         (top toolbar + player panel)"
    );
    assert_eq!(
        code.matches("self.toggle_audio_view_mode()").count(),
        2,
        "exactly two controls may flip the audio-view display mode"
    );
    // The table-thumbnail toggle (preview_btn) is a different control and
    // must still be present.
    assert_eq!(
        code.matches("let preview_btn = egui::SelectableLabel::new(").count(),
        1,
        "the table-preview waveform/spectrogram toggle must remain"
    );
}

// Regression guard for the rodio 0.19 audio-output compile errors:
// `SamplesBuffer<S>` takes S = the sample TYPE (f32), not an iterator —
// passing `data.into_iter()` produced E0277/E0599 ("Vec<f32>: rodio::Sample
// not satisfied"). Also guards against resurrecting the non-existent
// `Sink::set_speed_of_playback` API from older rodio.
#[test]
fn output_uses_samples_buffer_of_sample_type_not_iterator() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/player/output.rs");
    let code: String = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", path, e))
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");

    // SamplesBuffer is parameterized by the element type f32 everywhere.
    assert!(
        code.contains("SamplesBuffer<f32>"),
        "output.rs must use rodio::buffer::SamplesBuffer<f32> (S = sample type)"
    );
    // No iterator-as-sample-type anywhere (the exact mistake from the build log).
    assert!(
        !code.contains("IntoIter"),
        "SamplesBuffer must NOT be parameterized with an iterator type"
    );
    assert!(
        !code.contains(".into_iter())") && !code.contains("into_iter(),\n"),
        "SamplesBuffer::new must receive the Vec<f32> directly (D: Into<Vec<S>>)"
    );
    // rodio 0.19 has no speed-of-playback knob on Sink.
    assert!(
        !code.contains("set_speed_of_playback"),
        "rodio 0.19 Sink has no set_speed_of_playback; must not be called"
    );
}

// The Controller must expose exactly one definition of each accessor that
// previously got duplicated (E0592: current_frame/current_secs/resume_frame/reset).
#[test]
fn controller_has_no_duplicate_public_accessors() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/player/controller.rs");
    let code: String = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", path, e))
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");

    for name in ["current_frame", "current_secs", "resume_frame", "reset"] {
        let needle = format!("pub fn {}(", name);
        let count = code.matches(&needle).count();
        assert_eq!(count, 1, "`{}` defined {} times in controller.rs (must be exactly 1)", name, count);
    }
    // Stale references removed together with the duplicates.
    assert!(!code.contains("sync_from_output"), "controller.rs still calls removed sync_from_output");
    assert!(!code.contains("self.meter"), "controller.rs still references removed meter field");
    assert!(!code.contains("use super::output::AudioOutput"), "AudioOutput does not exist in output.rs (it is RodioOutput)");
}

#[test]
fn output_holder_keeps_stream_handle_alive() {
    // Regression: playback was silent and the playhead frozen because only
    // `OutputStream` was stored — the last `OutputStreamHandle` drop destroys
    // the cpal stream (cpal 0.15 ALSA backend calls StreamId::destroy on
    // handle drop), so the Sink appended sources into a dead endpoint.
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/player/output.rs");
    let code: String = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", path, e))
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");

    // The holder struct must own both the stream and a handle clone.
    assert!(code.contains("_stream: rodio::OutputStream"), "holder lost the OutputStream field");
    assert!(code.contains("_handle: rodio::OutputStreamHandle"), "holder must keep an OutputStreamHandle alive");
    // And the construction site must actually clone the handle into it.
    assert!(code.contains("_handle: handle.clone()"), "handle is not cloned into the holder at construction");
}

#[test]
fn ui_poll_reads_audio_clock_directly_and_syncs_controller() {
    // Regression: playback DID run (disk write on Play, stop on Stop) but the
    // UI showed nothing moving — poll_playhead relied solely on PlayheadUpdate
    // events and never advanced the state machine. The 50 ms poll must read
    // SharedOutput.current_frame directly, mirror it into the Controller
    // (sync_playback_position), drive the cursor, and keep egui repainting.
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../audioscout-app/src/app.rs");
    let code: String = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", path, e))
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");

    // poll_playhead takes the egui context so it can schedule repaints itself.
    assert!(code.contains("fn poll_playhead(&mut self, ctx: &egui::Context)"), "poll_playhead lost its ctx parameter");
    assert!(code.contains("self.poll_playhead(ctx);"), "update() does not pass ctx to poll_playhead");
    // The real audio clock is polled from SharedOutput every tick...
    assert!(code.contains("shared.current_frame(ch)"), "poll no longer reads the audio clock via current_frame");
    // ...mirrored into the transport state machine (fixes Play-after-Pause
    // restarting from 0)...
    assert!(code.contains("ctrl.sync_playback_position(frame)"), "poll does not sync the controller position");
    // ...and drives the waveform cursor + auto-scroll.
    assert!(code.contains("self.audio_view.set_cursor(frame);"), "poll does not move the cursor");
    // Repaint cadence keeps the poll alive while playing.
    assert!(code.contains("ctx.request_repaint_after(std::time::Duration::from_millis(50));"), "poll lost its repaint scheduling");
}

#[test]
fn controller_exposes_playback_position_sync() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/player/controller.rs");
    let code: String = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", path, e))
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");

    assert_eq!(code.matches("pub fn sync_playback_position(").count(), 1, "sync_playback_position missing/duplicated");
    // It must be a playback-side update: never touch resume_frame (§5.3
    // CursorSource::Playback semantics — Pause resumes from the real clock).
    let body = code.split("pub fn sync_playback_position(").nth(1).unwrap();
    let body = body.split("pub fn ").next().unwrap();
    assert!(!body.contains("resume_frame"), "playback sync must not overwrite the resume point");
    assert!(body.contains("self.state != TransportState::Playing"), "playback sync must no-op outside Playing");
}

#[test]
fn device_enumeration_uses_cpal_015_host_api_via_rodio_reexport() {
    // Regression over FOUR broken builds of the ALSA-'default' device scan:
    //   * bare `cpal::` paths — cpal is NOT a direct dependency of
    //     audioscout-core (E0433); the crate root IS reachable as
    //     `rodio::cpal` though, because rodio 0.19 does `pub use cpal::{self, ...}`;
    //   * `cpal::host_ids()` — removed in cpal 0.13+, the 0.15 free function is
    //     `available_hosts()` (E0425);
    //   * `Host::from_id(...)` — never existed; the constructor is the free
    //     function `host_from_id(HostId) -> Result<Host, HostUnavailable>` (E0599);
    //   * `rodio::supported_output_devices()` — does not exist in rodio 0.19 (E0425);
    //   * trait methods `devices()/name()/supported_output_configs()` only
    //     resolve with `cpal::traits::{HostTrait, DeviceTrait}` imported,
    //     otherwise inference fails (E0282).
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/player/output.rs");
    let code: String = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", path, e))
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");

    // Only ever via rodio's re-export, never the unlinked bare crate.
    let without_reexport = code.replace("rodio::cpal", "");
    assert!(!without_reexport.contains("cpal::"), "bare cpal:: path — cpal is not a direct dependency, use rodio::cpal::");

    // Correct cpal 0.15 host API surface.
    assert!(code.contains("rodio::cpal::available_hosts()"), "must enumerate hosts via cpal 0.15 available_hosts()");
    assert!(code.contains("rodio::cpal::host_from_id(host_id)"), "must build Host via the free fn host_from_id");
    assert!(!code.contains("host_ids()"), "leftover pre-0.13 cpal API host_ids()");
    assert!(!code.contains("Host::from_id"), "nonexistent cpal API Host::from_id");
    assert!(!code.contains("supported_output_devices()"), "nonexistent rodio 0.19 API supported_output_devices");
    // Trait imports live per-function next to the code that needs them (the
    // device picker deliberately imports ONLY HostTrait because DeviceTrait
    // methods there are called fully-qualified — a combined import there is
    // an unused-import warning). Assert each function's own import line.
    assert!(code.contains("fn list_all_named_devices")
        && code.split("fn list_all_named_devices").nth(1).unwrap().contains("use rodio::cpal::traits::{DeviceTrait, HostTrait};"),
        "list_all_named_devices must import both cpal traits for devices()/name()");
    let dev_body = code.split("fn device_has_output").nth(1).unwrap().split("\nfn ").next().unwrap();
    assert!(code.contains("fn device_has_output")
        && dev_body.contains("use rodio::cpal::traits::DeviceTrait;"),
        "device_has_output must import DeviceTrait for supported_output_configs()");
    let pick_head = code.split("fn pick_output_device").nth(1).unwrap().split("\n\n").next().unwrap();
    assert!(code.contains("fn pick_output_device")
        && pick_head.contains("use rodio::cpal::traits::HostTrait;"),
        "pick_output_device must import HostTrait for default_output_device()");

    // Output-capability filtering and device typing. Probing must be LAZY
    // (per candidate in pick_output_device), never eager for every device —
    // eager supported_output_configs() opens every ALSA PCM via alsa-lib and
    // spams dmix/dsnoop/JACK errors on desktop Debian.
    assert!(code.contains("supported_output_configs()"), "candidate devices must still be probed for output configs");
    assert!(code.contains("fn device_has_output(device: &rodio::cpal::Device) -> bool"), "lazy per-device output probe expected");
    assert!(code.contains("Vec<(rodio::cpal::Device, String)>"), "device list typed with the cpal::Device that OutputStream::try_from_device accepts");

    // Device preference order: env override → PulseAudio host → ALSA virtual
    // routings ("pulse"/"default"/"dmix") → system default fallback. The old
    // exact-match-only scan for a literally-named "default" device is gone:
    // cpal's ALSA backend does not enumerate it on typical Debian setups.
    assert_eq!(code.matches("fn pick_output_device").count(), 1);
    assert!(code.contains("AUDIOSCOUT_ALSA_DEVICE"), "manual device override via env must exist");
    // cpal 0.15 HostId has no as_str(); the Pulse route is detected through
    // the *device name* of each host's default_output_device(), plus the
    // ALSA-host virtual plugins ("pulse"/"default"/"dmix") preference list.
    assert!(
        code.contains("\"pulse\"")
        && code.contains("lname.contains(\"pulse\") || lname.contains(\"pipewire\")")
        && code.contains("const PREFERRED: [&str; 3] = [\"pulse\", \"default\", \"dmix\"];"),
        "PulseAudio routing must be preferred over raw ALSA PCMs (by device name)");
    assert!(!code.contains("fn find_alsa_default_device"), "old exact-\"default\" scanner removed");
    assert!(
        code.contains("rodio::OutputStream::try_from_device_config(device, cfg)")
        && code.contains("fn select_f32_output_config(device: &rodio::cpal::Device) -> Option<rodio::cpal::SupportedStreamConfig>"),
        "chosen device must be opened with an explicit f32 SupportedStreamConfig (try_from_device_config)"
    );
    assert!(code.contains("OutputStream::try_default()"), "fallback to the system default device must remain");
}


#[test]
fn playback_is_paced_by_a_sink_queue_budget_not_wall_clock_sleeps() {
    // Regression over the "0.5 s of sound then freeze" family of bugs:
    //   * rodio 0.19 `Sink` is NOT `Clone` — the controller/pump used to call
    //     `sink.clone()` (E0599) and hold MutexGuard borrows across it
    //     (E0505); the shared handle must be an `Arc<Sink>` instead;
    //   * feeding must be paced by how many chunks are still queued in the
    //     Sink (MeteredSource decrements on exhaustion), not by blind
    //     wall-clock sleeps — otherwise the pump either races ahead (disk
    //     hammering, giant allocations) or starves the queue (~0.5 s of
    //     audio, then silence);
    //   * `session` is an `Option<(u64,u64)>` Copy tuple guard — matching it
    //     as `Some(&(s,e))` fails E0308.
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/player/output.rs");
    let code: String = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", path, e))
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");

    // Shared sink handle: Arc-wrapped, never cloned via Clone.
    assert!(code.contains("Arc<Mutex<Option<Arc<rodio::Sink>>>>"), "pump slot must hold an Arc<Sink>, not a non-Clone Sink copy");
    let flat: Vec<&str> = code.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    let flat = flat.join(" ");
    assert!(flat.contains("let sink = match rodio::Sink::try_new(&handle) { Ok(s) => Arc::new(s)"), "Sink must be wrapped in Arc right after try_new");
    // `Arc<Sink>` clones ARE allowed (`Some(s) => s.clone()`); a bare
    // `sink_clone` binding or an append through one was the old broken shape.
    assert!(
        !flat.contains("let sink_clone") && !flat.contains("sink_clone.append"),
        "no rodio::Sink::clone into a local — Sink is not Clone in rodio 0.19");
    assert!(flat.contains("fn controller_thread( sink: Arc<rodio::Sink>,"), "controller thread takes the Arc<Sink> handle");

    // Pump pacing by queue budget.
    assert_eq!(code.matches("fn pump_thread").count(), 1);
    let pump_body = code.split("fn pump_thread").nth(1).unwrap();
    let pump_body = pump_body.split("\nfn ").next().unwrap();
    assert!(pump_body.contains("let queued_chunks = Arc::new(AtomicUsize::new(0));"), "pump must track queued chunk count locally");
    assert!(pump_body.contains("queued_chunks.load(Ordering::Acquire) >= max_queued_chunks"), "pump must wait when the sink queue is full");
    assert!(pump_body.contains("queued_chunks.fetch_add(1"), "each appended chunk reserves a queue slot");
    assert!(pump_body.contains("chunks: queued_chunks.clone(),"), "the chunk carries its queue-slot guard into MeteredSource");
    assert!(pump_body.contains("sink.append(metered);"), "pump appends through the Arc<Sink> directly");

    // MeteredSource releases the slot exactly once, when exhausted.
    let metered = code.split("impl Iterator for MeteredSource").nth(1).unwrap().split("impl ").next().unwrap();
    assert!(metered.contains("self.chunks.fetch_sub(1, Ordering::AcqRel);"), "exhausted chunk must release its queue slot");
    assert!(code.contains("chunks: Arc<AtomicUsize>,"), "MeteredSource holds the shared queue counter");

    // Session guard destructuring must avoid the reference pattern.
    assert!(!code.contains("Some(&(s, e))"), "Option<(u64,u64)> Copy tuple must be matched by value (E0308)");
    assert!(flat.contains("Some((s, e)) => s < e && pump.cursor_frame.load(Ordering::Relaxed) >= e"), "completion check reads the pump cursor vs session end");
}

// ---------------------------------------------------------------------------
// Regression: non-editable audio files must load into the player.
//
// Bug (user report): double-clicking an mp3 in the file list showed
// "File format not editable" and refused to load; the message also stuck in
// the status bar after subsequently loading a wav. Per the TZ, playback must
// work for every supported audio format; "editable" only gates destructive
// edit operations (mute/normalize/save).
// ---------------------------------------------------------------------------

fn app_src_for_editable_check() -> String {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("../audioscout-app/src/app.rs");
    std::fs::read_to_string(p).expect("app.rs readable")
}

#[test]
fn load_file_accepts_all_supported_audio_formats() {
    let src = app_src_for_editable_check();
    // Isolate the body of `fn load_file` up to the next top-level fn.
    let start = src.find("fn load_file(&mut self, path: &Path)").expect("load_file present");
    let rest = &src[start..];
    let end = rest.find("\n    fn ").or(rest.find("\n    /// Rebuild")).expect("load_file end");
    let body = &rest[..end];

    // The hard gate on *editability* must be gone from the load path...
    assert!(
        !body.contains("if !is_editable_file(path)"),
        "load_file must not refuse non-editable formats (§3.4: all audio files play)"
    );
    // ...replaced by a supported-audio-format check.
    assert!(
        body.contains("is_audio_file(path)"),
        "load_file should validate against the supported audio extension list"
    );
    // A successful load must clear any stale status message (sticky-message bug).
    assert!(
        body.contains("self.status_message.clear()"),
        "successful load must clear the sticky status message"
    );
    // Edit gating stays where it belongs — on the edit operations themselves.
    assert!(
        src.contains("if !is_editable_file(&self.document.source_path)"),
        "edit operations must still be gated on is_editable_file"
    );
}

#[test]
fn wma_is_supported_non_editable_format() {
    assert!(
        audioscout_core::AUDIO_EXTENSIONS.contains(&"wma"),
        "WMA must be listed among supported (playable) audio extensions"
    );
    assert!(
        !audioscout_core::EDITABLE_EXTENSIONS.contains(&"wma"),
        "WMA is decode-only: it must NOT be editable"
    );
    let p = std::path::Path::new("track.wma");
    assert!(audioscout_core::is_audio_file(p));
    assert!(!audioscout_core::is_editable_file(p));
    let mp3 = std::path::Path::new("song.MP3");
    assert!(audioscout_core::is_audio_file(mp3), "extension matching is case-insensitive");
}
