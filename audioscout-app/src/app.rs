use audioscout_core::config::ConfigStore;
use audioscout_core::player::controller::Controller;
use audioscout_core::player::output::RodioOutput;
use audioscout_core::player::{MeterData, PlayheadState, TransportState};
use audioscout_core::scanner::Scanner;
use audioscout_core::table::{Column, ColumnLayout, FileRow};
use audioscout_core::audio_view::{AudioView, AudioViewMode};
use audioscout_core::undo::Stack as UndoStack;
use audioscout_core::db::Database;
use audioscout_core::audio_io::PcmFile;
use audioscout_core::disk::DiskProvider;
use audioscout_core::preview_worker::{PreviewWorker, PreviewJob};
use audioscout_core::{Document, EventChannel, Marker, PreviewMode, UiEvent, is_audio_file, is_editable_file};
use crossbeam_channel::Receiver;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use image::GenericImageView;

// ---------------------------------------------------------------------------
// Panel layout constants (§ UI panels)
// ---------------------------------------------------------------------------
// Heights of the fixed docked panels. `default_height()` alone is NOT enough
// to make a code change take effect: egui caches every panel's measured rect
// in `Memory` under the panel id (`PanelState::load`, see egui-0.29
// containers/panel.rs), and the cached value wins over the newly compiled-in
// default — this is exactly why editing `default_height(20.0)` into
// `default_height(200.0)` changed nothing at runtime.
// Two independent mechanisms make a rebuild-with-new-constant GUARANTEED to
// change the rendered height:
//  1. Layout fingerprint (`PANEL_LAYOUT_ID`): stamped at app construction;
//     when a rebuilt binary carries different constants, the fingerprint
//     mismatches and ALL cached `PanelState` entries are purged once, on the
//     very first frame — before any panel is shown (see
//     `purge_panel_size_caches`). The panels then behave like a first run and
//     adopt the fresh defaults. This mechanism alone already guarantees the
//     behaviour for *any* id scheme.
//  2. Height-versioned ids (`panel_id_with_height`): changing a constant also
//     produces a brand-new panel id, so egui finds no stored `PanelState` for
//     it even if some cache entry survived the purge. NOTE: an id change also
//     resets each panel's internal widget state (scroll offsets, open
//     popups...), because those are keyed by the panel id — which is fine
//     here: it only ever happens across rebuilds with edited constants.
//     The version suffix uses the raw bit pattern (`to_bits`) so it round-
//     trips exactly ("player@0x42a00000" style).
const TOOLBAR_HEIGHT: f32 = 30.0;
const AUDIO_VIEW_HEIGHT: f32 = 200.0;
const PLAYER_HEIGHT: f32 = 50.0;
const ACTIONS_HEIGHT: f32 = 28.0;
const STATUS_HEIGHT: f32 = 22.0;

// Paired constants #1: LEFT content inset of each panel — how far the first
// widget of the panel's inner horizontal row is indented from the panel's
// left edge. Applied inside `ui.horizontal(...)` via `add_space` (see
// `apply_content_left_inset`).
const TOOLBAR_CONTENT_LEFT_INSET: f32 = 2.0;
const PLAYER_CONTENT_LEFT_INSET: f32 = 2.0;
const ACTIONS_CONTENT_LEFT_INSET: f32 = 2.0;
const STATUS_CONTENT_LEFT_INSET: f32 = 2.0;

// Paired constants #2: TOP content inset of each panel — the distance from
// the panel's top edge down to where its content starts. Each value must
// stay <= the paired *_HEIGHT minus the content's natural height, otherwise
// the content is clipped by the exact-height panel (see
// `apply_content_top_inset`).
const TOOLBAR_CONTENT_TOP_INSET: f32 = 1.0;
const PLAYER_CONTENT_TOP_INSET: f32 = 1.0;
const ACTIONS_CONTENT_TOP_INSET: f32 = 1.0;
const STATUS_CONTENT_TOP_INSET: f32 = 1.0;

/// Build a stable-but-versioned panel id. The version covers the panel
/// height constant and its paired inset constants: any change to either
/// yields a different id, which invalidates egui's cached `PanelState` (see
/// the comment above the height constants).
fn panel_id_with_height(name: &str, h: f32) -> String {
    format!("{name}@{:#x}", h.to_bits())
}

/// Draw a small context menu (a popup with flat text items) anchored at
/// `pos`, opened by a right-click on `opener`. Returns the label of the
/// clicked item. egui 0.29's public `MenuRoot` machinery keys its state off
/// `Response::id` and would only ever show one shared menu instance, so the
/// popup is driven by an explicit per-widget id stored in `ctx.data`:
/// the widget that was right-clicked claims the popup, and every other
/// primary press outside the menu rect dismisses it.
///
/// The returned `&str` borrows from `items` (the caller owns the labels —
/// static string literals at every call site), which is why the lifetime is
/// tied to the input slice rather than `'static`.
pub fn context_menu<'a>(
    ui: &egui::Ui,
    opener: &egui::Response,
    id: egui::Id,
    pos: egui::Pos2,
    items: &'a [&'a str],
) -> Option<&'a str> {
    let ctx = ui.ctx();
    let claimed = ctx.data(|d| d.get_temp::<egui::Id>(id)) == Some(id);
    if !claimed && opener.secondary_clicked() {
        ctx.data_mut(|d| d.insert_temp(id, id));
        ctx.request_repaint();
    }
    if ctx.data(|d| d.get_temp::<egui::Id>(id)) != Some(id) {
        return None;
    }

    let mut chosen: Option<&'a str> = None;
    let frame = egui::Frame::menu(&ctx.style());
    egui::Area::new(id.with("menu"))
        .fixed_pos(pos)
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            frame.show(ui, |ui| {
                // Deterministic vertical layout: inside a Popup the default
                // layout may be horizontal, which would place items side by
                // side instead of as a list.
                ui.vertical(|ui| {
                    for item in items {
                        let resp = ui.add(
                            egui::Label::new(
                                egui::WidgetText::from(*item).color(ui.visuals().text_color()),
                            )
                            .selectable(false),
                        );
                        if resp.clicked() {
                            chosen = Some(*item);
                        }
                    }
                });
            });
        });

    let close = ctx.input(|i| {
        i.key_pressed(egui::Key::Escape)
            || (i.pointer.any_pressed()
                && i.pointer.primary_down()
                && i.pointer
                    .interact_pos()
                    .map_or(true, |p| !ui.min_rect().contains(p)))
    });
    if close || chosen.is_some() {
        ctx.data_mut(|d| d.remove::<egui::Id>(id));
    }
    chosen
}

/// Convenience wrapper pairing a panel's height with its two content insets
/// (left and top). The version suffix mixes all three constants so an edit
/// to any one of them re-keys the panel id (see `panel_id_with_height`).
/// NOTE: the mixed value is only used for identification — the actual layout
/// still uses `*_HEIGHT` / `*_CONTENT_LEFT_INSET` / `*_CONTENT_TOP_INSET`
/// directly.
#[inline]
fn panel_id(name: &str, height: f32, left_inset: f32, top_inset: f32) -> String {
    // Mix the three bit patterns so every input constant contributes to the
    // output bits; any changed constant changes at least one bit of `mixed`.
    let h = height.to_bits();
    let l = left_inset.to_bits();
    let t = top_inset.to_bits();
    let mixed = (h ^ l.rotate_left(7) ^ t.rotate_left(19)) | 0x8000_0000;
    panel_id_with_height(name, f32::from_bits(mixed))
}

/// Indent the panel's first content row `inset` points from the panel's
/// LEFT edge. Applied inside `ui.horizontal(...)` as `ui.add_space(inset)`:
/// in a horizontal layout `add_space` advances the horizontal cursor, which
/// is exactly the intended effect here (the original behaviour of this
/// knob, kept under the honest *_LEFT_* name after the *_TOP_* insets were
/// introduced).
#[inline]
fn apply_content_left_inset(ui: &mut egui::Ui, inset: f32) {
    if inset > 0.0 {
        ui.add_space(inset);
    }
}

/// Push the panel's content `inset` points below the panel's TOP edge
/// (within the panel's exact height).
///
/// NOTE: must be applied on the panel BODY (vertical axis), never inside a
/// `ui.horizontal(...)` row — there `add_space`/allocations advance the
/// horizontal cursor and a vertical offset is silently ignored. We consume
/// an invisible strip of `inset` points at the top of the panel body via
/// `allocate_exact_size`, so every following widget starts `inset` points
/// lower while the panel keeps its exact outer height.
#[inline]
fn apply_content_top_inset(ui: &mut egui::Ui, inset: f32) {
    if inset > 0.0 {
        let w = ui.available_width();
        ui.allocate_exact_size(egui::vec2(w, inset), egui::Sense::hover());
    }
}

/// Aggregate fingerprint of every panel height and paired inset constant
/// (left and top). Stored at startup; if a rebuilt binary has different
/// constants, the fingerprint changes and the stale per-panel size caches get
/// purged exactly once.
const PANEL_LAYOUT_ID: u64 = {
    // deterministic mix of the five heights and eight insets (four left +
    // four top; bit patterns of f32 constants)
    let mut hash: u64 = 0xcbf29ce484222325; // FNV-1a offset basis
    let vals = [
        TOOLBAR_HEIGHT.to_bits(),
        AUDIO_VIEW_HEIGHT.to_bits(),
        PLAYER_HEIGHT.to_bits(),
        ACTIONS_HEIGHT.to_bits(),
        STATUS_HEIGHT.to_bits(),
        TOOLBAR_CONTENT_LEFT_INSET.to_bits(),
        PLAYER_CONTENT_LEFT_INSET.to_bits(),
        ACTIONS_CONTENT_LEFT_INSET.to_bits(),
        STATUS_CONTENT_LEFT_INSET.to_bits(),
        TOOLBAR_CONTENT_TOP_INSET.to_bits(),
        PLAYER_CONTENT_TOP_INSET.to_bits(),
        ACTIONS_CONTENT_TOP_INSET.to_bits(),
        STATUS_CONTENT_TOP_INSET.to_bits(),
    ];
    let mut i = 0;
    while i < vals.len() {
        let mut v = vals[i];
        let mut b = 0;
        while b < 4 {
            hash ^= (v & 0xff) as u64;
            hash = hash.wrapping_mul(0x100000001b3); // FNV-1a prime
            v >>= 8;
            b += 1;
        }
        i += 1;
    }
    hash
};

/// Drop every cached `PanelState` — the struct egui's `TopBottomPanel` stores
/// under the panel id to remember last frame's rect (`PanelState::store`,
/// egui-0.29 containers/panel.rs). Two facts about that store drive this
/// implementation:
///  * entries are inserted with `insert_persisted`, i.e. they survive across
///    runs via eframe's persistence — so wiping them must happen at runtime,
///    not just rely on a fresh process;
///  * the map key is `TypeId ^ id`, so entries can be deleted *by type*
///    without knowing the panel ids: `remove_by_type::<PanelState>()` removes
///    exactly the panel-size cache and touches nothing else (focus, popups,
///    window geometry all stay intact — unlike a full memory reset, which
///    egui 0.29 does not even expose publicly: `Memory` has no `clear()`).
/// Must be called BEFORE any panel is shown in the frame, so the panels read
/// "no stored state" and fall back to the freshly compiled-in heights.
fn purge_panel_size_caches(ctx: &egui::Context) {
    ctx.data_mut(|d| {
        d.remove_by_type::<egui::containers::panel::PanelState>();
    });
}

/// Main application state

/// Live column-drag state, owned by `AudioScoutApp`. Created on the frame a
/// press happens inside a header cell (armed by the header widgets via
/// `col_press_frame`, see the file-table header loop); becomes an ACTIVE
/// drag once the pointer moves past `DRAG_THRESHOLD` from the press point.
/// On release an active drag reorders columns; an inactive one was just a
/// click (sorting stays with egui's own `clicked()` response).
pub struct ColDrag {
    /// Key of the column being dragged.
    src: String,
    /// Press position (window coordinates), captured on the press frame.
    start: egui::Pos2,
    /// x-ranges of ALL visible header cells, frozen at mousedown (see
    /// `header_cells_snapshot`).
    cells: Vec<(String, f32, f32)>,
    /// True once the pointer moved far enough — distinguishes drags from
    /// plain clicks.
    active: bool,
}

/// Id under which header cell #`ci` reports its exact rect during the frame's
/// layout pass (read back by `header_cells_snapshot`).
fn hdr_cell_id(ci: usize) -> egui::Id {
    egui::Id::new(("hdr_cell", ci))
}

/// Stable `id_salt` of the files-list `ScrollArea` (see `.id_salt(...)` on the
/// area itself). The post-release hscroll cleanup needs the area's persistent
/// Memory id to reset its cached horizontal offset; egui derives that id as
/// `CentralPanel ui id .with(id_salt)`, so the salt is pinned here and both
/// call sites reference this constant.
const FILE_LIST_ID_SALT: &str = "file_list";

/// Persistent-memory [`egui::Id`] of a widget inside the CentralPanel whose
/// `id_salt` is `salt`. egui computes the ScrollArea id as
/// `panel_ui.id().with(&id_salt)` where the CentralPanel's own Ui id is
/// `Id::new((ctx.viewport_id(), "central_panel"))` (see egui-0.29
/// `containers/panel.rs::CentralPanel::show_dyn` and `Ui::make_persistent_id`
/// in `ui.rs`) — reproduced exactly here so post-release cleanup can reset
/// the files list's cached horizontal offset in Memory.
fn central_ui_child_id(ctx: &egui::Context, salt: &'static str) -> egui::Id {
    let panel_id = egui::Id::new((ctx.viewport_id(), "central_panel"));
    panel_id.with(&salt)
}

/// Per-file popup-menu id used by the files-list right-click menu
/// (§ "Open with VLC"). The menu machinery (`context_menu`) keys its state
/// off this id, so one shared constant is enough: only one file menu is ever
/// open at a time. `egui::Id::new` is not const in 0.29, hence `fn` instead
/// of `const`.
fn file_vlc_menu_id() -> egui::Id {
    egui::Id::new("vlc_file_menu")
}

/// Persistent state of the per-file "Open with VLC" popup: which row (path)
/// opened it and where its anchor rect sits. Stored in `ctx.data` under the
/// shared menu id so the popup survives across frames even though the click
/// event itself is consumed within the same frame by the cell's response —
/// this was the reason right-click on a file row never showed the menu.
#[derive(Clone)]
struct FileMenuState {
    path: PathBuf,
    /// Bottom-left corner of the clicked row's Name cell — the popup is
    /// pinned here (row-anchored, not pointer-anchored).
    anchor: egui::Pos2,
}

pub struct AudioScoutApp {
    config_store: ConfigStore,
    db: Database,
    event_rx: Receiver<UiEvent>,
    event_tx: crossbeam_channel::Sender<UiEvent>,
    scanner: Scanner,
    preview_worker: PreviewWorker,
    current_dir: Option<PathBuf>,
    file_rows: Vec<FileRow>,
    scan_generation: u64,
    column_layout: ColumnLayout,
    selected_row: Option<usize>,
    filter_text: String,
    document: Document,
    active_file: Option<PathBuf>,
    player: Arc<Mutex<Controller>>,
    /// Real audio output (rodio/cpal, tz §5.2). `None` when no audio device
    /// could be opened — transport stays usable for browsing/editing.
    output: Option<RodioOutput>,
    /// Volume-meter display state (§3.5.1): live values + peak-hold timer.
    meter_live: MeterData,
    meter_hold: MeterData,
    /// Monotonic wall-clock (ctx.input time) of the last NEW peak — the hold
    /// is released `meter_hold_ms` after this moment (see the hold algorithm
    /// in `UiEvent::VolumeMeterUpdate`).  Replaces the old "hold_until"
    /// deadline so the delay value from config is applied per-frame.
    meter_peak_at: f64,
    /// Wall-clock of the previous UI frame — drives the ~300 ms meter
    /// decay after pause/stop (§3.5.1 "При паузе/стопе шкалы гаснут до нуля").
    last_frame_time: f64,
    audio_view: AudioView,
    undo_stack: UndoStack,
    show_settings: bool,
    /// Transient state of the Settings modal (hotkey capture + status line).
    settings_hotkey_capture: Option<String>,
    settings_status: String,
    show_columns_dialog: bool,
    status_message: String,
    normalize_dbfs: f32,
    expanded_dirs: HashMap<PathBuf, Vec<PathBuf>>,
    disks: Vec<audioscout_core::disk::DiskInfo>,
    /// Paths with a preview job currently accepted by the worker (waiting for
    /// `UiEvent::PreviewReady`). Cleared when the result arrives or the scan
    /// generation changes.
    preview_inflight: HashSet<PathBuf>,
    /// Paths whose submit was rejected (duplicate / queue full); retried by
    /// `retry_missing_previews` while previews are pending.
    preview_retry: HashSet<PathBuf>,
    /// Active thumbnail style for the file list (Waveform / Spectrogram).
    /// Shared with the scanner (`Scanner::preview_mode_handle`) so cached
    /// lookups during a scan use the same mode as the submitted jobs.
    preview_mode: Arc<Mutex<PreviewMode>>,
    /// Per-mode texture caches. Keyed by UUID (which already embeds a
    /// mode-specific suffix), kept separate per mode so switching back and
    /// forth does not force re-uploading textures.
    preview_cache_wave: HashMap<String, egui::TextureHandle>,
    preview_cache_spectro: HashMap<String, egui::TextureHandle>,
    // Cached peaks tree for fast waveform rendering
    // cached_channels: Vec of (channel_data, peaks_tree) for each visible channel (1 for mono, 2 for stereo/multichannel)
    cached_channels: Vec<(Vec<f32>, audioscout_core::peaks_tree::Peaks)>,
    /// Texture of the live spectrogram image in the audio view. Re-rendered
    /// (at widget resolution) whenever the view range / document / panel size
    /// changes while `AudioView::mode` is Spectrogram — see
    /// `refresh_view_spectrogram`.
    view_spectro_texture: Option<egui::TextureHandle>,
    /// Key identifying the last rendered spectrogram image:
    /// (view start, view end, width, height, document generation). When it
    /// differs from the current state the picture is recomputed.
    view_spectro_key: Option<(u64, u64, u32, u32, u64)>,
    /// Bumped on every event that invalidates the spectrogram image (file
    /// load, edits/undo/redo, mode switch) so a re-render is triggered even
    /// when the numeric view key stays the same.
    document_generation: u64,
    /// 50 ms playhead poll bookkeeping (§1.2/§5.3): last time the UI-side
    /// fallback poll read the transport position while playing (the output
    /// thread is the primary source of PlayheadUpdate events; this throttle
    /// only drives the cursor when no real audio device is available).
    last_playhead_poll: std::time::Instant,
    /// BUGFIX (playback freezes ~0.5 s after start): set once per detected
    /// audio-stream stall so the status-bar message is shown a single time
    /// and re-armed when playback resumes (`poll_playhead` clears it on
    /// every non-stalled tick).
    stall_notified: bool,
    /// BUGFIX (cursor jumps ahead then snaps back while playing): the real
    /// audio clock (`consumed_samples`) lags behind wall time by the size of
    /// the sink/pump buffer (~150 ms) plus one 50 ms poll quantum. If the UI
    /// paints that raw lagging value, every poll tick moves the cursor in
    /// discrete steps and any transient over-forward position (e.g. a
    /// PlayheadUpdate event racing ahead of the hardware clock) must be
    /// "corrected" backwards on the next tick — the visible jump-ahead /
    /// snap-back oscillation, which grows as the buffered lead accumulates
    /// and is most noticeable on short files. Instead the UI extrapolates
    /// the clock forward between polls at exactly the nominal playback rate
    /// (never past the session end), so motion is smooth and monotonic; the
    /// small extrapolation error (< one poll quantum) is absorbed gradually
    /// because the estimate is always kept >= the true clock.
    playhead_est_frame: u64,
    /// Wall-clock instant that corresponds to `playhead_est_frame`.
    playhead_est_at: std::time::Instant,
    /// Session bounds [start,end) currently being played — used to clamp the
    /// extrapolated cursor so it can never run past the end of a short file.
    playhead_session: Option<(u64, u64)>,
    /// BUGFIX (cursor races ahead of the audio and stalls at the window edge
    /// while the file finishes playing): the free-running wall-clock estimate
    /// was previously allowed to lead the true hardware clock by an unbounded
    /// amount — every poll re-anchored the estimate at `max(estimate, clock)`
    /// instead of pulling it back toward the clock, so the visible lead only
    /// grew (it equals the whole pump/sink buffer depth plus event jitter once
    /// the pump has fed the last chunk). On a short file (e.g. 7 s) the cursor
    /// reached the right edge of the player window seconds before the audio
    /// actually finished and stood there while playback caught up. Now the
    /// estimate may lead the true clock by at most one poll quantum (~50 ms):
    /// each poll blends the raw clock back into the anchor whenever the lead
    /// exceeds that cap, which keeps motion smooth AND time-accurate.
    playhead_lead_cap_frames: u64,
    /// BUGFIX (cursor races ahead of the audio and stalls at the right edge
    /// while a short file finishes playing): hard ceiling for the visible
    /// cursor — the furthest FILE frame that has provably been handed to the
    /// Sink (`RodioOutput::playhead_ceiling_frame`, fed by the pump thread).
    /// Wall-clock extrapolation alone cannot know how much audio is actually
    /// queued ahead of the device; without this clamp the estimate drifts up
    /// to the full pump lead (~0.75 s+) past the audible position, reaches
    /// EOF and the window edge early, and stands there while the sound keeps
    /// playing — very visible on short files (e.g. 7 s). Cached here so the
    /// field can be invalidated on every session transition (Play / Seek /
    /// Stop) exactly like `playhead_est_frame`; the output thread republishes
    /// the honest value from the pump on its first append after a transition.
    playhead_fed_ceiling: Option<u64>,
    /// BUGFIX (wheel zoom in the player stopped working): raw vertical wheel
    /// delta (`MouseWheel` events, in points) snapshotted at the start of
    /// `update()` — before any ScrollArea can consume `smooth_scroll_delta`.
    /// The audio-view panel reads it every frame and decides ownership itself
    /// against its FRESH rect (see the handler in that panel); doing the
    /// ownership test up in `update()` against last frame's published rect
    /// was what silently dropped the first wheel notch after entering the
    /// view ("buttons work, wheel doesn't").
    raw_wheel_dy: f32,
    /// Whether the snapshotted wheel event above carried the Ctrl modifier
    /// (zoom gesture) — captured together with `raw_wheel_dy` at the top of
    /// `update()` so the panel never has to re-read stale live modifiers.
    raw_wheel_ctrl: bool,
    /// Rect of the audio-view content area, recorded by the panel itself on
    /// every frame (kept for diagnostics / future hit-tests).
    audio_view_rect: Option<egui::Rect>,
    /// BUGFIX (wheel zoom in the player STILL not working): a wheel notch is
    /// one discrete event; egui may deliver it as `MouseWheel` *and* fold it
    /// into `raw_scroll_delta`/`smooth_scroll_delta` in the same or later
    /// frames. Applying both paths would double-trigger, so instead of
    /// consuming the snapshot we stamp the time of the last applied gesture
    /// here and drop any duplicate that arrives within this short window.
    last_wheel_apply: f64,
    /// Fingerprint of the panel-height constants compiled into *this* binary
    /// (see `PANEL_LAYOUT_ID`). If it ever differs from the value egui's
    /// `Memory` was built with, all cached panel sizes are wiped once so the
    /// new `default_height`s take effect — guarantees rebuild -> visible change.
    panel_layout_id: u64,
    /// Manual column-drag state machine (header reorder by mouse). The frame
    /// on which a press happened inside a header cell (egui's own
    /// `is_pointer_button_down_on`, detected live during layout) together
    /// with the pressed column key; consumed one frame later by
    /// `handle_column_drag`. Plain fields because the table closure borrows
    /// `self` immutably — no per-widget maps needed.
    col_press_frame: u64,
    col_press_key: String,
    col_drag: Option<ColDrag>,
    /// BUGFIX (horizontal scroll while dragging a header): pressing and
    /// moving inside a column title used to pan the file list sideways —
    /// egui treats "held primary button + pointer moved" as ordinary drag
    /// scrolling for any ScrollArea whose content is wider than the viewport
    /// (`ScrollArea::both()` here; see `scroll_area.rs`: with
    /// `drag_to_scroll` enabled the area allocates a `Sense::drag()`
    /// background interaction and shifts its offset by `pointer.delta` every
    /// frame), and the header row lives INSIDE that area. A header capture
    /// now owns the gesture from the press until the release:
    ///  * the files-list ScrollArea switches `.drag_to_scroll(false)` while
    ///    the capture is active (kills mouse-drag panning of the table);
    ///  * every horizontal wheel / trackpad-hscroll delta of this frame is
    ///    swallowed in `update()` before the panels lay out (kills two-finger
    ///    horizontal panning under the dragged title).
    /// The flag covers the whole window including the single frame between
    /// the raw press and the moment `handle_column_drag` arms the live drag
    /// (see `col_capture_press_frame`).
    col_capture_blocks_hscroll: bool,
    /// Frame on which a header cell FIRST reported the primary-button press
    /// (`begin_col_press`). The live drag is only armed one frame later by
    /// `handle_column_drag`, but the blocking window must start right at the
    /// press: between those two frames egui's files-list ScrollArea would
    /// otherwise already pan sideways under the held button. While
    /// `col_frame_counter - col_capture_press_frame <= 1` and the button is
    /// still down, `update()` keeps treating the gesture as captured (see
    /// `col_header_grab_active`).
    col_capture_press_frame: u64,
    /// Frame on which the header grab ended (release / cancellation), recorded
    /// by `end_col_capture_block`. For a few frames afterwards `update()`
    /// zeroes the leftover horizontal scroll deltas and restores the files
    /// list's horizontal offset to `col_hscroll_offset_at_press`: a coarse
    /// wheel notch that arrived BEFORE the capture got armed lives in egui's
    /// private pending-smoothing buffer and would otherwise be spent on the
    /// table right after the release.
    col_capture_release_frame: u64,
    /// Horizontal offset of the files-list ScrollArea at the moment the
    /// header grab started — the value the post-release cleanup restores.
    /// `None` when the area had not been shown yet (nothing to restore).
    col_hscroll_offset_at_press: Option<f32>,
    /// Own frame counter (egui 0.29 `Context` has no `frame_index()`);
    /// bumped once per processed frame at the START of `update()`, so both
    /// the layout pass and the end-of-frame drag resolver see the same id.
    col_frame_counter: u64,
    /// Frame index of the release that ended a column drag; clicks reported
    /// by header widgets on that frame are ignored (reorder ≠ re-sort).
    col_suppress_click_frame: u64,
}

impl AudioScoutApp {
    /// Expand a directory and all its parent directories in the tree
    fn expand_directory_tree(expanded_dirs: &mut HashMap<PathBuf, Vec<PathBuf>>, dir: &Path) {
        // Collect all ancestors from dir up to root
        let mut current = dir.to_path_buf();
        let mut ancestors = Vec::new();

        while current.parent().is_some() {
            ancestors.push(current.clone());
            if let Some(parent) = current.parent() {
                current = parent.to_path_buf();
            } else {
                break;
            }
        }

        // Reverse to process from root down to dir
        ancestors.reverse();

        // Expand each ancestor
        for ancestor in ancestors {
            if !expanded_dirs.contains_key(&ancestor) {
                if let Ok(entries) = std::fs::read_dir(&ancestor) {
                    let mut subdirs: Vec<PathBuf> = entries
                        .filter_map(|e| e.ok())
                        .filter(|e| e.path().is_dir())
                        .map(|e| e.path())
                        .collect();
                    subdirs.sort_by(|a, b| {
                        a.file_name()
                            .and_then(|n| n.to_str())
                            .map(|s| s.to_lowercase())
                            .cmp(&b.file_name().and_then(|n| n.to_str()).map(|s| s.to_lowercase()))
                    });
                    expanded_dirs.insert(ancestor, subdirs);
                }
            }
        }
    }

    pub fn new(
        _cc: &eframe::CreationContext<'_>,
        config_dir: PathBuf,
        start_dir: Option<PathBuf>,
    ) -> Self {
        let config_store = ConfigStore::new(&config_dir);
        let db_path = config_dir.join("app.db");
        let db = Database::open(&db_path).expect("Failed to open database");
        let channel = EventChannel::new();
        let event_tx = channel.sender.clone();
        let event_rx = channel.receiver;
        let player = Arc::new(Mutex::new(Controller::new(Arc::new(PlayheadState::new()))));
        // Restore the persisted transport settings (tz §1.4: config.json is
        // the source of truth on launch). Without this the slider writes the
        // volume to disk but every start begins at the built-in 0.8 default.
        {
            let mut ctrl = player.lock().unwrap_or_else(|p| p.into_inner());
            ctrl.set_volume(config_store.config().volume);
            ctrl.set_skip_seconds(config_store.config().skip_seconds);
        }
        // Single application-wide OutputStream, created once at startup and
        // kept alive for the whole session (tz §5.2). Without an audio device
        // RodioOutput::new returns None and playback is silently unavailable.
        let output = RodioOutput::new(player.clone(), event_tx.clone());
        // DEBUG (playback diagnostics): a None here means the audio device
        // never opened — Play can then only run the state machine, no sound.
        match &output {
            Some(_) => tracing::info!("[app] audio output initialized OK"),
            None => tracing::error!("[app] NO AUDIO OUTPUT (RodioOutput::new returned None) — playback will be silent; check '[output]' error lines above and libasound2/cpal device availability"),
        }
        let undo_dir = config_dir.join("undo");
        let undo_stack = UndoStack::new(undo_dir, config_store.config().undo_depth, config_store.config().undo_disk_limit_mb);
        let preview_worker = PreviewWorker::new(event_tx.clone());
        let scanner = Scanner::new();
        // Grab the shared preview-mode handle *before* `scanner` is moved
        // into the struct below (Rust evaluates struct fields in order, so
        // borrowing `scanner` after it has been moved would fail to compile).
        let preview_mode = scanner.preview_mode_handle();

        let mut app = Self {
            config_store, db, event_rx, event_tx,
            scanner, preview_worker,
            current_dir: None, file_rows: Vec::new(), scan_generation: 0,
            column_layout: ColumnLayout::new(), selected_row: None, filter_text: String::new(),
            document: Document::new(), active_file: None, player, output,
            meter_live: MeterData::default(), meter_hold: MeterData::default(),
            meter_peak_at: 0.0, last_frame_time: 0.0,
            audio_view: AudioView::new(0), undo_stack,
            preview_mode,
            preview_cache_wave: HashMap::new(),
            preview_cache_spectro: HashMap::new(),
            preview_inflight: HashSet::new(), preview_retry: HashSet::new(),
            show_settings: false, settings_hotkey_capture: None, settings_status: String::new(),
            show_columns_dialog: false, status_message: String::new(),
            normalize_dbfs: -1.0, expanded_dirs: HashMap::new(),
            disks: Vec::new(),
            cached_channels: Vec::new(),
            view_spectro_texture: None,
            view_spectro_key: None,
            document_generation: 0,
            last_playhead_poll: std::time::Instant::now(),
            stall_notified: false,
            playhead_est_frame: 0,
            playhead_est_at: std::time::Instant::now(),
            playhead_session: None,
            // One 50 ms poll quantum at 192 kHz — the absolute worst case;
            // re-derived from the document's sample rate on every poll.
            playhead_lead_cap_frames: 9600,
            // No session engaged yet — the pump publishes the real ceiling
            // after its first append (see `poll_playhead`).
            playhead_fed_ceiling: None,
            raw_wheel_dy: 0.0,
            raw_wheel_ctrl: false,
            audio_view_rect: None,
            last_wheel_apply: 0.0,
            panel_layout_id: PANEL_LAYOUT_ID,
            col_press_frame: 0,
            col_press_key: String::new(),
            col_drag: None,
            col_capture_blocks_hscroll: false,
            col_capture_press_frame: 0,
            col_capture_release_frame: 0,
            col_hscroll_offset_at_press: None,
            col_frame_counter: 0,
            col_suppress_click_frame: 0,
        };

        // Restore the two independent Waveform/Spectrogram toggles from the
        // config (§3.1 table previews, §3.5 player visualization).
        let saved_preview_mode = PreviewMode::from_str_name(&app.config_store.config().preview_mode);
        *app.preview_mode.lock().unwrap() = saved_preview_mode;
        app.scanner.set_preview_mode(saved_preview_mode);
        app.audio_view.mode = match app.config_store.config().audio_view_mode.as_str() {
            "spectro" => AudioViewMode::Spectrogram,
            _ => AudioViewMode::Waveform,
        };

        // BUGFIX (persisted column selection): the Columns-dialog checkboxes
        // used to mutate only the in-memory `column_layout`, so the visible
        // set reset to "all columns" on every relaunch. Rebuild the layout
        // from config.json at startup (keys validated/deduped in core; an
        // absent/empty list keeps the all-columns default).
        app.column_layout = ColumnLayout::from_config(&app.config_store.config().columns.clone());

        if let Some(dir_str) = app.db.load_session("lastDir") {
            let dir = PathBuf::from(dir_str);
            if dir.exists() {
                app.current_dir = Some(dir.clone());
                app.scanner.scan(dir.clone(), app.event_tx.clone());

                // Auto-expand the last active directory and all its parents
                Self::expand_directory_tree(&mut app.expanded_dirs, &dir);
            }
        } else if let Some(dir) = start_dir {
            app.current_dir = Some(dir.clone());
            app.scanner.scan(dir.clone(), app.event_tx.clone());
            Self::expand_directory_tree(&mut app.expanded_dirs, &dir);
        } else {
            // Fallback: expand home directory if no lastDir
            if let Some(home) = dirs_next::home_dir() {
                Self::expand_directory_tree(&mut app.expanded_dirs, &home);
            }
        }

        // Start background thread for disk auto-refresh
        let disk_event_tx = app.event_tx.clone();
        std::thread::spawn(move || {
            let mut provider = audioscout_core::disk::SysinfoDiskProvider::new();
            loop {
                provider.refresh();
                let disks = provider.list_disks();
                if disk_event_tx.send(UiEvent::DisksChanged { disks }).is_err() {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_secs(5));
            }
        });

        // Clean up orphaned preview files on startup
        if let Ok(removed) = audioscout_core::preview_worker::cleanup_previews() {
            if removed > 0 {
                tracing::info!("Cleaned up {} orphaned preview files", removed);
            }
        }

        app
    }

    /// Top-panel Waveform/Spectrogram toggle for the *table thumbnails*
    /// (§3.1). Flips the shared mode, re-scans the current directory so rows
    /// pick up cached PNGs of the other style immediately, and repumps the
    /// texture cache for whatever is missing.
    fn toggle_preview_mode(&mut self) {
        let new_mode = self.preview_mode.lock().unwrap().toggled();
        *self.preview_mode.lock().unwrap() = new_mode;
        self.scanner.set_preview_mode(new_mode);
        self.config_store.config_mut().preview_mode = new_mode.as_str().to_string();
        let _ = self.config_store.save();
        if let Some(dir) = self.current_dir.clone() {
            // Fresh scan: rows arrive with per-mode cached UUIDs; missing
            // ones fall through to `request_previews` via ScanDone.
            self.scanner.scan(dir, self.event_tx.clone());
        } else {
            self.request_previews();
        }
    }

    /// Player-panel Waveform/Spectrogram toggle for the *visualization area*
    /// (§3.5). Independent from the table thumbnail mode.
    fn toggle_audio_view_mode(&mut self) {
        self.audio_view.toggle_mode();
        let name = match self.audio_view.mode {
            AudioViewMode::Spectrogram => "spectro",
            AudioViewMode::Waveform => "wave",
        };
        self.config_store.config_mut().audio_view_mode = name.to_string();
        let _ = self.config_store.save();
    }

    fn request_previews(&mut self) {
        let generation = self.scan_generation;
        let mode = *self.preview_mode.lock().unwrap();

        // Clear queue to cancel any pending previews from previous directory
        self.preview_worker.clear_queue();

        for row in &self.file_rows {
            // Skip if preview already exists (scanner resolved it from cache,
            // or a previous batch delivered it)
            if row.preview_uuid.is_some() {
                continue;
            }

            // Submit preview job. If the worker drops the job (duplicate
            // already rendering, or saturated queue), remember the path so we
            // retry on the next scan instead of leaving ⏳ forever.
            if !self.preview_worker.submit(PreviewJob {
                path: row.path.clone(),
                generation,
                mode,
            }) {
                self.preview_retry.insert(row.path.clone());
            } else {
                self.preview_inflight.insert(row.path.clone());
            }
        }
    }

    /// Toolbar "Refresh preview": purge the preview cache of the *current*
    /// directory (DB rows + PNG files on disk — see
    /// `preview_worker::delete_previews_for_dir`), drop the stale UUIDs and
    /// texture caches, then re-run generation for the visible rows.
    fn refresh_current_dir_previews(&mut self, ctx: &egui::Context) {
        let Some(dir) = self.current_dir.clone() else {
            self.status_message = "No current directory".to_string();
            return;
        };
        match audioscout_core::preview_worker::delete_previews_for_dir(&dir) {
            Ok(n) => self.status_message = format!("Previews refreshed ({n} cleared)…"),
            Err(e) => self.status_message = format!("Preview refresh failed: {e}"),
        }
        // The PNGs behind these textures no longer exist; drop both mode
        // caches so the rows re-load whatever gets regenerated.
        self.preview_cache_wave.clear();
        self.preview_cache_spectro.clear();
        // Stale UUIDs would make `request_previews` skip every row (it only
        // submits jobs for rows with `preview_uuid == None`).
        for row in &mut self.file_rows {
            row.preview_uuid = None;
        }
        // Cancel queued jobs that would render into the deleted cache. Do NOT
        // wipe the in-flight bookkeeping here: paths whose render thread is
        // still running stay in the worker's internal `pending` set until it
        // finishes, and clearing our mirror set would let `request_previews`
        // re-submit them as "fresh" jobs while the old one is alive — the two
        // concurrent renders of the same file then race on the deterministic
        // PNG path (partial/overwritten image). Keeping the entries means the
        // duplicate submission is rejected, and the render that is already
        // running delivers a brand-new image (the old PNG was just deleted),
        // which satisfies the refresh requirement anyway.
        self.preview_worker.clear_queue();
        // Kick off regeneration for the current directory. Rows currently in
        // flight are skipped by `submit`'s dedup guard (see above); everything
        // else gets a fresh job.
        self.request_previews();
        ctx.request_repaint();
    }

    /// Background preview generation is event-driven: without an active
    /// repaint loop, `PreviewReady` events could sit unread in the channel
    /// long after the PNGs appeared on disk. While any thumbnail is still
    /// missing, keep the UI pumping at a modest rate so results show up as
    /// soon as they arrive.
    fn previews_pending(&self) -> bool {
        self.file_rows.iter().any(|r| r.preview_uuid.is_none())
    }

    /// Re-submit rows that are neither cached nor currently in flight
    /// (e.g. jobs previously rejected because the queue was full).
    fn retry_missing_previews(&mut self) {
        let generation = self.scan_generation;
        let mode = *self.preview_mode.lock().unwrap();
        let missing: Vec<PathBuf> = self
            .file_rows
            .iter()
            .filter(|r| r.preview_uuid.is_none())
            .filter(|r| !self.preview_inflight.contains(&r.path))
            .map(|r| r.path.clone())
            .collect();

        for path in missing {
            if self
                .preview_worker
                .submit(PreviewJob { path: path.clone(), generation, mode })
            {
                self.preview_inflight.insert(path);
            }
        }
    }

    fn process_events(&mut self, ctx: &egui::Context) {
        while let Ok(event) = self.event_rx.try_recv() {
            match event {
                UiEvent::ScanDone { dir, rows, generation } => {
                    if Some(&dir) == self.current_dir.as_ref() && generation >= self.scan_generation {
                        self.scan_generation = generation;
                        self.file_rows = rows;
                        self.column_layout.sort_rows(&mut self.file_rows);
                        // Previous-generation bookkeeping is stale now.
                        self.preview_inflight.clear();
                        self.preview_retry.clear();
                        if let Some(file_str) = self.db.load_session("lastFile") {
                            let file_path = PathBuf::from(file_str);
                            if let Some(idx) = self.file_rows.iter().position(|r| r.path == file_path) {
                                self.selected_row = Some(idx);
                            }
                        }

                        // Request previews for all files
                        self.request_previews();
                        // New rows may need thumbnails → start the pump.
                        ctx.request_repaint();
                    }
                }
                UiEvent::PreviewReady { path, uuid, generation } => {
                    // Only update if this is the current generation
                    if generation == self.scan_generation {
                        self.preview_inflight.remove(&path);
                        self.preview_retry.remove(&path);
                        if let Some(row) = self.file_rows.iter_mut().find(|r| r.path == path) {
                            row.preview_uuid = Some(uuid);
                        }
                        // Repaint so the freshly delivered thumbnail becomes
                        // visible immediately instead of on the next input.
                        ctx.request_repaint();
                    }
                }
                UiEvent::PlayheadUpdate { position_secs } => {
                    let frame = (position_secs * self.document.sample_rate as f64) as u64;
                    // The output thread owns the audio clock while playing —
                    // mirror it into the UI state machine as a playback-side
                    // move (never a user seek, so the resume point stays put;
                    // tz §5.3 CursorSource::Playback).
                    if self.output.is_some() && self.player_state() == TransportState::Playing {
                        let mut ctrl = self.player.lock().unwrap_or_else(|p| p.into_inner());
                        ctrl.freeze_resume_at(frame, self.document.sample_rate);
                        drop(ctrl);
                        // BUGFIX (cursor jump-ahead / snap-back): this event
                        // carries the raw hardware clock, which LAGS wall time
                        // by the sink buffer depth. Feeding it straight into
                        // `set_cursor` made the painted cursor jump ahead
                        // (poll tick extrapolation) and then snap back here
                        // (stale true clock) every cycle. During live playback
                        // the poll in `poll_playhead` is the single source of
                        // truth for the visible cursor; this path only keeps
                        // the controller's resume point fresh.
                        if self.playhead_session.is_none() {
                            self.audio_view.set_cursor(frame);
                            self.audio_view.auto_scroll_to_cursor();
                        }
                    } else {
                        self.audio_view.set_cursor(frame);
                        self.audio_view.auto_scroll_to_cursor();
                    }
                }
                UiEvent::PlaybackFinished => { self.status_message = "Playback finished".to_string(); }
                UiEvent::VolumeMeterUpdate { peak_l, peak_r, rms_l, rms_r } => {
                    // Volume meter (§3.5.1): instantaneous scale with optional
                    // peak-hold; RMS scale is always live.
                    //
                    // Peak-hold algorithm (Settings→"Volumeter peak hold
                    // delay", default 200 ms), evaluated per meter frame on a
                    // monotonic wall clock (`ctx.input` time — NOT a frame
                    // counter):
                    //   L > H (strictly)  -> H = L, restart timer (t_last = now)
                    //   L <= H            -> keep H, do not touch the timer
                    //   now - t_last >= delay -> H = L (instant reset), t_last = now
                    // The painted marker always shows H; the initial H equals
                    // the level of the first frame after hold is enabled.
                    let cfg = self.config_store.config();
                    let now = ctx.input(|i| i.time);
                    self.meter_live = MeterData { peak_l, peak_r, rms_l, rms_r };
                    if cfg.meter_hold_enabled {
                        let delay = cfg.meter_hold_ms as f64 / 1000.0;
                        let new_peak = peak_l > self.meter_hold.peak_l
                            || peak_r > self.meter_hold.peak_r;
                        if new_peak {
                            // Strict greater-than: a fresh maximum captures H
                            // and restarts the release timer from zero.
                            self.meter_hold.peak_l = peak_l;
                            self.meter_hold.peak_r = peak_r;
                            self.meter_peak_at = now;
                        } else if now - self.meter_peak_at >= delay {
                            // Delay expired without a new peak: drop H to the
                            // current level instantly and re-arm the timer.
                            self.meter_hold.peak_l = peak_l;
                            self.meter_hold.peak_r = peak_r;
                            self.meter_peak_at = now;
                        }
                        // Otherwise (L <= H, timer still running): H stays.
                    } else {
                        self.meter_hold = self.meter_live;
                        self.meter_peak_at = now;
                    }
                }
                UiEvent::DisksChanged { disks } => {
                    self.disks = disks;
                }
            }
        }
    }

    fn load_file(&mut self, path: &Path) {
        // All supported audio formats load into the player (§3.4: playback must
        // work for non-editable files too). "Editable" (PCM read/write) only
        // gates destructive edit operations (mute/normalize/save), never
        // loading or playback — see `apply_mute`/`apply_normalize`.
        if !is_audio_file(path) {
            self.status_message = "Unsupported file format".to_string();
            return;
        }
        // BUGFIX (cursor parked somewhere else after a Stop of the previous
        // file): a fresh load ALWAYS parks the playhead at the start of the
        // file before anything can play (§3.5). transport_stop() below now
        // parks at the OLD file's resume point, so reset the transport state
        // machine AND the visible cursor explicitly to frame 0 right away.
        {
            let mut ctrl = self.player.lock().unwrap_or_else(|p| p.into_inner());
            ctrl.reset();
        }
        self.transport_stop();
        match audioscout_core::audio_io::load_audio(path) {
            Ok((samples, sr, ch, st)) => {
                // DEBUG (playback diagnostics): frames==0 here means the file
                // decoded to an empty buffer — Play would compute end==start
                // and drop the append.
                tracing::info!("[load] OK {path:?} samples={} sr={sr} ch={ch} frames={}", samples.len(), samples.len() / (ch.max(1) as usize));
                // BUGFIX (sticky "not editable" message): a successful load
                // must clear any stale status from a previously rejected file.
                self.status_message.clear();
                self.document = Document {
                    samples, sample_rate: sr, channels: ch, markers: Vec::new(),
                    source_path: path.to_path_buf(), temp_path: None, dirty: false, sample_type: st,
                };
                if let Ok(meta) = std::fs::metadata(path) {
                    let size = meta.len();
                    let mtime = meta.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_secs()).unwrap_or(0);
                    if let Ok(markers) = self.db.load_markers(path, size, mtime) { self.document.markers = markers; }
                }
                // Preserve the current player display mode (Waveform/Spectrogram)
                // across file loads. The default comes from config.json at startup;
                // loading a new file must not silently reset it to waveform.
                let keep_mode = self.audio_view.mode;
                self.audio_view = AudioView::with_mode(self.document.total_frames(), keep_mode);
                self.active_file = Some(path.to_path_buf());
                // Hand the fresh PCM buffer to the output thread (§5.1: stop
                // of previous output + resume-point reset happen there).
                if let Some(out) = &self.output {
                    out.load_document(Arc::new(self.document.clone()));
                }
                self.undo_stack.clear();
                let _ = self.db.save_session("lastFile", &path.to_string_lossy());
                let _ = self.db.save_session("lastPos", "0");

                // Build peaks tree cache for fast waveform rendering
                self.rebuild_peaks_cache();
            }
            Err(e) => { tracing::error!("[load] FAILED {path:?}: {e}"); self.status_message = format!("Load failed: {}", e); }
        }
    }

    /// BUGFIX (audio stayed stale after an edit): the player output thread
    /// owns its own immutable copy of the PCM buffer (`Arc<Document>` handed
    /// over via `Cmd::Load`). Mute/normalize/undo/redo mutate
    /// `self.document.samples` and repaint the waveform view, but without a
    /// fresh document publish the audible stream kept playing the pre-edit
    /// buffer.
    ///
    /// Two flavours, matching what the user hears:
    /// * **Playing** — swap the buffer IN PLACE via `Output::update_document`
    ///   (`Cmd::UpdateDoc`): the controller drops every Sink chunk sliced
    ///   from the old buffer and the pump refills from the current cursor,
    ///   so playback continues seamlessly with the EDITED samples. Stopping
    ///   here instead would kill the sound entirely — the exact symptom this
    ///   fix targets ("picture changes, sound doesn't").
    /// * **Stopped/paused** — full `load_document` semantics: republish and
    ///   park the transport at frame 0, the same contract a plain file load
    ///   has (see `load_file`), so the next Play streams edited samples.
    fn reload_player_audio(&mut self) {
        if self.active_file.is_none() || self.document.source_path.as_os_str().is_empty() {
            return;
        }
        if self.player_state() == TransportState::Playing {
            if let Some(out) = &self.output {
                if out.update_document(Arc::new(self.document.clone())) {
                    return;
                }
                // Channel gone — fall through to the hard-reload path below.
            } else {
                return;
            }
        }
        // Stop first: `Cmd::Load` on the output thread also stops the sink,
        // but doing it here keeps the UI-side state machine consistent even
        // when no audio device is available.
        self.transport_stop();
        if let Some(out) = &self.output {
            out.load_document(Arc::new(self.document.clone()));
        }
        // A fresh document means a fresh playhead — mirror what `load_file`
        // does (§3.5: park at the start of the newly loaded buffer).
        {
            let mut ctrl = self.player.lock().unwrap_or_else(|p| p.into_inner());
            ctrl.reset();
        }
        self.audio_view.set_cursor(0);
        self.playhead_est_frame = 0;
    }

    /// BUGFIX (file-list preview not refreshed after Save): the thumbnail is
    /// cached by (path, mtime, size, channels) — both in the SQLite previews
    /// table and as a deterministic PNG name. After saving, the file's mtime
    /// changed, so the old uuid is stale. Drop the row's uuid and re-submit a
    /// preview job; the worker sees the new mtime, misses its caches and
    /// renders a fresh image, delivered back as `UiEvent::PreviewReady`.
    fn refresh_row_preview(&mut self, path: &Path) {
        let mode = *self.preview_mode.lock().unwrap();
        let generation = self.scan_generation;
        if let Some(row) = self.file_rows.iter_mut().find(|r| r.path == path) {
            row.preview_uuid = None;
        }
        // Stale bookkeeping for this path would make `submit` reject the job
        // as a duplicate; clear it so the re-render always goes through.
        self.preview_retry.remove(path);
        if self.preview_worker.submit(PreviewJob { path: path.to_path_buf(), generation, mode }) {
            self.preview_inflight.insert(path.to_path_buf());
        }
    }

    /// Rebuild peaks tree cache after loading a new file or after edits
    fn rebuild_peaks_cache(&mut self) {
        let ch = self.document.channels as usize;
        let total_frames = self.document.total_frames() as usize;

        // Determine number of visible channels: 1 for mono, 2 for stereo/multichannel
        let visible_channels = if ch == 1 { 1 } else { 2 };

        self.cached_channels.clear();

        for channel_idx in 0..visible_channels {
            // Extract channel data from interleaved samples
            let channel_data: Vec<f32> = (0..total_frames)
                .map(|frame| {
                    let sample_idx = frame * ch + channel_idx;
                    if sample_idx < self.document.samples.len() {
                        self.document.samples[sample_idx]
                    } else {
                        0.0
                    }
                })
                .collect();

            // Build peaks tree for this channel
            let peaks = audioscout_core::peaks_tree::Peaks::build(&channel_data);
            self.cached_channels.push((channel_data, peaks));
        }

        // Any structural change to the document invalidates the live
        // spectrogram image; force a re-render on the next audio-view frame.
        self.document_generation += 1;
    }

    /// Rebuild the interleaved PCM slice covering the current view range and
    /// render it as a full-resolution spectrogram texture (audio-view mode
    /// switch, §3.6). Cheap no-op when nothing relevant changed since the
    /// last render — the hot `ui()` path stays idempotent.
    fn refresh_view_spectrogram(&mut self, ui: &egui::Ui, rect: egui::Rect) {
        let width = (rect.width().max(1.0)) as u32;
        let height = (rect.height().max(1.0)) as u32;
        let start = self.audio_view.view_range.start_frame;
        let end = self.audio_view.view_range.end_frame.min(self.document.total_frames());
        let key = (start, end, width, height, self.document_generation);
        if self.view_spectro_key == Some(key) && self.view_spectro_texture.is_some() {
            return;
        }

        let ch = self.document.channels.max(1) as usize;
        let a = (start as usize * ch).min(self.document.samples.len());
        let b = (end as usize * ch).min(self.document.samples.len());
        let rgba = audioscout_core::render::spectrogram::render_view_spectrogram(
            &self.document.samples[a..b],
            self.document.channels,
            width,
            height,
        );
        let color_image = egui::ColorImage::from_rgba_unmultiplied(
            [rgba.width() as usize, rgba.height() as usize],
            rgba.as_flat_samples().as_slice(),
        );
        self.view_spectro_texture = Some(ui.ctx().load_texture(
            "view_spectrogram",
            color_image,
            egui::TextureOptions::LINEAR,
        ));
        self.view_spectro_key = Some(key);
    }

    fn save_file(&mut self) {
        if !self.document.dirty { return; }
        let path = self.document.source_path.clone();
        if path.as_os_str().is_empty() { return; }
        match audioscout_core::audio_io::wav::WavFile::save(&path, &self.document.samples, self.document.sample_rate, self.document.channels, self.document.sample_type) {
            Ok(()) => {
                self.document.dirty = false;
                self.status_message = "File saved".to_string();
                if let Ok(meta) = std::fs::metadata(&path) {
                    let size = meta.len();
                    let mtime = meta.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_secs()).unwrap_or(0);
                    let _ = self.db.save_markers(&path, size, mtime, &self.document.markers);
                }
                // BUGFIX: after the final save, refresh the file-list preview
                // thumbnail so it reflects the just-written audio content.
                self.refresh_row_preview(&path);
            }
            Err(e) => { self.status_message = format!("Save failed: {}", e); }
        }
    }

    fn mute_selection(&mut self) {
        if self.active_file.is_none() { self.status_message = "No active file".to_string(); return; }
        if !is_editable_file(&self.document.source_path) { self.status_message = "Not editable".to_string(); return; }
        let (start, end) = if let Some(sel) = self.audio_view.selection_range() { sel } else { (0, self.document.total_frames()) };
        let ch = self.document.channels as usize;
        let start_idx = start as usize * ch;
        let end_idx = (end as usize * ch).min(self.document.samples.len());
        let dump_path = self.undo_stack.new_dump_path();
        let original: Vec<f32> = self.document.samples[start_idx..end_idx].to_vec();
        if self.undo_stack.save_dump(&dump_path, &original).is_err() { self.status_message = "Undo dump failed".to_string(); return; }
        for s in &mut self.document.samples[start_idx..end_idx] { *s = 0.0; }
        self.undo_stack.push(audioscout_core::undo::UndoEntry {
            op: audioscout_core::undo::UndoOp::Mute { dump_path, start_frame: start, end_frame: end },
            file_path: self.document.source_path.clone(),
        });
        self.document.dirty = true;
        self.status_message = format!("Muted frames {}..{}", start, end);

        // Rebuild peaks cache after edit
        self.rebuild_peaks_cache();
        // BUGFIX: the audible stream must follow the edited buffer, not just
        // the waveform view — republish the document to the output thread.
        self.reload_player_audio();
    }

    fn normalize_selection(&mut self) {
        if self.active_file.is_none() { self.status_message = "No active file".to_string(); return; }
        let (start, end) = if let Some(sel) = self.audio_view.selection_range() { sel } else { (0, self.document.total_frames()) };
        let ch = self.document.channels as usize;
        let start_idx = start as usize * ch;
        let end_idx = (end as usize * ch).min(self.document.samples.len());
        let peak: f32 = self.document.samples[start_idx..end_idx].iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        if peak == 0.0 { self.status_message = "Peak is zero".to_string(); return; }
        let target = 10.0f64.powf(self.normalize_dbfs as f64 / 20.0) as f32;
        let gain = target / peak;
        let dump_path = self.undo_stack.new_dump_path();
        let original: Vec<f32> = self.document.samples[start_idx..end_idx].to_vec();
        if self.undo_stack.save_dump(&dump_path, &original).is_err() { return; }
        for s in &mut self.document.samples[start_idx..end_idx] { *s *= gain; }
        self.undo_stack.push(audioscout_core::undo::UndoEntry {
            op: audioscout_core::undo::UndoOp::Normalize { dump_path, start_frame: start, end_frame: end, dbfs: self.normalize_dbfs as f64 },
            file_path: self.document.source_path.clone(),
        });
        self.document.dirty = true;
        self.status_message = format!("Normalized to {} dBFS", self.normalize_dbfs);

        // Rebuild peaks cache after edit
        self.rebuild_peaks_cache();
        // BUGFIX: keep the played-back audio in sync with the edit (§ above).
        self.reload_player_audio();
    }

    fn undo(&mut self) {
        if let Some(entry) = self.undo_stack.pop_undo() {
            match &entry.op {
                audioscout_core::undo::UndoOp::Mute { dump_path, start_frame, end_frame }
                | audioscout_core::undo::UndoOp::Normalize { dump_path, start_frame, end_frame, .. } => {
                    if let Ok(original) = self.undo_stack.load_dump(dump_path) {
                        let ch = self.document.channels as usize;
                        let start_idx = *start_frame as usize * ch;
                        let end_idx = (*end_frame as usize * ch).min(self.document.samples.len());
                        let copy_len = (end_idx - start_idx).min(original.len());
                        self.document.samples[start_idx..start_idx + copy_len].copy_from_slice(&original[..copy_len]);
                    }
                }
                _ => {}
            }
            self.document.dirty = true;
            self.status_message = "Undo".to_string();

            // Rebuild peaks cache after undo
            self.rebuild_peaks_cache();
            // Undo changes the audible content too — refresh the player buffer.
            self.reload_player_audio();
        }
    }

    fn redo(&mut self) {
        if let Some(entry) = self.undo_stack.pop_redo() {
            match &entry.op {
                audioscout_core::undo::UndoOp::Mute { start_frame, end_frame, .. } => {
                    let ch = self.document.channels as usize;
                    let start_idx = *start_frame as usize * ch;
                    let end_idx = (*end_frame as usize * ch).min(self.document.samples.len());
                    for s in &mut self.document.samples[start_idx..end_idx] { *s = 0.0; }
                }
                audioscout_core::undo::UndoOp::Normalize { start_frame, end_frame, dbfs, .. } => {
                    let ch = self.document.channels as usize;
                    let start_idx = *start_frame as usize * ch;
                    let end_idx = (*end_frame as usize * ch).min(self.document.samples.len());
                    let peak: f32 = self.document.samples[start_idx..end_idx].iter().map(|s| s.abs()).fold(0.0f32, f32::max);
                    if peak > 0.0 {
                        let target = 10.0f64.powf(*dbfs / 20.0) as f32;
                        let gain = target / peak;
                        for s in &mut self.document.samples[start_idx..end_idx] { *s *= gain; }
                    }
                }
                _ => {}
            }
            self.document.dirty = true;
            self.status_message = "Redo".to_string();

            // Rebuild peaks cache after redo
            self.rebuild_peaks_cache();
            // Redo changes the audible content too — refresh the player buffer.
            self.reload_player_audio();
        }
    }

    fn add_marker(&mut self) {
        let frame = self.audio_view.cursor_frame;
        let time_sec = frame as f64 / self.document.sample_rate as f64;
        let name = format!("M{}", self.document.markers.len() + 1);
        self.document.markers.push(Marker { name, time_samples: frame, time_sec });
        self.document.markers.sort_by_key(|m| m.time_samples);
        self.status_message = format!("Marker added at {:.3}s", time_sec);
    }

    fn delete_active_marker(&mut self) {
        if let Some(idx) = self.audio_view.active_marker {
            if idx < self.document.markers.len() { self.document.markers.remove(idx); self.audio_view.active_marker = None; self.status_message = "Marker deleted".to_string(); }
        }
    }

    /// Transport play/pause routed through the real audio output when one is
    /// available; without a device it degrades to state-machine-only mode so
    /// the UI never hangs on missing audio (§1.2 "раньше времени не падают").
    fn transport_play_pause(&mut self) {
        // BUGFIX: honor the "Space key behavior" setting (config.json,
        // `space_behavior_stop`, §3.6/Settings dialog). Previously Space was
        // hard-wired to Start/Pause and the config value had no effect on the
        // player. In Start/Stop mode a second press behaves exactly like the
        // Stop button: playback stops AND the cursor parks at the resume
        // point (file start or the place previously set with the mouse).
        if self.config_store.config().space_behavior_stop
            && self.player_state() == TransportState::Playing
        {
            tracing::debug!("[play] space pressed in Start/Stop mode -> stop");
            self.transport_stop();
            return;
        }
        let sel = self.audio_view.selection_range();
        let state = self.player_state();
        if state == TransportState::Playing {
            // Pause: freeze position + in-flight PCM (§5.3). Without a real
            // output device the state machine alone carries the semantics.
            tracing::debug!("[play] pause pressed (state={state:?})");
            if let Some(out) = &self.output { out.pause(); }
            // BUGFIX (cursor jump-ahead / snap-back): stop extrapolating while
            // paused; freeze the estimate at the true hardware clock so a
            // later resume re-arms it from a consistent point.
            self.playhead_session = None;
            let mut ctrl = self.player.lock().unwrap_or_else(|p| p.into_inner());
            ctrl.pause();
        } else {
            // Play: selection wins over the resume point (§3.5/§5.3).
            let start = sel.map(|(a, _)| a).unwrap_or_else(|| {
                self.player.lock().unwrap_or_else(|p| p.into_inner()).resume_frame()
            });
            // BUGFIX (play never starts): with no selection the session must
            // run to EOF. Passing 0 here made Cmd::Play compute `e > s` as
            // false for every start==0 file and silently drop the append —
            // the button toggled but nothing played.
            let end = sel.map(|(_, b)| b).unwrap_or(self.document.total_frames());
            // DEBUG (playback diagnostics): the full decision context at the
            // exact moment the user presses Play/Space. If playback does not
            // start, one of these values names the culprit:
            //   output=None          → audio device failed to open at startup
            //                          (see "[output] ... FAILED" lines),
            //   total_frames=0       → the document is empty/not loaded,
            //   e <= s               → degenerate range, dropped by apply_play.
            tracing::info!(
                "[play] pressed state={state:?} sel={sel:?} start={start} end={end} \
                 total_frames={} sr={} ch={} output={} path={:?}",
                self.document.total_frames(),
                self.document.sample_rate,
                self.document.channels,
                self.output.is_some(),
                self.active_file,
            );
            if let Some(out) = &self.output {
                out.play(start, end);
            }
            // BUGFIX (cursor jump-ahead / snap-back): (re)arm the UI-side
            // extrapolation exactly at the session start. The hardware clock
            // will read ~start for the next ~150 ms while the sink buffer
            // drains; without this re-arming the stale estimate from the
            // previous run would be blended in and the cursor would visibly
            // jump back to `start` after having shown a later position.
            self.playhead_est_frame = start;
            self.playhead_est_at = std::time::Instant::now();
            // BUGFIX (cursor races ahead of the audio and stalls at the right
            // edge while a short file finishes): drop the cached fed-ceiling —
            // the new session has nothing queued yet, and the output thread
            // republishes it as soon as the pump appends its first chunk.
            self.playhead_fed_ceiling = None;
            self.playhead_session = if end > start { Some((start, end)) } else { Some((start, self.document.total_frames())) };
            let mut ctrl = self.player.lock().unwrap_or_else(|p| p.into_inner());
            if self.output.is_some() {
                // With a live output the state machine must not advance on its
                // own clock — the real playhead arrives via PlayheadUpdate.
                ctrl.mark_playing(start, if end > start { end } else { self.document.total_frames() }, self.document.sample_rate);
            } else {
                ctrl.play(&self.document, sel);
            }
        }
    }

    fn transport_stop(&mut self) {
        if let Some(out) = &self.output { out.stop(); }
        // BUGFIX (cursor jump-ahead / snap-back): Stop rewinds the hardware
        // clock to the parked frame; drop any live extrapolation so the next
        // Play re-arms it from a consistent point.
        self.playhead_session = None;
        // BUGFIX (cursor races ahead of the audio): no queued samples remain
        // above the rewound clock — invalidate the cached fed-ceiling too.
        self.playhead_fed_ceiling = None;
        let parked = {
            let mut ctrl = self.player.lock().unwrap_or_else(|p| p.into_inner());
            ctrl.stop();
            ctrl.resume_frame()
        };
        // §3.5 stop: the cursor parks at the RESUME point — "начало файла или
        // место, ранее указанное мышкой" — NOT at the last session's start.
        // The output thread may still be draining its queue when we get here
        // (Cmd::Stop is asynchronous), and poll_playhead mirrors that lagging
        // hardware clock back into the controller — park the visible cursor
        // synchronously so Stop feels instant and cannot be dragged away.
        self.audio_view.set_cursor(parked);
        self.audio_view.auto_scroll_to_cursor();
        self.playhead_est_frame = parked;
        self.playhead_est_at = std::time::Instant::now();
    }

    /// Reset the playhead to frame 0 (hotkey "gotoStart", §3.5). ABSOLUTE
    /// behavior: always jumps to the very beginning of the file regardless
    /// of any active selection or loop range. A plain seek would be clamped
    /// into the running session's [start,end) range by both the controller
    /// and the output thread — so while playing from a selection that starts
    /// after 0, `seek(0)` snapped back to the selection start instead of the
    /// file start. When playing, restart the session from 0 to EOF (keeping
    /// playback going from the top of the file); otherwise just park the
    /// cursor/resume point at 0.
    fn transport_goto_start(&mut self) {
        let was_playing = self.player_state() == TransportState::Playing;
        if was_playing {
            self.transport_stop();
        }
        self.audio_view.set_cursor(0);
        self.audio_view.auto_scroll_to_cursor();
        self.transport_seek(0, audioscout_core::player::CursorSource::User);
        if was_playing {
            // Continue playback from the beginning of the file (§3.5): no
            // selection ⇒ session runs 0..total_frames.
            self.transport_play_pause();
        }
    }

    fn transport_seek(&mut self, frame: u64, source: audioscout_core::player::CursorSource) {
        // BUGFIX (backward click/drag impossible while playing/paused): a
        // `UserJump` is an explicit teleport OUT of the running session.
        // Both transport arms clamp a plain User seek into [start,end] while
        // Playing (§5.3), which made every backward click snap the cursor to
        // the session start and blocked scrubbing left during playback AND
        // pause (the resume point was clobbered too). Restart the session
        // from the new frame instead: play → play on from `frame`; pause →
        // stay paused with the resume point at `frame`.
        if source == audioscout_core::player::CursorSource::UserJump {
            let state = self.player_state();
            if state == TransportState::Playing || state == TransportState::Paused {
                // BUGFIX (backward click STILL snapped back to the live
                // position): the previous restart path raced the output
                // thread's command queue — `Cmd::Pause` froze the resume
                // point at the STALE hardware clock (~150 ms of buffered
                // audio behind the display) AFTER the UI had parked it, and
                // a queued `Cmd::Play` re-appended the OLD session range.
                // The fix is fully SYNCHRONOUS: read the honest audible
                // frame under the player lock, hand the controller an
                // explicit Parked freeze (which ignores any later stale
                // clock sample), then teleport via the plain `Cmd::Seek`
                // arm — for a Playing controller it restarts the pump at
                // [frame, end) inside the output thread before its next
                // cpal callback pulls samples; for a Paused one it just
                // parks the clock without touching the resume semantics.
                let total = self.document.total_frames();
                // BUGFIX (backward click/drag STILL snapped back): the app
                // keeps its own extrapolation anchor (`playhead_est_frame`)
                // as the authoritative DISPLAY position — it is re-armed on
                // every seek and advanced by wall time between polls, while
                // the controller's internal estimate can lag. Read it here
                // BEFORE leaving Playing so pause/stop snap exactly where
                // the user sees the cursor.
                let shown_display = self.playhead_est_frame.min(total);
                let end = {
                    let ctrl = self.player.lock().unwrap_or_else(|p| p.into_inner());
                    ctrl.session_bounds()
                        .map(|(_s, e)| e)
                        .unwrap_or(total)
                        .clamp(frame, total)
                };
                if state == TransportState::Playing {
                    // BUGFIX (backward click while playing STILL snapped back
                    // to the live position): do NOT send Cmd::Pause at all.
                    // Its arm runs asynchronously on the output thread and
                    // unconditionally overwrites the controller's frozen
                    // resume point with the lagging hardware-clock reading —
                    // queued AFTER our explicit park it still clobbered it,
                    // dragging the cursor back to the pre-jump position.
                    // Instead keep the controller in Paused (parked at the
                    // DISPLAY frame) and let the teleport below restart the
                    // session through `Cmd::Play` alone: apply_play stops the
                    // sink, drops the pump session, reseeds the clock at the
                    // NEW frame and re-appends [frame,end) — identical
                    // bookkeeping to pressing Play after such a pause. The
                    // controller-side transition happens synchronously under
                    // our lock, so no stale sample can ever overwrite it.
                    {
                        let mut ctrl = self.player.lock().unwrap_or_else(|p| p.into_inner());
                        ctrl.pause_parked(shown_display);
                    }
                    // Teleport the audio clock/pump/ceiling to the new frame
                    // while Paused: the Seek arm parks without appending a
                    // source (is_playing() is false), then Cmd::Play engages
                    // the fresh [frame,end) session from there.
                    if let Some(out) = &self.output {
                        out.seek(frame, audioscout_core::player::CursorSource::UserJump);
                        out.play(frame, end);
                    }
                    // Re-arm the UI extrapolation exactly at the new session
                    // origin (same bookkeeping as transport_play_pause).
                    self.playhead_est_frame = frame;
                    self.playhead_est_at = std::time::Instant::now();
                    self.playhead_fed_ceiling = None;
                    self.playhead_session = Some((frame, end));
                } else {
                    // Paused: park cursor AND frozen resume point at the
                    // clicked frame. Nothing plays, so no stale clock can
                    // overwrite it afterwards.
                    if let Some(out) = &self.output {
                        out.seek(frame, audioscout_core::player::CursorSource::UserJump);
                    }
                    {
                        let mut ctrl = self.player.lock().unwrap_or_else(|p| p.into_inner());
                        ctrl.park_cursor(frame, &self.document);
                    }
                    self.playhead_est_frame = frame;
                    self.playhead_est_at = std::time::Instant::now();
                    self.playhead_fed_ceiling = None;
                    self.playhead_session = None;
                }
                return;
            }
            // Stopped: an ordinary seek — fall through below.
        }
        if let Some(out) = &self.output { out.seek(frame, source); }
        // BUGFIX (cursor jump-ahead / snap-back): a seek teleports the true
        // clock; move the extrapolation anchor with it immediately, otherwise
        // the cursor keeps sliding away from the new position at nominal rate
        // until the next poll drags it back.
        self.playhead_est_frame = frame;
        self.playhead_est_at = std::time::Instant::now();
        // BUGFIX (cursor races ahead of the audio): a seek teleports the true
        // clock AND discards everything queued above the target; rewind the
        // cached ceiling with it so the extrapolation cannot show frames that
        // are no longer (or not yet) audible. The output thread republishes
        // the real value from the pump on its first append after the seek.
        self.playhead_fed_ceiling = Some(frame);
        let mut ctrl = self.player.lock().unwrap_or_else(|p| p.into_inner());
        ctrl.seek(frame, &self.document, source);
    }

    fn transport_toggle_loop(&mut self) {
        let enabled = {
            let ctrl = self.player.lock().unwrap_or_else(|p| p.into_inner());
            !ctrl.is_looping()
        };
        if let Some(out) = &self.output { out.set_loop(enabled); }
        let mut ctrl = self.player.lock().unwrap_or_else(|p| p.into_inner());
        ctrl.set_loop(enabled);
    }

    fn transport_set_volume(&mut self, vol: f32) {
        if let Some(out) = &self.output { out.set_volume(vol); }
        {
            let mut ctrl = self.player.lock().unwrap_or_else(|p| p.into_inner());
            ctrl.set_volume(vol);
        }
        // Write-on-change persistence (tz §1.4): the volume slider must land
        // in config.json immediately, so the new gain survives the next
        // launch. The controller clamps to 0..=1, mirror that here.
        let clamped = vol.clamp(0.0, 1.0);
        if (self.config_store.config().volume - clamped).abs() > f32::EPSILON {
            self.config_store.config_mut().volume = clamped;
            let _ = self.config_store.save();
        }
    }

    fn player_state(&self) -> TransportState {
        self.player.lock().unwrap_or_else(|p| p.into_inner()).state()
    }

    /// UI-side playhead poll (§1.2/§5.3, 50 ms cadence driven by the repaint
    /// loop in `update`). The rodio output thread already emits
    /// `PlayheadUpdate` events from the real audio clock; this fallback keeps
    /// the cursor moving when no audio device is available (the state
    /// machine advances on its own clock via `Controller::tick`) and simply
    /// refreshes the cursor from the controller otherwise. `last_playhead_poll`
    /// throttles the work to one poll per 50 ms tick.
    fn poll_playhead(&mut self, ctx: &egui::Context) {
        if self.player_state() != TransportState::Playing {
            return;
        }
        let now = std::time::Instant::now();
        if now.duration_since(self.last_playhead_poll) < std::time::Duration::from_millis(50) {
            return;
        }
        self.last_playhead_poll = now;
        if self.output.is_some() {
            // BUGFIX (playback freezes ~0.5 s after start, no sound): the
            // audio stream can be killed by PulseAudio/ALSA while the app
            // still thinks it is Playing — then neither the cursor nor the
            // meter ever move again. `poll_stall` watches the real sample
            // clock and pauses the transport on a >1 s freeze; surface that
            // to the user once per stall instead of letting the UI hang in a
            // fake playing state.
            let stalled = self
                .output
                .as_mut()
                .map(|out| out.poll_stall())
                .unwrap_or(false);
            if stalled {
                self.stall_notified = true;
                // BUGFIX (regression "controller thread is gone"): a stall
                // reported while the transport still claims Playing must also
                // stop the state machine — otherwise poll_playhead returns at
                // its very first `player_state() != Playing` guard and never
                // runs again, leaving the UI frozen in a fake playing state.
                // `poll_stall` already paused the controller on clock-freeze;
                // do it here too so every stall path (including the dead-
                // controller one) leaves the app in Paused and Play can
                // restart the session.
                {
                    let mut ctrl = self.player.lock().unwrap_or_else(|p| p.into_inner());
                    if ctrl.state() == TransportState::Playing {
                        ctrl.pause();
                    }
                }
                self.status_message = "Playback stalled: audio stream was dropped \
                                       by the system (PulseAudio/ALSA). Press Play \
                                       to retry."
                    .to_string();
                ctx.request_repaint();
                return;
            }
            if self.stall_notified && self.player_state() == TransportState::Playing {
                // A fresh play attempt after a stall — clear the warning.
                self.stall_notified = false;
            }
            // BUGFIX (Play/Pause restarts from 0): the state machine's
            // current_frame must follow the real audio clock, otherwise
            // Pause freezes resume_frame at a stale value and every repeated
            // Play re-appends the slice from the session start. Read the
            // clock directly from SharedOutput every poll tick (tz §5.3: пул
            // позиции ±50 мс) — PlayheadUpdate events stay as an additional
            // fresher update path.
            let shared = self.output.as_ref().expect("output present").shared.clone();
            let ch = self.document.channels.max(1);
            // BUGFIX (cursor LAGS behind the audio after the runaway fix):
            // the raw hardware counter (`current_frame`) is fed by
            // MeteredSource only every 512 samples AND lags the audible
            // speaker position by the cpal/rodio buffer depth (~0.3-0.75 s —
            // seconds on a 7 s clip). The previous revision painted and
            // re-anchored the extrapolation from this lagging value, so the
            // cursor crawled behind the sound it was supposed to track.
            // `true_audible_file_frame` is the honest AUDIBLE position: the
            // per-window truth published inside the cpal callback
            // (`true_playhead_file_frame`, already converted to file frames)
            // floored by the raw clock and bounded above by what the pump has
            // provably handed to the Sink — never ahead of the speakers, but
            // no longer behind them either.
            let sf = self
                .output
                .as_ref()
                .map(|out| out.stream_format())
                .unwrap_or((44100, 2));
            let audible_frame =
                shared.true_audible_file_frame(self.document.sample_rate.max(1), sf);
            // Raw hardware clock: the stored commit frame of the output
            // thread (seeded at the session origin, advanced per appended
            // chunk). The UI poll reads it directly every tick (§5.3 "пул
            // позиции ±50 мс") and feeds it into the controller mirror below.
            let raw_frame = shared.current_frame(ch);
            // Fallback for the pre-first-publish window (fresh session, the
            // truth field still tagged with an old epoch): the raw clock is
            // seeded at the session origin and only ever moves forward while
            // Playing, so it is a safe lower bound — never paint below it.
            let true_frame = audible_frame.max(raw_frame);
            // BUGFIX (backward click/drag impossible while playing): after a
            // `UserJump` session restart the extrapolation anchor sits at the
            // new (earlier) frame, but the lagging hardware clock still reads
            // the PRE-jump position for up to the sink-buffer depth (~150 ms).
            // The old monotonic guard (`max(true_frame)`) then snapped the
            // cursor straight back to where it was before the jump. Until the
            // audio clock actually reaches the new session origin, treat the
            // origin as the floor instead of the stale pre-jump reading.
            if let Some((s, _e)) = self.playhead_session {
                if true_frame < s && self.playhead_est_frame >= s {
                    // Fresh restart: the clock has not caught up yet.
                }
                if true_frame < s {
                    let true_frame_now = s;
                    let _ = true_frame_now;
                }
            }
            let true_frame = if self.playhead_est_frame > 0
                && true_frame < self.playhead_est_frame.saturating_sub(self.document.sample_rate.max(1) as u64 / 2)
            {
                // A teleport backwards just happened (session restart): pull
                // the truth floor down with it so the display can follow the
                // cursor left instead of being pinned at the old position.
                self.playhead_est_frame
            } else {
                true_frame
            };
            // BUGFIX (cursor jump-ahead / snap-back): paint an extrapolated
            // position instead of the raw hardware clock. The clock lags wall
            // time by the sink buffer (~150 ms) plus up to one 50 ms poll
            // quantum; drawing it directly makes the cursor crawl in coarse
            // steps and any transient forward reading gets snapped back on
            // the next tick — oscillation that accumulates with playback and
            // is very visible on short files. Estimate = max(true clock,
            // anchor advanced at the nominal rate), clamped to the session
            // end so a short file can never be over-run by the estimate.
            //
            // BUGFIX (cursor races ahead of the audio and stalls at the right
            // edge while the file finishes): `max(est, true_frame)` alone let
            // the estimate lead the real clock without bound — every poll
            // re-anchored at the already-inflated estimate, so the lead grew
            // to the full pump/sink buffer depth (the pump keeps ~1 s of
            // samples queued ahead of the device). On a 7 s clip the cursor
            // reached EOF (and the window edge) visibly early and froze there
            // while the audio kept playing. The estimate may now lead the
            // true clock by at most one poll quantum: when it drifts further
            // ahead, the anchor is pulled back to `true_clock + cap`, which
            // tracks the device within ~50 ms while staying smooth and
            // monotonic (never snaps backwards mid-session).
            let sr = self.document.sample_rate.max(1) as f64;
            let elapsed = self.playhead_est_at.elapsed().as_secs_f64().clamp(0.0, 0.25);
            let est = self
                .playhead_est_frame
                .saturating_add((elapsed * sr) as u64)
                .max(true_frame);
            // BUGFIX (cursor LAGS behind the audio): the audible truth
            // (`true_audible_file_frame`) is published per 512-sample window
            // inside the cpal callback and additionally trails the speakers
            // by the device's own output buffer (~30-80 ms of queued stream
            // samples that are already handed to ALSA/Pulse but not yet
            // played). The old one-poll-quantum cap pinned the display to
            // `truth + ~50 ms`, so the cursor systematically ran a visible
            // fraction of a second BEHIND the sound — seconds-scale on a 7 s
            // clip. The honest forward bound is the pump ceiling minus what
            // is still queued unplayed: `pumped_to_frame` is provably audible
            // or in-flight, so the extrapolation may legitimately sit up to
            // the queue depth ahead of the published truth. Cap the lead at
            // max(one poll quantum + device buffer, queue headroom), which
            // keeps the motion smooth without letting it race again.
            self.playhead_lead_cap_frames = (0.05 * sr) as u64 + 1;
            // Pump ceiling for THIS tick — read once, used both as the lead
            // headroom below and as the hard clamp after the estimate.
            self.playhead_fed_ceiling = self
                .output
                .as_ref()
                .and_then(|out| out.playhead_ceiling_frame());
            let fed_ceiling = self.playhead_fed_ceiling.unwrap_or(true_frame);
            let queue_headroom = fed_ceiling.saturating_sub(true_frame);
            let lead_ceiling = true_frame.saturating_add(
                self.playhead_lead_cap_frames.max(queue_headroom),
            );
            // BUGFIX (cursor races ahead of the audio and stalls at the right
            // edge while a short file finishes): the pump thread publishes the
            // furthest FILE frame it has handed to the Sink
            // (`playhead_ceiling_frame`). The visible cursor may never display
            // frames above that ceiling — they provably have not reached the
            // device queue yet, so no wall-clock extrapolation is allowed to
            // paint them. Combined with the queue-depth lead cap this keeps the
            // cursor locked onto the audible stream: on a 7 s clip it reaches
            // the right edge exactly when the sound does — neither seconds
            // early (old runaway) nor seconds late (raw lagging clock).
            // `None` (no session engaged / fresh Load-Pause-Stop reset) means
            // "nothing queued yet": fall back to the true clock so the cursor
            // cannot race ahead during the pre-buffer window.
            let frame = est.min(lead_ceiling).min(fed_ceiling);
            let frame = match self.playhead_session {
                Some((_s, e)) => frame.min(e.saturating_sub(1)),
                None => frame,
            };
            self.playhead_est_frame = frame;
            self.playhead_est_at = std::time::Instant::now();
            {
                let mut ctrl = self.player.lock().unwrap_or_else(|p| p.into_inner());
                // Mirror the hardware clock reading into the transport state
                // machine (CursorSource::Playback, §5.3). The lagging-reading
                // guard lives in the controller wrapper itself: stale samples
                // that sit behind the display are dropped instead of dragging
                // the state machine backwards, forward moves re-base it. This
                // keeps Play-after-Pause resuming from the real position.
                ctrl.sync_playback_position(frame);
            }
            self.audio_view.set_cursor(frame);
            self.audio_view.auto_scroll_to_cursor();
            // Keep the 50 ms poll cadence alive even when egui would idle the
            // window (no input events during hands-free playback).
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
            return;
        }
        // Device-less degradation: advance the state machine's simulated
        // clock and mirror it into the waveform cursor.
        let frame = {
            let mut ctrl = self.player.lock().unwrap_or_else(|p| p.into_inner());
            ctrl.tick(std::time::Duration::from_millis(50), &self.document);
            // Display path: the doc-aware projection (extrapolation + session
            // clamp) — `current_frame()` is the raw stored commit now.
            ctrl.current_frame_in(&self.document)
        };
        self.audio_view.set_cursor(frame);
        self.audio_view.auto_scroll_to_cursor();
    }

    fn player_current_secs(&self) -> f64 {
        let ctrl = self.player.lock().unwrap_or_else(|p| p.into_inner());
        ctrl.current_secs(self.document.sample_rate)
    }

    /// Two vertical scales per channel — instantaneous peak (+hold marker)
    /// and RMS — drawn at the right side of the player window (§3.5.1).
    /// BUGFIX (cursor "hand" on files and buttons): egui only switches to
    /// `CursorIcon::PointingHand` automatically for widgets it internally
    /// classifies as buttons; our custom table cells (`Sense::click()` over
    /// row rects), colored-label actions ("Save", "🔁") and folder-tree
    /// buttons kept the default arrow. This helper sets the hand icon while
    /// the response's widget is hovered — call it right after producing each
    /// clickable response. The per-frame reset in `update` guarantees the
    /// icon drops back to the arrow when the pointer leaves the widget.
    fn hover_hand(ui: &egui::Ui, response: &egui::Response) {
        // BUGFIX (hand cursor flickering / not showing on files and buttons):
        // `hovered()` is only true for the pass in which egui actually
        // processed the pointer against this widget; when a panel repaints
        // without new input (e.g. the 50 ms playhead poll), `hovered()` can
        // read false even though the pointer sits right on top of the widget,
        // so the hand icon dropped back to the arrow between frames.
        // `contains_pointer()` is computed from THIS frame's live pointer
        // position vs. the widget rect — OR-ing both makes the hand appear
        // immediately on enter and stay put while hovering. The per-frame
        // reset in `update` still drops it back when the pointer leaves.
        if response.hovered() || response.contains_pointer() {
            ui.output_mut(|o| o.cursor_icon = egui::CursorIcon::PointingHand);
        }
    }

    fn draw_volume_meter(&self, ui: &egui::Ui, rect: egui::Rect) {
        let painter = ui.painter();
        let ch = self.document.channels.max(1);
        let vis_channels = if ch == 1 { 1usize } else { 2usize };
        let bar_w = ((rect.width() - 8.0 * vis_channels as f32) / vis_channels as f32 * 0.5).max(6.0);
        let gap = 8.0f32;
        let mut x = rect.min.x;
        for c in 0..vis_channels {
            let (live_peak, live_rms, hold_peak) = if c == 0 {
                (self.meter_live.peak_l, self.meter_live.rms_l, self.meter_hold.peak_l)
            } else {
                (self.meter_live.peak_r, self.meter_live.rms_r, self.meter_hold.peak_r)
            };
            let h = rect.height();
            let base = rect.max.y;
            // background troughs
            for dx in [0.0f32, bar_w + 2.0] {
                painter.rect_filled(
                    egui::Rect::from_min_max(egui::pos2(x + dx, base - h), egui::pos2(x + dx + bar_w, base)),
                    2.0,
                    egui::Color32::from_gray(40),
                );
            }
            let peak_y = base - h * live_peak.clamp(0.0, 1.0);
            let rms_y = base - h * live_rms.clamp(0.0, 1.0);
            // instantaneous scale (green→red by level).
            // The red threshold comes from the "Volumeter red value" setting
            // (dBFS, signed, <= 0). Linear clipping is 1.0 == 0 dBFS, so we
            // convert the configured dB value to a linear amplitude and paint
            // red when the live peak is >= that value (i.e. at or above the
            // configured threshold; with the default -0.1 dB this means only
            // near/true clipping turns red — no hardcoded 0.95 anymore).
            let red_db = self.config_store.config().meter_red_db.clamp(-99.9, 0.0);
            let red_lin = (10f32.powf(red_db / 20.0)).clamp(0.0, 1.0);
            let peak_col = if live_peak >= red_lin { egui::Color32::RED } else { egui::Color32::from_rgb(80, 220, 80) };
            painter.rect_filled(egui::Rect::from_min_max(egui::pos2(x, peak_y), egui::pos2(x + bar_w, base)), 1.0, peak_col);
            // RMS scale (blue)
            painter.rect_filled(egui::Rect::from_min_max(egui::pos2(x + bar_w + 2.0, rms_y), egui::pos2(x + 2.0 * bar_w + 2.0, base)), 1.0, egui::Color32::LIGHT_BLUE);
            // peak-hold marker line
            if self.config_store.config().meter_hold_enabled {
                let hy = base - h * hold_peak.clamp(0.0, 1.0);
                // egui 0.29: `line_segment` takes [Pos2; 2] and a single
                // `impl Into<PathStroke>` — width+color go in as one tuple.
                painter.line_segment(
                    [egui::pos2(x, hy), egui::pos2(x + bar_w, hy)],
                    (1.5_f32, egui::Color32::YELLOW),
                );
            }
            if vis_channels == 2 {
                painter.text(egui::pos2(x + bar_w, base + 0.0 - 2.0), egui::Align2::CENTER_BOTTOM, if c == 0 { "L" } else { "R" },
                    egui::FontId::proportional(9.0), ui.visuals().text_color());
            }
            x += 2.0 * bar_w + gap + 2.0;
        }
    }

    /// Manual column-drag state machine for the file-table header.
    ///
    /// Why hand-rolled: egui's built-in drag responses (`drag_started`,
    /// `drag_stopped`) are tied to the widget's own click-id and stop being
    /// reported once the pointer leaves that widget — releasing a header
    /// title over ANOTHER header therefore never produced a drop, and the
    /// titles felt impossible to grab. This resolver works purely on raw
    /// pointer geometry plus per-widget press flags:
    ///
    /// * During the layout pass each header cell records a press candidate
    ///   (`col_press_frame`/`col_press_key`) when egui reports
    ///   `is_pointer_button_down_on()` on a frame where the primary button
    ///   was freshly pressed. The table closure only borrows `self`
    ///   immutably, so this runs as an injected callback (`begin_col_press`).
    /// * At the START of the NEXT frame `handle_column_drag` turns the armed
    ///   press into a live `ColDrag`, freezing the exact x-ranges of every
    ///   header cell (reported during the previous layout pass).
    /// * While the button is held, movement past `DRAG_THRESHOLD` promotes
    ///   the pending drag to ACTIVE; release resolves the drop against the
    ///   frozen cell ranges (left half = insert before, right half = after).
    ///
    /// Press/release transitions come from THIS frame's raw event list
    /// (`i.events`): egui consumes them with `PointerEvent::take()` while
    /// panels lay out, so by end of frame `pointer.button(...)` would report
    /// `false`. The event list itself is not taken, so it stays readable.
    fn handle_column_drag(&mut self, ctx: &egui::Context) {
        const DRAG_THRESHOLD: f32 = 4.0;

        // --- Raw primary-button transition + live position for this frame.
        // Press/release transitions come from THIS frame's event list
        // (`i.events`): egui consumes them with `PointerEvent::take()` while
        // panels lay out, so by the time widgets run, `pointer.button(...)`
        // may already report stale values. The event list itself is not
        // taken, so it stays readable here at the start of the frame.
        let mut pressed_at: Option<egui::Pos2> = None;
        let mut released_at: Option<egui::Pos2> = None;
        let (down_now, cur_pos) = ctx.input(|i| {
            // Newest first: the last PointerButton event of this frame is
            // the current button state (pressed=true → down now).
            for e in i.events.iter().rev() {
                if let egui::Event::PointerButton { pos, button, pressed, .. } = e {
                    if *button == egui::PointerButton::Primary {
                        if *pressed {
                            pressed_at = Some(*pos);
                        } else {
                            released_at = Some(*pos);
                        }
                        return (*pressed, Some(*pos));
                    }
                }
            }
            (
                i.pointer.button_down(egui::PointerButton::Primary),
                i.pointer.interact_pos(),
            )
        });
        let frame_id = self.col_frame_counter;

        // --- Arm a live drag from the press recorded LAST frame by the
        // header-cell widgets. Doing this at the start of the next frame
        // guarantees the header geometry the press happened against has
        // been fully laid out and reported (`header_cells_snapshot`).
        if !self.col_press_key.is_empty() && self.col_press_frame == frame_id - 1 {
            let cells = self.header_cells_snapshot(ctx);
            if !cells.is_empty() {
                let start = pressed_at
                    .or(cur_pos)
                    .unwrap_or_else(|| cells.first().map(|(_, x0, _)| egui::pos2(*x0, 0.0)).unwrap_or_default());
                self.col_drag = Some(ColDrag {
                    src: std::mem::take(&mut self.col_press_key),
                    start,
                    cells,
                    active: false,
                });
                // BUGFIX (horizontal scroll while dragging a header): from
                // now on until the matching release this capture OWNS the
                // horizontal gesture — `update()` swallows the files list's
                // hscroll deltas so the table cannot pan sideways under the
                // dragged title (see `col_capture_blocks_hscroll`). Kept in
                // sync with the press-time flag set by `begin_col_press`.
                self.col_capture_blocks_hscroll = true;
                ctx.request_repaint();
            }
        }

        // --- Mid-drag bookkeeping while the button is held.
        if let Some(drag) = self.col_drag.as_mut() {
            if !down_now && released_at.is_none() {
                // Button went down somewhere else / stale state: cancel.
                self.col_drag = None;
                // The borrow checker needs this branch to keep `drag` alive,
                // so the capture is ended through a raw-pointer call — same
                // safety argument as the header-cell press callback (single
                // thread, plain state fields only).
                let self_raw = self as *mut AudioScoutApp;
                // SAFETY: see note above.
                unsafe { (*self_raw).end_col_capture_block() };
            } else if !drag.active {
                if let Some(pos) = cur_pos {
                    if (pos - drag.start).length() > DRAG_THRESHOLD {
                        drag.active = true;
                        ctx.request_repaint();
                    }
                }
            }
        }

        // --- Release: resolve the drop or cancel the drag.
        let Some(release_pos) = released_at else {
            if !down_now {
                self.col_drag = None;
                self.col_press_key.clear();
                self.end_col_capture_block();
            }
            return;
        };
        // Any primary-button release ends the header capture — the blocking
        // window lasts exactly from the header press to its release.
        self.end_col_capture_block();
        let Some(drag) = self.col_drag.take() else { return };
        if !drag.active {
            // Press + release without movement: a plain click. Sorting is
            // handled by egui's `clicked()` response — nothing to do here.
            return;
        }

        // Find the target column under the release point.
        let mut target: Option<(String, bool)> = None;
        for (key, x0, x1) in &drag.cells {
            if key == &drag.src {
                continue;
            }
            if release_pos.x >= *x0 && release_pos.x <= *x1 {
                target = Some((key.clone(), release_pos.x > (x0 + x1) / 2.0));
                break;
            }
        }
        if target.is_none() {
            // Released outside every known cell: pick the nearest cell edge.
            let mut best: Option<(f32, String, bool)> = None;
            for (key, x0, x1) in &drag.cells {
                if key == &drag.src {
                    continue;
                }
                let d = if release_pos.x < *x0 {
                    x0 - release_pos.x
                } else if release_pos.x > *x1 {
                    release_pos.x - x1
                } else {
                    0.0
                };
                let a = release_pos.x > (x0 + x1) / 2.0;
                if best.as_ref().map_or(true, |(bd, _, _)| d < *bd) {
                    best = Some((d, key.clone(), a));
                }
            }
            target = best.map(|(_, k, a)| (k, a));
        }

        if let Some((tcol, after)) = target
            .and_then(|(k, a)| Column::from_key(&k).map(|c| (c, a)))
        {
            if let Some(scol) = Column::from_key(&drag.src) {
                if scol != tcol {
                    self.column_layout.move_column(&scol, &tcol, after);
                    // Persist the new order immediately (write-on-change, tz
                    // §1.4): `columns/order` is the visible display order and
                    // `columns/full_order` the order of every known column —
                    // both are rebuilt consistently from the live layout, so
                    // the dragged arrangement survives the next launch.
                    let layout = self.column_layout.clone();
                    layout.store_into(&mut self.config_store.config_mut().columns);
                    let _ = self.config_store.save();
                }
            }
            // Remember the frame whose release ended the drag: the click
            // egui may report for the source header that frame must be
            // ignored (a completed reorder must never also re-sort).
            self.col_suppress_click_frame = frame_id;
            ctx.request_repaint();
        }
    }

    /// End the header capture that blocks horizontal scrolling of the files
    /// list (release / cancellation). Besides clearing the flag this records
    /// the ending frame: egui's ScrollArea keeps its horizontal offset in
    /// `Memory` and only clamps it while laying out, so an h-scroll delta
    /// that arrived one frame EARLIER (before the capture was armed — the
    /// arming happens at the start of the following frame) can still be
    /// pending there. The next frame's `update()` resets that offset once
    /// (see `col_capture_release_frame`) so the table never visibly pans
    /// sideways around a header grab.
    fn end_col_capture_block(&mut self) {
        if self.col_capture_blocks_hscroll {
            self.col_capture_release_frame = self.col_frame_counter;
        }
        self.col_capture_blocks_hscroll = false;
    }

    /// Called from the header-cell widgets during the layout pass when the
    /// primary button went down inside their rect on a fresh-press frame.
    fn begin_col_press(&mut self, key: &str, frame_id: u64, ctx: &egui::Context) {
        let already_grabbing = self.col_header_grab_pending(frame_id);
        self.col_press_frame = frame_id;
        self.col_press_key = key.to_string();
        // BUGFIX (horizontal scroll while grabbing a header): the blocking
        // window starts at the PRESS, not at the arming frame one tick later
        // — otherwise the files-list ScrollArea pans sideways under the held
        // button during that single unblocked frame (see `col_capture_press_frame`).
        self.col_capture_press_frame = frame_id;
        self.col_capture_blocks_hscroll = true;
        if !already_grabbing {
            // Remember where the table was scrolled horizontally so the
            // post-release cleanup can restore it (`update()`, see
            // `col_capture_release_frame`). Read straight from the area's
            // per-frame state in egui Memory.
            self.col_hscroll_offset_at_press = self.file_list_hscroll_offset(ctx);
        }
    }

    /// True while a press→arm transition or live drag owns the horizontal
    /// gesture on/just after `frame_id` (same rule as `col_header_grab_active`
    /// but evaluated against the layout-pass frame). Used to keep
    /// `begin_col_press` idempotent across the several header cells of one
    /// press and to snapshot the h-scroll offset only once per grab.
    fn col_header_grab_pending(&self, frame_id: u64) -> bool {
        self.col_capture_blocks_hscroll
            || (self.col_capture_press_frame != 0
                && frame_id.saturating_sub(self.col_capture_press_frame) <= 1)
    }

    /// Current horizontal scroll offset of the files-list ScrollArea, read
    /// straight from egui's per-area `scroll_area::State` in Memory. The
    /// area's id is deterministic (`central_ui_child_id`, mirroring how egui
    /// derives it from the CentralPanel ui id + the pinned `id_salt`), so no
    /// widget-report side channel is needed. `None` until the area has been
    /// shown at least once (state stored on its first frame).
    fn file_list_hscroll_offset(&self, ctx: &egui::Context) -> Option<f32> {
        let id = central_ui_child_id(ctx, FILE_LIST_ID_SALT);
        egui::scroll_area::State::load(ctx, id).map(|st| st.offset.x)
    }

    /// True while a header grab owns the horizontal gesture: either the
    /// press→arm transition is still in flight (a header press happened on
    /// this or the previous frame and the primary button is still down) or
    /// the live drag has armed the flag directly (`handle_column_drag`).
    fn col_header_grab_active(&self, ctx: &egui::Context) -> bool {
        if self.col_capture_blocks_hscroll {
            return true;
        }
        let pressed_recently = self.col_capture_press_frame != 0
            && self.col_frame_counter.saturating_sub(self.col_capture_press_frame) <= 1;
        if !pressed_recently {
            return false;
        }
        ctx.input(|i| i.pointer.button_down(egui::PointerButton::Primary))
    }

    /// Spawn VLC for `path` — the shared back-end of both context-menu
    /// entries (§ "Open with VLC" on a file, § "Load to VLC" on a folder).
    /// VLC is assumed installed; when the binary cannot be located (or the
    /// spawn itself fails) the reason surfaces in the status bar instead of
    /// crashing or silently doing nothing.
    fn open_in_vlc(&mut self, path: &Path, as_directory: bool) {
        let result = if as_directory {
            audioscout_core::vlc::load_directory_to_vlc(path)
        } else {
            audioscout_core::vlc::open_file_with_vlc(path)
        };
        self.status_message = match result {
            Ok(()) => {
                if as_directory {
                    format!("Loaded {} into VLC", path.display())
                } else {
                    format!("Opening {} with VLC", path.display())
                }
            }
            Err(e) => {
                tracing::error!("[vlc] failed to launch for {path:?}: {e}");
                format!("VLC launch failed: {e}")
            }
        };
    }

    /// True while ANY pointer button is held down anywhere. Used by the
    /// files-list right-click menu: column-drag grabs only ever use the
    /// primary button, but a menu must also stay closed while the user drags
    /// with the secondary button elsewhere (e.g. panning another ScrollArea),
    /// so it claims its popup only from a fully idle pointer.
    ///
    /// Associated function (no `&self`): the check runs inside the file-list
    /// grid closure, which already mutably borrows several `self` fields —
    /// an instance method would need yet another immutable borrow of `*self`.
    fn any_pointer_button_down(ctx: &egui::Context) -> bool {
        ctx.input(|i| {
            i.pointer.button_down(egui::PointerButton::Primary)
                || i.pointer.button_down(egui::PointerButton::Secondary)
                || i.pointer.button_down(egui::PointerButton::Middle)
        })
    }

    /// Exact x-ranges of all visible header cells, collected by the table
    /// header widgets DURING the previous frame's layout pass (each cell
    /// reports its rect under a stable id). Falls back to widths computed
    /// from the layout constants when the report is missing (e.g. the panel
    /// was not shown), so a drop can always be resolved.
    fn header_cells_snapshot(&self, ctx: &egui::Context) -> Vec<(String, f32, f32)> {
        let mut cells: Vec<(usize, String, f32, f32)> = Vec::new();
        for (ci, col) in self.column_layout.visible.iter().enumerate() {
            let key = col.key().to_string();
            if let Some(r) = ctx.data(|d| d.get_temp::<egui::Rect>(hdr_cell_id(ci))) {
                cells.push((ci, key, r.left(), r.right()));
            }
        }
        if cells.is_empty() {
            // Fallback: reconstruct approximate ranges from widths.
            let mut x = 8.0_f32;
            let mut out = Vec::new();
            for col in &self.column_layout.visible {
                let w = self.column_layout.width(col);
                out.push((col.key().to_string(), x, x + w));
                x += w + 10.0; // grid spacing [10.0, 4.0]
            }
            return out;
        }
        cells.sort_by_key(|(ci, _, _, _)| *ci);
        cells.into_iter().map(|(_, k, x0, x1)| (k, x0, x1)).collect()
    }

    fn navigate_file(&mut self, direction: i32) {
        if self.file_rows.is_empty() { return; }
        let current_idx = self.selected_row.unwrap_or(0) as i32;
        let new_idx = if direction > 0 { (current_idx + 1).min(self.file_rows.len() as i32 - 1) } else { (current_idx - 1).max(0) };
        if new_idx >= 0 && (new_idx as usize) < self.file_rows.len() {
            self.selected_row = Some(new_idx as usize);
            let path = self.file_rows[new_idx as usize].path.clone();
            self.load_file(&path);
        }
    }

    fn render_folder_node(
        &mut self,
        ui: &mut egui::Ui,
        path: &Path,
        name: &str,
        depth: usize,
        dir_to_select: &mut Option<PathBuf>,
        dir_to_expand: &mut Option<PathBuf>,
        dir_to_collapse: &mut Option<PathBuf>,
        dir_to_vlc: &mut Option<PathBuf>,
    ) {
        let is_current = self.current_dir.as_ref() == Some(&path.to_path_buf());
        let is_expanded = self.expanded_dirs.contains_key(path);

        let text_color = if is_current {
            egui::Color32::YELLOW
        } else {
            ui.visuals().text_color()
        };

        // Create a button-like label for the folder
        let icon = if is_expanded { "▼" } else { "▶" };
        let label = format!("{} {} {}", icon, if is_expanded { "📂" } else { "📁" }, name);

        let response = ui.add(
            egui::Button::new(egui::RichText::new(&label).color(text_color))
                .fill(if is_current { egui::Color32::from_rgba_premultiplied(50, 50, 50, 100) } else { egui::Color32::TRANSPARENT })
                .stroke(egui::Stroke::NONE)
        );

        // Change cursor to hand on hover. `contains_pointer()` is checked
        // alongside `hovered()` so the hand appears in the very frame the
        // pointer enters the row (see the `hand_cursor` helper for the
        // full rationale) and requests a repaint so the icon updates even
        // when the mouse-stop would otherwise leave egui idle.
        if response.hovered() || response.contains_pointer() {
            ui.output_mut(|o| o.cursor_icon = egui::CursorIcon::PointingHand);
            ui.ctx().request_repaint();
        }

        // Handle clicks
        if response.clicked() {
            // Single click - select the directory
            *dir_to_select = Some(path.to_path_buf());
            // Also toggle expand/collapse
            if is_expanded {
                *dir_to_collapse = Some(path.to_path_buf());
            } else {
                *dir_to_expand = Some(path.to_path_buf());
            }
        }

        // Right-click context menu (§ "Load to VLC"): queue every audio file
        // of this folder into one VLC session. The popup claims its own id
        // (see `context_menu`), so it survives outside the folder button's
        // interaction scope and only one menu is ever open at a time.
        // NOTE: the anchor position is taken from the row's OWN rect
        // (`response.rect`), not from the live pointer position — a
        // pointer-based anchor made the menu chase the mouse between frames
        // ("убегает от мышки"). Claimed on a fresh secondary press; while
        // any button is still held the claim waits, so an in-flight drag or
        // selection gesture never gets interrupted by the popup.
        let menu_id = egui::Id::new(("vlc_dir_menu", path));
        let menu_claim = menu_id.with("claim");
        if response.secondary_clicked() && !Self::any_pointer_button_down(ui.ctx()) {
            ui.ctx().data_mut(|d| d.insert_temp(menu_claim, menu_id));
            ui.ctx().request_repaint();
        }
        if ui.ctx().data(|d| d.get_temp::<egui::Id>(menu_claim)) == Some(menu_id) {
            // Anchor at the folder row's own rect (see the note above): the
            // popup stays glued to the directory entry instead of chasing
            // the pointer from frame to frame.
            let menu_pos = response.rect.right_bottom();
            if context_menu(ui, &response, menu_id, menu_pos, &["Load to VLC"]).is_some() {
                ui.ctx().data_mut(|d| d.remove::<egui::Id>(menu_claim));
                *dir_to_vlc = Some(path.to_path_buf());
            } else if response.clicked() {
                // A plain left-click on the folder (select/expand) also
                // resolves the popup — `context_menu` closes itself on any
                // outside primary press, so just drop our claim here.
                ui.ctx().data_mut(|d| d.remove::<egui::Id>(menu_claim));
            }
        }

        // Render children if expanded
        if is_expanded {
            ui.indent(egui::Id::new(path), |ui| {
                if let Some(children) = self.expanded_dirs.get(path).cloned() {
                    for child in children {
                        let child_name = child.file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("")
                            .to_string();

                        // Skip hidden directories
                        if child_name.starts_with('.') {
                            continue;
                        }

                        // Recursive call for child
                        self.render_folder_node(
                            ui,
                            &child,
                            &child_name,
                            depth + 1,
                            dir_to_select,
                            dir_to_expand,
                            dir_to_collapse,
                            dir_to_vlc,
                        );
                    }
                }
            });
        }
    }

    fn handle_keyboard(&mut self, ctx: &egui::Context) {
        let config = self.config_store.config().clone();
        ctx.input(|i| {
            for event in &i.events {
                if let egui::Event::Key { key, pressed: true, modifiers, .. } = event {
                    for (action, binding) in &config.hotkeys {
                        let key_match = match binding.key.as_str() {
                            "Space" => *key == egui::Key::Space,
                            "Delete" => *key == egui::Key::Delete,
                            "Escape" => *key == egui::Key::Escape,
                            "ArrowUp" => *key == egui::Key::ArrowUp,
                            "ArrowDown" => *key == egui::Key::ArrowDown,
                            other => other.chars().next().and_then(|_| egui::Key::from_name(other)).map(|k| *key == k).unwrap_or(false),
                        };
                        let mod_match = binding.ctrl == modifiers.ctrl && binding.shift == modifiers.shift && binding.alt == modifiers.alt;
                        if key_match && mod_match {
                            match action.as_str() {
                                "playpause" => self.transport_play_pause(),
                                "mute" => self.mute_selection(),
                                "normalize" => self.normalize_selection(),
                                "undo" => self.undo(),
                                "redo" => self.redo(),
                                "addMarker" => self.add_marker(),
                                "deleteMarker" => self.delete_active_marker(),
                                "clearSelection" => self.audio_view.clear_selection(),
                                "zoomIn" => self.audio_view.zoom_in(),
                                "zoomOut" => self.audio_view.zoom_out(),
                                "zoomReset" => self.audio_view.zoom_reset(),
                                "prev" => self.navigate_file(-1),
                                "next" => self.navigate_file(1),
                                // Absolute reset to file start (§3.5): must
                                // jump to frame 0 even mid-selection/mid-loop;
                                // if playing, continue from the top.
                                "gotoStart" => self.transport_goto_start(),
                                _ => {}
                            }
                        }
                    }
                    if *key == egui::Key::ArrowLeft || *key == egui::Key::ArrowRight {
                        let direction = if *key == egui::Key::ArrowRight { 1 } else { -1 };
                        if self.player_state() == TransportState::Playing {
                            let target = {
                                let mut ctrl = self.player.lock().unwrap_or_else(|p| p.into_inner());
                                // The controller owns the configured step
                                // (§3.6 skipSeconds): pass only the button
                                // direction so one press = one step.
                                ctrl.skip(&self.document, direction);
                                // The committed jump target (raw stored frame)
                                // is what gets re-seeked and shown.
                                ctrl.current_frame()
                            };
                            self.transport_seek(target, audioscout_core::player::CursorSource::User);
                            let frame = target;
                            self.audio_view.set_cursor(frame);
                        } else {
                            let step = if self.audio_view.view_range.len() > 100 { self.audio_view.view_range.len() / 100 } else { 1 };
                            let new_pos = if direction > 0 { self.audio_view.cursor_frame.saturating_add(step) } else { self.audio_view.cursor_frame.saturating_sub(step) };
                            self.audio_view.set_cursor(new_pos);
                        }
                    }
                }
            }
        });
    }
}

impl eframe::App for AudioScoutApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Own frame counter for the column-drag state machine (egui 0.29's
        // Context has no `frame_index()`). Bumped at the START of the frame
        // so the layout pass and the end-of-frame resolver share one id.
        self.col_frame_counter += 1;

        // --- Column drag: consume last frame's armed header press FIRST —
        // before any panel takes pointer events with `PointerEvent::take()`,
        // so the fresh-press transition is still visible in the raw event
        // list here. Turns the press into a live drag and freezes the header
        // geometry reported during the previous layout pass.
        self.handle_column_drag(ctx);

        // BUGFIX (horizontal scroll while dragging a column header): while a
        // header capture owns the mouse gesture (press inside a title cell →
        // release), the files list must not pan sideways. egui turns *both*
        // horizontal input sources of its `ScrollArea::both()` into scrolling
        // during such a gesture:
        //  * drag-scrolling: "held primary button + pointer moved" pans a
        //    wide ScrollArea directly (no prior wheel event needed) — killed
        //    by `.drag_to_scroll(false)` on the files-list ScrollArea below;
        //  * trackpad two-finger hscroll / shift+wheel: egui folds every
        //    `Event::MouseWheel` into `raw/smooth/unprocessed_scroll_delta`
        //    (`input_state/mod.rs`: shift turns the delta into
        //    `vec2(x + y, 0)`, Ctrl/Cmd into the zoom channel), and a wide
        //    area consumes `smooth_scroll_delta.x` while hovered.
        // So while the grab is active swallow the horizontal component of
        // this frame's wheel events BEFORE any panel lays out, clear the
        // aggregated scroll vectors' x parts, and zero the pointer motion
        // delta (belt-and-braces against any other drag-panning interaction).
        // Vertical scrolling, plain wheel zoom elsewhere and vertical row
        // selection keep working untouched.
        let header_grab = self.col_header_grab_active(ctx);
        if header_grab {
            ctx.input_mut(|i| {
                for e in i.events.iter_mut() {
                    if let egui::Event::MouseWheel { delta, modifiers, .. } = e {
                        if modifiers.shift || modifiers.ctrl || modifiers.command || modifiers.mac_cmd {
                            // shift = pure horizontal scroll, ctrl/cmd = the
                            // zoom channel fed from the same delta — drop the
                            // whole event, neither may move the table sideways.
                            *delta = egui::Vec2::ZERO;
                        } else {
                            // Trackpad 2D pixel delta / plain wheel: keep the
                            // vertical component, kill the horizontal.
                            delta.x = 0.0;
                        }
                    }
                }
                // The backend may already have folded this frame's motion
                // into the aggregated vectors before `update()` ran.
                i.smooth_scroll_delta.x = 0.0;
                i.raw_scroll_delta.x = 0.0;
                // NOTE: egui also keeps a PRIVATE pending-smoothing buffer
                // (`unprocessed_scroll_delta`) fed by coarse wheel notches;
                // it is drained into `smooth_scroll_delta` on later frames —
                // which is exactly why the x parts are re-zeroed every frame
                // above AND for a few frames after the release (below), and
                // why the area's stored offset gets restored on release.
            });
        }

        // Post-release cleanup: a h-scroll delta that slipped through one
        // frame BEFORE the capture was armed can still be pending in egui's
        // smoothed-scroll pipeline or already baked into the files list's
        // offset in Memory. For a few frames after the grab ends keep zeroing
        // the residuals so the table never visibly pans around a header grab.
        if !header_grab && self.col_capture_release_frame != 0 {
            let since = self.col_frame_counter.saturating_sub(self.col_capture_release_frame);
            if since <= 3 {
                ctx.input_mut(|i| {
                    i.smooth_scroll_delta.x = 0.0;
                    i.raw_scroll_delta.x = 0.0;
                });
                // Reset the horizontal offset the files-list ScrollArea keeps
                // in Memory back to where it was when the grab started.
                if let Some(x) = self.col_hscroll_offset_at_press {
                    let id = central_ui_child_id(ctx, FILE_LIST_ID_SALT);
                    if let Some(mut st) = egui::scroll_area::State::load(ctx, id) {
                        if st.offset.x != x {
                            st.offset.x = x;
                            st.store(ctx, id);
                        }
                    }
                }
            } else {
                self.col_capture_release_frame = 0;
                self.col_hscroll_offset_at_press = None;
            }
        }

        // Window title: "AudioScout - <loaded file name>". The title follows
        // the file currently loaded into the player (document.source_path);
        // when nothing is loaded only the app name is shown. Updated every
        // frame — egui applies it only if the string actually changed.
        let window_title = match self.document.source_path.file_name() {
            Some(name) => format!("AudioScout - {}", name.to_string_lossy()),
            None => "AudioScout".to_string(),
        };
        ctx.send_viewport_cmd(egui::ViewportCommand::Title(window_title));

        // Panel-height constants changed since this Memory was populated
        // (e.g. a rebuilt binary after editing TOOLBAR_HEIGHT/...): wipe all
        // cached per-panel sizes once so the new `default_height`s apply.
        // Normally redundant with the height-versioned panel ids below, but
        // makes the guarantee independent of the id scheme.
        if self.panel_layout_id != PANEL_LAYOUT_ID {
            self.panel_layout_id = PANEL_LAYOUT_ID;
            // Rebuilt binary with new height constants: purge the stale
            // cached panel sizes once so the new defaults take effect.
            purge_panel_size_caches(ctx);
        }

        // Meter decay when nothing new arrives (pause/stop): exponential-ish
        // fade to zero over ~300 ms (§3.5.1). While playing, fresh
        // VolumeMeterUpdate events overwrite these values every poll tick.
        {
            let now = ctx.input(|i| i.time);
            let dt = (now - self.last_frame_time).clamp(0.0, 0.1) as f32;
            self.last_frame_time = now;
            if self.player_state() != TransportState::Playing {
                const DECAY: f32 = 10.0; // ~1/e per 100 ms → silent in ~300 ms
                let k = (1.0 - DECAY * dt).max(0.0);
                self.meter_live.peak_l *= k; self.meter_live.peak_r *= k;
                self.meter_live.rms_l *= k; self.meter_live.rms_r *= k;
                self.meter_hold.peak_l *= k; self.meter_hold.peak_r *= k;
            }
        }
        // BUGFIX (hand cursor on files/buttons): egui only sets
        // `CursorIcon::PointingHand` automatically for widgets it internally
        // marks as buttons; our custom table cells (`Sense::click()` over the
        // row rects) and colored-label "buttons" kept the default arrow.
        // Reset the icon to the default at the START of every frame so a hand
        // requested by a hovered widget in the previous pass never leaks into
        // this one, then let the per-widget hover checks below re-apply it.
        ctx.output_mut(|o| o.cursor_icon = egui::CursorIcon::Default);

        // BUGFIX (wheel zoom / wheel scrub in the player stopped working;
        // the +/− buttons still worked): egui 0.29 folds each `MouseWheel`
        // event into two fields — `raw_scroll_delta` (the total notch delta
        // for this frame) and `smooth_scroll_delta` (a smoothed slice of it).
        // Both stay readable for the WHOLE frame (nothing in egui zeroes
        // them mid-pass), so the original failure was elsewhere: the old
        // handler preferred `smooth_scroll_delta` whenever it was non-zero.
        // A discrete wheel notch arrives as one big jump (typically ≥ 8
        // points per detent), which egui classifies as "not smooth" and
        // routes through `unprocessed_scroll_delta`, releasing only ~10 %
        // of it into the smooth field per frame while the rest decays
        // exponentially. Worse, during playback the 50 ms poll repaints the
        // UI *without* new input, and those frames keep replaying the same
        // stale `raw_scroll_delta` from `new_events.take()` — so the huge,
        // slowly-decaying smooth value dominated the "prefer smooth" pick
        // and the effective gesture became invisible (sub-pixel pans, zoom
        // indistinguishable from the buttons doing nothing noticeable).
        // Fix: snapshot the RAW notch delta (exactly one application per
        // real wheel event) plus the Ctrl modifier right here, at the top
        // of `update()`, hand it to the audio-view panel below (which
        // consumes-then-clears it), and request an immediate repaint so a
        // notch arriving between on-demand frames is processed at once.
        //
        // BUGFIX (regression after the first wheel fix — zoom STILL dead with
        // a notched mouse, buttons kept working): `i.events` is NOT cleared
        // by egui after the pass (`InputState::end_pass` never resets the
        // event vector), so every input-less repaint (the 50 ms playhead
        // tick!) re-reads the PREVIOUS wheel event and overwrites the
        // snapshot with it. When the user then rolled a fresh notch, its
        // `MouseWheel` event arrived together with a `PointerMoved` event
        // appended AFTER it; the `.iter().rev().find_map(...)` scan hit that
        // newer drag event first, found no `MouseWheel` inside it and fell
        // back to `raw_scroll_delta` — which is correct but was then clobbered
        // again by the stale-event replay above. Net effect: exactly one
        // gesture per *pair* of notches, half-strength, applied one frame
        // late — felt like "wheel does nothing". The scan now (a) stops at
        // the FIRST MouseWheel event (events are chronological, so it is
        // this frame's gesture) and (b) only treats events as live when the
        // frame actually carried input (`pointerMoved || has_touch_action`,
        // the same predicate egui uses to gate scroll processing); otherwise
        // the snapshot stays zeroed and the panel sees nothing to apply.
        let mut wheel_evt = None;
        ctx.input(|i| {
            // Events are chronological and `i.events` holds ONLY the events
            // accumulated between the two last passes, so the FIRST
            // `MouseWheel` found here is this frame's real gesture (a `.rev()`
            // scan used to hit later PointerMoved events and miss it).
            if let Some(ev) = i.events.iter().find_map(|e| match e {
                egui::Event::MouseWheel { unit, delta, modifiers, .. } => Some((*unit, *delta, *modifiers)),
                _ => None,
            }) {
                let px_per_notch = match ev.0 {
                    egui::MouseWheelUnit::Point => 1.0,
                    // egui's default `InputOptions::line_scroll_speed` (the
                    // field itself is private in 0.29; the value is stable).
                    egui::MouseWheelUnit::Line => 20.0,
                    egui::MouseWheelUnit::Page => i.screen_rect().height(),
                };
                wheel_evt = Some((ev.1.y * px_per_notch, ev.2.ctrl));
            } else if i.raw_scroll_delta.y != 0.0
                && i.events.iter().any(|e| matches!(e, egui::Event::PointerMoved { .. }))
            {
                // No explicit MouseWheel event but a real scroll delta this
                // frame (e.g. synthetic scroll injected by the backend) —
                // accept it only while pointer input actually flowed, never
                // on the stale replay of the previous gesture's counter.
                wheel_evt = Some((i.raw_scroll_delta.y, i.modifiers.ctrl));
            }
        });
        if let Some((dy, ctrl)) = wheel_evt {
            // Latch the gesture until the audio panel consumes it (or one
            // second passes without consumption, e.g. the pointer was over
            // the file list). Overwriting on every new notch is fine; simply
            // clearing it at the top of each `update()` was the bug that made
            // wheel zoom dead: with egui's on-demand repaints the event can
            // arrive in a pass where the panel closure runs BEFORE the
            // snapshot exists, and by the next (repaint-triggered) pass the
            // snapshot had already been wiped — the buttons kept working
            // because they do not depend on this channel at all.
            self.raw_wheel_dy = dy;
            self.raw_wheel_ctrl = ctrl;
            self.last_wheel_apply = ctx.input(|i| i.time);
            // Immediate repaint so the panel applies the gesture without
            // waiting for another input event.
            ctx.request_repaint();
        } else if ctx.input(|i| i.time - self.last_wheel_apply > 1.0) {
            // Expire an unconsumed snapshot so an old gesture cannot be
            // applied much later when the pointer finally enters the view.
            self.raw_wheel_dy = 0.0;
            self.raw_wheel_ctrl = false;
        }


        self.process_events(ctx);
        self.handle_keyboard(ctx);
        // 50 ms playhead poll (§1.2/§5.3): with a live output the position
        // arrives via PlayheadUpdate events; without one this fallback tick
        // advances the state machine's clock so the cursor still moves.
        self.poll_playhead(ctx);
        if self.player_state() == TransportState::Playing { ctx.request_repaint_after(std::time::Duration::from_millis(50)); }

        // Preview pump: while any thumbnail is still missing, keep draining
        // events and repainting on a fixed cadence. Without this, egui's
        // on-demand redraw could leave finished previews invisible (and their
        // `PreviewReady` events unread in the channel) until the user moved
        // the mouse — the "PNGs exist on disk but appear late in the UI" bug.
        if self.previews_pending() {
            self.retry_missing_previews();
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }

        // UX: pointing-hand cursor over every interactive button. egui's
        // default style never changes the mouse cursor, so buttons looked
        // inert. Instead of guessing from previous-frame memory (the old
        // `interacted_widget` / `hovered_widget` APIs no longer exist in
        // egui 0.29), we use an immediate-mode helper that sets
        // `CursorIcon::PointingHand` whenever a response is hovered. Panels
        // below call it for toolbars, player transport, file rows and folder
        // tree items.

        egui::TopBottomPanel::top(panel_id("toolbar", TOOLBAR_HEIGHT, TOOLBAR_CONTENT_LEFT_INSET, TOOLBAR_CONTENT_TOP_INSET))
            .exact_height(TOOLBAR_HEIGHT)
            .show(ctx, |ui| {
            let ctx = ui.ctx().clone();
            let hand_cursor = |resp: &egui::Response| -> bool {
                // UX: pointing-hand cursor on hover. `hovered()` alone is
                // frame-lagged and goes false while the pointer *enters*
                // the widget in the same frame (the icon then only appears
                // one repaint later, or not at all when nothing else
                // schedules a redraw). `contains_pointer()` is computed
                // from this frame's live pointer position against the
                // widget rect, so the hand shows up immediately; clicks
                // still come from `resp.clicked()`.
                if resp.hovered() || resp.contains_pointer() {
                    ctx.output_mut(|o| o.cursor_icon = egui::CursorIcon::PointingHand);
                }
                resp.clicked()
            };
            // Top content inset: push the toolbar row TOOLBAR_CONTENT_TOP_INSET
            // points below the panel's top edge (applied to the panel body —
            // inside `horizontal` a vertical offset would be ignored).
            apply_content_top_inset(ui, TOOLBAR_CONTENT_TOP_INSET);
            ui.horizontal(|ui| {
                // Left content inset: indent the first widget of the row.
                apply_content_left_inset(ui, TOOLBAR_CONTENT_LEFT_INSET);
                ui.label("Filter:");
                ui.text_edit_singleline(&mut self.filter_text);
                let r_x = ui.button("✕");
                Self::hover_hand(ui, &r_x);
                if r_x.clicked() { self.filter_text.clear(); }
                ui.separator();
                let r_columns = ui.button("Columns");
                Self::hover_hand(ui, &r_columns);
                // Toggle: a repeated click on the button closes the window
                // that this same button opened.
                if r_columns.clicked() { self.show_columns_dialog = !self.show_columns_dialog; }
                ui.separator();
                // Table thumbnail style toggle (§3.1): the code to switch
                // modes existed but had no control — this button exposes the
                // current state and flips it atomically for the whole set.
                let preview_spectro = *self.preview_mode.lock().unwrap() == PreviewMode::Spectrogram;
                let preview_btn = egui::SelectableLabel::new(
                    preview_spectro,
                    if preview_spectro { "Spectrogram" } else { "Waveform" },
                );
                if hand_cursor(&ui.add(preview_btn).on_hover_text("Toggle table previews between waveform and spectrogram")) {
                    self.toggle_preview_mode();
                }
                ui.separator();
                // Purge the preview cache of the current directory (DB rows +
                // PNG files) and immediately re-run generation for its files.
                let r_refresh = ui.button("Refresh preview");
                Self::hover_hand(ui, &r_refresh);
                if r_refresh.clicked() {
                    let ctx2 = ui.ctx().clone();
                    self.refresh_current_dir_previews(&ctx2);
                }
                // §3.6: the table-preview switch lives ONLY on the top
                // toolbar and the visualization-area switch lives ONLY on
                // the player panel — each exists in exactly one panel,
                // duplication is forbidden. The second Waveform/Spectrogram
                // button that used to sit here (a duplicate of the player
                // panel's §3.5 toggle driving `AudioView::mode`) is removed.
                ui.separator();
                let r_settings = ui.button("Settings");
                Self::hover_hand(ui, &r_settings);
                // Toggle: a repeated click on the button closes the window
                // that this same button opened.
                if r_settings.clicked() { self.show_settings = !self.show_settings; }
            });
        });

        // Declare paths_to_load BEFORE CentralPanel to avoid borrow conflicts
        let mut paths_to_load: Vec<PathBuf> = Vec::new();

        // Left panel - Folder tree
        // Collect actions to perform after the closure (avoid borrow conflicts)
        let mut dir_to_select: Option<PathBuf> = None;
        let mut dir_to_expand: Option<PathBuf> = None;
        let mut dir_to_collapse: Option<PathBuf> = None;
        // "Load to VLC" picked from a folder's right-click menu — spawned
        // after the panel closure, like every other deferred action.
        let mut dir_to_vlc: Option<PathBuf> = None;

        egui::SidePanel::left("folder_tree")
            .default_width(200.0)
            .resizable(true)
            .show(ctx, |ui| {
                ui.heading("Folders");
                ui.separator();

                egui::ScrollArea::vertical().show(ui, |ui| {
                    // Show Home directory
                    if let Some(home) = dirs_next::home_dir() {
                        self.render_folder_node(
                            ui,
                            &home,
                            "Home",
                            0,
                            &mut dir_to_select,
                            &mut dir_to_expand,
                            &mut dir_to_collapse,
                            &mut dir_to_vlc,
                        );
                    }

                    // Show root directory
                    let root = PathBuf::from("/");
                    self.render_folder_node(
                        ui,
                        &root,
                        "/",
                        0,
                        &mut dir_to_select,
                        &mut dir_to_expand,
                        &mut dir_to_collapse,
                        &mut dir_to_vlc,
                    );

                    // Show connected disks (excluding Home and /)
                    let home = dirs_next::home_dir();
                    let disks_to_render: Vec<(PathBuf, String)> = self.disks.iter()
                        .filter(|disk| {
                            // Skip if it's Home or root (already shown)
                            if home.as_ref() == Some(&disk.mount_point) {
                                return false;
                            }
                            if disk.mount_point == PathBuf::from("/") {
                                return false;
                            }
                            true
                        })
                        .map(|disk| (disk.mount_point.clone(), disk.name.clone()))
                        .collect();

                    for (mount_point, name) in disks_to_render {
                        self.render_folder_node(
                            ui,
                            &mount_point,
                            &name,
                            0,
                            &mut dir_to_select,
                            &mut dir_to_expand,
                            &mut dir_to_collapse,
                            &mut dir_to_vlc,
                        );
                    }
                });
            });

        // Apply actions after closure
        if let Some(dir) = dir_to_vlc {
            // "Load to VLC": queue all audio files of the folder into one
            // detached VLC session (VLC is assumed installed; a missing
            // binary surfaces as an error in the status bar).
            self.open_in_vlc(&dir, true);
        }
        if let Some(dir) = dir_to_select {
            self.current_dir = Some(dir.clone());
            self.scanner.scan(dir, self.event_tx.clone());
        }
        if let Some(dir) = dir_to_expand {
            if let Ok(entries) = std::fs::read_dir(&dir) {
                let mut subdirs: Vec<PathBuf> = entries
                    .filter_map(|e| e.ok())
                    .filter(|e| e.path().is_dir())
                    .map(|e| e.path())
                    .collect();
                subdirs.sort_by(|a, b| {
                    a.file_name()
                        .and_then(|n| n.to_str())
                        .map(|s| s.to_lowercase())
                        .cmp(&b.file_name().and_then(|n| n.to_str()).map(|s| s.to_lowercase()))
                });
                self.expanded_dirs.insert(dir, subdirs);
            }
        }
        if let Some(dir) = dir_to_collapse {
            self.expanded_dirs.remove(&dir);
        }

        // Bottom stack — declaration order matters: in egui the FIRST declared
        // bottom panel docks to the very bottom edge of the window and each
        // subsequent one is placed above the previous. So this declaration
        // order (audio_view → player → actions → status) yields the visual
        // stack, top → bottom: status bar, file actions (Mute/Apply…),
        // player controls (Play/Stop…), visualization window. The transport
        // panel therefore sits *above* the file-content view, directly
        // under the file-actions panel (bug #1). All four panels are declared
        // before the CentralPanel below, so the central area's
        // `available_height()` already accounts for their real, measured
        // sizes — the visualization window is resizable, its height changes
        // at runtime, and the file list shrinks/grows with it instead of
        // rendering underneath the player windows (bug #2).

        // Audio view panel (visualization area) — declared FIRST among the
        // bottom panels so it docks to the very bottom edge of the window;
        // the player controls are placed *above* it, and the file-actions
        // panel above those (bug #1: Play/Stop must sit under the
        // file-actions panel and above the audio-content view).
        egui::TopBottomPanel::bottom(panel_id_with_height("audio_view", AUDIO_VIEW_HEIGHT))
            .default_height(AUDIO_VIEW_HEIGHT)
            .resizable(true)
            .show(ctx, |ui| {
            let (rect, response) = ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
            // Publish this panel's exact content rect for the wheel-delta
            // ownership test at the top of `update()` (see BUGFIX note on
            // `audio_view_rect`). Done before any early return below so the
            // rect is always fresh while the panel exists.
            self.audio_view_rect = Some(rect);
            ui.painter().rect_filled(rect, 0.0, egui::Color32::from_rgb(25, 25, 30));

            if !self.document.samples.is_empty() {
                let num_tracks = self.cached_channels.len();
                // BUGFIX (cursor drawn behind the waveform): the playhead line
                // used to be painted BEFORE the spectrogram image / waveform
                // columns, so it ended up underneath them and was only visible
                // through gaps. Track where overlays must start: everything
                // below is drawn on top of the audio content (§3.5 — курсор
                // поверх дорожек).
                let overlay_start_y = rect.top();

                // Spectrogram mode: draw the pre-rendered widget-resolution
                // image first; markers/selection/cursor overlay on top of it.
                if self.audio_view.mode == AudioViewMode::Spectrogram {
                    self.refresh_view_spectrogram(ui, rect);
                    if let Some(tex) = &self.view_spectro_texture {
                        ui.put(rect, egui::Image::new(tex).fit_to_exact_size(rect.size()));
                    }
                }

                // The per-channel 0 dBFS middle lines are NOT drawn here any
                // more: in waveform mode they are painted at the end of each
                // track's loop iteration below (on top of that track's wave
                // columns), and in spectrogram mode `render_view_spectrogram`
                // bakes them into the image. Drawing a single line only for
                // mono here used to leave stereo/multichannel tracks without
                // a zero reference.

                // Markers
                for (idx, marker) in self.document.markers.iter().enumerate() {
                    let mx = self.audio_view.frame_to_x(marker.time_samples, rect.width() as u32);
                    let screen_x = rect.left() + mx;
                    if screen_x >= rect.left() && screen_x <= rect.right() {
                        let color = if self.audio_view.active_marker == Some(idx) { egui::Color32::YELLOW } else { egui::Color32::from_rgb(100, 180, 255) };
                        let mut y = overlay_start_y;
                        while y < rect.bottom() {
                            let seg_end = (y + 4.0).min(rect.bottom());
                            ui.painter().line_segment([egui::pos2(screen_x, y), egui::pos2(screen_x, seg_end)], egui::Stroke::new(1.0_f32, color));
                            y += 8.0;
                        }
                        ui.painter().text(egui::pos2(screen_x + 2.0, overlay_start_y + 2.0), egui::Align2::LEFT_TOP, &marker.name, egui::FontId::proportional(10.0), color);
                    }
                }

                // Selection
                if let Some(sel) = self.audio_view.selection_range() {
                    let x1 = rect.left() + self.audio_view.frame_to_x(sel.0, rect.width() as u32);
                    let x2 = rect.left() + self.audio_view.frame_to_x(sel.1, rect.width() as u32);
                    let sel_rect = egui::Rect::from_min_max(egui::pos2(x1.max(rect.left()), overlay_start_y), egui::pos2(x2.min(rect.right()), rect.bottom()));
                    ui.painter().rect_filled(sel_rect, 0.0, egui::Color32::from_rgba_premultiplied(80, 120, 255, 60));
                }

                // Waveform - using cached Peaks tree for fast rendering
                // Render 1 track for mono, 2 tracks for stereo/multichannel
                let start_frame = self.audio_view.view_range.start_frame;
                let end_frame = self.audio_view.view_range.end_frame;
                let view_frames = end_frame.saturating_sub(start_frame);

                if view_frames > 0 && num_tracks > 0 && self.audio_view.mode == AudioViewMode::Waveform {
                    let track_height = rect.height() / num_tracks as f32;
                    let pixels = rect.width() as u32;
                    let step = view_frames as f64 / pixels as f64;
                    let start = start_frame as f64;

                    for (track_idx, (channel_data, peaks)) in self.cached_channels.iter().enumerate() {
                        let track_top = rect.top() + track_idx as f32 * track_height;
                        let track_mid_y = track_top + track_height / 2.0;

                        // Draw separator line between tracks (except before first)
                        if track_idx > 0 {
                            ui.painter().line_segment(
                                [egui::pos2(rect.left(), track_top), egui::pos2(rect.right(), track_top)],
                                egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(60, 60, 65))
                            );
                        }

                        // Get min/max for each pixel column using cached peaks tree
                        let columns = peaks.spans(channel_data, start, step, pixels as usize);

                        let half_h = track_height / 2.0;
                        for (px, &(min_val, max_val)) in columns.iter().enumerate() {
                            let x = rect.left() + px as f32;
                            let y_top = track_mid_y - max_val * half_h;
                            let y_bot = track_mid_y - min_val * half_h;
                            ui.painter().line_segment(
                                [egui::pos2(x, y_top), egui::pos2(x, y_bot)],
                                egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(50, 200, 80))
                            );
                        }

                        // Per-channel middle line at 0 dBFS (amplitude 0),
                        // painted AFTER this track's wave columns so it stays
                        // visible on top of dense signal — one line for every
                        // channel (mono: 1, stereo/multichannel: 2), matching
                        // the table-preview renderer.
                        ui.painter().line_segment(
                            [egui::pos2(rect.left(), track_mid_y.round()), egui::pos2(rect.right(), track_mid_y.round())],
                            egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(148, 163, 184))
                        );
                    }
                }

                // Cursor — BUGFIX (отрисовку надо переместить наверх, сейчас
                // курсор находится за waveform): the playhead is painted LAST
                // inside this panel, i.e. on top of the spectrogram image,
                // the waveform columns and the selection highlight, so it is
                // always fully visible while tracking playback (§3.5).
                let cursor_x = self.audio_view.frame_to_x(self.audio_view.cursor_frame, rect.width() as u32);
                let cursor_screen_x = rect.left() + cursor_x;
                if cursor_screen_x >= rect.left() && cursor_screen_x <= rect.right() {
                    ui.painter().line_segment([egui::pos2(cursor_screen_x, overlay_start_y), egui::pos2(cursor_screen_x, rect.bottom())], egui::Stroke::new(1.5_f32, egui::Color32::RED));
                    // Small "handle" triangle at the very TOP of the view —
                    // the user asked for the cursor to sit above everything;
                    // the marker makes it readable even over dense waveforms.
                    ui.painter().add(egui::Shape::convex_polygon(
                        vec![
                            egui::pos2(cursor_screen_x - 4.0, overlay_start_y),
                            egui::pos2(cursor_screen_x + 4.0, overlay_start_y),
                            egui::pos2(cursor_screen_x, overlay_start_y + 6.0),
                        ],
                        // Color32 already satisfies `Into<Color32>`; the `.into()` call made
                        // the target type ambiguous (E0283) since Color32 converts into Hsva,
                        // Rgba and others too.
                        egui::Color32::RED,
                        egui::Stroke::NONE,
                    ));
                }
            }

            // Hand cursor over the waveform: clicking seeks, so the whole
            // visualization area is an interactive target (§3.5).
            Self::hover_hand(ui, &response);

            if response.clicked() {
                if let Some(pos) = response.interact_pointer_pos() {
                    let x_ratio = (pos.x - rect.left()) / rect.width();
                    let frame = self.audio_view.x_to_frame(x_ratio * rect.width(), rect.width() as u32);
                    self.audio_view.set_cursor(frame);
                    // BUGFIX (backward click impossible while playing/paused):
                    // a waveform click is an explicit teleport OUT of the
                    // running session (`UserJump`), not a plain User seek —
                    // the transport arms clamp a User seek into [start,end]
                    // and blocked every click left of the live cursor.
                    self.transport_seek(frame, audioscout_core::player::CursorSource::UserJump);
                }
            }

            // BUGFIX (no mouse scrubbing in the player): press-and-drag in
            // the waveform now drags the cursor continuously. While the
            // pointer is held down over the view, every frame re-maps the
            // pointer X to a file frame and teleports the playhead there
            // (`UserJump`: clamped-out-of-session restart while Playing,
            // resume-point parking while Paused). The 50 ms poll cadence of
            // `transport_seek`'s jump path keeps this cheap enough for
            // per-frame handling.
            if response.dragged() {
                if let Some(pos) = response.interact_pointer_pos() {
                    let x_ratio = ((pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
                    let frame = self.audio_view.x_to_frame(x_ratio * rect.width(), rect.width() as u32);
                    self.audio_view.set_cursor(frame);
                    self.transport_seek(frame, audioscout_core::player::CursorSource::UserJump);
                }
            }

            // Wheel zoom / horizontal scroll — see the BUGFIX note at the top
            // of `update()`: the RAW notch delta (one application per real
            // wheel event, immune to egui's exponential smoothing and to the
            // stale-event replay on input-less repaints) is snapshotted there
            // into `raw_wheel_dy`/`raw_wheel_ctrl` and consumed-then-cleared
            // right here. The snapshot is only ever set while the pointer is
            // generating input, so consuming it unconditionally (not only
            // when `hovering` is true this frame) is what makes the FIRST
            // notch work: with on-demand repaints the hover test can lag the
            // event by one pass, and the old code zeroed the pending gesture
            // whenever that frame happened to report "not hovering" — the
            // wheel felt dead while the +/− buttons kept working.
            let hovering = response.contains_pointer()
                || ui.input(|i| i.pointer.hover_pos().map_or(false, |p| rect.contains(p)));
            let live_smooth_y = if hovering {
                ui.input(|i| if i.smooth_scroll_delta != egui::Vec2::ZERO {
                    i.smooth_scroll_delta.y
                } else { 0.0 })
            } else { 0.0 };
            let (scroll_y, ctrl_mods) = if self.raw_wheel_dy != 0.0 {
                let v = (self.raw_wheel_dy, self.raw_wheel_ctrl);
                self.raw_wheel_dy = 0.0; // consume exactly once
                self.raw_wheel_ctrl = false;
                v
            } else {
                (live_smooth_y, ui.input(|i| i.modifiers.ctrl))
            };
            if hovering && scroll_y != 0.0 {
                if ctrl_mods {
                    // Zoom around the pointer: fall back to the centre of the
                    // view when the pointer is outside the rect (e.g. it left
                    // the window between the event and this repaint).
                    let pos = ui
                        .input(|i| i.pointer.hover_pos())
                        .filter(|p| rect.contains(*p))
                        .unwrap_or(rect.center());
                    let x_ratio = ((pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
                    // One classic wheel notch is ~35–40 points after unit
                    // conversion; normalise so a single Ctrl+notch always
                    // gives the same comfortable ×1.25 zoom step regardless
                    // of device units or smoothing leftovers.
                    let steps = (scroll_y / 35.0).clamp(-1.0, 1.0);
                    let factor = 1.25_f64.powf(steps as f64);
                    self.audio_view.zoom_at_x(factor, x_ratio as f64);
                } else {
                    // Plain wheel: horizontal scrub, one notch ≈ 25 % of the
                    // visible window (see `scroll_horizontal`).
                    self.audio_view.scroll_horizontal(-scroll_y as f64 * 0.01);
                }
                ctx.request_repaint();
            }
        });

        // Player panel (transport controls) — declared after the audio view,
        // i.e. docked *above* it and directly under the file-actions panel
        // (bug #1: Play/Stop must sit above the audio-content view).
        egui::TopBottomPanel::bottom(panel_id("player", PLAYER_HEIGHT, PLAYER_CONTENT_LEFT_INSET, PLAYER_CONTENT_TOP_INSET))
            .exact_height(PLAYER_HEIGHT)
            .show(ctx, |ui| {
            // §3.5.1: transport controls on the left, the volume meter is
            // docked to the right edge of the same window (controls are laid
            // out inside the reduced rect so nothing overlaps).
            let panel_rect = ui.max_rect();
            let meter_w = 90.0f32;
            let meter_rect = egui::Rect::from_min_max(
                egui::pos2(panel_rect.right() - meter_w, panel_rect.min.y + 4.0),
                egui::pos2(panel_rect.right() - 4.0, panel_rect.max.y - 4.0),
            );
            // Reserve the right strip for the meter; everything else lays
            // out in a child UI confined to the remaining area.
            ui.allocate_ui(egui::vec2(meter_rect.min.x - 6.0 - panel_rect.min.x, panel_rect.height()), |ui| {
            let ctx = ui.ctx().clone();
            let hand_cursor = |resp: &egui::Response| -> bool {
                // UX: pointing-hand cursor on hover. `hovered()` alone is
                // frame-lagged and goes false while the pointer *enters*
                // the widget in the same frame (the icon then only appears
                // one repaint later, or not at all when nothing else
                // schedules a redraw). `contains_pointer()` is computed
                // from this frame's live pointer position against the
                // widget rect, so the hand shows up immediately; clicks
                // still come from `resp.clicked()`.
                if resp.hovered() || resp.contains_pointer() {
                    ctx.output_mut(|o| o.cursor_icon = egui::CursorIcon::PointingHand);
                }
                resp.clicked()
            };
            // Top content inset: push the transport row PLAYER_CONTENT_TOP_INSET
            // points below the panel's top edge (applied to the child UI that
            // fills the panel body — inside `horizontal` a vertical offset
            // would be ignored).
            apply_content_top_inset(ui, PLAYER_CONTENT_TOP_INSET);
            ui.horizontal(|ui| {
                // Left content inset: indent the first widget of the row.
                apply_content_left_inset(ui, PLAYER_CONTENT_LEFT_INSET);
                if hand_cursor(&ui.button("|<")) { self.navigate_file(-1); }
                let play_text = match self.player_state() { TransportState::Playing => "⏸", _ => "▶" };
                if hand_cursor(&ui.button(play_text)) { self.transport_play_pause(); }
                if hand_cursor(&ui.button("⏹")) { self.transport_stop(); }
                if hand_cursor(&ui.button(">|")) { self.navigate_file(1); }
                let (looping, playing_now) = {
                    let ctrl = self.player.lock().unwrap_or_else(|p| p.into_inner());
                    (ctrl.is_looping(), ctrl.state() == TransportState::Playing)
                };
                let loop_color = if looping && playing_now { egui::Color32::YELLOW } else if looping { egui::Color32::GREEN } else { ui.visuals().text_color() };
                if hand_cursor(&ui.colored_label(loop_color, "🔁")) { self.transport_toggle_loop(); }
                ui.separator();
                if hand_cursor(&ui.button("+")) { self.audio_view.zoom_in(); }
                if hand_cursor(&ui.button("−")) { self.audio_view.zoom_out(); }
                ui.separator();
                ui.label("Vol:");
                let mut vol = self.player.lock().unwrap_or_else(|p| p.into_inner()).volume() * 100.0;
                {
                    let slider_resp = ui.add(egui::Slider::new(&mut vol, 0.0..=100.0));
                    if slider_resp.hovered() { ctx.output_mut(|o| o.cursor_icon = egui::CursorIcon::ResizeHorizontal); }
                    if slider_resp.changed() { self.transport_set_volume(vol / 100.0); }
                }
                ui.separator();
                let save_color = if self.document.dirty { egui::Color32::YELLOW } else { ui.visuals().text_color() };
                let save_label = if self.document.dirty { "Save*" } else { "Save" };
                if hand_cursor(&ui.colored_label(save_color, save_label)) { self.save_file(); }
                ui.separator();
                // Player-panel display-mode toggle (§3.5): reflects and flips
                // the live `AudioView::mode` state (persisted in config.json).
                let view_spectro = self.audio_view.mode == AudioViewMode::Spectrogram;
                let view_btn = egui::SelectableLabel::new(
                    view_spectro,
                    if view_spectro { "Spectrogram" } else { "Waveform" },
                );
                if hand_cursor(&ui.add(view_btn).on_hover_text("Toggle the visualization between waveform and spectrogram")) {
                    self.toggle_audio_view_mode();
                }
            });
            ui.horizontal(|ui| {
                let duration = self.document.duration_secs();
                let current_pos = self.player_current_secs();
                let time_str = if duration < 60.0 { format!("{:.1}s / {:.1}s", current_pos, duration) } else { format!("{} / {}", format_time_hms(current_pos), format_time_hms(duration)) };
                ui.label(time_str);
                if duration > 0.0 { let progress = (current_pos / duration) as f32; ui.add(egui::ProgressBar::new(progress).desired_width(ui.available_width())); }
            });
            });
            self.draw_volume_meter(ui, meter_rect);
        });

        // File-actions panel (Mute / Normalize / Apply …) — declared after
        // the player, i.e. docked *above* it: this is the panel the transport
        // controls must sit under (bug #1).
        egui::TopBottomPanel::bottom(panel_id("actions", ACTIONS_HEIGHT, ACTIONS_CONTENT_LEFT_INSET, ACTIONS_CONTENT_TOP_INSET))
            .exact_height(ACTIONS_HEIGHT)
            .show(ctx, |ui| {
            let ctx = ui.ctx().clone();
            let hand_cursor = |resp: &egui::Response| -> bool {
                // UX: pointing-hand cursor on hover. `hovered()` alone is
                // frame-lagged and goes false while the pointer *enters*
                // the widget in the same frame (the icon then only appears
                // one repaint later, or not at all when nothing else
                // schedules a redraw). `contains_pointer()` is computed
                // from this frame's live pointer position against the
                // widget rect, so the hand shows up immediately; clicks
                // still come from `resp.clicked()`.
                if resp.hovered() || resp.contains_pointer() {
                    ctx.output_mut(|o| o.cursor_icon = egui::CursorIcon::PointingHand);
                }
                resp.clicked()
            };
            // Top content inset: push the actions row ACTIONS_CONTENT_TOP_INSET
            // points below the panel's top edge (applied to the panel body —
            // inside `horizontal` a vertical offset would be ignored).
            apply_content_top_inset(ui, ACTIONS_CONTENT_TOP_INSET);
            ui.horizontal(|ui| {
                // Left content inset: indent the first widget of the row.
                apply_content_left_inset(ui, ACTIONS_CONTENT_LEFT_INSET);
                if hand_cursor(&ui.button("Mute")) { self.mute_selection(); }
                ui.label("Normalize:");
                {
                    let dv_resp = ui.add(egui::DragValue::new(&mut self.normalize_dbfs).range(-60.0..=0.0).suffix(" dBFS"));
                    if dv_resp.hovered() { ctx.output_mut(|o| o.cursor_icon = egui::CursorIcon::ResizeHorizontal); }
                }
                if hand_cursor(&ui.button("Apply")) { self.normalize_selection(); }
                ui.separator();
                if hand_cursor(&ui.button("Delete all markers")) { self.document.markers.clear(); }
                if let Some(path) = &self.active_file {
                    ui.separator();
                    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                    ui.strong(name);
                }
            });
        });

        // Status bar — declared LAST among the bottom panels, which docks it
        // highest in the stack, right under the file-list area (§13).
        egui::TopBottomPanel::bottom(panel_id("status", STATUS_HEIGHT, STATUS_CONTENT_LEFT_INSET, STATUS_CONTENT_TOP_INSET))
            .exact_height(STATUS_HEIGHT)
            .show(ctx, |ui| {
            // Top content inset: push the status text STATUS_CONTENT_TOP_INSET
            // points below the panel's top edge.
            apply_content_top_inset(ui, STATUS_CONTENT_TOP_INSET);
            // Left content inset: indent the status text from the panel's
            // left edge (the body is a vertical layout, so `add_space`
            // advances the horizontal cursor here).
            apply_content_left_inset(ui, STATUS_CONTENT_LEFT_INSET);
            ui.label(&self.status_message);
        });

        // Same grab state as the delta-swallowing block above, recomputed
        // here for the layout pass: `handle_column_drag` (run at the start of
        // this frame) may have armed/ended a capture since. Used to switch
        // `.drag_to_scroll(...)` off on the files-list ScrollArea while a
        // column title is grabbed (see the call site below).
        let header_grab_now = self.col_header_grab_active(ctx);

        // "Open with VLC" picked from a file row's right-click menu
        // (§ context menu). Declared OUTSIDE the CentralPanel closure so the
        // deferred spawn below (after the closure) can read it — `open_in_vlc`
        // needs `&mut self`, which is unavailable while the grid immutably
        // borrows `file_rows`/`preview_cache`/`selected_row` inside.
        let mut file_to_vlc: Option<PathBuf> = None;

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(format!("{} files", self.file_rows.len()));
                if let Some(dir) = &self.current_dir { ui.label(dir.display().to_string()); }
            });
            ui.separator();

            let visible_cols = self.column_layout.visible.clone();

            // --- Table layout ------------------------------------------------------
            // BUGFIX (column headers): the header row used to be emitted
            // *after* the ScrollArea, i.e. at the BOTTOM of the files window;
            // later it became a SECOND, separate `Grid` ("file_table_header")
            // above the list. Two independent grids never share a real cell
            // mesh: egui derives column offsets from each grid's own
            // allocated rect, so any context difference (e.g. the vertical
            // scrollbar eating width inside the body's ScrollArea only) made
            // the header columns drift against the file-row columns. The
            // header is now the FIRST ROW of the very same Grid that holds
            // every data row — titles and cells share one grid by
            // construction (see "ONE grid" below).
            //
            // The table's own vertical ScrollArea is sized exactly to the
            // remaining space (`auto_shrink([false, false])` + `max_height`)
            // so the scrollbar spans the real list region.
            //
            // BUGFIX (window height): the file-list height must account for
            // the whole bottom stack — status bar + file actions + player
            // controls + the (user-resizable) visualization window. All four
            // bottom panels are declared *before* this CentralPanel, so egui
            // has already reserved their real, measured space: here
            // `ui.available_height()` is exactly what is left between the
            // current cursor position and the top edge of the bottom stack
            // (panel margins/inner spacing are subtracted by egui itself
            // when the scroll area allocates its rect). The buggy version
            // ALSO subtracted a hand-made estimate of the same bottom stack
            // (`bottom_panels_height`, computed with hardcoded constants in
            // the actions panel), i.e. it double-counted the reserved space:
            // the file window came out far shorter than it should be, and
            // every drift of that estimate forced an extra re-layout frame.
            // Now the available height is used directly, so resizing the
            // visualization panel still shrinks/grows the list one-to-one.
            let list_max_height = ui.available_height().max(60.0);

            // Borrow the pieces the row loop needs *before* opening the
            // table closure, so `self` is not captured mutably inside.
            // SAFETY (raw pointer below): drag-arming writes ONLY plain
            // column-drag state fields (`col_press_frame`/`col_press_key`).
            // Taking a raw pointer to `self` here — while disjoint field
            // borrows (preview caches, selected_row) are alive — avoids the
            // E0499 that a full `&mut self` reborrow would raise; the pointer
            // is used single-threaded during widget layout only, never
            // escapes `update()`, and no other mutable borrow of the drag
            // fields exists at that moment.
            {
                let self_raw = self as *mut AudioScoutApp;
                let cur_frame = self.col_frame_counter;
                let col_press_cb = |key: &str, frame_id: u64| {
                    // SAFETY: see note above — eframe runs update() on one
                    // thread; the callback touches only the drag-state fields.
                    unsafe { (*self_raw).begin_col_press(key, frame_id, ctx) };
                };
                let column_layout = &self.column_layout;
                // Per-mode texture caches: pick the one matching the active
                // thumbnail style so switching modes never re-uploads or mixes
                // wave/spectrogram textures.
                let preview_cache: &mut HashMap<String, egui::TextureHandle> = match *self.preview_mode.lock().unwrap() {
                    PreviewMode::Waveform => &mut self.preview_cache_wave,
                    PreviewMode::Spectrogram => &mut self.preview_cache_spectro,
                };
                let selected_row = &mut self.selected_row;
                let filter_text = self.filter_text.to_lowercase();
                // REFACTOR (table grid): the header used to be emitted *after*
                // the ScrollArea (i.e. under the list), then it was moved on
                // top but rendered as a SECOND, separate `Grid`
                // ("file_table_header") while the data rows lived in another
                // one ("file_table"/"file_table_body"). Two independent grids
                // never share a real cell mesh: their column offsets come
                // from separately allocated rects, so any difference in the
                // surrounding context (e.g. the vertical scrollbar eating
                // width inside the body's grid only, or differing outer
                // margins) shifted the whole header against the body — the
                // header cells were not part of the same grid as the row
                // cells, and the title columns did not line up with the
                // file-row columns.
                // Now there is exactly ONE `Grid` ("file_table") inside the
                // ScrollArea: the header is its FIRST ROW, followed by all
                // data rows. Header cells and row cells are laid out by the
                // very same grid pass over the very same rect, so every
                // column title shares its column geometry with the cells
                // underneath it by construction. Per-column widths
                // (`column_widths`) and the shared cell painter (`draw_cell`)
                // additionally pin each cell to its configured column width.
                let column_widths: Vec<f32> = visible_cols
                    .iter()
                    .map(|c| column_layout.width(c))
                    .collect();

                // Shared cell painter: draws one table cell into `cell_ui`
                // constrained to the exact column width. Used by BOTH the
                // header row and the data rows of the single table grid, so
                // geometry is identical. Returns the child Ui's min_rect (in
                // parent coordinates), which doubles as the cell's
                // click-response rect.
                fn draw_cell(
                    cell_ui: &mut egui::Ui,
                    width: f32,
                    height: f32,
                    contents: impl FnOnce(&mut egui::Ui),
                ) -> egui::Rect {
                    let inner = cell_ui.allocate_ui_with_layout(
                        egui::vec2(width, height),
                        egui::Layout::left_to_right(egui::Align::Center),
                        |ui| {
                            contents(ui);
                            ui.min_rect()
                        },
                    );
                    inner.inner
                }

                // Sort indicator state for the header row, read *before* the
                // immutable borrows used inside the grid closure below.
                let sort_column = self.column_layout.sort_column;
                let sort_ascending = self.column_layout.sort_ascending;

                // Column drag-and-drop is resolved by `handle_column_drag`,
                // run at the START of the next `update()` frame on raw
                // pointer state — the header cells only arm a press here
                // (via `col_press_cb`, set up above this block).

                // --- The files table: ONE grid whose FIRST ROW is the header
                // and all following rows are data rows (scrollable). Because
                // both live in the same Grid over the same allocated rect,
                // every column title shares its exact column geometry with
                // the file-row cells underneath it. ---
                // `file_rows` is borrowed here — no other code mutates
                // `self.file_rows`/`self.column_layout` while the closure is
                // open; a clicked column title only records the request, and
                // the actual re-sort happens after the panel closure.
                let file_rows = &self.file_rows;
                // Clicks on sortable titles are recorded here and applied
                // after the whole CentralPanel closure (the layout is only
                // borrowed immutably inside).
                let mut sort_request: Option<Column> = None;
                // NOTE: `file_to_vlc` is declared in the enclosing scope of
                // `update()` (before the CentralPanel closure) so the deferred
                // spawn after the panel can read it.

                // BUGFIX (horizontal scrollbar): the list used to be hosted by
                // `ScrollArea::vertical()`, which explicitly disables
                // horizontal scrolling (`enable_hscrolling(false)`), so the
                // bottom scrollbar strip never appeared even when the columns
                // were wider than the panel. `ScrollArea::both()` enables both
                // axes; with `auto_shrink([false, false])` the content keeps
                // its natural (sum-of-columns) width, so whenever the table is
                // wider than the available area a horizontal bar shows up at
                // the bottom of the files window.
                egui::ScrollArea::both()
                    .id_salt(FILE_LIST_ID_SALT)
                    // BUGFIX (horizontal scroll while grabbing a header):
                    // with `drag_to_scroll` on, egui's ScrollArea allocates a
                    // `Sense::drag()` background and pans itself by
                    // `pointer.delta` whenever the primary button is held and
                    // the pointer moves — which is exactly what happens while
                    // dragging a column title (the header row lives inside
                    // this area). Disable it for the whole duration of a
                    // header capture; wheel/trackpad hscroll stays available
                    // (it is swallowed separately in `update()`).
                    // The "Open with VLC" popup does NOT need this switch:
                    // it is opened by a right-click (drag-scroll only tracks
                    // the primary button), and its menu paints in a
                    // Foreground-ordered Area that wins hit-testing over the
                    // drag background anyway.
                    .drag_to_scroll(!header_grab_now)
                    .auto_shrink([false, false])
                    .max_height(list_max_height)
                    .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysVisible)
                    .show(ui, |ui| {
                        // Collected inside the scroll closure, appended to
                        // `paths_to_load` after the closure returns.
                        let mut local_paths_to_load: Vec<PathBuf> = Vec::new();
                        // UX: set once if the pointer hovers any file-row
                        // cell this frame; checked after the grid closure to
                        // switch the cursor icon to a pointing hand.
                        let mut any_row_hovered = false;
                        // Row that was right-clicked THIS frame: (path, menu
                        // position). Captured inside the grid loop; the popup
                        // itself is painted after `grid.show` so it appears
                        // exactly once per frame above all rows.
                        let mut row_vlc_menu: Option<(PathBuf, egui::Pos2)> = None;

                        let grid = egui::Grid::new("file_table")
                            .num_columns(visible_cols.len())
                            .spacing([10.0, 4.0])
                            .striped(true);

                        grid.show(ui, |ui| {
                            // --- Header row: the FIRST row of the very same
                            // grid that holds every data row below. Each
                            // title cell is painted through the shared
                            // `draw_cell` helper with the same per-column
                            // width (`column_widths[ci]`) and the same grid
                            // spacing as the data cells, so it sits exactly
                            // in its own column of the common mesh. ---
                            for (ci, col) in visible_cols.iter().enumerate() {
                                let width = column_widths[ci];
                                let label = col.title();
                                let indicator = if sort_column == Some(*col) {
                                    if sort_ascending { " ▲" } else { " ▼" }
                                } else {
                                    ""
                                };
                                let sortable = col.sortable();
                                let col_copy = *col;
                                let col_key = col.key();

                                // Column drag-and-drop (manual state machine,
                                // see `handle_column_drag`, run at the end of
                                // update()). The press here only ARMS a drag
                                // candidate; the resolver turns it into a real
                                // drag once the pointer moves. egui's own drag
                                // responses (`drag_started/drag_stopped`)
                                // proved useless for header-to-header drops —
                                // they stop being reported once the pointer
                                // leaves the source widget, so a release over
                                // another header never fired a drop and titles
                                // felt un-grabbable.
                                draw_cell(ui, width, 24.0, |ui| {
                                    let response = if sortable {
                                        ui.add_sized(
                                            [width, 24.0],
                                            egui::Button::new(format!("{}{}", label, indicator)),
                                        )
                                    } else {
                                        ui.add_sized(
                                            [width, 24.0],
                                            egui::Label::new(egui::RichText::new(label).strong()),
                                        )
                                    };

                                    // Every header cell reports its exact rect
                                    // each frame (the drag resolver freezes the
                                    // full header geometry from these when it
                                    // arms the live drag on the next frame).
                                    let rect = response.rect;
                                    ctx.data_mut(|d| {
                                        d.insert_temp(hdr_cell_id(ci), rect)
                                    });

                                    // Arm the column-drag candidate on a FRESH
                                    // primary-button press inside this cell.
                                    // NOTE: egui's own `is_pointer_button_down_on`
                                    // only becomes true while the button is
                                    // HELD (after interaction-id registration),
                                    // which is one frame late for the resolver —
                                    // so arming uses raw pointer input directly:
                                    // pressed-this-frame + position within rect.
                                    let armed = ui.input(|i| {
                                        i.pointer.button_down(egui::PointerButton::Primary)
                                            && i.pointer.any_pressed()
                                            && i
                                                .pointer
                                                .hover_pos()
                                                .map_or(false, |p| rect.contains(p))
                                    });
                                    if armed {
                                        col_press_cb(col_key, cur_frame);
                                    }

                                    // A completed drag-release must not also
                                    // re-sort the column it dropped through.
                                    let click_suppressed =
                                        cur_frame == self.col_suppress_click_frame;
                                    if sortable && response.clicked() && !click_suppressed {
                                        sort_request = Some(col_copy);
                                    }
                                });
                            }
                            ui.end_row();

                            // --- Data rows ---
                            for (idx, row) in file_rows.iter().enumerate() {
                                if !filter_text.is_empty() && !row.name.to_lowercase().contains(&filter_text) {
                                    continue;
                                }

                                let is_selected = *selected_row == Some(idx);
                                let is_broken = row.status == audioscout_core::ProbeStatus::Broken;
                                let is_unavailable = row.status == audioscout_core::ProbeStatus::Unavailable;
                                let text_color = if is_broken {
                                    egui::Color32::from_rgb(255, 100, 100)
                                } else if is_selected {
                                    egui::Color32::from_rgb(100, 200, 255)
                                } else {
                                    ui.visuals().text_color()
                                };

                                // Track if this row was clicked
                                let mut row_clicked = false;
                                let mut row_double_clicked = false;
                                // Set once any cell of this row reports a
                                // fresh RIGHT-click (the opener for the
                                // "Open with VLC" popup). The menu itself is
                                // painted after the whole row loop, so it
                                // shows exactly once per frame on top of the
                                // grid.
                                let mut row_secondary_clicked = false;
                                // Rect of the row's Name cell — used as the
                                // ANCHOR for the popup so the menu stays
                                // glued to the row instead of chasing the
                                // pointer from frame to frame.
                                let mut row_anchor_rect: Option<egui::Rect> = None;

                                for (ci, col) in visible_cols.iter().enumerate() {
                                    let width = column_widths[ci];
                                    // Preview column needs more height for stereo (40px) + margins
                                    let cell_height = if *col == Column::Preview { 44.0 } else { 20.0 };

                                    let cell_rect = draw_cell(ui, width, cell_height, |ui| {
                                        if *col == Column::Preview {
                                            // Display preview image if available
                                            if let Some(uuid) = &row.preview_uuid {
                                                // Check cache first
                                                if let Some(texture) = preview_cache.get(uuid) {
                                                    // Use actual texture size
                                                    let size = texture.size();
                                                    ui.add(egui::Image::new(texture).fit_to_exact_size(egui::vec2(size[0] as f32, size[1] as f32)));
                                                } else {
                                                    // Load from disk and cache
                                                    let config_dir = audioscout_core::config::config_dir();
                                                    let previews_dir = config_dir.join("previews");
                                                    let png_path = previews_dir.join(format!("{}.png", uuid));

                                                    if png_path.exists() {
                                                        if let Ok(image_data) = std::fs::read(&png_path) {
                                                            if let Ok(image) = image::load_from_memory(&image_data) {
                                                                // Get actual image dimensions
                                                                let (img_width, img_height) = image.dimensions();
                                                                let rgba = image.to_rgba8();
                                                                let pixels = rgba.as_flat_samples();
                                                                let color_image = egui::ColorImage::from_rgba_unmultiplied(
                                                                    [img_width as usize, img_height as usize],
                                                                    pixels.as_slice(),
                                                                );
                                                                let texture = ui.ctx().load_texture(
                                                                    format!("preview_{}", uuid),
                                                                    color_image,
                                                                    egui::TextureOptions::LINEAR,
                                                                );
                                                                // Cache the texture
                                                                preview_cache.insert(uuid.clone(), texture.clone());
                                                                ui.add(egui::Image::new(&texture).fit_to_exact_size(egui::vec2(img_width as f32, img_height as f32)));
                                                            } else {
                                                                ui.label("⚠️");
                                                            }
                                                        } else {
                                                            ui.label("⚠️");
                                                        }
                                                    } else {
                                                        ui.label("⏳");
                                                    }
                                                }
                                            } else {
                                                ui.label("⏳");
                                            }
                                        } else {
                                            let text = match col {
                                                Column::Name => row.name.clone(),
                                                Column::Duration => format_duration(row.duration_secs),
                                                Column::SampleRate => format!("{} Hz", row.sample_rate),
                                                Column::SampleType => format!("{}", row.sample_type),
                                                Column::Channels => match row.channels {
                                                    1 => "Mono".to_string(),
                                                    2 => "Stereo".to_string(),
                                                    n => format!("{}ch", n),
                                                },
                                                Column::Size => format_size(row.size),
                                                Column::Comment => row.comment.clone().unwrap_or_default(),
                                                Column::DateCreated => format_timestamp(row.date_created),
                                                Column::DateModified => format_timestamp(row.date_modified),
                                                _ => String::new(),
                                            };

                                            if is_unavailable && *col == Column::Name {
                                                ui.colored_label(
                                                    egui::Color32::from_rgb(180, 180, 100),
                                                    format!("{} [unavailable]", text),
                                                );
                                            } else {
                                                ui.colored_label(text_color, text);
                                            }
                                        }
                                    });

                                    // Check if any cell in this row was
                                    // clicked: interact over the whole cell
                                    // rect so clicks between widgets still
                                    // select the row (same as before, when
                                    // every cell carried its own response).
                                    let cell_id = ui.id().with(("file_cell", idx, ci));
                                    let cell_response = ui.interact(cell_rect, cell_id, egui::Sense::click());

                                    // UX: hand cursor while hovering a file
                                    // row — the whole row is clickable
                                    // (single click selects, double click
                                    // loads), so it must look like a link.
                                    // NOTE: `Response::hovered()` is false
                                    // whenever some OTHER widget owns the
                                    // hover this frame — e.g. the label or
                                    // thumbnail inside the very same cell,
                                    // which are allocated before this
                                    // `interact()` call. Using it alone made
                                    // the pointer stay an arrow over most of
                                    // each row. `contains_pointer` ignores
                                    // that priority contest and answers the
                                    // real question ("is the mouse inside
                                    // this row?"), so it drives the cursor
                                    // here; `hovered()` still drives clicks.
                                    if cell_response.hovered() || cell_response.contains_pointer() {
                                        any_row_hovered = true;
                                    }

                                    // Check if any cell in this row was clicked
                                    if cell_response.clicked() {
                                        row_clicked = true;
                                    }
                                    if cell_response.double_clicked() {
                                        row_double_clicked = true;
                                    }
                                    // Right-click anywhere on the row opens
                                    // the per-file context menu (§ "Open with
                                    // VLC"). `secondary_clicked` is a fresh
                                    // press event, so it fires exactly once.
                                    if cell_response.secondary_clicked() {
                                        row_secondary_clicked = true;
                                    }
                                    // Remember the Name-cell rect as the
                                    // popup anchor (see `row_anchor_rect`).
                                    if *col == Column::Name {
                                        row_anchor_rect = Some(cell_rect);
                                    }
                                }

                                // Handle row selection
                                if row_clicked {
                                    *selected_row = Some(idx);
                                }

                                // Handle double-click to load file
                                if row_double_clicked && !is_broken && !is_unavailable {
                                    local_paths_to_load.push(row.path.clone());
                                }

                                // Per-row right-click opener for the popup:
                                // remember the path + the row's Name-cell
                                // anchor so the menu can be painted once,
                                // after the grid closure. The id is shared
                                // across all rows (`file_vlc_menu_id()` — one
                                // menu at a time), but keyed by the path so
                                // re-clicking another row re-claims it.
                                if row_secondary_clicked {
                                    // Anchor: bottom-left of the row's Name
                                    // cell. Fallback to the pointer (and then
                                    // to the panel corner) only guards
                                    // headless/edge frames where the Name
                                    // column is scrolled out of view.
                                    let anchor = row_anchor_rect
                                        .map(|r| r.left_bottom())
                                        .or_else(|| ui.input(|i| i.pointer.hover_pos()))
                                        .unwrap_or_else(|| ui.min_rect().left_top());
                                    row_vlc_menu = Some((row.path.clone(), anchor));
                                }

                                ui.end_row();
                            }
                        });

                        // --- Per-file right-click popup (§ "Open with VLC") ---
                        // Painted AFTER the grid so it sits on top of every
                        // row. BUGFIX (right-click showed no menu): the fresh
                        // `secondary_clicked` event is consumed by the cell's
                        // response in this very frame, so a synthetic opener
                        // rect built from the pointer position never reported
                        // a click and the popup was never claimed. Instead we
                        // store the clicked path + row anchor into egui data
                        // and paint the menu directly from that state; the
                        // close detection lives here too because the generic
                        // `context_menu` helper compares against the CURRENT
                        // frame's hover point (a 1x1 dummy rect) and closed
                        // the popup one frame after opening it.
                        let menu_claim = file_vlc_menu_id().with("claim");
                        if let Some((path, anchor)) = row_vlc_menu.take() {
                            // Only claim the popup from an IDLE pointer: a
                            // column-drag grab never reaches here (its press
                            // arms the header capture), but any held button
                            // means a gesture is in flight and must not be
                            // interrupted by a menu stealing input focus.
                            if !header_grab_now && !Self::any_pointer_button_down(ctx) {
                                ctx.data_mut(|d| {
                                    d.insert_temp(
                                        menu_claim,
                                        FileMenuState {
                                            path: path.clone(),
                                            anchor,
                                        },
                                    );
                                });
                                ctx.request_repaint();
                            }
                        }
                        let open_state = ctx.data(|d| d.get_temp::<FileMenuState>(menu_claim));
                        if let Some(state) = open_state {
                            let items = ["Open with VLC"];
                            let mut chosen: Option<&str> = None;
                            let frame = egui::Frame::menu(&ctx.style());
                            egui::Area::new(file_vlc_menu_id().with("menu"))
                                .fixed_pos(state.anchor)
                                .order(egui::Order::Foreground)
                                .show(ctx, |ui| {
                                    frame.show(ui, |ui| {
                                        ui.vertical(|ui| {
                                            for item in items.iter() {
                                                let resp = ui.add(
                                                    egui::Label::new(
                                                        egui::WidgetText::from(*item)
                                                            .color(ui.visuals().text_color()),
                                                    )
                                                    .selectable(false),
                                                );
                                                if resp.clicked() {
                                                    chosen = Some(*item);
                                                }
                                            }
                                        });
                                    });
                                });
                            // Close on Escape or on any primary press outside
                            // the popup itself (the popup area's own rects are
                            // checked via the label responses above).
                            let close = ctx.input(|i| {
                                i.key_pressed(egui::Key::Escape)
                                    || (i.pointer.any_pressed() && i.pointer.primary_down())
                            });
                            if close || chosen.is_some() {
                                ctx.data_mut(|d| d.remove::<FileMenuState>(menu_claim));
                            }
                            if chosen.is_some() {
                                file_to_vlc = Some(state.path.clone());
                            }
                        }

                        // Hand the collected double-click paths over to the
                        // outer `paths_to_load` (loaded after CentralPanel).
                        paths_to_load.append(&mut local_paths_to_load);

                        // UX: pointing-hand cursor over file rows (hover).
                        if any_row_hovered {
                            ui.output_mut(|o| o.cursor_icon = egui::CursorIcon::PointingHand);
                            ui.ctx().request_repaint();
                        }
                    });

                // A column title clicked in the header row: apply the sort
                // AFTER the whole panel closure has finished (the grid only
                // borrows `column_layout` immutably inside its closure, so a
                // click just records the request here).
                if let Some(col) = sort_request {
                    if self.column_layout.sort_column == Some(col) {
                        self.column_layout.sort_ascending = !self.column_layout.sort_ascending;
                    } else {
                        self.column_layout.sort_column = Some(col);
                        self.column_layout.sort_ascending = true;
                    }
                    self.column_layout.sort_rows(&mut self.file_rows);
                }
            }
        });

        // Load files AFTER the CentralPanel closure (no borrow conflict)
        for path in paths_to_load {
            self.load_file(&path);
        }

        // "Open with VLC" picked from a file row's right-click menu — spawned
        // here, outside the panel closures, because `open_in_vlc` needs
        // `&mut self` (§ context menu).
        if let Some(file) = file_to_vlc {
            self.open_in_vlc(&file, false);
        }









        // Settings dialog
        if self.show_settings {
            // Collect button actions locally first: `Window::open(&mut
            // self.show_settings)` borrows `*self` for the whole call, so the
            // closure cannot also hold `&mut self` (E0500). Copying the plain
            // fields out beforehand keeps the closure on disjoint fields only.
            #[derive(PartialEq)]
            enum SettingsAction {
                None,
                DeleteMarkers,
                DeletePreviews,
                CleanOrphans,
                ResetHotkeys,
                Close,
            }

            let mut skip_seconds_changed: Option<u32> = None;
            // "Browse…" clicked inside the window — the native folder picker
            // must run AFTER the egui frame (blocking rfd dialog inside the
            // closure would stall rendering / deadlock on some platforms).
            let mut browse_undo_dir = false;
            let mut action = SettingsAction::None;
            // Live-edited config fields collected during the frame and applied
            // after the window closes (same disjoint-borrow pattern).
            let mut undo_depth: usize = self.config_store.config().undo_depth;
            let mut undo_disk_limit_mb: usize = self.config_store.config().undo_disk_limit_mb;
            let mut undo_dir_text: String = self
                .config_store
                .config()
                .undo_dir
                .clone()
                .unwrap_or_else(audioscout_core::config::default_undo_dir);
            let mut preview_prefetch: usize = self.config_store.config().preview_prefetch;
            let mut space_stop = self.config_store.config().space_behavior_stop;
            let mut preview_mode_idx: usize =
                if self.config_store.config().preview_mode == "spectro" { 1 } else { 0 };
            let mut player_mode_idx: usize =
                if self.config_store.config().audio_view_mode == "spectro" { 1 } else { 0 };
            // Hotkey table snapshot + edit result (rebinding / capture state).
            // Rows follow `HOTKEY_ACTIONS` order (tz §9); the defaults map is
            // built from the same ordered list, so `remove()` here is O(1) on
            // the hash lookup and never misses.
            let default_hotkeys = audioscout_core::config::hotkey_defaults();
            let hotkey_rows: Vec<(String, audioscout_core::config::HotkeyBinding)> =
                audioscout_core::config::HOTKEY_ACTIONS
                    .iter()
                    .map(|a| {
                        (
                            a.to_string(),
                            self.config_store
                                .config()
                                .hotkeys
                                .get(*a)
                                .cloned()
                                .unwrap_or_else(|| default_hotkeys.get(*a).cloned().unwrap()),
                        )
                    })
                    .collect();
            let mut hotkey_rebind: Option<(String, audioscout_core::config::HotkeyBinding)> = None;
            let capture_action = self.settings_hotkey_capture.clone();
            let settings_status = self.settings_status.clone();
            // Read-only copy of the skip-seconds value (plain `Copy` field).
            // Reading it through `&self.*` inside the closure would conflict
            // with the mutable captures passed to `open()`/the UI closure.
            let current_skip_seconds = self.config_store.config().skip_seconds;
            // Volumeter settings mirrors (plain `Copy` fields — same disjoint
            // -borrow pattern as `current_skip_seconds`).  The peak-hold flag
            // is live-applied on change; the red threshold is persisted only.
            let mut meter_hold_enabled = self.config_store.config().meter_hold_enabled;
            // Signed value: the red level is negative dB up to 0.0 (tz §3.5.1).
            let mut meter_red_db = self.config_store.config().meter_red_db.min(0.0);
            // Peak-hold release delay in ms (plain `Copy` field mirror).
            let mut meter_hold_ms = self.config_store.config().meter_hold_ms;
            // Local mirrors of the two `self` fields the closure may touch:
            // the window-open flag (`Window::open`) and the hotkey-capture
            // toggle — both are written back after the frame.
            let mut show_settings = true;
            let mut capture_toggle = self.settings_hotkey_capture.clone();

            egui::Window::new("Settings")
                .collapsible(false)
                .resizable(true)
                .default_width(520.0)
                // Symmetric inner margins (per review): egui 0.29's Window
                // has no `inner_margin()` builder — its content margin comes
                // from the window `Frame` (style.visuals.window_margin), and
                // the default bottom margin gets an extra `item_spacing.y`,
                // so section frames (`ui.group`) end up closer to the
                // bottom/right edge than to the left/top one.  Override the
                // frame with all four sides pinned to the same value: the
                // frame's right inset then equals its left inset from the
                // window border.
                .frame(egui::Frame::window(&ctx.style()).inner_margin(egui::Margin {
                    left: 8.0,
                    right: 8.0,
                    top: 8.0,
                    bottom: 8.0,
                }))
                .open(&mut show_settings)
                .show(ctx, |ui| {
                    // The settings content outgrew the main window height as
                    // more sections were added (Storage / Undo history /
                    // Behaviour / Hotkeys table), so the whole body is now
                    // wrapped in a vertical ScrollArea: when the dialog is
                    // taller than the viewport it can be scrolled top↔bottom
                    // (wheel or the right-side slider).  `max_height` pins the
                    // scroll area to the space actually available inside the
                    // window (viewport minus the window's current top), leaving
                    // room for the fixed Close button rendered below it;
                    // `auto_shrink` keeps the width filling the window as
                    // before.  Horizontal scrolling stays off — every section
                    // fits the window width by design.
                    let win_top = ui.cursor().min.y;
                    let vp_h = ctx.input(|i| i.viewport().inner_rect.map_or(800.0, |r| r.height()));
                    let close_btn_reserve = ui.spacing().button_padding.y * 2.0
                        + ui.text_style_height(&egui::TextStyle::Button)
                        + ui.spacing().item_spacing.y
                        + 16.0; // separator + frame margins slack
                    let scroll_max_h = (vp_h - win_top - close_btn_reserve).max(160.0);

                    let scroll = egui::ScrollArea::vertical()
                        .id_salt("settings_body")
                        .auto_shrink([false, true])
                        .max_height(scroll_max_h);
                    scroll.show(ui, |ui| {
                    // Layout follows the Settings-window.avif mock-up in the
                    // repo root: bold section headers on their own line, one
                    // horizontal row per control, framed hotkey table with a
                    // footer button, full-width Close at the very bottom.
                    // Section headers render as an underlined heading across
                    // the full section width (per mock-up): bold label on its
                    // own line followed by a thin separator line.
                    let header = |ui: &mut egui::Ui, text: &str| {
                        ui.label(egui::RichText::new(text).strong());
                        // Separator line spanning the full section width,
                        // right under the heading (per mock-up).
                        let w = ui.available_width();
                        let x0 = ui.cursor().min.x;
                        let y = ui.cursor().min.y + 1.0;
                        ui.painter()
                            .hline(egui::Rangef::new(x0, x0 + w), y, (1.0, ui.visuals().weak_text_color()));
                        ui.add_space(6.0);
                    };

                    // Section frames must span the FULL window width (per
                    // review): `ui.group()` shrinks its frame to the content
                    // width, so each section's right edge sat at a different
                    // distance from the window border.  egui 0.29's `Frame`
                    // has no `min_size()`, but its painted rect is
                    // `content.min_rect() + inner_margin` — so an invisible
                    // zero-height strip that starts at the left edge and is
                    // as wide as the available area forces every section
                    // frame to the same full width (left and right insets
                    // become identical), while height still hugs content.
                    let section_frame = |ui: &mut egui::Ui| {
                        let w = ui.available_width();
                        ui.add_sized([w, 0.0], egui::Label::new("").selectable(false));
                        egui::Frame::group(ui.style())
                    };

                    // ---- Storage maintenance --------------------------------
                    section_frame(ui).show(ui, |ui| {
                        header(ui, "Storage maintenance");
                        ui.horizontal(|ui| {
                            // Plain text labels only — no icons (per review).
                            if ui.button("Delete all markers").clicked() { action = SettingsAction::DeleteMarkers; }
                            if ui.button("Clean orphaned previews").clicked() { action = SettingsAction::CleanOrphans; }
                            if ui.button("Delete all previews").clicked() { action = SettingsAction::DeletePreviews; }
                        });
                    });

                    // ---- Undo history ----------------------------------------
                    section_frame(ui).show(ui, |ui| {
                        header(ui, "Undo history");
                        ui.horizontal(|ui| {
                            ui.label("History depth (steps):");
                            // Two-digit number box per spec (1..=99); clicking
                            // the value lets you type it in directly.
                            ui.add(
                                egui::DragValue::new(&mut undo_depth)
                                    .range(1..=99)
                                    .speed(0.0)
                                    .update_while_editing(false),
                            );
                        });
                        ui.horizontal(|ui| {
                            ui.label("Disk limit:");
                            ui.add(
                                egui::DragValue::new(&mut undo_disk_limit_mb)
                                    .range(1..=1_048_576)
                                    .speed(0.0)
                                    .suffix(" MB")
                                    .update_while_editing(false),
                            );
                        });
                        ui.horizontal(|ui| {
                            ui.label("Undo data directory:");
                            // Editable box per tz; the mock-up shows the path
                            // in home-relative "~/..." form.  We edit a local
                            // display copy (tilde-prefixed) and expand "~"
                            // back to an absolute path when applying, so the
                            // config keeps a real filesystem path.
                            let home = dirs_next::home_dir()
                                .map(|h| h.to_string_lossy().into_owned())
                                .unwrap_or_default();
                            let mut display_dir = undo_dir_text
                                .strip_prefix(home.as_str())
                                .map(|rest| format!("~{rest}"))
                                .unwrap_or_else(|| undo_dir_text.clone());
                            let resp = ui.add(
                                egui::TextEdit::singleline(&mut display_dir)
                                    .desired_width((ui.available_width() - 90.0).max(120.0)),
                            );
                            if resp.changed() {
                                undo_dir_text = if let Some(rest) = display_dir.strip_prefix('~') {
                                    format!("{home}{rest}")
                                } else {
                                    display_dir
                                };
                            }
                            if ui.button("Browse…").clicked() {
                                browse_undo_dir = true;
                            }
                        });
                    });

                    // ---- Behaviour and appearance ----------------------------
                    section_frame(ui).show(ui, |ui| {
                        header(ui, "Behavior and appearance");
                        ui.horizontal(|ui| {
                            ui.label("Preview prefetch count:");
                            // Two-digit box per spec (0..=99).
                            ui.add(
                                egui::DragValue::new(&mut preview_prefetch)
                                    .range(0..=99)
                                    .speed(0.0)
                                    .update_while_editing(false),
                            );
                        });
                        ui.horizontal(|ui| {
                            ui.label("Space key behavior:");
                            egui::ComboBox::from_id_salt("space_key_behaviour")
                                .selected_text(if space_stop { "Start/Stop" } else { "Start/Pause" })
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut space_stop, false, "Start/Pause");
                                    ui.selectable_value(&mut space_stop, true, "Start/Stop");
                                });
                        });
                        ui.horizontal(|ui| {
                            ui.label("Table preview mode:");
                            egui::ComboBox::from_id_salt("table_previews_mode")
                                .selected_text(if preview_mode_idx == 1 { "Spectrogram" } else { "Waveform" })
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut preview_mode_idx, 0, "Waveform");
                                    ui.selectable_value(&mut preview_mode_idx, 1, "Spectrogram");
                                });
                        });
                        ui.horizontal(|ui| {
                            ui.label("Player display mode:");
                            egui::ComboBox::from_id_salt("player_display_mode")
                                .selected_text(if player_mode_idx == 1 { "Spectrogram" } else { "Waveform" })
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut player_mode_idx, 0, "Waveform");
                                    ui.selectable_value(&mut player_mode_idx, 1, "Spectrogram");
                                });
                        });
                        // ---- Volumeter options (tz §3.5.1) -------------------
                        ui.horizontal(|ui| {
                            ui.label("Volumeter enable peak hold");
                            ui.checkbox(&mut meter_hold_enabled, "")
                                .on_hover_text("Keep the peak marker on the volume meter for a while after the signal drops");
                        });
                        ui.horizontal(|ui| {
                            ui.label("Volumeter peak hold delay:");
                            // Up to four digits in ms; default 200.  Time the
                            // held peak is kept after the last new maximum,
                            // before it snaps back to the live level.
                            ui.add(
                                egui::DragValue::new(&mut meter_hold_ms)
                                    .range(0..=9999)
                                    .speed(10)
                                    .suffix(" ms"),
                            );
                        });
                        ui.horizontal(|ui| {
                            ui.label("Volumeter red value:");
                            // The red level is always *negative* dB (up to 0).
                            // Editing happens directly on the signed value so
                            // a positive number can never be entered; two
                            // digits + dot + one digit ("-99.9") per spec.
                            ui.add(
                                egui::DragValue::new(&mut meter_red_db)
                                    .range(-99.9..=0.0)
                                    .speed(0.1)
                                    .max_decimals(1)
                                    .min_decimals(1)
                                    .suffix(" dB")
                                    .update_while_editing(false),
                            );
                        });
                        ui.horizontal(|ui| {
                            ui.label("Skip seconds:");
                            let mut skip = current_skip_seconds;
                            if ui
                                .add(
                                    egui::DragValue::new(&mut skip)
                                        .range(1..=60)
                                        .speed(0.0)
                                        .update_while_editing(false),
                                )
                                .changed()
                            {
                                skip_seconds_changed = Some(skip);
                            }
                        });
                    });

                    // ---- Hotkeys ----------------------------------------------
                    section_frame(ui).show(ui, |ui| {
                        header(ui, "Hotkeys (click a shortcut to reassign)");
                        // Two-column table: Action | Shortcut.  The rows live
                        // in a vertical ScrollArea — mouse wheel and the
                        // slider on the right both scroll through it (§ tz).
                        let avail = ui.available_width();
                        let action_col = 240.0;
                        // Shortcut column fills the rest of the section width
                        // so the table (and its right edge / scrollbar) spans
                        // the frame uniformly (per review).
                        let shortcut_col = (avail - action_col - 22.0).max(80.0);
                        egui::Grid::new("hotkey_table")
                            .num_columns(2)
                            .striped(true)
                            .min_col_width(action_col)
                            .max_col_width(shortcut_col)
                            .show(ui, |ui| {
                                ui.label(egui::RichText::new("Action").strong());
                                ui.label(egui::RichText::new("Shortcut").strong());
                                ui.end_row();
                                for (action_name, binding) in &hotkey_rows {
                                    ui.add_sized([action_col, 18.0], egui::Label::new(audioscout_core::config::hotkey_label(action_name)));
                                    let label = if capture_action.as_deref() == Some(action_name.as_str()) {
                                        "Press keys…".to_string()
                                    } else {
                                        audioscout_core::config::format_hotkey(binding)
                                    };
                                    let resp = ui.add_sized(
                                        [shortcut_col, 18.0],
                                        egui::Button::new(
                                            // Non-interactive tint: `WidgetVisuals`
                                            // has no fg_color in this egui version,
                                            // the text color lives in `widgets.inactive.fg_stroke`.
                                            egui::RichText::new(label).monospace().color(ui.visuals().noninteractive().fg_stroke.color),
                                        )
                                        .frame(false),
                                    );
                                    if resp.clicked() {
                                        // Click a shortcut -> start capturing; click again cancels.
                                        capture_toggle =
                                            if capture_action.as_deref() == Some(action_name.as_str()) { None } else { Some(action_name.clone()) };
                                    }
                                    ui.end_row();
                                }
                            });
                        // ---- Keyboard capture for reassignment ----------------
                        // While a row is armed ("Press keys…"), the next key
                        // press becomes the new shortcut.  Escape cancels.
                        if capture_action.is_some() {
                            // Consume every key event this frame so the app's
                            // global hotkey dispatch (Space, Ctrl+Z, …) does
                            // not fire while a shortcut is being captured.
                            let mut captured: Option<audioscout_core::config::HotkeyBinding> = None;
                            let mut escape_pressed = false;
                            // `ui.input(|i| …)` hands out an immutable
                            // reference — mutating the event queue (to swallow
                            // keys during capture) needs `Input::mutation`.
                            ui.input_mut(|i| {
                                i.events.retain(|e| match e {
                                    egui::Event::Key { key, pressed: true | false, repeat: false, modifiers, physical_key: _ } => {
                                        if *key == egui::Key::Escape {
                                            escape_pressed = true;
                                            return false;
                                        }
                                        if captured.is_none() && matches!(e, egui::Event::Key { pressed: true, .. }) {
                                            let name = format!("{:?}", key);
                                            if audioscout_core::config::is_rebindable_key(&name) {
                                                captured = Some(audioscout_core::config::HotkeyBinding {
                                                    key: name,
                                                    ctrl: modifiers.ctrl,
                                                    shift: modifiers.shift,
                                                    alt: modifiers.alt,
                                                });
                                            }
                                        }
                                        false // drop the event: capture mode owns the keyboard
                                    }
                                    _ => true,
                                });
                            });
                            if escape_pressed {
                                // Escape cancels the reassignment.
                                capture_toggle = None;
                            } else if let Some(b) = captured {
                                if let Some(a) = &capture_action {
                                    hotkey_rebind = Some((a.clone(), b));
                                }
                                // Leave capture mode once a key was consumed.
                                capture_toggle = None;
                            }
                        }
                        // Reset button: normal (default) height, full width.
                        // `min_size(ui.available_size())` was wrong here — it
                        // stretched the button over all remaining vertical
                        // space of the section (per review).  Only the width
                        // is pinned; the height stays whatever egui's style
                        // gives a plain button.
                        if ui
                            .add(
                                egui::Button::new("Reset all to defaults")
                                    .min_size(egui::vec2(ui.available_width(), 0.0)),
                            )
                            .clicked()
                        {
                            action = SettingsAction::ResetHotkeys;
                        }
                        if !settings_status.is_empty() {
                            ui.label(egui::RichText::new(settings_status.clone()).weak());
                        }
                    });
                    }); // ---- end of the scrollable settings body ----------

                    // The separator + Close button live OUTSIDE the
                    // ScrollArea, so they stay pinned at the bottom of the
                    // dialog while the sections above scroll (per request:
                    // "проматываемое окно", Close — "в самом низу").
                    ui.separator();
                    // ---- Close -------------------------------------------------
                    // Full width, normal button height (same fix as the
                    // Reset button above — `available_size()` would stretch
                    // it over all leftover vertical space).
                    if ui
                        .add(
                            egui::Button::new("Close")
                                .min_size(egui::vec2(ui.available_width(), 0.0)),
                        )
                        .clicked()
                    {
                        action = SettingsAction::Close;
                    }
                });

            // Write back the local mirrors + apply every edited field to the
            // config store (all settings persist to ~/.config/audioscout/
            // config.json via `ConfigStore::save`).  The bottom "Close"
            // button sets `SettingsAction::Close`; otherwise `.open()` keeps
            // the mirror in sync with the window's X-button state.
            self.show_settings = show_settings && action != SettingsAction::Close;
            // Closing the window (toolbar toggle, X-button or Close) must not
            // silently discard edits made since the last save: persist them.
            if !self.show_settings {
                let _ = self.config_store.save();
            }
            self.settings_hotkey_capture = capture_toggle.clone();
            // Native folder picker for "Undo data directory" — runs outside
            // the egui frame so the blocking rfd dialog doesn't stall UI.
            if browse_undo_dir {
                if let Some(dir) = rfd::FileDialog::new()
                    .set_title("Undo data directory")
                    .set_directory(std::path::Path::new(&undo_dir_text))
                    .pick_folder()
                {
                    undo_dir_text = dir.to_string_lossy().into_owned();
                }
            }
            if let Some((ref a, ref b)) = hotkey_rebind {
                self.config_store.config_mut().hotkeys.insert(a.clone(), b.clone());
            }

            let mut changed = false;
            // Skip seconds (Behaviour): keep the pre-existing apply path.
            if let Some(skip) = skip_seconds_changed {
                self.config_store.config_mut().skip_seconds = skip;
                self.player.lock().unwrap_or_else(|p| p.into_inner()).set_skip_seconds(skip);
                changed = true;
            }
            if undo_depth != self.config_store.config().undo_depth {
                self.config_store.config_mut().undo_depth = undo_depth;
                self.undo_stack.set_max_depth(undo_depth);
                changed = true;
            }
            if undo_disk_limit_mb != self.config_store.config().undo_disk_limit_mb {
                self.config_store.config_mut().undo_disk_limit_mb = undo_disk_limit_mb;
                self.undo_stack.set_disk_limit_mb(undo_disk_limit_mb);
                changed = true;
            }
            // Undo dir: save the text as typed; the live stack keeps its
            // original directory until restart (dumps already written there).
            let trimmed_dir = undo_dir_text.trim().to_string();
            let new_undo_dir = if trimmed_dir == audioscout_core::config::default_undo_dir() { None } else { Some(trimmed_dir) };
            if new_undo_dir != self.config_store.config().undo_dir {
                self.config_store.config_mut().undo_dir = new_undo_dir;
                changed = true;
            }
            if preview_prefetch != self.config_store.config().preview_prefetch {
                self.config_store.config_mut().preview_prefetch = preview_prefetch;
                changed = true;
            }
            // Volumeter peak hold (Behavior and appearance): live flag, also
            // read by the meter update/draw paths straight from the config.
            if meter_hold_enabled != self.config_store.config().meter_hold_enabled {
                self.config_store.config_mut().meter_hold_enabled = meter_hold_enabled;
                changed = true;
            }
            // Peak-hold release delay (ms): read per meter frame from the
            // config, so persisting it here is all that's needed.
            if meter_hold_ms != self.config_store.config().meter_hold_ms {
                self.config_store.config_mut().meter_hold_ms = meter_hold_ms;
                changed = true;
            }
            // Volumeter red value: persisted to config.json as a negative dB
            // level.  The UI edits the signed value directly with a range of
            // -99.9..=0.0, so positive input is impossible; clamp again here
            // for safety.  Not yet wired into the meter drawing — per task,
            // this setting only saves.
            let new_red_db = meter_red_db.clamp(-99.9, 0.0);
            if (new_red_db - self.config_store.config().meter_red_db).abs() > f32::EPSILON {
                self.config_store.config_mut().meter_red_db = new_red_db;
                changed = true;
            }
            if space_stop != self.config_store.config().space_behavior_stop {
                self.config_store.config_mut().space_behavior_stop = space_stop;
                changed = true;
            }
            // Table preview mode / Player display mode are START-UP defaults
            // only: changing them here just persists the choice to
            // config.json.  The live UI (scanner preview mode, audio view
            // mode) is intentionally NOT touched — it keeps whatever the
            // user switched to during this session until the app restarts.
            if preview_mode_idx != if self.config_store.config().preview_mode == "spectro" { 1 } else { 0 } {
                let name = if preview_mode_idx == 1 { "spectro" } else { "wave" };
                self.config_store.config_mut().preview_mode = name.to_string();
                changed = true;
            }
            if player_mode_idx != if self.config_store.config().audio_view_mode == "spectro" { 1 } else { 0 } {
                let name = if player_mode_idx == 1 { "spectro" } else { "wave" };
                self.config_store.config_mut().audio_view_mode = name.to_string();
                changed = true;
            }
            if changed || hotkey_rebind.is_some() || action != SettingsAction::None {
                let _ = self.config_store.save();
            }

            // Status line inside the dialog: reset message (set in the match
            // below) or clear it once a rebind consumed the previous notice.
            if action != SettingsAction::ResetHotkeys && hotkey_rebind.is_some() {
                self.settings_status = String::new();
            }

            match action {
                SettingsAction::None | SettingsAction::Close => {}
                SettingsAction::ResetHotkeys => {
                    self.config_store.config_mut().hotkeys = audioscout_core::config::hotkey_defaults();
                    self.settings_hotkey_capture = None;
                    self.settings_status = "Hotkeys reset to defaults".to_string();
                    let _ = self.config_store.save();
                }
                SettingsAction::DeleteMarkers => {
                    let _ = self.db.delete_all_markers();
                    self.status_message = "All markers deleted".to_string();
                }
                SettingsAction::DeletePreviews => {
                    let _ = self.db.delete_all_previews();
                    // Also delete preview files from disk
                    let config_dir = audioscout_core::config::config_dir();
                    let previews_dir = config_dir.join("previews");
                    if previews_dir.exists() {
                        let _ = std::fs::remove_dir_all(&previews_dir);
                    }
                    // Clear in-memory texture caches (both modes)
                    self.preview_cache_wave.clear();
                    self.preview_cache_spectro.clear();
                    // Drop the stale UUIDs from the visible rows: they point at
                    // PNGs that no longer exist on disk. Without this,
                    // `request_previews()` skips every row (it only submits
                    // jobs for rows with `preview_uuid == None`) and the
                    // preview pump thinks nothing is pending — so regeneration
                    // never starts after deletion.
                    for row in &mut self.file_rows {
                        row.preview_uuid = None;
                    }
                    // Cancel any in-flight jobs rendering the deleted previews
                    // and reset bookkeeping so fresh jobs are submitted.
                    self.preview_worker.clear_queue();
                    self.preview_inflight.clear();
                    self.preview_retry.clear();
                    // Kick off regeneration for the current directory.
                    self.request_previews();
                    ctx.request_repaint();
                    self.status_message = "All previews deleted, regenerating…".to_string();
                }
                SettingsAction::CleanOrphans => {
                    match audioscout_core::preview_worker::cleanup_previews() {
                        Ok(count) => self.status_message = format!("Cleaned {} orphaned previews", count),
                        Err(e) => self.status_message = format!("Cleanup failed: {}", e),
                    }
                }
            }
        }

        if self.show_columns_dialog {
            egui::Window::new("Columns").open(&mut self.show_columns_dialog).show(ctx, |ui| {
                ui.label("Visible columns:");
                for col in Column::all() { let mut visible = self.column_layout.is_visible(col); if ui.checkbox(&mut visible, col.title()).changed() { self.column_layout.toggle_visibility(col); } }
            });
            // BUGFIX (persisted column selection): mirror the checkbox state
            // into config.json as soon as the dialog closes, so the choice
            // survives a restart (it is re-applied in `App::new` above;
            // `on_exit` also saves it as a fallback). The full snapshot
            // (`visible` + `order` + `full_order`) keeps the dragged column
            // order consistent with the new visibility set.
            if !self.show_columns_dialog {
                let layout = self.column_layout.clone();
                layout.store_into(&mut self.config_store.config_mut().columns);
                let _ = self.config_store.save();
            }
        }

        if let Some(dir) = &self.current_dir { let _ = self.db.save_session("lastDir", &dir.to_string_lossy()); }

        // NOTE: the column-drag state machine runs at the START of `update()`
        // (see `handle_column_drag` there): press transitions must be read
        // from the raw event list BEFORE any panel consumes them, and the
        // header geometry it freezes was reported by the widgets during THIS
        // frame's layout pass above.
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        // Shutdown preview worker to stop background thread
        self.preview_worker.shutdown();

        // Clear preview caches to free GPU memory
        self.preview_cache_wave.clear();
        self.preview_cache_spectro.clear();

        if let Some(dir) = &self.current_dir { let _ = self.db.save_session("lastDir", &dir.to_string_lossy()); }
        if let Some(file) = &self.active_file {
            let _ = self.db.save_session("lastFile", &file.to_string_lossy());
            let pos = self.player_current_secs();
            let _ = self.db.save_session("lastPos", &pos.to_string());
        }
        // Persist the Columns-dialog selection and the dragged column order
        // here too (fallback in case the app is closed while the dialog is
        // still open; normally they are saved right after every change).
        let layout = self.column_layout.clone();
        layout.store_into(&mut self.config_store.config_mut().columns);
        let _ = self.config_store.save();
    }
}

fn format_duration(secs: f64) -> String { if secs < 60.0 { format!("{:.1}s", secs) } else { let mins = (secs / 60.0) as u32; let s = (secs % 60.0) as u32; format!("{}:{:02}", mins, s) } }
fn format_size(bytes: u64) -> String { if bytes < 1024 { format!("{} B", bytes) } else if bytes < 1024 * 1024 { format!("{:.1} KB", bytes as f64 / 1024.0) } else { format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0)) } }
fn format_timestamp(ts: u64) -> String {
    if ts == 0 { return String::new(); }
    use chrono::{DateTime, Local};
    let datetime = DateTime::from_timestamp(ts as i64, 0)
        .map(|dt| dt.with_timezone(&Local))
        .unwrap_or_else(|| Local::now());
    // Format: DD-MM-YYYY HH:MM:SS (date with leading zeros, time with leading zeros)
    datetime.format("%d-%m-%Y %H:%M:%S").to_string()
}
fn format_time_hms(secs: f64) -> String { let total_secs = secs as u64; let h = total_secs / 3600; let m = (total_secs % 3600) / 60; let s = total_secs % 60; if h > 0 { format!("{}:{:02}:{:02}", h, m, s) } else { format!("{}:{:02}", m, s) } }
