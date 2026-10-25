//! Waveform preview rendering.
//!
//! Новая реализация генерации превью для окна списка файлов,
//! написанная по образцу CLI-примера из каталога `waveform-preview`:
//! - декомпозиция interleaved-сэмплов на planar-каналы;
//! - выбор отображаемых дорожек (моно → 1, стерео/мульти → первые 2);
//! - построение дерева Peaks для каждой дорожки;
//! - построчный рендер min/max (peak) или RMS-столбцов вокруг центральной
//!   линии дорожки с разделителями между дорожками (тёмная/светлая темы).

use image::{Rgba, RgbaImage};

use crate::peaks_tree::Peaks;

/// Rendering color palette (mirrors the Dark/Light styles of the
/// `waveform-preview` example).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Style {
    Dark,
    Light,
}

/// Amplitude reduction mode per pixel column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Min/max envelope per column.
    Peak,
    /// Symmetric RMS bar per column.
    Rms,
}

/// Options controlling how a preview waveform image is rendered.
pub struct RenderOptions {
    pub width: u32,
    pub height: u32,
    pub style: Style,
    pub mode: Mode,
}

impl Default for RenderOptions {
    fn default() -> Self {
        // Table preview cell size from the spec: 340x20 px, dark theme, peaks.
        Self {
            width: 340,
            height: 20,
            style: Style::Dark,
            mode: Mode::Peak,
        }
    }
}

/// Picks the channel indexes to render as tracks:
/// - 1 channel (mono)  → [0]
/// - 2 channels        → [0, 1]
/// - >2 channels       → [0, 1] (first two), same rule as the example
pub fn select_channels(total_channels: usize) -> Vec<usize> {
    match total_channels {
        0 => vec![],
        1 => vec![0],
        _ => vec![0, 1],
    }
}

/// Splits interleaved samples into per-channel planar vectors.
pub fn deinterleave(samples: &[f32], channels: usize) -> Vec<Vec<f32>> {
    let ch = channels.max(1);
    let frames = samples.len() / ch;
    let mut planar: Vec<Vec<f32>> = (0..ch).map(|_| Vec::with_capacity(frames)).collect();
    for frame in 0..frames {
        for ch_idx in 0..ch {
            let s = samples[frame * ch + ch_idx];
            planar[ch_idx].push(if s.is_finite() { s } else { 0.0 });
        }
    }
    planar
}

/// Renders a waveform preview image from interleaved audio samples.
///
/// This is the entry point used by the file-list preview worker:
/// mono files produce a single track, stereo/multichannel files produce
/// two stacked tracks with a separator line between them.
pub fn render_waveform(
    samples: &[f32],
    channels: u16,
    width: u32,
    height: u32,
    style: Style,
    mode: Mode,
) -> RgbaImage {
    let planar = deinterleave(samples, channels as usize);
    let selected = select_channels(planar.len());
    let channel_slices: Vec<&[f32]> =
        selected.iter().map(|&ch_idx| planar[ch_idx].as_slice()).collect();
    let peaks_list: Vec<Peaks> =
        selected.iter().map(|&ch_idx| Peaks::build(&planar[ch_idx])).collect();

    let opts = RenderOptions {
        width,
        height,
        style,
        mode,
    };
    render_tracks(&channel_slices, &peaks_list, &opts)
}

/// Convenience wrapper matching the default table-preview options.
pub fn render_waveform_preview(samples: &[f32], channels: u16) -> RgbaImage {
    let opts = RenderOptions::default();
    render_waveform(
        samples,
        channels,
        opts.width,
        opts.height,
        opts.style,
        opts.mode,
    )
}

/// Renders a waveform preview image directly from a precomputed min/max
/// envelope (see `crate::audio_io::Envelope`).
///
/// Same visual output as `render_waveform` in `Mode::Peak`, but without
/// materialising or re-reducing the full sample signal — this is the fast
/// path used by the file-list preview worker for large files.
pub fn render_envelope(env: &crate::audio_io::Envelope, width: u32, height: u32, style: Style) -> RgbaImage {
    // Which channel tracks to display: mono → 1, stereo/multichannel → first 2.
    let selected = select_channels(env.channels);
    let n = env.frames;

    let mut img = RgbaImage::new(width, height);
    let (bg, fg, grid, separator) = palette(style);

    for pixel in img.pixels_mut() {
        *pixel = bg;
    }
    if selected.is_empty() || width == 0 || height == 0 || n == 0 {
        return img;
    }

    let num_tracks = selected.len();
    let track_height = height / num_tracks as u32;
    if track_height == 0 {
        return img;
    }

    for (track_idx, &ch_idx) in selected.iter().enumerate() {
        let y_start = track_idx as u32 * track_height;
        let y_end = if track_idx == num_tracks - 1 {
            height
        } else {
            (track_idx as u32 + 1) * track_height
        };
        let track_h = y_end - y_start;

        if track_idx > 0 {
            for x in 0..width {
                *img.get_pixel_mut(x, y_start) = separator;
            }
        }

        let cy = y_start + track_h / 2;
        for x in 0..width {
            *img.get_pixel_mut(x, cy) = grid;
        }

        // Map envelope frames onto pixel columns, merging min/max per column
        // exactly like Peaks::spans does for the raw-signal path.
        let lo = &env.lo[ch_idx * n..(ch_idx + 1) * n];
        let hi = &env.hi[ch_idx * n..(ch_idx + 1) * n];

        let margin = 2u32.min(track_h / 4);
        let draw_h = track_h.saturating_sub(margin * 2) as f32;

        for x in 0..width {
            let a = (x as usize * n) / width as usize;
            let b = (((x + 1) as usize * n) / width as usize).max(a + 1).min(n);
            let mut col_lo = f32::INFINITY;
            let mut col_hi = f32::NEG_INFINITY;
            for i in a..b {
                if lo[i] < col_lo { col_lo = lo[i]; }
                if hi[i] > col_hi { col_hi = hi[i]; }
            }
            if col_lo > col_hi {
                continue; // empty column: leave the grid line visible
            }
            let y_top = cy as f32 - col_hi.clamp(-1.0, 1.0) * (draw_h / 2.0);
            let y_bot = cy as f32 - col_lo.clamp(-1.0, 1.0) * (draw_h / 2.0);
            let y_top = y_top.max((y_start + margin) as f32) as u32;
            let y_bot = (y_bot as u32).min(y_end.saturating_sub(1));
            for y in y_top..=y_bot {
                *img.get_pixel_mut(x, y) = fg;
            }
        }
    }

    img
}

/// Color palette shared by both renderers (mirrors the Dark/Light styles of
/// the `waveform-preview` example).
fn palette(style: Style) -> (Rgba<u8>, Rgba<u8>, Rgba<u8>, Rgba<u8>) {
    match style {
        Style::Dark => (
            Rgba([13, 18, 32, 255]),
            Rgba([16, 185, 129, 255]),
            Rgba([31, 41, 55, 255]),
            Rgba([55, 65, 81, 255]),
        ),
        Style::Light => (
            Rgba([255, 255, 255, 255]),
            Rgba([5, 150, 105, 255]),
            Rgba([229, 231, 235, 255]),
            Rgba([209, 213, 219, 255]),
        ),
    }
}

/// Core renderer following the `waveform-preview` example: draws one track
/// per channel slice, each with its own center line and peak/RMS columns.
fn render_tracks(channels: &[&[f32]], peaks_list: &[Peaks], opts: &RenderOptions) -> RgbaImage {
    let w = opts.width;
    let h = opts.height;
    let mut img = RgbaImage::new(w, h);

    let (bg, fg, grid, separator) = palette(opts.style);

    // Background
    for pixel in img.pixels_mut() {
        *pixel = bg;
    }

    if channels.is_empty() || w == 0 || h == 0 {
        return img;
    }

    let num_tracks = channels.len();
    let track_height = h / num_tracks as u32;
    if track_height == 0 {
        return img;
    }

    for (track_idx, (samples, peaks)) in channels.iter().zip(peaks_list.iter()).enumerate() {
        let y_start = track_idx as u32 * track_height;
        let y_end = if track_idx == num_tracks - 1 {
            h
        } else {
            (track_idx as u32 + 1) * track_height
        };
        let track_h = y_end - y_start;

        // Separator between tracks (except before the first one)
        if track_idx > 0 {
            for x in 0..w {
                *img.get_pixel_mut(x, y_start) = separator;
            }
        }

        // Track center line
        let cy = y_start + track_h / 2;
        for x in 0..w {
            *img.get_pixel_mut(x, cy) = grid;
        }

        if samples.is_empty() {
            continue;
        }

        // One min/max (or RMS) column per horizontal pixel
        let step = samples.len() as f64 / w as f64;
        let columns: Vec<(f32, f32)> = match opts.mode {
            Mode::Peak => peaks.spans(samples, 0.0, step, w as usize),
            Mode::Rms => peaks
                .rms_spans(samples, 0.0, step, w as usize)
                .into_iter()
                .map(|r| (-r, r))
                .collect(),
        };

        let margin = 2u32.min(track_h / 4);
        let draw_h = track_h.saturating_sub(margin * 2) as f32;
        for (x, &(lo, hi)) in columns.iter().enumerate() {
            let x = x as u32;
            if x >= w {
                break;
            }
            let y_top = cy as f32 - hi.clamp(-1.0, 1.0) * (draw_h / 2.0);
            let y_bot = cy as f32 - lo.clamp(-1.0, 1.0) * (draw_h / 2.0);
            let y_top = y_top.max((y_start + margin) as f32) as u32;
            let y_bot = (y_bot as u32).min(y_end.saturating_sub(1));
            for y in y_top..=y_bot {
                *img.get_pixel_mut(x, y) = fg;
            }
        }
    }

    img
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_empty() {
        let img = render_waveform(&[], 1, 340, 20, Style::Dark, Mode::Peak);
        assert_eq!(img.width(), 340);
        assert_eq!(img.height(), 20);
    }

    #[test]
    fn test_render_sine_mono() {
        let samples: Vec<f32> = (0..44100)
            .map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 44100.0).sin() * 0.5)
            .collect();
        let img = render_waveform(&samples, 1, 340, 20, Style::Dark, Mode::Peak);
        assert_eq!(img.dimensions(), (340, 20));
        // Some pixels must be painted with the foreground color
        assert!(img.pixels().any(|p| p == &Rgba([16, 185, 129, 255])));
    }

    #[test]
    fn test_render_stereo_two_tracks() {
        // Interleaved stereo: L = +0.5, R = -0.5
        let samples: Vec<f32> = (0..44100)
            .flat_map(|_| [0.5f32, -0.5])
            .collect();
        let img = render_waveform(&samples, 2, 340, 40, Style::Dark, Mode::Peak);
        assert_eq!(img.dimensions(), (340, 40));
        // Separator row between the two tracks at y = 20
        assert_eq!(img.get_pixel(10, 20), &Rgba([55, 65, 81, 255]));
    }

    #[test]
    fn test_rms_mode() {
        let samples: Vec<f32> = (0..44100)
            .map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 44100.0).sin() * 0.5)
            .collect();
        let img = render_waveform(&samples, 1, 340, 20, Style::Light, Mode::Rms);
        assert_eq!(img.dimensions(), (340, 20));
    }

    #[test]
    fn test_deinterleave() {
        let interleaved = vec![0.1, 0.2, 0.3, 0.4];
        let planar = deinterleave(&interleaved, 2);
        assert_eq!(planar.len(), 2);
        assert_eq!(planar[0], vec![0.1, 0.3]);
        assert_eq!(planar[1], vec![0.2, 0.4]);
    }

    #[test]
    fn test_select_channels() {
        assert_eq!(select_channels(1), vec![0]);
        assert_eq!(select_channels(2), vec![0, 1]);
        assert_eq!(select_channels(6), vec![0, 1]);
        assert!(select_channels(0).is_empty());
    }
}
