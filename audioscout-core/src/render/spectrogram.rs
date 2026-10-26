//! Spectrogram preview rendering.
//!
//! Реализация по образцу `render::waveform`: превью для ячейки Preview
//! таблицы (340×20 px, тёмная тема). В отличие от волны, здесь нужен
//! частотно-временной анализ, поэтому используется собственный radix-2 FFT
//! (без внешних зависимостей): окно Hann, размер 512 сэмпла с шагом 256,
//! логарифмические по частоте бины (`t²`-маппинг) и heat colormap.

use image::{Rgba, RgbaImage};

/// In-place iterative radix-2 Cooley–Tukey FFT.
///
/// `re`/`im` must have power-of-two length; bit-reversal permutation is
/// applied first, then butterflies per stage.
fn fft_radix2(re: &mut [f32], im: &mut [f32]) {
    let n = re.len();
    if n <= 1 {
        return;
    }
    debug_assert!(n.is_power_of_two());

    // Bit-reversal permutation.
    let mut j: usize = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    // Butterflies.
    let mut len = 2;
    while len <= n {
        let ang = -core::f32::consts::TAU / len as f32;
        let (wr_step, wi_step) = (ang.cos(), ang.sin());
        let half = len / 2;
        let mut start = 0;
        while start < n {
            let (mut wr, mut wi) = (1.0f32, 0.0f32);
            for k in 0..half {
                let a = start + k;
                let b = a + half;
                let xr = re[b] * wr - im[b] * wi;
                let xi = re[b] * wi + im[b] * wr;
                re[b] = re[a] - xr;
                im[b] = im[a] - xi;
                re[a] += xr;
                im[a] += xi;
                // Advance twiddle factor incrementally.
                let nwr = wr * wr_step - wi * wi_step;
                wi = wr * wi_step + wi * wr_step;
                wr = nwr;
            }
            start += len;
        }
        len <<= 1;
    }
}

/// Map each of the `bins` output rows to a centre frequency via the
/// logarithmic `t²` curve (`t ∈ [0, 1]` → freq fraction `t²`), so low
/// frequencies get more vertical resolution — same visual model as the
/// classic spectrogram views.
fn log_bin_fractions(bins: usize) -> Vec<f32> {
    (0..bins)
        .map(|i| {
            let t = (bins - 1 - i) as f32 / (bins.saturating_sub(1)).max(1) as f32;
            (t * t).clamp(0.0, 1.0)
        })
        .collect()
}

/// Heat colormap: black → red → yellow → white.
fn heat(v: f32) -> Rgba<u8> {
    let v = v.clamp(0.0, 1.0);
    let r = (2.0 * v).min(1.0);
    let g = (2.0 * v - 0.65).clamp(0.0, 1.0);
    let b = if v > 0.85 { (3.0 * v - 2.55).clamp(0.0, 1.0) } else { v * v * 0.4 };
    Rgba([(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8, 255])
}

/// Render a spectrogram thumbnail covering the **entire** file.
///
/// - mono → whole canvas; stereo/multichannel → first two tracks stacked
///   vertically (same channel-selection rule as the waveform previews);
/// - window size 512 (radix-2), hop 256, Hann window;
/// - per-column magnitude normalised by the global max (dB-ish log scale).
///
/// The analysis hop is derived from the signal length (`total_frames /
/// width`), not fixed at 256: with a fixed hop only the first `width × 256`
/// frames were analysed and the picture stopped at the beginning of long
/// files. Now every one of the `width` columns averages a slice of the
/// timeline, so the whole track is visible regardless of duration.
pub fn render_spectrogram_preview(samples: &[f32], channels: u16, width: u32, height: u32) -> RgbaImage {
    const FFT_SIZE: usize = 512;

    let ch = (channels.max(1) as usize).min(2);
    let track_height = (height as usize) / ch;
    let bins = track_height.max(1);

    let mut img = RgbaImage::from_pixel(width, height, Rgba([10, 10, 14, 255]));

    // Hann window (periodic).
    let hann: Vec<f32> = (0..FFT_SIZE)
        .map(|i| 0.5 - 0.5 * (core::f32::consts::TAU * i as f32 / FFT_SIZE as f32).cos())
        .collect();

    let total_frames = samples.len() / ch.max(1);
    if total_frames < FFT_SIZE / 2 || width == 0 || height == 0 {
        return img;
    }

    // One analysis column per horizontal pixel; hop spans the full signal so
    // even a 3-hour file fits into 340 columns (each column then averages
    // several windows). Bounded below by the classic hop for short signals.
    let cols = width;
    let hop = (total_frames / cols as usize).max(FFT_SIZE / 4);

    let bin_fracs = log_bin_fractions(bins);

    // Pass 1: compute per-track magnitude maps, remember the global max.
    let mut track_maps: Vec<Vec<f32>> = Vec::with_capacity(ch);
    let mut global_max = 1e-9f32;

    for track in 0..ch {
        let mut map = vec![0.0f32; (cols as usize) * bins];
        for col in 0..cols as usize {
            // This column covers the timeline slice [slice_start, slice_end);
            // average several FFT windows spread over it so long files still
            // fit into `width` columns without dropping the tail.
            let last_start = total_frames.saturating_sub(FFT_SIZE);
            let slice_start = (col * hop).min(last_start);
            let slice_end = ((col + 1) * hop).clamp(slice_start + FFT_SIZE / 2, total_frames);
            let n_windows = ((slice_end - slice_start) / (FFT_SIZE / 2)).min(8).max(1);

            // Accumulate magnitudes across the column's windows.
            let mut acc = vec![0.0f32; bins];
            let mut re = vec![0.0f32; FFT_SIZE];
            let mut im = vec![0.0f32; FFT_SIZE];
            for w in 0..n_windows {
                let frame_start = if n_windows == 1 {
                    slice_start
                } else {
                    let span = (slice_end - slice_start - FFT_SIZE) as u64;
                    slice_start + (span * w as u64 / n_windows as u64) as usize
                };
                let frame_start = frame_start.min(last_start);
                // Fill the FFT buffers with windowed samples of this track.
                for i in 0..FFT_SIZE {
                    let frame = frame_start + i;
                    let s = if frame < total_frames {
                        samples[frame * ch + track]
                    } else {
                        0.0
                    };
                    re[i] = s * hann[i];
                    im[i] = 0.0;
                }
                fft_radix2(&mut re, &mut im);

                // Magnitudes at the log-spaced centre frequencies.
                let nyq_bin = FFT_SIZE / 2;
                for (bi, frac) in bin_fracs.iter().enumerate() {
                    let bin = ((*frac * (nyq_bin - 1) as f32).round() as usize).min(nyq_bin - 1);
                    acc[bi] += (re[bin] * re[bin] + im[bin] * im[bin]).sqrt();
                }
            }
            let inv = 1.0 / n_windows as f32;
            for bi in 0..bins {
                let mag = acc[bi] * inv;
                map[col * bins + bi] = mag;
                if mag > global_max {
                    global_max = mag;
                }
            }
        }
        track_maps.push(map);
    }

    // Pass 2: paint. Log-ish scaling makes quiet content visible.
    for (track, map) in track_maps.iter().enumerate() {
        let top_row = track * track_height;
        for col in 0..cols as usize {
            // Stretch/compress columns onto the full canvas width.
            let x = if cols as usize == width as usize {
                col
            } else {
                col * width as usize / cols as usize
            };
            for bi in 0..bins {
                let v = map[col * bins + bi] / global_max;
                let v = (v.ln().max(-6.0) + 6.0) / 6.0; // −60 dB floor → [0,1]
                // Row 0 = highest frequency → top of the track.
                let y = (top_row + bi).min((height as usize) - 1);
                img.put_pixel(x as u32, y as u32, heat(v));
            }
        }
    }

    // Track separator (matches the waveform style).
    if ch > 1 && height > 2 {
        let sep = track_height as u32;
        for x in 0..width {
            img.put_pixel(x, sep, Rgba([60, 60, 65, 255]));
        }
    }

    img
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_matches_known_signal() {
        // Pure sine at bin 8 of a 64-point FFT → peak magnitude at index 8.
        let n = 64;
        let mut re: Vec<f32> = (0..n)
            .map(|i| ((core::f32::consts::TAU * 8.0 * i as f32 / n as f32).cos()))
            .collect();
        let mut im = vec![0.0f32; n];
        fft_radix2(&mut re, &mut im);
        let mags: Vec<f32> = re.iter().zip(&im).map(|(r, i)| (r * r + i * i).sqrt()).collect();
        let peak = mags.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
        assert_eq!(peak.0, 8);
    }

    #[test]
    fn renders_mono_and_stereo() {
        let mono: Vec<f32> = (0..4000).map(|i| (i as f32 * 0.05).sin()).collect();
        let img = render_spectrogram_preview(&mono, 1, 340, 20);
        assert_eq!(img.dimensions(), (340, 20));

        let stereo: Vec<f32> = (0..4000).flat_map(|i| [(i as f32 * 0.05).sin(), (i as f32 * 0.1).cos()]).collect();
        let img = render_spectrogram_preview(&stereo, 2, 340, 40);
        assert_eq!(img.dimensions(), (340, 40));
    }
}
