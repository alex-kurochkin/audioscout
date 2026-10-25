use image::{ImageBuffer, Rgba, RgbaImage};
use crate::peaks_tree::Peaks;

/// Render waveform preview image using Peaks tree for fast rendering
/// Supports mono (1 track) and stereo/multichannel (2 tracks)
pub fn render_waveform(samples: &[f32], sample_rate: u32, channels: u16, width: u32, height: u32) -> RgbaImage {
    let ch = channels.max(1) as usize;
    let total_frames = if ch == 0 { 0 } else { samples.len() / ch };

    // Determine number of visible tracks: 1 for mono, 2 for stereo/multichannel
    let num_tracks = if ch == 1 { 1 } else { 2 };

    let mut img = ImageBuffer::new(width, height);

    // Dark background
    for pixel in img.pixels_mut() {
        *pixel = Rgba([13, 18, 32, 255]);
    }

    if total_frames == 0 || width == 0 || height == 0 {
        return img;
    }

    // Green waveform color
    let fg = Rgba([16, 185, 129, 255]);
    let separator_color = Rgba([31, 41, 55, 255]);

    let track_height = height / num_tracks as u32;
    let margin = 2u32;
    let draw_h = track_height.saturating_sub(margin * 2) as f32;

    for track_idx in 0..num_tracks {
        let track_top = track_idx as u32 * track_height;
        let track_mid_y = track_top as f32 + track_height as f32 / 2.0;

        // Draw separator line between tracks (except before first)
        if track_idx > 0 {
            for x in 0..width {
                *img.get_pixel_mut(x, track_top) = separator_color;
            }
        }

        // Extract channel data from interleaved samples
        let channel_idx = track_idx.min(ch - 1);
        let channel_ Vec<f32> = (0..total_frames)
            .map(|frame| {
                let sample_idx = frame * ch + channel_idx;
                if sample_idx < samples.len() {
                    samples[sample_idx]
                } else {
                    0.0
                }
            })
            .collect();

        // Build peaks tree for this channel
        let peaks = Peaks::build(&channel_data);

        let step = channel_data.len() as f64 / width as f64;
        let columns = peaks.spans(&channel_data, 0.0, step, width as usize);

        for (x, &(lo, hi)) in columns.iter().enumerate() {
            let x = x as u32;
            let y_top = track_mid_y - hi.clamp(-1.0, 1.0) * (draw_h / 2.0);
            let y_bot = track_mid_y - lo.clamp(-1.0, 1.0) * (draw_h / 2.0);
            let y_top = y_top.max((track_top + margin) as f32) as u32;
            let y_bot = (y_bot as u32).min((track_top + track_height).saturating_sub(1));

            for y in y_top..=y_bot {
                *img.get_pixel_mut(x, y) = fg;
            }
        }
    }

    img
}

/// Render waveform for audio view (higher resolution, with stereo support)
pub fn render_waveform_view(
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
    start_frame: u64,
    end_frame: u64,
    width: u32,
    height: u32,
    stereo: bool,
) -> RgbaImage {
    let ch = channels.max(1) as usize;
    let total_frames = if ch == 0 { 0 } else { samples.len() / ch };
    let view_frames = (end_frame - start_frame) as usize;

    let row_height = if stereo { height / 2 } else { height };
    let mut img = ImageBuffer::new(width, height);

    // Dark background
    for pixel in img.pixels_mut() {
        *pixel = Rgba([30, 30, 35, 255]);
    }

    if view_frames == 0 || width == 0 || total_frames == 0 {
        return img;
    }

    let fg = Rgba([50, 220, 80, 255]);

    for row in 0..if stereo { 2 } else { 1 } {
        let row_offset = row * row_height as usize;
        let channel = row;

        // Extract channel data
        let channel_samples: Vec<f32> = (0..total_frames)
            .map(|i| {
                let idx = i * ch + channel.min(ch - 1);
                if idx < samples.len() { samples[idx] } else { 0.0 }
            })
            .collect();

        // Build peaks tree for this channel
        let peaks = Peaks::build(&channel_samples);

        let mid_y = row_offset as f32 + row_height as f32 / 2.0;
        let step = view_frames as f64 / width as f64;
        let start = start_frame as f64;

        let columns = peaks.spans(&channel_samples, start, step, width as usize);

        let margin = 2u32;
        let draw_h = row_height.saturating_sub(margin * 2) as f32;

        for (x, &(lo, hi)) in columns.iter().enumerate() {
            let x = x as u32;
            let y_top = mid_y - hi.clamp(-1.0, 1.0) * (draw_h / 2.0);
            let y_bot = mid_y - lo.clamp(-1.0, 1.0) * (draw_h / 2.0);
            let y_top = y_top.max((row_offset + margin as usize) as f32) as u32;
            let y_bot = (y_bot as u32).min((row_offset + row_height as usize).saturating_sub(1) as u32);

            for y in y_top..=y_bot {
                if y < height {
                    *img.get_pixel_mut(x, y) = fg;
                }
            }
        }

        // Draw center line
        let center = mid_y as u32;
        if center < height {
            for x in 0..width {
                img.put_pixel(x, center, Rgba([60, 60, 65, 255]));
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
        let img = render_waveform(&[], 44100, 1, 340, 20);
        assert_eq!(img.width(), 340);
        assert_eq!(img.height(), 20);
    }

    #[test]
    fn test_render_sine() {
        let samples: Vec<f32> = (0..44100)
            .map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 44100.0).sin() * 0.5)
            .collect();
        let img = render_waveform(&samples, 44100, 1, 340, 20);
        assert_eq!(img.width(), 340);
    }
}
