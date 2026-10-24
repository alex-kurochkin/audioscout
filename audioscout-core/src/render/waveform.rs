use image::{ImageBuffer, Rgba, RgbaImage};

pub fn render_waveform(samples: &[f32], sample_rate: u32, channels: u16, width: u32, height: u32) -> RgbaImage {
    let ch = channels.max(1) as usize;
    let total_frames = if ch == 0 { 0 } else { samples.len() / ch };
    let mut img = ImageBuffer::new(width, height);
    for pixel in img.pixels_mut() { *pixel = Rgba([20, 20, 25, 255]); }
    if total_frames == 0 || width == 0 || height == 0 { return img; }
    let frames_per_pixel = (total_frames as f64 / width as f64).max(1.0);
    let mid_y = height as f32 / 2.0;
    let wave_color = Rgba([50, 200, 80, 255]);
    for x in 0..width {
        let frame_start = (x as f64 * frames_per_pixel) as usize;
        let frame_end = ((x as f64 + 1.0) * frames_per_pixel).min(total_frames as f64) as usize;
        if frame_start >= total_frames { break; }
        let mut min_val: f32 = 0.0; let mut max_val: f32 = 0.0;
        for frame in frame_start..frame_end {
            let sample_idx = frame * ch;
            if sample_idx < samples.len() {
                let val = samples[sample_idx];
                if val < min_val { min_val = val; }
                if val > max_val { max_val = val; }
            }
        }
        let y_min = (mid_y - max_val * mid_y).clamp(0.0, height as f32 - 1.0) as u32;
        let y_max = (mid_y - min_val * mid_y).clamp(0.0, height as f32 - 1.0) as u32;
        for y in y_min..=y_max { if y < height { img.put_pixel(x, y, wave_color); } }
    }
    img
}

pub fn render_waveform_view(samples: &[f32], sample_rate: u32, channels: u16, start_frame: u64, end_frame: u64, width: u32, height: u32, stereo: bool) -> RgbaImage {
    let ch = channels.max(1) as usize;
    let total_frames = if ch == 0 { 0 } else { samples.len() / ch };
    let view_frames = (end_frame - start_frame) as usize;
    let row_height = if stereo { height / 2 } else { height };
    let mut img = ImageBuffer::new(width, height);
    for pixel in img.pixels_mut() { *pixel = Rgba([30, 30, 35, 255]); }
    if view_frames == 0 || width == 0 || total_frames == 0 { return img; }
    let frames_per_pixel = (view_frames as f64 / width as f64).max(1.0);
    let wave_color = Rgba([50, 220, 80, 255]);
    for row in 0..if stereo { 2 } else { 1 } {
        let row_offset = row * row_height as usize;
        let channel = row;
        let mid_y = row_offset as f32 + row_height as f32 / 2.0;
        for x in 0..width {
            let frame_start = start_frame as usize + (x as f64 * frames_per_pixel) as usize;
            let frame_end = (start_frame as f64 + (x as f64 + 1.0) * frames_per_pixel).min(end_frame as f64) as usize;
            let mut min_val: f32 = 0.0; let mut max_val: f32 = 0.0;
            for frame in frame_start..frame_end {
                let sample_idx = frame * ch + channel.min(ch - 1);
                if sample_idx < samples.len() {
                    let val = samples[sample_idx];
                    if val < min_val { min_val = val; }
                    if val > max_val { max_val = val; }
                }
            }
            let y_min = (mid_y - max_val * (row_height as f32 / 2.0)).clamp(row_offset as f32, (row_offset + row_height as usize - 1) as f32) as u32;
            let y_max = (mid_y - min_val * (row_height as f32 / 2.0)).clamp(row_offset as f32, (row_offset + row_height as usize - 1) as f32) as u32;
            for y in y_min..=y_max { if y < height { img.put_pixel(x, y, wave_color); } }
        }
        let center = mid_y as u32;
        if center < height { for x in 0..width { img.put_pixel(x, center, Rgba([60, 60, 65, 255])); } }
    }
    img
}
