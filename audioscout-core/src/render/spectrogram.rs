use image::{ImageBuffer, Rgba, RgbaImage};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;

const FFT_SIZE: usize = 512;

pub fn render_spectrogram(samples: &[f32], _sample_rate: u32, channels: u16, width: u32, height: u32) -> RgbaImage {
    let ch = channels.max(1) as usize;
    let total_frames = if ch == 0 { 0 } else { samples.len() / ch };
    let mut img = ImageBuffer::new(width, height);
    for pixel in img.pixels_mut() { *pixel = Rgba([20, 20, 25, 255]); }
    if total_frames < FFT_SIZE || width == 0 || height == 0 { return img; }
    let mono: Vec<f32> = (0..total_frames).map(|i| samples[i * ch]).collect();
    let hop_size = (total_frames / width as usize).max(1);
    let freq_bins = FFT_SIZE / 2;
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);
    let window: Vec<f32> = (0..FFT_SIZE).map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / FFT_SIZE as f32).cos())).collect();
    for x in 0..width as usize {
        let center = x * hop_size + FFT_SIZE / 2;
        if center + FFT_SIZE / 2 > total_frames { break; }
        let mut fft_buffer: Vec<Complex<f32>> = (0..FFT_SIZE).map(|i| {
            let sample_idx = center - FFT_SIZE / 2 + i;
            let val = if sample_idx < total_frames { mono[sample_idx] * window[i] } else { 0.0 };
            Complex::new(val, 0.0)
        }).collect();
        fft.process(&mut fft_buffer);
        for y in 0..height as usize {
            let t = y as f32 / height as f32;
            let freq_bin = (t * t * freq_bins as f32) as usize;
            let freq_bin = freq_bin.min(freq_bins - 1);
            let magnitude = fft_buffer[freq_bin].norm();
            let db = 20.0 * magnitude.max(1e-10).log10();
            let normalized = ((db + 80.0) / 80.0).clamp(0.0, 1.0);
            let color = heat_colormap(normalized as f32);
            img.put_pixel(x as u32, (height as usize - 1 - y) as u32, color);
        }
    }
    img
}

pub fn render_spectrogram_view(samples: &[f32], _sample_rate: u32, channels: u16, start_frame: u64, end_frame: u64, width: u32, height: u32) -> RgbaImage {
    let ch = channels.max(1) as usize;
    let view_len = (end_frame - start_frame) as usize;
    let mut img = ImageBuffer::new(width, height);
    for pixel in img.pixels_mut() { *pixel = Rgba([30, 30, 35, 255]); }
    if view_len < FFT_SIZE || width == 0 || height == 0 { return img; }
    let mono: Vec<f32> = (start_frame as usize..end_frame as usize).map(|i| {
        let idx = i * ch; if idx < samples.len() { samples[idx] } else { 0.0 }
    }).collect();
    let hop_size = (view_len / width as usize).max(1);
    let freq_bins = FFT_SIZE / 2;
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);
    let window: Vec<f32> = (0..FFT_SIZE).map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / FFT_SIZE as f32).cos())).collect();
    for x in 0..width as usize {
        let center = x * hop_size + FFT_SIZE / 2;
        if center + FFT_SIZE / 2 > view_len { break; }
        let mut fft_buffer: Vec<Complex<f32>> = (0..FFT_SIZE).map(|i| {
            let sample_idx = center - FFT_SIZE / 2 + i;
            let val = if sample_idx < view_len { mono[sample_idx] * window[i] } else { 0.0 };
            Complex::new(val, 0.0)
        }).collect();
        fft.process(&mut fft_buffer);
        for y in 0..height as usize {
            let t = y as f32 / height as f32;
            let freq_bin = (t * t * freq_bins as f32) as usize;
            let freq_bin = freq_bin.min(freq_bins - 1);
            let magnitude = fft_buffer[freq_bin].norm();
            let db = 20.0 * magnitude.max(1e-10).log10();
            let normalized = ((db + 80.0) / 80.0).clamp(0.0, 1.0);
            let color = heat_colormap(normalized as f32);
            img.put_pixel(x as u32, (height as usize - 1 - y) as u32, color);
        }
    }
    img
}

fn heat_colormap(value: f32) -> Rgba<u8> {
    let v = value.clamp(0.0, 1.0);
    let (r, g, b) = if v < 0.25 { let t = v / 0.25; (0.0, 0.0, t) }
    else if v < 0.5 { let t = (v - 0.25) / 0.25; (t, 0.0, 1.0 - t * 0.5) }
    else if v < 0.75 { let t = (v - 0.5) / 0.25; (1.0, t, 0.5 - t * 0.5) }
    else { let t = (v - 0.75) / 0.25; (1.0, 0.5 + t * 0.5, t) };
    Rgba([(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8, 255])
}
