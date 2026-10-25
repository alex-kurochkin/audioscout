use clap::Parser;
use image::{Rgba, RgbaImage};
use std::fs;
use std::path::PathBuf;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

const BLOCK: usize = 32;
const FANOUT: usize = 4;

// ─── Peaks tree ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
struct Block {
    min: f32,
    max: f32,
    squares: f32,
}

impl Block {
    const EMPTY: Block = Block {
        min: f32::INFINITY,
        max: f32::NEG_INFINITY,
        squares: 0.0,
    };

    fn point(s: f32) -> Self {
        Block { min: s, max: s, squares: if s.is_finite() { s * s } else { 0.0 } }
    }

    fn of(samples: &[f32]) -> Self {
        samples.iter().fold(Self::EMPTY, |b, &s| b.merge(Self::point(s)))
    }

    fn merge(self, o: Block) -> Block {
        Block {
            min: self.min.min(o.min),
            max: self.max.max(o.max),
            squares: self.squares + o.squares,
        }
    }

    fn range(self) -> (f32, f32) {
        if self.min > self.max { (0.0, 0.0) } else { (self.min, self.max) }
    }

    fn rms(self, count: usize) -> f32 {
        if count == 0 { return 0.0; }
        (self.squares / count as f32).sqrt()
    }
}

struct Peaks {
    len: usize,
    levels: Vec<Vec<Block>>,
}

impl Peaks {
    fn build(samples: &[f32]) -> Self {
        if samples.is_empty() {
            return Peaks { len: 0, levels: vec![] };
        }
        let mut levels: Vec<Vec<Block>> = vec![
            samples.chunks(BLOCK).map(Block::of).collect()
        ];
        while let Some(last) = levels.last().filter(|l| l.len() > 1) {
            let next = last.chunks(FANOUT)
                .map(|c| c.iter().fold(Block::EMPTY, |a, &b| a.merge(b)))
                .collect();
            levels.push(next);
        }
        Peaks { len: samples.len(), levels }
    }

    fn minmax(&self, samples: &[f32], start: usize, end: usize) -> (f32, f32) {
        self.sum(samples, start, end).range()
    }

    fn sum(&self, samples: &[f32], start: usize, end: usize) -> Block {
        let end = end.min(self.len).min(samples.len());
        let (first, last) = (start.div_ceil(BLOCK), end / BLOCK);
        let raw = |a: usize, b: usize| Block::of(samples.get(a..b).unwrap_or_default());
        if first >= last { return raw(start, end); }
        raw(start, first * BLOCK)
            .merge(raw(last * BLOCK, end))
            .merge(self.blocks(first, last))
    }

    fn blocks(&self, mut lo: usize, mut hi: usize) -> Block {
        let mut acc = Block::EMPTY;
        for (lvl, level) in self.levels.iter().enumerate() {
            let mut add = |i: usize| {
                acc = acc.merge(level.get(i).copied().unwrap_or(Block::EMPTY));
            };
            if lvl + 1 == self.levels.len() || hi.saturating_sub(lo) < 2 * FANOUT {
                (lo..hi).for_each(add);
                break;
            }
            while lo % FANOUT != 0 { add(lo); lo += 1; }
            while hi % FANOUT != 0 { hi -= 1; add(hi); }
            lo /= FANOUT;
            hi /= FANOUT;
        }
        acc
    }

    fn spans(&self, samples: &[f32], start: f64, step: f64, cols: usize) -> Vec<(f32, f32)> {
        (0..cols).map(|c| {
            let a = (start + step * c as f64).max(0.0) as usize;
            let b = (start + step * (c + 1) as f64).ceil() as usize;
            self.minmax(samples, a, b)
        }).collect()
    }

    fn rms_spans(&self, samples: &[f32], start: f64, step: f64, cols: usize) -> Vec<f32> {
        (0..cols).map(|c| {
            let a = (start + step * c as f64).floor() as usize;
            let b = ((start + step * (c + 1) as f64).ceil() as usize)
                .min(self.len).min(samples.len());
            let block = self.sum(samples, a, b);
            block.rms(b.saturating_sub(a))
        }).collect()
    }
}

// ─── Audio decode ──────────────────────────────────────────────────────────

fn decode_audio(path: &PathBuf) -> Result<(u32, Vec<f32>, usize), String> {
    let file = fs::File::open(path)
        .map_err(|e| format!("не удалось открыть файл: {e}"))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| format!("не удалось определить формат: {e}"))?;

    let mut reader = probed.format;

    let track = reader.default_track()
        .ok_or_else(|| "аудио-трек не найден".to_string())?;

    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate
        .ok_or_else(|| "sample rate неизвестен".to_string())?;
    let channels = track.codec_params.channels
        .map(|c| c.count()).unwrap_or(1);

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| format!("кодек не поддерживается: {e}"))?;

    // SampleBuffer создаётся ОДИН раз при первом декодированном пакете
    let mut sample_buf: Option<SampleBuffer<f32>> = None;
    let mut all_samples: Vec<f32> = Vec::new();

    loop {
        let packet = match reader.next_packet() {
            Ok(p) => p,
            Err(Error::IoError(_)) => break,
            Err(e) => return Err(format!("ошибка чтения пакета: {e}")),
        };
        if packet.track_id() != track_id { continue; }

        match decoder.decode(&packet) {
            Ok(audio_buf) => {
                // Создаём sample buffer при первом пакете
                if sample_buf.is_none() {
                    let spec = *audio_buf.spec();
                    let duration = audio_buf.capacity() as u64;
                    sample_buf = Some(SampleBuffer::<f32>::new(duration, spec));
                }

                // Копируем в interleaved f32
                if let Some(buf) = &mut sample_buf {
                    buf.copy_interleaved_ref(audio_buf);
                    all_samples.extend_from_slice(buf.samples());
                }
            }
            Err(Error::DecodeError(m)) => {
                eprintln!("пропускаем повреждённый пакет: {m}");
            }
            Err(e) => return Err(format!("ошибка декодирования: {e}")),
        }
    }

    if all_samples.is_empty() {
        return Err("файл не содержит аудио-данных".to_string());
    }

    Ok((sample_rate, all_samples, channels))
}

/// Mixdown interleaved → mono f32
fn mixdown_to_mono(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 { return interleaved.to_vec(); }
    let frames = interleaved.len() / channels;
    let inv = 1.0 / channels as f32;
    (0..frames).map(|f| {
        let mut sum = 0.0f32;
        for ch in 0..channels {
            sum += interleaved[f * channels + ch];
        }
        sum * inv
    }).collect()
}

// ─── Render ────────────────────────────────────────────────────────────────

enum Style { Dark, Light }
enum Mode { Peak, Rms }

struct RenderOptions {
    width: u32,
    height: u32,
    style: Style,
    mode: Mode,
}

fn render_waveform(samples: &[f32], peaks: &Peaks, opts: &RenderOptions) -> RgbaImage {
    let w = opts.width;
    let h = opts.height;
    let mut img = RgbaImage::new(w, h);

    let (bg, fg, grid) = match opts.style {
        Style::Dark => (
            Rgba([13, 18, 32, 255]),
            Rgba([16, 185, 129, 255]),
            Rgba([31, 41, 55, 255]),
        ),
        Style::Light => (
            Rgba([255, 255, 255, 255]),
            Rgba([5, 150, 105, 255]),
            Rgba([229, 231, 235, 255]),
        ),
    };

    for pixel in img.pixels_mut() { *pixel = bg; }

    let cy = h / 2;
    for x in 0..w {
        *img.get_pixel_mut(x, cy) = grid;
    }

    let step = samples.len() as f64 / w as f64;

    let columns: Vec<(f32, f32)> = match opts.mode {
        Mode::Peak => peaks.spans(samples, 0.0, step, w as usize),
        Mode::Rms => peaks.rms_spans(samples, 0.0, step, w as usize)
            .into_iter().map(|r| (-r, r)).collect(),
    };

    let margin = 4u32;
    let draw_h = h.saturating_sub(margin * 2) as f32;
    for (x, &(lo, hi)) in columns.iter().enumerate() {
        let x = x as u32;
        let y_top = cy as f32 - hi.clamp(-1.0, 1.0) * (draw_h / 2.0);
        let y_bot = cy as f32 - lo.clamp(-1.0, 1.0) * (draw_h / 2.0);
        let y_top = y_top.max(margin as f32) as u32;
        let y_bot = (y_bot as u32).min(h.saturating_sub(1));
        for y in y_top..=y_bot {
            *img.get_pixel_mut(x, y) = fg;
        }
    }
    img
}

// ─── CLI ───────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "waveform-preview", about = "Генератор PNG превью звуковой волны")]
struct Cli {
    input: PathBuf,
    #[arg(short, long, default_value = "waveform.png")]
    output: PathBuf,
    #[arg(long, default_value = "1200")]
    width: u32,
    #[arg(long, default_value = "200")]
    height: u32,
    #[arg(long, default_value = "dark")]
    style: String,
    #[arg(long, default_value = "peak")]
    mode: String,
}

fn main() {
    let cli = Cli::parse();

    println!("🎵 waveform-preview");
    println!("   вход:  {}", cli.input.display());
    println!("   выход: {}", cli.output.display());

    println!("⏳ Декодирование аудио...");
    let t0 = std::time::Instant::now();
    let (sample_rate, interleaved, channels) = match decode_audio(&cli.input) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("❌ Ошибка: {e}");
            std::process::exit(1);
        }
    };
    let mixdown = mixdown_to_mono(&interleaved, channels);
    println!("   ✓ {} семплов, {} Hz, {} канал(ов), {:.2}s ({}ms)",
        mixdown.len(), sample_rate, channels,
        mixdown.len() as f64 / sample_rate as f64,
        t0.elapsed().as_millis());

    println!("⏳ Построение дерева Peaks...");
    let t1 = std::time::Instant::now();
    let peaks = Peaks::build(&mixdown);
    println!("   ✓ {} уровней, {} блоков Level 0 ({}ms)",
        peaks.levels.len(),
        peaks.levels.first().map(|l| l.len()).unwrap_or(0),
        t1.elapsed().as_millis());

    println!("⏳ Рендеринг waveform ({}x{})...", cli.width, cli.height);
    let t2 = std::time::Instant::now();
    let style = match cli.style.as_str() {
        "light" => Style::Light,
        _ => Style::Dark,
    };
    let mode = match cli.mode.as_str() {
        "rms" => Mode::Rms,
        _ => Mode::Peak,
    };
    let opts = RenderOptions { width: cli.width, height: cli.height, style, mode };
    let img = render_waveform(&mixdown, &peaks, &opts);
    println!("   ✓ Рендеринг ({}ms)", t2.elapsed().as_millis());

    match img.save(&cli.output) {
        Ok(_) => {
            let size = fs::metadata(&cli.output).map(|m| m.len()).unwrap_or(0);
            println!("✅ Готово! {} ({} KB)", cli.output.display(), size / 1024);
        }
        Err(e) => {
            eprintln!("❌ Не удалось сохранить PNG: {e}");
            std::process::exit(1);
        }
    }
}
