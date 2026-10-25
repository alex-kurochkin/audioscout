import { useState } from 'react'

const cargoToml = `[package]
name = "waveform-preview"
version = "0.1.0"
edition = "2021"

[dependencies]
# Декодирование WAV (нативно) + MP3/OGG/FLAC/AAC (через symphonia)
symphonia = { version = "0.5", features = ["all"] }
# Генерация PNG
image = { version = "0.25", default-features = false, features = ["png"] }
# Парсинг аргументов CLI
clap = { version = "4", features = ["derive"] }
`

const mainRust = `//! waveform-preview — CLI генератор PNG превью звуковой волны.
//!
//! Читает WAV (нативно) или MP3/OGG/FLAC/AAC (через symphonia),
//! строит многоуровневое дерево пиков (как в SoundCraft) и рендерит
//! waveform в PNG.
//!
//! Использование:
//!   cargo run --release -- input.mp3 -o preview.png
//!   cargo run --release -- input.wav -o preview.png --width 2000 --height 300
//!   cargo run --release -- input.mp3 --style dark --mode rms

use clap::Parser;
use image::{Rgba, RgbaImage};
use std::fs;
use std::path::PathBuf;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

// ─── Константы дерева пиков (как в SoundCraft) ─────────────────────────────
const BLOCK: usize = 32;     // семплов в базовом блоке (Level 0)
const FANOUT: usize = 4;     // блоков объединяется в следующий уровень

// ─── Блок дерева пиков ─────────────────────────────────────────────────────
#[derive(Debug, Clone, Copy)]
struct Block {
    min: f32,
    max: f32,
    squares: f32, // сумма квадратов для RMS
}

impl Block {
    const EMPTY: Block = Block {
        min: f32::INFINITY,
        max: f32::NEG_INFINITY,
        squares: 0.0,
    };

    fn point(s: f32) -> Self {
        Block {
            min: s,
            max: s,
            squares: if s.is_finite() { s * s } else { 0.0 },
        }
    }

    fn of(samples: &[f32]) -> Self {
        samples
            .iter()
            .fold(Self::EMPTY, |b, &s| b.merge(Self::point(s)))
    }

    fn merge(self, o: Block) -> Block {
        Block {
            min: self.min.min(o.min),
            max: self.max.max(o.max),
            squares: self.squares + o.squares,
        }
    }

    fn range(self) -> (f32, f32) {
        if self.min > self.max {
            (0.0, 0.0)
        } else {
            (self.min, self.max)
        }
    }

    fn rms(self, count: usize) -> f32 {
        if count == 0 {
            return 0.0;
        }
        (self.squares / count as f32).sqrt()
    }
}

// ─── Многоуровневое дерево пиков ───────────────────────────────────────────
struct Peaks {
    len: usize,
    levels: Vec<Vec<Block>>,
}

impl Peaks {
    /// Строит дерево за O(n) — один проход по семплам.
    fn build(samples: &[f32]) -> Self {
        if samples.is_empty() {
            return Peaks { len: 0, levels: vec![] };
        }
        // Level 0: блоки по BLOCK семплов
        let mut levels: Vec<Vec<Block>> =
            vec![samples.chunks(BLOCK).map(Block::of).collect()];
        // Каждый следующий уровень: FANOUT блоков → 1 блок
        while let Some(last) = levels.last().filter(|l| l.len() > 1) {
            let next = last
                .chunks(FANOUT)
                .map(|c| {
                    c.iter()
                        .fold(Block::EMPTY, |a, &b| a.merge(b))
                })
                .collect();
            levels.push(next);
        }
        Peaks {
            len: samples.len(),
            levels,
        }
    }

    /// Возвращает (min, max) для диапазона [start, end).
    /// Использует coarsest level, который помещается — O(log n).
    fn minmax(&self, samples: &[f32], start: usize, end: usize) -> (f32, f32) {
        self.sum(samples, start, end).range()
    }

    fn sum(&self, samples: &[f32], start: usize, end: usize) -> Block {
        let end = end.min(self.len).min(samples.len());
        let (first, last) = (start.div_ceil(BLOCK), end / BLOCK);
        let raw = |a: usize, b: usize| {
            Block::of(samples.get(a..b).unwrap_or_default())
        };
        if first >= last {
            return raw(start, end);
        }
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
            if lvl + 1 == self.levels.len()
                || hi.saturating_sub(lo) < 2 * FANOUT
            {
                (lo..hi).for_each(add);
                break;
            }
            while lo % FANOUT != 0 {
                add(lo);
                lo += 1;
            }
            while hi % FANOUT != 0 {
                hi -= 1;
                add(hi);
            }
            lo /= FANOUT;
            hi /= FANOUT;
        }
        acc
    }

    /// (min, max) для каждого из cols столбцов шириной step семплов.
    fn spans(
        &self,
        samples: &[f32],
        start: f64,
        step: f64,
        cols: usize,
    ) -> Vec<(f32, f32)> {
        (0..cols)
            .map(|c| {
                let a = (start + step * c as f64).max(0.0) as usize;
                let b = (start + step * (c + 1) as f64)
                    .ceil() as usize;
                self.minmax(samples, a, b)
            })
            .collect()
    }

    /// RMS для каждого столбца (Power view).
    fn rms_spans(
        &self,
        samples: &[f32],
        start: f64,
        step: f64,
        cols: usize,
    ) -> Vec<f32> {
        (0..cols)
            .map(|c| {
                let a = (start + step * c as f64).floor() as usize;
                let b = ((start + step * (c + 1) as f64).ceil() as usize)
                    .min(self.len)
                    .min(samples.len());
                let block = self.sum(samples, a, b);
                block.rms(b.saturating_sub(a))
            })
            .collect()
    }
}

// ─── Декодирование аудио ───────────────────────────────────────────────────
/// Декодирует аудиофайл в planar f32 (один Vec на канал).
/// WAV читается нативным парсером symphonia; MP3/OGG/FLAC/AAC — через кодеки.
fn decode_audio(path: &PathBuf) -> Result<(u32, Vec<Vec<f32>>), String> {
    let file = fs::File::open(path)
        .map_err(|e| format!("не удалось открыть файл: {e}"))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    // Подсказка по расширению
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    // Probe формата
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| format!("не удалось определить формат: {e}"))?;

    let mut reader = probed.format;

    // Найти первый аудио-трек
    let track = reader
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| "аудио-трек не найден".to_string())?;

    let track_id = track.id;
    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| "sample rate неизвестен".to_string())?;
    let channels = track
        .codec_params
        .channels
        .map(|c| c.count())
        .unwrap_or(1);

    // Создать декодер
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| format!("кодек не поддерживается: {e}"))?;

    // Planar f32: один Vec на канал
    let mut planar: Vec<Vec<f32>> = (0..channels).map(|_| Vec::new()).collect();
    let mut scratch: Vec<f32> = Vec::new();

    // Packet-by-packet decode loop
    loop {
        let packet = match reader.next_packet() {
            Ok(p) => p,
            Err(Error::IoError(_)) => break,
            Err(e) => return Err(format!("ошибка чтения пакета: {e}")),
        };
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = decoded.spec();
                let n = decoded.frames() as usize;
                // Копируем в interleaved scratch
                scratch.resize(n * channels, 0.0);
                decoded
                    .copy_to_interleaved(&mut scratch[..n * channels], spec);
                // Разделяем по каналам (planar)
                for ch in 0..channels {
                    for f in 0..n {
                        let s = scratch[f * channels + ch];
                        planar[ch].push(if s.is_finite() { s } else { 0.0 });
                    }
                }
            }
            Err(Error::DecodeError(m)) => {
                eprintln!("пропускаем повреждённый пакет: {m}");
            }
            Err(e) => return Err(format!("ошибка декодирования: {e}")),
        }
    }

    if planar[0].is_empty() {
        return Err("файл не содержит аудио-данных".to_string());
    }

    Ok((sample_rate, planar))
}

// ─── Рендеринг waveform в PNG ──────────────────────────────────────────────
struct RenderOptions {
    width: u32,
    height: u32,
    style: Style,
    mode: Mode,
}

#[derive(Clone, Copy)]
enum Style {
    Dark,
    Light,
}

#[derive(Clone, Copy)]
enum Mode {
    Peak,
    Rms,
}

fn render_waveform(
    samples: &[f32],
    peaks: &Peaks,
    opts: &RenderOptions,
) -> RgbaImage {
    let w = opts.width;
    let h = opts.height;
    let mut img = RgbaImage::new(w, h);

    // Цвета
    let (bg, fg, grid) = match opts.style {
        Style::Dark => (
            Rgba([13, 18, 32, 255]),     // dark bg
            Rgba([16, 185, 129, 255]),   // emerald
            Rgba([31, 41, 55, 255]),     // grid
        ),
        Style::Light => (
            Rgba([255, 255, 255, 255]),
            Rgba([5, 150, 105, 255]),
            Rgba([229, 231, 235, 255]),
        ),
    };

    // Фон
    for pixel in img.pixels_mut() {
        *pixel = bg;
    }

    // Центральная линия
    let cy = h / 2;
    for x in 0..w {
        if let Some(p) = img.get_pixel_mut(x, cy) {
            *p = grid;
        }
    }

    // Шаг: семплов на пиксель
    let step = samples.len() as f64 / w as f64;

    // Получаем spans (min, max) для каждого столбца
    let columns: Vec<(f32, f32)> = match opts.mode {
        Mode::Peak => peaks.spans(samples, 0.0, step, w as usize),
        Mode::Rms => peaks
            .rms_spans(samples, 0.0, step, w as usize)
            .into_iter()
            .map(|r| (-r, r))
            .collect(),
    };

    // Рисуем столбцы
    let margin = 4u32;
    let draw_h = h.saturating_sub(margin * 2) as f32;
    for (x, &(lo, hi)) in columns.iter().enumerate() {
        let x = x as u32;
        // Нормализуем в [-1, 1] → [0, draw_h]
        let y_top = cy as f32 - hi.clamp(-1.0, 1.0) * (draw_h / 2.0);
        let y_bot = cy as f32 - lo.clamp(-1.0, 1.0) * (draw_h / 2.0);
        let y_top = y_top.max(margin as f32) as u32;
        let y_bot = (y_bot as u32).min(h.saturating_sub(1));
        for y in y_top..=y_bot {
            if let Some(p) = img.get_pixel_mut(x, y) {
                *p = fg;
            }
        }
    }

    img
}

// ─── CLI ───────────────────────────────────────────────────────────────────
#[derive(Parser)]
#[command(
    name = "waveform-preview",
    about = "Генератор PNG превью звуковой волны (алгоритм SoundCraft)"
)]
struct Cli {
    /// Путь к аудиофайлу (WAV, MP3, OGG, FLAC, AAC)
    input: PathBuf,

    /// Путь к выходному PNG
    #[arg(short, long, default_value = "waveform.png")]
    output: PathBuf,

    /// Ширина изображения в пикселях
    #[arg(long, default_value = "1200")]
    width: u32,

    /// Высота изображения в пикселях
    #[arg(long, default_value = "200")]
    height: u32,

    /// Стиль: dark или light
    #[arg(long, default_value = "dark")]
    style: String,

    /// Режим: peak или rms
    #[arg(long, default_value = "peak")]
    mode: String,
}

fn main() {
    let cli = Cli::parse();

    println!("🎵 waveform-preview");
    println!("   вход:  {}", cli.input.display());
    println!("   выход: {}", cli.output.display());

    // 1. Декодирование
    println!("⏳ Декодирование аудио...");
    let t0 = std::time::Instant::now();
    let (sample_rate, channels) = match decode_audio(&cli.input) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("❌ Ошибка: {e}");
            std::process::exit(1);
        }
    };
    let mixdown: Vec<f32> = mixdown_to_mono(&channels);
    println!(
        "   ✓ {} семплов, {} Hz, {} канал(ов), {:.2}s ({}ms)",
        mixdown.len(),
        sample_rate,
        channels.len(),
        mixdown.len() as f64 / sample_rate as f64,
        t0.elapsed().as_millis()
    );

    // 2. Построение дерева пиков
    println!("⏳ Построение дерева Peaks...");
    let t1 = std::time::Instant::now();
    let peaks = Peaks::build(&mixdown);
    println!(
        "   ✓ {} уровней, {} блоков Level 0 ({}ms)",
        peaks.levels.len(),
        peaks.levels.first().map(|l| l.len()).unwrap_or(0),
        t1.elapsed().as_millis()
    );

    // 3. Рендеринг PNG
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
    let opts = RenderOptions {
        width: cli.width,
        height: cli.height,
        style,
        mode,
    };
    let img = render_waveform(&mixdown, &peaks, &opts);
    println!("   ✓ Рендеринг ({}ms)", t2.elapsed().as_millis());

    // 4. Сохранение PNG
    match img.save(&cli.output) {
        Ok(_) => {
            let size = fs::metadata(&cli.output)
                .map(|m| m.len())
                .unwrap_or(0);
            println!(
                "✅ Готово! {} ({} KB)",
                cli.output.display(),
                size / 1024
            );
        }
        Err(e) => {
            eprintln!("❌ Не удалось сохранить PNG: {e}");
            std::process::exit(1);
        }
    }
}

/// Сводит многоканальное аудио в моно (усреднение).
fn mixdown_to_mono(channels: &[Vec<f32>]) -> Vec<f32> {
    if channels.is_empty() {
        return vec![];
    }
    if channels.len() == 1 {
        return channels[0].clone();
    }
    let len = channels.iter().map(|c| c.len()).min().unwrap_or(0);
    let inv = 1.0 / channels.len() as f32;
    (0..len)
        .map(|i| {
            let sum: f32 = channels.iter().map(|c| c[i]).sum();
            sum * inv
        })
        .collect()
}
`

const readme = `# waveform-preview

CLI-утилита для генерации PNG превью звуковой волны.
Использует алгоритм многоуровневого дерева пиков из SoundCraft DAW.

## Возможности

- **Форматы**: WAV, MP3, OGG Vorbis, FLAC, AAC, ALAC, CAF
- **Алгоритм**: SoundCraft Peaks tree (BLOCK=32, FANOUT=4, O(log n) запросы)
- **Режимы**: Peak (min/max) и RMS (Power view)
- **Стили**: Dark и Light тема
- **Производительность**: ~10-50ms для WAV, ~100-500ms для MP3 (5 мин файл)

## Сборка

\`\`\`bash
cargo build --release
\`\`\`

## Использование

\`\`\`bash
# Базовый запуск
./waveform-preview input.mp3 -o preview.png

# Настройка размера
./waveform-preview input.wav -o preview.png --width 2000 --height 300

# RMS режим (Power view, как в Pro Tools)
./waveform-preview input.mp3 --mode rms

# Светлая тема
./waveform-preview input.flac --style light
\`\`\`

## Алгоритм

1. **Декодирование** — symphonia читает любой формат → planar f32
2. **Mixdown** — многоканальное → моно (усреднение)
3. **Peaks::build()** — O(n) один проход, строит дерево:
   - Level 0: 32 семпла → Block {min, max, squares}
   - Level N: 4 блока → 1 блок
4. **Peaks::spans()** — O(log n) запрос на столбец
5. **Рендеринг** — столбцы в PNG через crate \`image\`

## Зависимости

- \`symphonia\` — декодирование аудио
- \`image\` — генерация PNG
- \`clap\` — CLI аргументы
`

export default function CLIProgram() {
  const [activeFile, setActiveFile] = useState<'cargo' | 'main' | 'readme'>('main')

  const files = {
    cargo: { name: 'Cargo.toml', content: cargoToml, lang: 'toml' },
    main: { name: 'src/main.rs', content: mainRust, lang: 'rust' },
    readme: { name: 'README.md', content: readme, lang: 'markdown' },
  }

  const current = files[activeFile]

  return (
    <div className="space-y-6">
      <div className="rounded-xl bg-[#111827] border border-gray-800 p-6">
        <div className="flex items-center gap-3 mb-4">
          <span className="text-3xl">🦀</span>
          <div>
            <h3 className="text-xl font-semibold">Rust CLI программа</h3>
            <p className="text-sm text-gray-400">
              Готовый исходный код для копирования. Использует те же алгоритмы, что и SoundCraft.
            </p>
          </div>
        </div>

        <div className="rounded-lg bg-amber-500/10 border border-amber-500/30 p-4 mb-6">
          <div className="text-sm text-amber-200">
            <strong>💡 Как запустить:</strong>
            <pre className="mt-2 text-xs font-mono bg-black/30 rounded p-3 overflow-x-auto text-amber-100">
{`# Создайте проект
cargo new waveform-preview
cd waveform-preview

# Скопируйте Cargo.toml и src/main.rs из вкладок ниже

# Запустите
cargo run --release -- your-audio.mp3 -o preview.png`}
            </pre>
          </div>
        </div>

        {/* File tabs */}
        <div className="flex gap-1 mb-2 border-b border-gray-800">
          {(Object.keys(files) as Array<keyof typeof files>).map(key => (
            <button
              key={key}
              onClick={() => setActiveFile(key)}
              className={`px-4 py-2 text-sm font-mono rounded-t-lg transition-all ${
                activeFile === key
                  ? 'bg-[#0a0e17] text-emerald-400 border border-gray-800 border-b-0'
                  : 'text-gray-500 hover:text-gray-300'
              }`}
            >
              {files[key].name}
            </button>
          ))}
        </div>

        {/* Code */}
        <div className="relative rounded-b-xl rounded-tr-xl bg-[#0a0e17] border border-gray-800 overflow-hidden">
          <div className="absolute top-2 right-2 z-10">
            <button
              onClick={() => navigator.clipboard.writeText(current.content)}
              className="px-3 py-1 text-xs bg-gray-800 hover:bg-gray-700 text-gray-300 rounded transition-colors"
            >
              📋 Копировать
            </button>
          </div>
          <div className="p-4 overflow-x-auto max-h-[600px] overflow-y-auto">
            <pre className="text-sm leading-relaxed">
              <code>
                {current.content.split('\n').map((line, i) => (
                  <div key={i} className="flex hover:bg-white/5">
                    <span className="inline-block w-10 text-right mr-4 text-gray-600 select-none text-xs">
                      {i + 1}
                    </span>
                    <span className="text-gray-300">{line || ' '}</span>
                  </div>
                ))}
              </code>
            </pre>
          </div>
        </div>
      </div>

      {/* Build instructions */}
      <div className="rounded-xl bg-[#111827] border border-gray-800 p-6">
        <h3 className="text-lg font-semibold mb-4">🔨 Как скомпилировать и запустить</h3>

        <div className="space-y-4">
          {/* Step 1 */}
          <BuildStep
            num={1}
            title="Установить Rust (если ещё не установлен)"
            cmd={`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Перезапустить терминал или выполнить:
source $HOME/.cargo/env

# Проверить версию
rustc --version   # должно быть 1.70+`}
            note="На Windows используйте rustup-init.exe с сайта https://rustup.rs. Также потребуется Visual Studio C++ Build Tools."
          />

          {/* Step 2 */}
          <BuildStep
            num={2}
            title="Создать проект"
            cmd={`cargo new waveform-preview
cd waveform-preview`}
          />

          {/* Step 3 */}
          <BuildStep
            num={3}
            title="Скопировать исходный код"
            note="Откройте вкладку Cargo.toml выше → нажмите «Копировать» → вставьте в файл Cargo.toml. То же самое с src/main.rs."
          />

          {/* Step 4 */}
          <BuildStep
            num={4}
            title="Скомпилировать (release — быстрая версия)"
            cmd={`# Первый запуск скачает и соберёт все зависимости (~1-2 мин)
cargo build --release

# Готовый бинарник появится в:
#   ./target/release/waveform-preview        (Linux/macOS)
#   ./target/release/waveform-preview.exe    (Windows)`}
            note="Release-сборка с оптимизациями работает в 10-50× быстрее debug. Для больших файлов (часы аудио) это критично."
          />

          {/* Step 5 */}
          <BuildStep
            num={5}
            title="Запустить"
            cmd={`# Базовый запуск
./target/release/waveform-preview my-audio.mp3 -o preview.png

# Или через cargo run (компилирует на лету)
cargo run --release -- my-audio.mp3 -o preview.png

# С настройками
./target/release/waveform-preview song.wav \\
    --width 2000 --height 300 \\
    --style dark --mode rms \\
    -o cover.png`}
          />
        </div>

        {/* Platform-specific */}
        <div className="mt-6 grid md:grid-cols-3 gap-3">
          <PlatformCard
            os="🐧 Linux"
            deps="Никаких дополнительных зависимостей не нужно — всё собирается из исходников."
            extra="Для минимального размера бинарника можно добавить в Cargo.toml:
[profile.release]
strip = true
lto = true
codegen-units = 1"
          />
          <PlatformCard
            os="🍎 macOS"
            deps="Нужен Xcode Command Line Tools:
xcode-select --install"
            extra="Для universal binary (Intel + Apple Silicon):
cargo build --release --target universal2-apple-darwin"
          />
          <PlatformCard
            os="🪟 Windows"
            deps="Нужны Visual Studio Build Tools с компонентом «Разработка классических приложений на C++»"
            extra="Запуск из PowerShell:
.\\target\\release\\waveform-preview.exe input.mp3 -o out.png"
          />
        </div>

        {/* Troubleshooting */}
        <div className="mt-6 rounded-lg bg-amber-500/5 border border-amber-500/20 p-4">
          <h4 className="text-sm font-semibold text-amber-400 mb-2">⚠️ Возможные проблемы</h4>
          <div className="text-sm text-gray-400 space-y-2">
            <div>
              <code className="text-amber-300 text-xs">error: linker `cc` not found</code>
              <div className="text-xs mt-0.5">→ Установите C-компилятор: <code>sudo apt install build-essential</code> (Ubuntu) или <code>xcode-select --install</code> (macOS)</div>
            </div>
            <div>
              <code className="text-amber-300 text-xs">error: failed to download from `https://crates.io`</code>
              <div className="text-xs mt-0.5">→ Проверьте интернет-соединение. При первом запуске cargo скачивает зависимости с crates.io</div>
            </div>
            <div>
              <code className="text-amber-300 text-xs">Не удалось определить формат</code>
              <div className="text-xs mt-0.5">→ Убедитесь, что файл не повреждён. Попробуйте другой MP3 или WAV.</div>
            </div>
          </div>
        </div>
      </div>

      {/* Usage examples */}
      <div className="rounded-xl bg-[#111827] border border-gray-800 p-6">
        <h3 className="text-lg font-semibold mb-4">📖 Примеры использования</h3>
        <div className="space-y-3">
          <UsageExample
            cmd="waveform-preview song.mp3"
            desc="Базовый запуск: 1200×200, dark theme, peak mode"
          />
          <UsageExample
            cmd="waveform-preview podcast.wav -o cover.png --width 1600 --height 400"
            desc="Большой размер для обложки"
          />
          <UsageExample
            cmd="waveform-preview track.flac --mode rms --style light"
            desc="RMS (Power view) в светлой теме"
          />
          <UsageExample
            cmd="waveform-preview *.mp3  # через xargs"
            desc="Пакетная обработка: find . -name '*.mp3' | xargs -I {} waveform-preview {} -o {}.png"
            desc2="find . -name '*.mp3' | xargs -I {} waveform-preview {} -o {}.png"
          />
        </div>
      </div>
    </div>
  )
}

function UsageExample({ cmd, desc, desc2 }: { cmd: string; desc: string; desc2?: string }) {
  return (
    <div className="bg-[#0d1220] rounded-lg p-3 border border-gray-800">
      <code className="text-sm text-emerald-400 font-mono">$ {cmd}</code>
      <div className="text-xs text-gray-500 mt-1">{desc}</div>
      {desc2 && <code className="text-xs text-gray-600 font-mono mt-1 block">{desc2}</code>}
    </div>
  )
}

function BuildStep({ num, title, cmd, note }: { num: number; title: string; cmd?: string; note?: string }) {
  return (
    <div className="bg-[#0d1220] rounded-lg border border-gray-800 overflow-hidden">
      <div className="flex items-center gap-3 px-4 py-3 border-b border-gray-800">
        <div className="w-7 h-7 rounded-full bg-emerald-500/20 text-emerald-400 flex items-center justify-center text-sm font-bold">
          {num}
        </div>
        <h4 className="font-semibold text-white text-sm">{title}</h4>
      </div>
      {cmd && (
        <div className="p-4 bg-black/30">
          <pre className="text-xs font-mono text-gray-300 whitespace-pre-wrap leading-relaxed overflow-x-auto">
            <code>{cmd}</code>
          </pre>
        </div>
      )}
      {note && (
        <div className="px-4 py-3 text-xs text-gray-400 border-t border-gray-800 bg-gray-900/30">
          💡 {note}
        </div>
      )}
    </div>
  )
}

function PlatformCard({ os, deps, extra }: { os: string; deps: string; extra?: string }) {
  return (
    <div className="rounded-lg bg-[#0d1220] border border-gray-800 p-4">
      <div className="font-semibold text-white text-sm mb-2">{os}</div>
      <pre className="text-xs font-mono text-gray-400 whitespace-pre-wrap mb-2">{deps}</pre>
      {extra && (
        <div className="text-xs text-gray-500 pt-2 border-t border-gray-800">
          <pre className="font-mono whitespace-pre-wrap">{extra}</pre>
        </div>
      )}
    </div>
  )
}
