pub mod wav;
pub mod aiff;
pub mod caf;
pub mod ffmpeg;

use crate::{AudioMeta, ProbeStatus, SampleType};
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AudioIoError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),
    #[error("Decode error: {0}")]
    DecodeError(String),
    #[error("File is broken: {0}")]
    BrokenFile(String),
    #[error("FFmpeg not available")]
    FfmpegNotAvailable,
    #[error("Save error: {0}")]
    SaveError(String),
}

pub type Result<T> = std::result::Result<T, AudioIoError>;

pub trait PcmFile {
    fn load(path: &Path) -> Result<(Vec<f32>, u32, u16, SampleType)>;
    fn save(path: &Path, samples: &[f32], sample_rate: u32, channels: u16, sample_type: SampleType) -> Result<()>;
    fn probe(path: &Path) -> Result<AudioMeta>;
}

pub fn load_audio(path: &Path) -> Result<(Vec<f32>, u32, u16, SampleType)> {
    let ext = path.extension().and_then(|e| e.to_str()).map(|e| e.to_lowercase()).unwrap_or_default();
    match ext.as_str() {
        "wav" => wav::WavFile::load(path),
        "aiff" | "aif" => aiff::AiffFile::load(path),
        "caf" => caf::CafFile::load(path),
        _ => load_with_symphonia(path),
    }
}

/// Per-channel min/max envelope reduced directly from the decoder output.
///
/// Mirrors the peak-envelope semantics of the `waveform-preview` example
/// (`Block::of` + `Block::range`): every emitted point carries the true
/// min/max of the samples it covers, so no peaks are lost even though the
/// full signal is never materialised.
pub struct Envelope {
    pub channels: usize,
    pub frames: usize,
    /// Planar layout: `lo[c]` / `hi[c]` hold the per-channel minima/maxima.
    pub lo: Vec<Vec<f32>>,
    pub hi: Vec<Vec<f32>>,
}

/// Streaming min/max decimator feeding an [`Envelope`].
///
/// Deliberately minimal: all hot paths are branch-free folds over plain
/// slices (`iter().fold`), because per-sample loops with branches were what
/// made the previous implementation crawl on large files.
struct EnvAccum {
    channels: usize,
    block_frames: usize,
    /// Total frames fed into the accumulator — used by callers to enforce a
    /// hard decode cap so a pathological file can never spin forever.
    frames_in: usize,
    /// Raw interleaved samples carried between decoder packets so that
    /// envelope blocks never straddle a packet boundary incorrectly.
    carry: Vec<f32>,
    lo: Vec<Vec<f32>>,
    hi: Vec<Vec<f32>>,
}

impl EnvAccum {
    fn new(channels: usize, block_frames: usize) -> Self {
        let ch = channels.max(1);
        Self {
            channels: ch,
            block_frames: block_frames.max(1),
            frames_in: 0,
            carry: Vec::new(),
            lo: (0..ch).map(|_| Vec::new()).collect(),
            hi: (0..ch).map(|_| Vec::new()).collect(),
        }
    }

    /// Reduces one frame-major interleaved block `[start, end)` to a single
    /// `(min, max)` point per channel — the same fold as the example's
    /// `Block::of(...).range()`, branch-free so it vectorises.
    #[inline]
    fn reduce_block(&mut self, data: &[f32], start: usize, end: usize) {
        let ch = self.channels;
        let mut mins = [f32::INFINITY; 8];
        let mut maxs = [f32::NEG_INFINITY; 8];
        let n = ch.min(8);
        for frame in data[start..end].chunks(ch) {
            for c in 0..n {
                let s = frame[c];
                mins[c] = mins[c].min(s);
                maxs[c] = maxs[c].max(s);
            }
        }
        for c in 0..ch {
            let (mn, mx) = (mins[c % 8], maxs[c % 8]);
            // NaN-poisoned or empty block → flat line at zero (mirrors Block::range()).
            let (lo, hi) = if mn <= mx { (mn, mx) } else { (0.0, 0.0) };
            self.lo[c].push(lo);
            self.hi[c].push(hi);
        }
    }

    /// Reduces whole blocks sitting at the front of `carry`.
    /// (`mem::take` avoids borrowing `self.carry` while mutating `self.lo/hi`.)
    fn drain_carry(&mut self) {
        let stride = self.block_frames * self.channels;
        let taken = std::mem::take(&mut self.carry);
        let complete = taken.len() - taken.len() % self.channels;
        let mut idx = 0usize;
        while idx + stride <= complete {
            self.reduce_block(&taken, idx, idx + stride);
            idx += stride;
        }
        self.carry = taken[idx..].to_vec();
    }

    /// Feeds interleaved samples (frame-major). Partial blocks stay in
    /// `carry` until enough data arrives.
    fn push(&mut self, samples: &[f32]) {
        if samples.is_empty() {
            return;
        }
        let ch = self.channels;
        self.frames_in += samples.len() / ch;
        let stride = self.block_frames * ch;

        if !self.carry.is_empty() {
            self.carry.extend_from_slice(samples);
            self.drain_carry();
            return;
        }

        let complete = samples.len() / ch * ch;
        let mut idx = 0usize;
        while idx + stride <= complete {
            self.reduce_block(samples, idx, idx + stride);
            idx += stride;
        }
        if idx < samples.len() {
            self.carry.extend_from_slice(&samples[idx..]);
        }
    }

    fn finish(&mut self) {
        let ch = self.channels;
        let complete = (self.carry.len() / ch) * ch;
        if complete == 0 {
            return;
        }
        self.drain_carry();
        // Trailing partial block → one final envelope point per channel.
        if !self.carry.is_empty() {
            let tail = std::mem::take(&mut self.carry);
            self.reduce_block(&tail, 0, tail.len());
        }
    }

    fn frames_out(&self) -> usize {
        self.lo.first().map(|v| v.len()).unwrap_or(0)
    }

    /// Total number of audio frames fed so far (for decode-cap enforcement).
    fn frames_in(&self) -> usize {
        self.frames_in
    }

    fn into_envelope(self) -> Envelope {
        Envelope { channels: self.channels, frames: self.frames_out(), lo: self.lo, hi: self.hi }
    }
}

/// Hard cap for a single envelope decode.
///
/// A damaged stream (or an endless source) must never stall the preview
/// pipeline forever: once we have folded this many frames into the envelope
/// we stop reading and render what we have. At typical thumbnail resolutions
/// the cap corresponds to many hours of audio, so valid files are unaffected.
const ENVELOPE_MAX_FRAMES: usize = 200_000_000;

/// Decodes `path` and reduces it to a per-channel min/max envelope with
/// roughly `target_frames` points per channel, streaming through the decoder.
///
/// Fast path modelled on the `waveform-preview` example: a ~340-px thumbnail
/// needs only a few hundred envelope points per track, so we fold peaks while
/// reading instead of buffering hundreds of megabytes of samples.
///
/// Every code path below is *bounded*: the loops stop at end-of-stream, on
/// the first decode error, or at [`ENVELOPE_MAX_FRAMES`] — there is no
/// unbounded "keep waiting" state that could make preview generation hang
/// indefinitely behind a stuck reader.
pub fn load_audio_envelope(path: &Path, target_frames: usize) -> Result<(Envelope, u32)> {
    let ext = path.extension().and_then(|e| e.to_str()).map(|e| e.to_lowercase()).unwrap_or_default();
    match ext.as_str() {
        "wav" => envelope_from_hound(path, target_frames.max(1)),
        _ => envelope_with_symphonia(path, target_frames.max(1)),
    }
}

fn envelope_from_hound(path: &Path, target_frames: usize) -> Result<(Envelope, u32)> {
    let reader = hound::WavReader::open(path)
        .map_err(|e| AudioIoError::DecodeError(format!("WAV open: {}", e)))?;
    let spec = reader.spec();
    let ch = spec.channels.max(1) as usize;
    let total_samples = reader.len() as usize;
    let block = (total_samples / ch).div_ceil(target_frames).max(1);
    drop(reader);

    // BufReader: hound's WavReader issues one read() per sample otherwise.
    let file = std::io::BufReader::with_capacity(
        1 << 20,
        std::fs::File::open(path)
            .map_err(|e| AudioIoError::DecodeError(format!("WAV open: {}", e)))?,
    );
    let mut reader = hound::WavReader::new(file)
        .map_err(|e| AudioIoError::DecodeError(format!("WAV open: {}", e)))?;

    let mut acc = EnvAccum::new(ch, block);
    const CHUNK: usize = 1 << 14; // samples per batch

    match spec.sample_format {
        hound::SampleFormat::Int => match spec.bits_per_sample {
            8 | 16 => pump_scaled::<i16>(&mut reader, if spec.bits_per_sample == 8 { 1.0 / 128.0 } else { 1.0 / 32768.0 }, &mut acc),
            24 | 32 => pump_scaled::<i32>(&mut reader, if spec.bits_per_sample == 24 { 1.0 / 8388608.0 } else { 1.0 / 2147483648.0 }, &mut acc),
            bits => return Err(AudioIoError::UnsupportedFormat(format!("Unsupported bits: {}", bits))),
        },
        hound::SampleFormat::Float => {
            let mut it = reader.samples::<f32>();
            let mut buf: Vec<f32> = Vec::with_capacity(CHUNK);
            loop {
                buf.clear();
                while buf.len() < CHUNK {
                    match it.next() {
                        Some(Ok(v)) => buf.push(v),
                        // End of stream *or* a corrupt sample: stop reading.
                        // Never retry — a broken tail must not become an
                        // infinite loop that starves the whole preview queue.
                        _ => break,
                    }
                }
                if buf.is_empty() {
                    break;
                }
                acc.push(&buf);
                if buf.len() < CHUNK || acc.frames_in() >= ENVELOPE_MAX_FRAMES {
                    break;
                }
            }
        }
    }

    acc.finish();
    Ok((acc.into_envelope(), spec.sample_rate))
}

/// Reads PCM samples in chunks, scales them to f32 and feeds the envelope
/// accumulator — no full-file buffering anywhere.
///
/// Bounded on three independent conditions: clean end of stream, first
/// sample-read error, and the [`ENVELOPE_MAX_FRAMES`] safety cap. The old
/// version silently swallowed errors (`Some(Ok(v)) => …, _ => break` inside
/// a fixed-size refill was fine, but the outer loop had no frame cap, so a
/// pathological file could keep the worker busy forever and every later
/// preview job would appear to "just wait").
fn pump_scaled<T>(reader: &mut hound::WavReader<std::io::BufReader<std::fs::File>>, scale: f32, acc: &mut EnvAccum)
where
    T: hound::Sample + Copy + HoundToFloat,
{
    let mut it = reader.samples::<T>();
    let mut raw: Vec<T> = Vec::with_capacity(1 << 14);
    let mut buf: Vec<f32> = Vec::with_capacity(1 << 14);
    loop {
        raw.clear();
        buf.clear();
        while raw.len() < (1 << 14) {
            match it.next() {
                Some(Ok(v)) => raw.push(v),
                // EOF or corrupt data → stop; do not spin on the iterator.
                _ => break,
            }
        }
        if raw.is_empty() {
            break;
        }
        buf.extend(raw.iter().map(|&v| v.to_f32(scale)));
        acc.push(&buf);
        if raw.len() < (1 << 14) || acc.frames_in() >= ENVELOPE_MAX_FRAMES {
            break;
        }
    }
}

/// Minimal scaling trait bridging hound sample types to f32.
trait HoundToFloat: Sized {
    fn to_f32(self, scale: f32) -> f32;
}
impl HoundToFloat for i16 {
    #[inline]
    fn to_f32(self, scale: f32) -> f32 { self as f32 * scale }
}
impl HoundToFloat for i32 {
    #[inline]
    fn to_f32(self, scale: f32) -> f32 { self as f32 * scale }
}

fn envelope_with_symphonia(path: &Path, target_frames: usize) -> Result<(Envelope, u32)> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::errors::Error;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = std::fs::File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) { hint.with_extension(ext); }

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| AudioIoError::DecodeError(format!("Probe failed: {}", e)))?;

    let mut format = probed.format;
    let track = format.default_track()
        .ok_or_else(|| AudioIoError::DecodeError("No audio track found".to_string()))?;

    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(44100);
    let ch = track.codec_params.channels.map(|c| c.count()).unwrap_or(1).max(1);
    let total_frames_hint = track.codec_params.n_frames.unwrap_or(0) as usize;
    let block = if total_frames_hint > 0 {
        (total_frames_hint / target_frames).max(1)
    } else {
        // Unknown length: keep the envelope dense but bounded.
        2048
    };

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| AudioIoError::DecodeError(format!("Codec init failed: {}", e)))?;

    let mut acc = EnvAccum::new(ch, block);
    let mut sample_buf: Option<SampleBuffer<f32>> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(_) => break,
        };
        if packet.track_id() != track_id { continue; }

        match decoder.decode(&packet) {
            Ok(audio_buf) => {
                if sample_buf.is_none() {
                    let spec = *audio_buf.spec();
                    let duration = audio_buf.capacity() as u64;
                    sample_buf = Some(SampleBuffer::<f32>::new(duration, spec));
                }
                if let Some(buf) = &mut sample_buf {
                    buf.copy_interleaved_ref(audio_buf);
                    acc.push(buf.samples());
                }
            }
            Err(Error::DecodeError(m)) => {
                tracing::warn!("Skipping corrupted packet: {}", m);
            }
            Err(_) => continue,
        }

        // Hard decode cap: a damaged stream that keeps yielding packets must
        // not keep this worker (and every preview queued behind it) busy
        // forever. Stop early and render what we have.
        if acc.frames_in() >= ENVELOPE_MAX_FRAMES {
            tracing::warn!("Envelope decode capped at {} frames: {}", ENVELOPE_MAX_FRAMES, path.display());
            break;
        }
    }

    acc.finish();
    if acc.frames_out() == 0 {
        return Err(AudioIoError::DecodeError("File contains no audio data".to_string()));
    }
    Ok((acc.into_envelope(), sample_rate))
}

fn load_with_symphonia(path: &Path) -> Result<(Vec<f32>, u32, u16, SampleType)> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::errors::Error;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = std::fs::File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) { hint.with_extension(ext); }

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| AudioIoError::DecodeError(format!("Probe failed: {}", e)))?;

    let mut format = probed.format;
    let track = format.default_track()
        .ok_or_else(|| AudioIoError::DecodeError("No audio track found".to_string()))?;

    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(44100);
    let channels = track.codec_params.channels.map(|c| c.count()).unwrap_or(1);

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| AudioIoError::DecodeError(format!("Codec init failed: {}", e)))?;

    // SampleBuffer создаётся ОДИН раз при первом декодированном пакете
    let mut sample_buf: Option<SampleBuffer<f32>> = None;
    let mut all_samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(Error::IoError(_)) => break,
            Err(_) => break,
        };

        // Skip packets from other tracks
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
                tracing::warn!("Skipping corrupted packet: {}", m);
            }
            Err(_) => continue,
        }
    }

    if all_samples.is_empty() {
        return Err(AudioIoError::DecodeError("File contains no audio data".to_string()));
    }

    Ok((all_samples, sample_rate, channels as u16, SampleType::Float32))
}

pub fn probe_file(path: &Path) -> AudioMeta {
    // Try to get metadata from various sources
    let mut meta = if let Ok(meta) = probe_with_symphonia(path) {
        meta
    } else if let Ok(meta) = probe_with_lofty(path) {
        meta
    } else if let Ok(meta) = probe_with_ffprobe(path) {
        meta
    } else {
        let ext = path.extension().and_then(|e| e.to_str()).map(|e| e.to_lowercase()).unwrap_or_default();
        let status = match ext.as_str() { "wma" | "wv" => ProbeStatus::Unavailable, _ => ProbeStatus::Broken };
        AudioMeta { duration_secs: 0.0, sample_rate: 0, channels: 0, sample_type: SampleType::Compressed, comment: None, status }
    };

    // Always try to extract comments via lofty (works for all formats)
    if meta.comment.is_none() {
        if let Ok(tagged_file) = lofty::read_from_path(path) {
            meta.comment = extract_comment(&tagged_file);
        }
    }

    meta
}

fn probe_with_symphonia(path: &Path) -> Result<AudioMeta> {
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = std::fs::File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) { hint.with_extension(ext); }

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| AudioIoError::DecodeError(format!("Probe: {}", e)))?;

    let track = probed.format.tracks().first()
        .ok_or_else(|| AudioIoError::DecodeError("No track".to_string()))?;
    let params = &track.codec_params;
    let duration_secs = params.n_frames.unwrap_or(0) as f64 / params.sample_rate.unwrap_or(44100) as f64;
    let sample_rate = params.sample_rate.unwrap_or(44100);
    let channels = params.channels.map(|c| c.count() as u16).unwrap_or(1);
    let sample_type = match params.bits_per_sample {
        Some(8) => SampleType::UInt8, Some(16) => SampleType::Int16,
        Some(24) => SampleType::Int24, Some(32) => SampleType::Int32,
        _ => SampleType::Float32,
    };

    Ok(AudioMeta { duration_secs, sample_rate, channels, sample_type, comment: None, status: ProbeStatus::Ok })
}

fn probe_with_lofty(path: &Path) -> Result<AudioMeta> {
    use lofty::file::AudioFile;

    let tagged_file = lofty::read_from_path(path)
        .map_err(|e| AudioIoError::DecodeError(format!("Lofty: {}", e)))?;

    let properties = tagged_file.properties();
    let duration_secs = properties.duration().as_secs_f64();
    let sample_rate = properties.sample_rate().unwrap_or(44100);
    let channels = properties.channels().unwrap_or(1) as u16;
    let bit_depth = properties.bit_depth().unwrap_or(16);
    let sample_type = match bit_depth {
        8 => SampleType::UInt8, 16 => SampleType::Int16,
        24 => SampleType::Int24, 32 => SampleType::Float32,
        _ => SampleType::Float32,
    };

    // Extract comment from various tag formats
    let comment: Option<String> = extract_comment(&tagged_file);

    Ok(AudioMeta { duration_secs, sample_rate, channels, sample_type, comment, status: ProbeStatus::Ok })
}

/// Extract comment/description from audio file tags
/// Supports: WAV (ICMT), FLAC (DESCRIPTION), ID3 (COMM), etc.
fn extract_comment(tagged_file: &lofty::file::TaggedFile) -> Option<String> {
    use lofty::file::TaggedFileExt;
    use lofty::tag::{Accessor, ItemKey, ItemValue};

    // Try all tags (not just primary/first) to find comments
    for tag in tagged_file.tags() {
        // Try standard comment keys in order of preference
        // 1. ItemKey::Comment (WAV ICMT, ID3 COMM, etc.)
        if let Some(comment) = tag.get_string(&ItemKey::Comment) {
            return Some(comment.to_string());
        }

        // 2. ItemKey::Description (FLAC Vorbis DESCRIPTION)
        if let Some(desc) = tag.get_string(&ItemKey::Description) {
            return Some(desc.to_string());
        }

        // 3. Try Accessor trait method
        if let Some(comment) = tag.comment() {
            return Some(comment.to_string());
        }

        // 4. Search through all items for comment-like keys
        for item in tag.items() {
            let item_key_str = format!("{:?}", item.key());
            let key_lower = item_key_str.to_lowercase();

            // Check if key contains "comment" or "description"
            if key_lower.contains("comment") || key_lower.contains("description") {
                // Try to extract text value
                match item.value() {
                    ItemValue::Text(text) | ItemValue::Locator(text) => {
                        if !text.is_empty() {
                            return Some(text.to_string());
                        }
                    }
                    _ => {}
                }
            }
        }

        // 5. For ID3 tags, try to find COMM frame directly
        // ID3v2 stores comments in COMM frames with language and description
        for item in tag.items() {
            match item.value() {
                ItemValue::Text(text) | ItemValue::Locator(text) => {
                    let key_str = format!("{:?}", item.key());
                    // If we found any text item with "COMM" or similar in key
                    if key_str.contains("COMM") || key_str.contains("Comment") {
                        if !text.is_empty() {
                            return Some(text.to_string());
                        }
                    }
                }
                _ => {}
            }
        }
    }

    None
}

fn probe_with_ffprobe(path: &Path) -> Result<AudioMeta> {
    ffmpeg::ffprobe(path)
}
