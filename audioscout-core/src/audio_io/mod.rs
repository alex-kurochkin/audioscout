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

/// Streaming min/max decimator: accumulates per-channel `[min, max]` pairs for
/// a fixed-size input block of frames and emits one envelope frame
/// `(min, max)` per channel when the block is full.
///
/// This reproduces the peak-envelope semantics of the `waveform-preview`
/// example (which builds its Peaks tree over the *full* signal) without ever
/// holding the whole decoded file in memory: every emitted point carries the
/// true min/max of the skipped region, so no peaks are lost regardless of how
/// aggressively the signal is decimated.
pub struct MinMaxDecimator {
    channels: usize,
    block_frames: usize,
    mins: Vec<f32>,
    maxs: Vec<f32>,
    filled: usize,
    // Emitted (min, max) values, stored channel-major: lo[c*n .. (c+1)*n].
    out_lo: Vec<f32>,
    out_hi: Vec<f32>,
}

impl MinMaxDecimator {
    pub fn new(channels: usize, block_frames: usize) -> Self {
        let ch = channels.max(1);
        Self {
            channels: ch,
            block_frames: block_frames.max(1),
            mins: vec![f32::INFINITY; ch],
            maxs: vec![f32::NEG_INFINITY; ch],
            filled: 0,
            out_lo: Vec::new(),
            out_hi: Vec::new(),
        }
    }

    /// Feeds interleaved samples (frame-major, `channels` values per frame).
    /// A trailing partial frame is ignored — the caller should keep leftover
    /// samples and feed them again after appending more data.
    pub fn push(&mut self, samples: &[f32]) {
        let ch = self.channels;
        let frames = samples.len() / ch;
        if frames == 0 {
            return;
        }
        // Process whole blocks directly from the input slice (branch-free
        // inner loop), then carry any remainder in the accumulator.
        let mut off = 0usize;
        while self.filled + frames - off >= self.block_frames {
            let take = self.block_frames - self.filled;
            let end = off + take;
            accumulate(&mut self.mins, &mut self.maxs, &samples[off * ch..end * ch], ch);
            off = end;
            self.filled = 0;
            self.flush();
        }
        if off < frames {
            accumulate(&mut self.mins, &mut self.maxs, &samples[off * ch..frames * ch], ch);
            self.filled += frames - off;
        }
    }

    fn flush(&mut self) {
        for c in 0..self.channels {
            // Empty/poisoned block → flat line at zero (mirrors Block::range()).
            let (lo, hi) = if self.mins[c] <= self.maxs[c] {
                (self.mins[c], self.maxs[c])
            } else {
                (0.0, 0.0)
            };
            self.out_lo.push(lo);
            self.out_hi.push(hi);
            self.mins[c] = f32::INFINITY;
            self.maxs[c] = f32::NEG_INFINITY;
        }
        self.filled = 0;
    }

    /// Flushes the partially-filled block (call once at end of stream).
    pub fn finish(&mut self) {
        if self.filled > 0 {
            self.flush();
        }
    }

    pub fn frames_out(&self) -> usize {
        self.out_lo.len() / self.channels
    }

    /// Consumes the decimator and returns the channel-major min/max envelope.
    pub fn into_envelope(self) -> Envelope {
        Envelope {
            channels: self.channels,
            frames: self.frames_out(),
            lo: self.out_lo,
            hi: self.out_hi,
        }
    }
}

#[inline]
fn accumulate(mins: &mut [f32], maxs: &mut [f32], samples: &[f32], ch: usize) {
    let frames = samples.len() / ch;
    for f in 0..frames {
        let base = f * ch;
        for c in 0..ch {
            let s = samples[base + c];
            let s = if s.is_finite() { s } else { 0.0 };
            if s < mins[c] { mins[c] = s; }
            if s > maxs[c] { maxs[c] = s; }
        }
    }
}

/// Channel-major min/max envelope produced by [`MinMaxDecimator`].
pub struct Envelope {
    pub channels: usize,
    pub frames: usize,
    /// `lo[c * frames + i]` — minimum of channel `c` at envelope frame `i`.
    pub lo: Vec<f32>,
    pub hi: Vec<f32>,
}

/// Loads audio and reduces it to a min/max envelope with at most
/// `max_frames_per_channel` envelope frames per channel, streaming through
/// the decoder so the full waveform never has to be materialised.
///
/// Used by the preview renderer: a small table thumbnail (~340 px wide) needs
/// only a few hundred envelope points per track — decoding and buffering the
/// entire multi-hundred-megabyte sample array is what made rendering crawl on
/// large files.
pub fn load_audio_envelope(
    path: &Path,
    max_frames_per_channel: usize,
) -> Result<(Envelope, u32)> {
    let ext = path.extension().and_then(|e| e.to_str()).map(|e| e.to_lowercase()).unwrap_or_default();
    match ext.as_str() {
        "wav" => envelope_from_hound(path, max_frames_per_channel),
        _ => envelope_with_symphonia(path, max_frames_per_channel),
    }
}

fn envelope_from_hound(path: &Path, max_frames: usize) -> Result<(Envelope, u32)> {
    let reader = hound::WavReader::open(path)
        .map_err(|e| AudioIoError::DecodeError(format!("WAV open: {}", e)))?;
    let spec = reader.spec();
    let ch = spec.channels.max(1) as usize;
    let total_samples = reader.len() as usize;
    let block = (total_samples / ch).div_ceil(max_frames.max(1)).max(1);
    let mut dec = MinMaxDecimator::new(ch, block);

    // Buffered reader: hound's WavReader does one read() call per sample
    // without buffering, which dominates decode time for PCM files.
    let file = std::io::BufReader::with_capacity(
        1 << 20,
        std::fs::File::open(path)
            .map_err(|e| AudioIoError::DecodeError(format!("WAV open: {}", e)))?,
    );
    let mut reader = hound::WavReader::new(file)
        .map_err(|e| AudioIoError::DecodeError(format!("WAV open: {}", e)))?;

    match spec.sample_format {
        hound::SampleFormat::Int => match spec.bits_per_sample {
            8 => {
                let scale = 1.0 / 128.0;
                feed_iter(&mut reader.into_samples::<i16>().map(move |s| s.map(|v| v as f32 * scale)), &mut dec);
            }
            16 => {
                let scale = 1.0 / 32768.0;
                feed_iter(&mut reader.into_samples::<i16>().map(move |s| s.map(|v| v as f32 * scale)), &mut dec);
            }
            24 | 32 => {
                let scale = if spec.bits_per_sample == 24 { 1.0 / 8388608.0 } else { 1.0 / 2147483648.0 };
                feed_iter(&mut reader.into_samples::<i32>().map(move |s| s.map(|v| v as f32 * scale)), &mut dec);
            }
            bits => return Err(AudioIoError::UnsupportedFormat(format!("Unsupported bits: {}", bits))),
        },
        hound::SampleFormat::Float => {
            feed_iter(&mut reader.into_samples::<f32>(), &mut dec);
        }
    }

    dec.finish();
    Ok((dec.into_envelope(), spec.sample_rate))
}

/// Drives a fallible sample iterator through the decimator in chunks,
/// buffering only a small window of samples at a time.
fn feed_iter<I>(mut iter: I, dec: &mut MinMaxDecimator)
where
    I: Iterator<Item = std::result::Result<f32, hound::Error>>,
{
    const CHUNK: usize = 1 << 15;
    let ch = dec.channels;
    let mut buf: Vec<f32> = Vec::with_capacity(CHUNK + ch);
    while let Some(v) = iter.next() {
        if let Ok(s) = v {
            buf.push(s);
        }
        if buf.len() >= CHUNK + ch {
            let usable = (buf.len() / ch) * ch;
            dec.push(&buf[..usable]);
            buf.clear();
        }
    }
    let usable = (buf.len() / ch) * ch;
    dec.push(&buf[..usable]);
}

fn envelope_with_symphonia(path: &Path, max_frames: usize) -> Result<(Envelope, u32)> {
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
        (total_frames_hint / max_frames.max(1)).max(1)
    } else {
        // Unknown length: pick a block that keeps the envelope dense enough.
        1024
    };

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| AudioIoError::DecodeError(format!("Codec init failed: {}", e)))?;

    let mut dec = MinMaxDecimator::new(ch, block);
    let mut sample_buf: Option<SampleBuffer<f32>> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(Error::IoError(_)) => break,
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
                    dec.push(buf.samples());
                }
            }
            Err(Error::DecodeError(m)) => {
                tracing::warn!("Skipping corrupted packet: {}", m);
            }
            Err(_) => continue,
        }
    }

    dec.finish();
    if dec.frames_out() == 0 {
        return Err(AudioIoError::DecodeError("File contains no audio data".to_string()));
    }
    Ok((dec.into_envelope(), sample_rate))
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
