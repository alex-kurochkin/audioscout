pub mod envelope;
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
    // Formats symphonia cannot decode at all (tz §2.4): straight to the
    // external-ffmpeg fallback → cached session-temp WAV → hound read.
    if matches!(ext.as_str(), "wma" | "wv") {
        return load_via_ffmpeg(path);
    }
    match ext.as_str() {
        "wav" => wav::WavFile::load(path),
        "aiff" | "aif" => aiff::AiffFile::load(path),
        "caf" => caf::CafFile::load(path),
        _ => match load_with_symphonia(path) {
            // Native chain failed — last resort is ffmpeg (§2.4:
            // native → temp-WAV → reload).
            Err(_) if ffmpeg::ffmpeg_available() => load_via_ffmpeg(path),
            other => other,
        },
    }
}

/// ffmpeg-fallback chain: decode into the session temp cache (`pcm_f32le`,
/// source sample rate untouched) and reload the resulting WAV natively.
fn load_via_ffmpeg(path: &Path) -> Result<(Vec<f32>, u32, u16, SampleType)> {
    let tmp = ffmpeg::decode_to_temp_wav(path)?;
    // pcm_f32le output always round-trips through the WAV reader; whatever
    // exotic sub-chunking survives, hound reports the concrete type.
    wav::WavFile::load(&tmp)
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
pub(crate) struct EnvAccum {
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
    pub(crate) fn new(channels: usize, block_frames: usize) -> Self {
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
    pub(crate) fn push(&mut self, samples: &[f32]) {
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

    pub(crate) fn finish(&mut self) {
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
    pub(crate) fn frames_in(&self) -> usize {
        self.frames_in
    }

    pub(crate) fn into_envelope(self) -> Envelope {
        Envelope { channels: self.channels, frames: self.frames_out(), lo: self.lo, hi: self.hi }
    }
}

/// Cheap header-only probe used by the preview pipeline: reads just enough
/// of the file to learn the channel count / sample rate. Never decodes audio
/// data, so it is safe to call before committing to a full envelope pass.
pub fn probe_file_header(path: &Path) -> AudioMeta {
    let zero = || AudioMeta {
        duration_secs: 0.0,
        sample_rate: 0,
        channels: 0,
        sample_type: SampleType::Compressed,
        comment: None,
        status: ProbeStatus::Broken,
    };
    let ext = path.extension().and_then(|e| e.to_str()).map(|e| e.to_lowercase()).unwrap_or_default();
    if ext == "wav" {
        return match hound::WavReader::open(path) {
            Ok(r) => {
                let spec = r.spec();
                AudioMeta {
                    duration_secs: r.len() as f64 / spec.sample_rate.max(1) as f64,
                    sample_rate: spec.sample_rate,
                    channels: spec.channels,
                    sample_type: match spec.sample_format {
                        hound::SampleFormat::Float => SampleType::Float32,
                        hound::SampleFormat::Int => match spec.bits_per_sample {
                            8 => SampleType::UInt8,
                            16 => SampleType::Int16,
                            24 => SampleType::Int24,
                            32 => SampleType::Int32,
                            _ => SampleType::Compressed,
                        },
                    },
                    comment: None,
                    status: ProbeStatus::Ok,
                }
            }
            Err(_) => zero(),
        };
    }
    match probe_with_symphonia(path) {
        Ok(m) => m,
        Err(_) => zero(),
    }
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
        // wma/wv are technically fine — they simply need the external ffmpeg
        // decoder (§2.3). With ffprobe present we can still show duration,
        // rate and tags even when decoding is impossible, so the row renders
        // as Unavailable (neutral colour, load/playback blocked) instead of
        // Broken.
        if matches!(ext.as_str(), "wma" | "wv") {
            match ffmpeg::ffprobe_meta(path) {
                Ok(Some(mut meta)) => {
                    meta.status = ProbeStatus::Unavailable;
                    meta
                }
                _ => AudioMeta { duration_secs: 0.0, sample_rate: 0, channels: 0, sample_type: SampleType::Compressed, comment: None, status: ProbeStatus::Unavailable },
            }
        } else {
            AudioMeta { duration_secs: 0.0, sample_rate: 0, channels: 0, sample_type: SampleType::Compressed, comment: None, status: ProbeStatus::Broken }
        }
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
pub(crate) fn extract_comment(tagged_file: &lofty::file::TaggedFile) -> Option<String> {
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
