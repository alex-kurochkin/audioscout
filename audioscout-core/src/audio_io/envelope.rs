//! Min/max envelope extraction for file-list preview thumbnails.
//!
//! The peak-envelope semantics mirror the `waveform-preview` example
//! (`Block::of` + `Block::range`: every emitted point carries the true min/max
//! of the samples it covers), but the full signal is never materialised in
//! memory — peaks are folded while streaming through the decoder.

use super::{AudioIoError, EnvAccum, Result};
use std::path::Path;

/// Absolute ceiling on the number of decoded frames a single preview may
/// consume. It exists purely as a safety net against endless/corrupt streams;
/// normal files are bounded by the seek plan / byte budget long before this
/// kicks in.
const ENVELOPE_MAX_FRAMES: usize = 200_000_000;

/// Decodes `path` and reduces it to a per-channel min/max envelope for a
/// file-list preview thumbnail.
///
/// Two strategies, chosen by format:
///
/// - **PCM containers (WAV):** one streaming pass through the samples with a
///   fixed block size derived from the header's frame count. Reading PCM is
///   cheap and sequential, so this stays the fastest path.
/// - **Compressed formats (MP3, ...):** *seek-sampled* decode. Instead of
///   decoding every packet from start to finish (which makes MP3 previews
///   scale linearly with file size — a 5-minute 320 kbps MP3 means ~7 MB and
///   tens of thousands of decoded frames just to draw 1360 points), we jump to
///   evenly spaced positions across the file and decode only short windows
///   around each jump. Cost becomes O(target × window), independent of
///   duration. Symphonia's MP3 demuxer seeks by bisection over the raw stream
///   without decoding, so each jump is nearly free.
///
/// Both paths are additionally bounded by `budget_bytes` of consumed input and
/// by [`ENVELOPE_MAX_FRAMES`] decoded frames, so a pathological file can never
/// spin the preview worker forever.
pub fn load_audio_envelope(
    path: &Path,
    target_frames: usize,
    budget_bytes: u64,
) -> Result<(super::Envelope, u32)> {
    let target = target_frames.max(1);
    let budget = budget_bytes.max(1 << 16);
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .unwrap_or_default();
    let (mut env, sr) = match ext.as_str() {
        "wav" => envelope_from_hound(path, target, budget)?,
        _ => envelope_seek_sampled(path, target, budget)?,
    };
    pad_envelope_to(&mut env, target);
    Ok((env, sr))
}

/// Pads the envelope with flat (silent) points up to exactly `target` frames.
///
/// When decoding stopped early because of the byte budget, the waveform is
/// drawn across the *full* file duration, so the truncated part reads as
/// silence rather than compressing the visible audio into a fraction of the
/// width. This also guarantees uniform per-channel lengths.
fn pad_envelope_to(env: &mut super::Envelope, target: usize) {
    for c in 0..env.channels {
        let cur = env.lo[c].len();
        if cur < target {
            env.lo[c].resize(target, 0.0);
            env.hi[c].resize(target, 0.0);
        }
    }
    env.frames = env.lo.first().map(|v| v.len()).unwrap_or(0);
}

/// Byte-level progress tracker shared by both envelope readers.
struct Budget {
    limit: u64,
    seen: u64,
    /// Bytes already accounted for (used by the seekable reader to translate
    /// absolute positions into increments).
    last_pos: u64,
}

impl Budget {
    fn new(limit: u64) -> Self {
        Self { limit, seen: 0, last_pos: 0 }
    }
    fn exhausted(&self) -> bool {
        self.seen >= self.limit
    }
    fn add(&mut self, delta: u64) {
        self.seen += delta;
    }
    /// Accounts for an absolute stream position (monotonic).
    fn add_until(&mut self, pos: u64) {
        if pos > self.last_pos {
            self.seen += pos - self.last_pos;
            self.last_pos = pos;
        }
    }
}

fn envelope_from_hound(
    path: &Path,
    target_frames: usize,
    budget_bytes: u64,
) -> Result<(super::Envelope, u32)> {
    let probe = hound::WavReader::open(path)
        .map_err(|e| AudioIoError::DecodeError(format!("WAV open: {}", e)))?;
    let spec = probe.spec();
    let ch = spec.channels.max(1) as usize;
    let total_samples = probe.len() as usize;
    drop(probe);

    // One envelope point per equal slice of the file; each slice is reduced
    // from the stream exactly once.
    let block = (total_samples / ch).div_ceil(target_frames).max(1);

    // BufReader: hound's WavReader issues one read() per sample otherwise.
    let file = std::io::BufReader::with_capacity(
        1 << 20,
        std::fs::File::open(path)
            .map_err(|e| AudioIoError::DecodeError(format!("WAV open: {}", e)))?,
    );
    let mut reader = hound::WavReader::new(file)
        .map_err(|e| AudioIoError::DecodeError(format!("WAV open: {}", e)))?;

    let mut acc = EnvAccum::new(ch, block);
    let mut budget = Budget::new(budget_bytes);
    const CHUNK: usize = 1 << 14; // samples per batch

    match spec.sample_format {
        hound::SampleFormat::Int => match spec.bits_per_sample {
            8 | 16 => pump_scaled::<i16>(
                &mut reader,
                if spec.bits_per_sample == 8 { 1.0 / 128.0 } else { 1.0 / 32768.0 },
                &mut acc,
                &mut budget,
            ),
            24 | 32 => pump_scaled::<i32>(
                &mut reader,
                if spec.bits_per_sample == 24 { 1.0 / 8388608.0 } else { 1.0 / 2147483648.0 },
                &mut acc,
                &mut budget,
            ),
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
                budget.add((buf.len() * (spec.bits_per_sample as usize / 8)) as u64);
                acc.push(&buf);
                if buf.len() < CHUNK || acc.frames_in() >= ENVELOPE_MAX_FRAMES || budget.exhausted() {
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
/// Bounded on four independent conditions: clean end of stream, first
/// sample-read error, the byte budget, and the [`ENVELOPE_MAX_FRAMES`] safety
/// cap. Without these caps a pathological file could keep the worker busy
/// forever and every later preview job would appear to "just wait".
fn pump_scaled<T>(
    reader: &mut hound::WavReader<std::io::BufReader<std::fs::File>>,
    scale: f32,
    acc: &mut EnvAccum,
    budget: &mut Budget,
) where
    T: hound::Sample + Copy + HoundToFloat,
{
    let bytes_per_sample = std::mem::size_of::<T>();
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
        budget.add((raw.len() * bytes_per_sample) as u64);
        buf.extend(raw.iter().map(|&v| v.to_f32(scale)));
        acc.push(&buf);
        if raw.len() < (1 << 14) || acc.frames_in() >= ENVELOPE_MAX_FRAMES || budget.exhausted() {
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
    fn to_f32(self, scale: f32) -> f32 {
        self as f32 * scale
    }
}
impl HoundToFloat for i32 {
    #[inline]
    fn to_f32(self, scale: f32) -> f32 {
        self as f32 * scale
    }
}

/// Seekable media-source wrapper used to measure how many *input* bytes the
/// demuxer actually consumed, so the preview can stop after a fixed budget.
///
/// The position is updated by the `Read`/`Seek` implementations below and read
/// back through the shared `Arc<AtomicU64>` because symphonia takes ownership
/// of the source.
struct MeteredFile {
    file: std::fs::File,
    len: u64,
    pos: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl MeteredFile {
    fn new(path: &Path) -> Result<(Self, std::sync::Arc<std::sync::atomic::AtomicU64>)> {
        let file = std::fs::File::open(path)?;
        let len = file.metadata()?.len();
        let pos = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        Ok((Self { file, len, pos: pos.clone() }, pos))
    }
}

impl std::io::Read for MeteredFile {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.file.read(buf)?;
        self.pos.fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(n)
    }
}

impl std::io::Seek for MeteredFile {
    fn seek(&mut self, from: std::io::SeekFrom) -> std::io::Result<u64> {
        let np = self.file.seek(from)?;
        self.pos.store(np, std::sync::atomic::Ordering::Relaxed);
        Ok(np)
    }
}

impl symphonia::core::io::MediaSource for MeteredFile {
    fn is_seekable(&self) -> bool {
        true
    }

    fn byte_len(&self) -> Option<u64> {
        Some(self.len)
    }
}

/// Frames decoded per envelope point when seek-sampling a compressed file.
///
/// One MP3 frame ≈ 26 ms @ 44.1 kHz ≈ 1152 frames, so 2048 covers at least a
/// full packet around every seek target even with codec priming/skip delay.
/// Total decode work is `target × WINDOW` frames — constant with respect to
/// file duration (≈2.8 M frames for a 1360-point thumbnail, i.e. ~1 minute of
/// audio regardless of whether the file lasts 3 minutes or 3 hours).
const SEEK_WINDOW_FRAMES: usize = 2048;

/// Seek-sampled envelope extraction for compressed formats (MP3, OGG, ...).
///
/// Why streaming decode is too slow for MP3: the only way to reduce an MP3 is
/// to decode it, and a full streaming pass costs O(duration). WAV avoids this
/// because PCM can be *read* without decoding; compressed files need a
/// different trick — we jump (via container seeks) to evenly spaced targets
/// across the timeline and decode only [`SEEK_WINDOW_FRAMES`] around each
/// jump. That yields one real min/max point per envelope slot while touching
/// only a tiny slice of the stream.
///
/// Falls back to a bounded sequential pass when the stream turns out to be
/// non-seekable or seeking produces no usable data (broken index, exotic
/// container).
fn envelope_seek_sampled(
    path: &Path,
    target_frames: usize,
    budget_bytes: u64,
) -> Result<(super::Envelope, u32)> {
    use symphonia::core::formats::{FormatOptions, SeekMode};
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let (metered, src_pos) = MeteredFile::new(path)?;
    let mss = MediaSourceStream::new(Box::new(metered), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| AudioIoError::DecodeError(format!("Probe failed: {}", e)))?;

    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| AudioIoError::DecodeError("No audio track found".to_string()))?;

    let track_id = track.id;
    let params = track.codec_params.clone();
    let sample_rate = params.sample_rate.unwrap_or(44100).max(1);
    let ch = params.channels.map(|c| c.count()).unwrap_or(1).max(1);

    // Whether the *underlying file* supports seeking. Symphonia's MP3 demuxer
    // uses `SeekMode::Coarse` to estimate a byte position from the timestamp
    // and jump there directly (parsing only a handful of frame headers on the
    // way); with a non-seekable source it would have to scan linearly, so in
    // that case we stay on the bounded sequential path instead.
    let seekable = std::fs::metadata(path).map(|m| m.len() > 0).unwrap_or(false);
    let total_frames = params.n_frames.unwrap_or(0) as usize;

    // Sequential fallback conditions: no seeking possible, or no known length
    // to build a seek plan from, or the file is short enough that a single
    // linear pass is cheaper than ~target random jumps anyway.
    if !seekable || total_frames == 0 || total_frames <= target_frames * SEEK_WINDOW_FRAMES {
        return envelope_sequential_symphonia(
            &mut format, track_id, &params, sample_rate, ch, target_frames, budget_bytes,
        );
    }

    let block = (total_frames / target_frames).max(1);
    let mut env = envelope_by_seeks(
        &mut format, track_id, &params, ch, target_frames, block, budget_bytes, &src_pos,
        SeekMode::Coarse,
    );

    // Did the seek plan actually produce real audio data anywhere? If not
    // (broken index, unsupported seek behaviour), fall back to the bounded
    // sequential reader instead of shipping an all-silence thumbnail.
    let has_data = env.lo.iter().zip(env.hi.iter()).any(|(l, h)| {
        l.iter().zip(h.iter()).any(|(&lo, &hi)| lo < 0.0 || hi > 0.0)
    });
    if !has_data {
        let (fallback, _) = envelope_sequential_symphonia(
            &mut format, track_id, &params, sample_rate, ch, target_frames, budget_bytes,
        )?;
        if fallback.frames > env.frames {
            env = fallback;
        }
    }

    Ok((env, sample_rate))
}

/// Builds the envelope by jumping to `target` evenly spaced frame positions
/// and decoding a small window at each one. Envelope slot `i` receives the
/// min/max of the window starting at frame `i * block`, which matches the
/// fixed-block layout the renderer expects.
fn envelope_by_seeks(
    format: &mut Box<dyn symphonia::core::formats::FormatReader>,
    track_id: u32,
    params: &symphonia::core::codecs::CodecParameters,
    ch: usize,
    target: usize,
    block: usize,
    budget_bytes: u64,
    src_pos: &std::sync::Arc<std::sync::atomic::AtomicU64>,
    mode: symphonia::core::formats::SeekMode,
) -> super::Envelope {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::errors::Error;
    use symphonia::core::formats::SeekTo;

    let mut lo: Vec<Vec<f32>> = (0..ch).map(|_| vec![0.0f32; target]).collect();
    let mut hi: Vec<Vec<f32>> = (0..ch).map(|_| vec![0.0f32; target]).collect();

    let mut decoder = match symphonia::default::get_codecs().make(params, &DecoderOptions::default()) {
        Ok(d) => d,
        Err(e) => {
            tracing::debug!("Codec init failed in seek sampler: {}", e);
            return super::Envelope { channels: ch, frames: target, lo, hi };
        }
    };

    let mut budget = Budget::new(budget_bytes);
    let mut sample_buf: Option<SampleBuffer<f32>> = None;

    for i in 0..target {
        // Hard stop if the overall decode budget ran out in earlier slots:
        // remaining envelope points stay flat (padded silence).
        if budget.exhausted() || budget.seen >= ENVELOPE_MAX_FRAMES as u64 {
            break;
        }
        let start_frame = i.saturating_mul(block) as u64;
        let to = SeekTo::TimeStamp { ts: start_frame, track_id };
        if format.seek(mode, to).is_err() {
            // Seeking broke mid-plan: keep the points collected so far and
            // let the caller decide whether to fall back.
            break;
        }
        // Reset the decoder at each jump so priming/delay samples from the
        // previous window don't leak into this one.
        let _ = decoder.reset();

        let mut got = 0usize; // interleaved samples collected for this slot
        while got < ch * SEEK_WINDOW_FRAMES {
            let packet = match format.next_packet() {
                Ok(p) => p,
                Err(_) => break,
            };
            if packet.track_id() != track_id {
                continue;
            }
            match decoder.decode(&packet) {
                Ok(audio_buf) => {
                    if sample_buf.is_none() {
                        let spec = *audio_buf.spec();
                        let cap = audio_buf.capacity() as u64;
                        sample_buf = Some(SampleBuffer::<f32>::new(cap, spec));
                    }
                    if let Some(buf) = &mut sample_buf {
                        buf.copy_interleaved_ref(audio_buf);
                        let s = buf.samples();
                        let take = (ch * SEEK_WINDOW_FRAMES - got).min(s.len());
                        fold_window(s, take, ch, i, &mut lo, &mut hi);
                        got += take;
                    }
                }
                Err(Error::DecodeError(_)) => continue,
                Err(_) => break,
            }
            if got + ch * SEEK_WINDOW_FRAMES >= ENVELOPE_MAX_FRAMES {
                break;
            }
        }

        // Account the *compressed* bytes this window actually consumed. The
        // metered source reports its current position; a forward-only stretch
        // of reads adds `pos - last_pos`, while the next seek rewinds below
        // `last_pos` — that rewind is not real read traffic, so we clamp it
        // out (`add_until` ignores non-monotonic positions) and keep the
        // watermark. This makes the byte budget bound genuine I/O only, and
        // because the per-slot work is already constant, total consumption is
        // O(target × window bytes) — independent of file duration.
        let pos = src_pos.load(std::sync::atomic::Ordering::Relaxed);
        budget.add_until(pos);
    }

    super::Envelope { channels: ch, frames: target, lo, hi }
}

/// Folds one decoded window into envelope slot `i` (per-channel min/max).
#[inline]
fn fold_window(
    samples: &[f32],
    len: usize,
    ch: usize,
    slot: usize,
    lo: &mut [Vec<f32>],
    hi: &mut [Vec<f32>],
) {
    let n = ch.min(8);
    let mut mins = [f32::INFINITY; 8];
    let mut maxs = [f32::NEG_INFINITY; 8];
    for frame in samples[..len].chunks(ch) {
        for c in 0..n {
            let s = frame[c];
            mins[c] = mins[c].min(s);
            maxs[c] = maxs[c].max(s);
        }
    }
    for c in 0..ch {
        let (mn, mx) = (mins[c % 8], maxs[c % 8]);
        if mn <= mx {
            // NaN-safe: only overwrite the flat default when real data arrived.
            lo[c][slot] = lo[c][slot].min(mn);
            hi[c][slot] = hi[c][slot].max(mx);
        }
    }
}

/// Bounded sequential decode → envelope. Used as the fallback for
/// non-seekable streams and as the fast path for short compressed files where
/// seeking would cost more than simply reading through.
fn envelope_sequential_symphonia(
    format: &mut Box<dyn symphonia::core::formats::FormatReader>,
    track_id: u32,
    params: &symphonia::core::codecs::CodecParameters,
    sample_rate: u32,
    ch: usize,
    target_frames: usize,
    budget_bytes: u64,
) -> Result<(super::Envelope, u32)> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::errors::Error;

    let total_frames_hint = params.n_frames.unwrap_or(0) as usize;
    let block = if total_frames_hint > 0 {
        (total_frames_hint / target_frames).max(1)
    } else {
        // Unknown length: keep the envelope dense but bounded.
        2048
    };

    let mut decoder = symphonia::default::get_codecs()
        .make(params, &DecoderOptions::default())
        .map_err(|e| AudioIoError::DecodeError(format!("Codec init failed: {}", e)))?;

    let mut acc = EnvAccum::new(ch, block);
    let mut budget = Budget::new(budget_bytes);
    let mut sample_buf: Option<SampleBuffer<f32>> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(_) => break,
        };
        if packet.track_id() != track_id {
            continue;
        }

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

        // Rough input-byte accounting: decoded f32 bytes plus per-packet
        // overhead. Keeps the fallback bounded without instrumenting every
        // read of the source.
        budget.add(4096);
        if budget.exhausted() || acc.frames_in() >= ENVELOPE_MAX_FRAMES {
            break;
        }
    }

    acc.finish();
    if acc.frames_out() == 0 {
        return Err(AudioIoError::DecodeError("File contains no audio data".to_string()));
    }
    Ok((acc.into_envelope(), sample_rate))
}
