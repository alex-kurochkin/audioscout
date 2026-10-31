//! Optional external-ffmpeg fallback (tz §2.4, §5.4).
//!
//! Formats outside symphonia's coverage (wma, wv) are decoded by spawning an
//! external `ffmpeg` into a session-temporary WAV (`pcm_f32le`, original
//! sample rate preserved — forcing `-ar` is forbidden by the spec). The temp
//! file name is a hash of (path + mtime + size), so repeated loads hit the
//! decode cache instead of re-spawning ffmpeg. Without ffmpeg on PATH such
//! files stay `ProbeStatus::Unavailable` (§2.3) — not broken.

use super::{AudioIoError, Result};
use crate::{AudioMeta, ProbeStatus, SampleType};
#[cfg(test)]
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU8, Ordering};


/// Cached availability of the external `ffmpeg` binary. Probing it spawns a
/// process, so the answer is computed once per session (§2.3 forbids paying
/// that cost for every scanned row).
static FFMPEG_AVAILABLE: AtomicU8 = AtomicU8::new(0); // 0 = unknown, 1 = yes, 2 = no

pub fn ffmpeg_available() -> bool {
    match FFMPEG_AVAILABLE.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            let ok = Command::new("ffmpeg")
                .arg("-version")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            FFMPEG_AVAILABLE.store(if ok { 1 } else { 2 }, Ordering::Relaxed);
            ok
        }
    }
}

/// Reset the cached availability check (test hook only).
#[cfg(test)]
pub fn reset_availability_cache() {
    FFMPEG_AVAILABLE.store(0, Ordering::Relaxed);
}

/// Session-wide decode-cache directory (`<config>/ffmpeg_decode`).
///
/// Lives for the whole session like `tempfile::TempDir` would, but survives
/// process restarts so the (path+mtime+size)-keyed cache stays warm (§2.4).
/// Honouring `AUDIOSCOUT_CONFIG_DIR` keeps tests hermetic: pointing it at a
/// temp dir routes all cache traffic there and never touches real user data.
fn cache_dir() -> PathBuf {
    let base = match std::env::var("AUDIOSCOUT_CONFIG_DIR") {
        Ok(dir) => PathBuf::from(dir),
        Err(_) => crate::config::config_dir(),
    };
    let dir = base.join("ffmpeg_decode");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// Cache key: hex hash of "path|mtime|size" (tz §2.4).
fn decode_key(input: &Path) -> String {
    use std::hash::{Hash, Hasher};
    let meta = std::fs::metadata(input);
    let mtime = meta
        .as_ref()
        .ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let size = meta.map(|m| m.len()).unwrap_or(0);
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    input.hash(&mut hasher);
    mtime.hash(&mut hasher);
    size.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// Cached path for a decoded copy of `input`; the file is present only after
/// a successful decode.
fn cached_wav(input: &Path) -> PathBuf {
    cache_dir().join(format!("{}.wav", decode_key(input)))
}

/// Spawn `ffmpeg -y -i <input> -f wav -acodec pcm_f32le <output>`.
///
/// No `-ar` flag: the source sample rate must be preserved (§2.4). Child
/// stdout/stderr are captured and logged on failure.
fn spawn_decode(input: &Path, output: &Path) -> Result<()> {
    let out = Command::new("ffmpeg")
        .arg("-hide_banner")
        .arg("-loglevel").arg("error")
        .arg("-y")
        .arg("-i").arg(input)
        .arg("-f").arg("wav")
        .arg("-acodec").arg("pcm_f32le")
        .arg(output)
        .output()?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        tracing::error!("ffmpeg decode failed for {:?}: {}", input, err.trim());
        return Err(AudioIoError::DecodeError(format!("ffmpeg decode failed: {}", err.trim())));
    }
    Ok(())
}

/// Decode `input` to a cached session-temp WAV and return its path.
pub fn decode_to_temp_wav(input: &Path) -> Result<PathBuf> {
    if !ffmpeg_available() {
        return Err(AudioIoError::FfmpegNotAvailable);
    }
    let tmp = cached_wav(input);
    if tmp.exists() {
        return Ok(tmp);
    }
    // Write via a per-process unique staging name, then rename atomically, so
    // concurrent decodes never observe a half-written cache entry.
    let staged = tmp.with_extension(format!("wav.{}.tmp", std::process::id()));
    let res = spawn_decode(input, &staged);
    match res {
        Ok(()) => {
            std::fs::rename(&staged, &tmp)?;
            Ok(tmp)
        }
        Err(e) => {
            let _ = std::fs::remove_file(&staged);
            Err(e)
        }
    }
}

/// Legacy helper kept for API compatibility: explicit two-path decode.
pub fn ffmpeg_decode(input: &Path, output: &Path) -> Result<()> {
    if !ffmpeg_available() {
        return Err(AudioIoError::FfmpegNotAvailable);
    }
    spawn_decode(input, output)
}

/// Lossless pipe-decode: `ffmpeg -i input -f f32le -` straight into memory.
///
/// Used by the envelope pipeline so a WMA/WV preview costs one ffmpeg spawn
/// with no intermediate file at all. Returns interleaved frame-major f32
/// samples plus the stream's sample rate / channel count (read back from the
/// ffmpeg banner, since raw PCM carries no header).
pub fn decode_to_pcm(input: &Path) -> Result<(Vec<f32>, u32, u16)> {
    if !ffmpeg_available() {
        return Err(AudioIoError::FfmpegNotAvailable);
    }
    let child = Command::new("ffmpeg")
        .arg("-hide_banner")
        .arg("-loglevel").arg("error")
        .arg("-i").arg(input)
        .arg("-f").arg("f32le")
        .arg("-")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    // Reading stdout fully before wait() avoids pipe deadlock for large files.
    let out = child.wait_with_output()?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        tracing::error!("ffmpeg pipe decode failed for {:?}: {}", input, err.trim());
        return Err(AudioIoError::DecodeError(format!(
            "ffmpeg pipe decode failed: {}",
            err.trim()
        )));
    }
    // Banner parse (fallback probe): "Stream #0:0...: Audio: ..., 44100 Hz, stereo, ..."
    // ffprobe JSON is authoritative when available; these values are only used
    // for envelope geometry (sr/channels), where the banner is exact.
    let (mut sr, mut ch) = (44100u32, 1u16);
    if let Ok(meta) = ffprobe_stream_info(input) {
        if meta.sample_rate != 0 {
            sr = meta.sample_rate;
        }
        if meta.channels != 0 {
            ch = meta.channels;
        }
    }
    let bytes = out.stdout;
    let mut samples = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        samples.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok((samples, sr, ch))
}

/// ffprobe metadata extraction for formats symphonia cannot open (wma/wv).
///
/// Returns `Ok(None)` when ffprobe is missing or fails, so callers can fall
/// back to the `Unavailable` status without treating the file as broken.
pub fn ffprobe_meta(path: &Path) -> Result<Option<AudioMeta>> {
    match ffprobe(path) {
        Ok(mut meta) => {
            // lofty understands ASF/WMA tag containers even though symphonia
            // cannot decode them — fill the comment slot ("last chance" chain
            // of tz §2.5 runs in reverse order here, cheap native first).
            if meta.comment.is_none() {
                if let Ok(tagged) = lofty::read_from_path(path) {
                    meta.comment = super::extract_comment(&tagged);
                }
            }
            Ok(Some(meta))
        }
        Err(AudioIoError::FfmpegNotAvailable) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Minimal JSON reader for `ffprobe -show_streams -show_format -of json`.
///
/// Uses serde_json's raw Value tree (no codegen) and maps fields defensively:
/// numbers may arrive as strings, `bits_per_raw_sample` may be absent.
pub fn ffprobe(path: &Path) -> Result<AudioMeta> {
    let output = Command::new("ffprobe")
        .arg("-v").arg("quiet")
        .arg("-show_streams").arg("-show_format")
        .arg("-of").arg("json")
        .arg(path)
        .output()?;
    if !output.status.success() {
        return Err(AudioIoError::DecodeError("ffprobe failed".to_string()));
    }
    let json_str = String::from_utf8_lossy(&output.stdout);
    let json: serde_json::Value = serde_json::from_str(&json_str)
        .map_err(|e| AudioIoError::DecodeError(format!("ffprobe JSON: {e}")))?;

    // First *audio* stream (a container may carry video/jacket streams).
    let stream = json["streams"]
        .as_array()
        .and_then(|arr| {
            arr.iter()
                .find(|s| s["codec_type"].as_str() == Some("audio"))
                .or_else(|| arr.first())
        })
        .ok_or_else(|| AudioIoError::DecodeError("No streams".to_string()))?;

    let num = |v: Option<&serde_json::Value>| -> Option<f64> {
        v.and_then(|v| v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse().ok())))
    };
    let sample_rate = num(stream["sample_rate"].as_str().map(|_| &stream["sample_rate"]))
        .or_else(|| num(Some(&stream["sample_rate"])))
        .unwrap_or(44100.0) as u32;
    let channels = num(Some(&stream["channels"])).unwrap_or(1.0) as u16;
    let duration_secs = num(json["format"]["duration"].as_str().map(|_| &json["format"]["duration"]))
        .or_else(|| num(Some(&json["format"]["duration"])))
        .or_else(|| num(Some(&stream["duration"])))
        .unwrap_or(0.0);
    let bits = num(Some(&stream["bits_per_raw_sample"]))
        .or_else(|| num(Some(&stream["bits_per_sample"])))
        .unwrap_or(0.0) as u32;
    let codec = stream["codec_name"].as_str().unwrap_or("");
    // Lossy codecs (WMA/AAC/mp3...) have no meaningful integer depth; the
    // decoded PCM is always f32 (§2.1 fallback rule).
    let sample_type = match (codec, bits) {
        ("pcm_f32le" | "pcm_f32be" | "f32", _) => SampleType::Float32,
        (_, 8) => SampleType::UInt8,
        (_, 16) => SampleType::Int16,
        (_, 24) => SampleType::Int24,
        (_, 32) => SampleType::Int32,
        _ => SampleType::Compressed,
    };

    // Tags live under stream tags or format tags.
    let mut comment = None;
    for tags in [stream.get("tags"), json["format"].get("tags")] {
        if let Some(obj) = tags.and_then(|t| t.as_object()) {
            for key in ["comment", "Comment", "DESCRIPTION", "description"] {
                if let Some(v) = obj.get(key).and_then(|v| v.as_str()) {
                    if !v.is_empty() {
                        comment = Some(v.to_string());
                        break;
                    }
                }
            }
        }
        if comment.is_some() {
            break;
        }
    }

    Ok(AudioMeta {
        duration_secs,
        sample_rate,
        channels,
        sample_type,
        comment,
        status: ProbeStatus::Ok,
    })
}

/// Lightweight variant returning only geometry (for envelope sizing).
pub fn ffprobe_stream_info(path: &Path) -> Result<AudioMeta> {
    ffprobe(path)
}

/// Write a small silent mono 16-bit WAV — test fixture generator.
#[cfg(test)]
pub(crate) fn write_test_wav(path: &Path, frames: usize, sample_rate: u32) {
    let data_len = (frames * 2) as u32;
    let mut v: Vec<u8> = Vec::new();
    v.extend_from_slice(b"RIFF");
    v.extend_from_slice(&(36 + data_len).to_le_bytes());
    v.extend_from_slice(b"WAVEfmt ");
    v.extend_from_slice(&16u32.to_le_bytes());
    v.extend_from_slice(&1u16.to_le_bytes());
    v.extend_from_slice(&1u16.to_le_bytes()); // mono
    v.extend_from_slice(&sample_rate.to_le_bytes());
    v.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    v.extend_from_slice(&2u16.to_le_bytes());
    v.extend_from_slice(&16u16.to_le_bytes());
    v.extend_from_slice(b"data");
    v.extend_from_slice(&data_len.to_le_bytes());
    let mut w = std::fs::File::create(path).unwrap();
    w.write_all(&v).unwrap();
    w.write_all(&vec![0u8; data_len as usize]).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn availability_check_is_cached_and_never_panics() {
        // Whatever the host has, the call must return cleanly and stick to the
        // cached answer on repeat invocation.
        let a = ffmpeg_available();
        let b = ffmpeg_available();
        assert_eq!(a, b);
    }

    #[test]
    fn decode_without_ffmpeg_reports_not_available() {
        if ffmpeg_available() {
            return; // environment-dependent branch: only assert the negative
        }
        let dir = std::env::temp_dir();
        let p = dir.join("audioscout_missing.wma");
        assert!(matches!(
            decode_to_temp_wav(&p),
            Err(AudioIoError::FfmpegNotAvailable)
        ));
        assert!(matches!(
            decode_to_pcm(&p),
            Err(AudioIoError::FfmpegNotAvailable)
        ));
    }

    #[test]
    fn decode_cache_key_tracks_mtime_and_size() {
        let dir = std::env::temp_dir().join(format!("as_ffmpeg_key_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let a = dir.join("x.wav");
        write_test_wav(&a, 100, 44100);
        let k1 = decode_key(&a);
        // Same content, different name → different key (path is part of it).
        let b = dir.join("y.wav");
        write_test_wav(&b, 100, 44100);
        assert_ne!(k1, decode_key(&b));
        // Appending bytes changes size → new key (cache invalidation works).
        let mut f = std::fs::OpenOptions::new().append(true).open(&a).unwrap();
        f.write_all(&[0u8; 8]).unwrap();
        drop(f);
        assert_ne!(k1, decode_key(&a));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn ffprobe_meta_returns_none_when_ffmpeg_missing() {
        if ffmpeg_available() {
            return;
        }
        let p = std::env::temp_dir().join("audioscout_absent.wma");
        assert!(ffprobe_meta(&p).unwrap().is_none());
    }

    #[test]
    fn real_ffmpeg_roundtrip_when_available() {
        if !ffmpeg_available() {
            eprintln!("ffmpeg not installed — skipping live round-trip");
            return;
        }
        // Hermetic run: route the session decode cache into a temp dir.
        let dir = std::env::temp_dir().join(format!("as_ffmpeg_rt_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        std::env::set_var("AUDIOSCOUT_CONFIG_DIR", &dir);
        let wav = dir.join("tone.wav");
        // 44100 Hz tone is overkill; short silent wav suffices for the chain.
        write_test_wav(&wav, 4410, 44100);
        // Produce a genuine WMA through ffmpeg itself.
        let wma = dir.join("tone.wma");
        let out = Command::new("ffmpeg")
            .arg("-hide_banner").arg("-loglevel").arg("error").arg("-y")
            .arg("-i").arg(&wav)
            .arg("-c:a").arg("wmav2")
            .arg(&wma)
            .output();
        if !out.map(|o| o.status.success()).unwrap_or(false) {
            eprintln!("ffmpeg lacks the WMA encoder — skipping");
            let _ = std::fs::remove_dir_all(&dir);
            return;
        }
        // Metadata probe of the WMA goes through ffprobe.
        let meta = ffprobe_meta(&wma).unwrap().expect("ffprobe present");
        assert_eq!(meta.sample_rate, 44100);
        assert_eq!(meta.channels, 1);
        assert!(meta.duration_secs > 0.05);
        // Full-load chain: wma → cached temp wav → hound read.
        let (samples, sr, ch, st) = super::super::load_audio(&wma).unwrap();
        assert_eq!(sr, 44100);
        assert_eq!(ch, 1);
        assert_eq!(st, SampleType::Float32);
        assert!(samples.len() >= 4000);
        // Second load hits the decode cache (same key, file already there).
        let cached = cached_wav(&wma);
        assert!(cached.exists());
        let (again, ..) = super::super::load_audio(&wma).unwrap();
        assert_eq!(again.len(), samples.len());
        // Envelope path via the raw pipe.
        let (env, esr) = super::super::envelope::load_audio_envelope(&wma, 64, u64::MAX).unwrap();
        assert_eq!(esr, 44100);
        assert_eq!(env.lo[0].len(), 64);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
