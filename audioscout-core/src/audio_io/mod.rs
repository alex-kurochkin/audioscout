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

fn load_with_symphonia(path: &Path) -> Result<(Vec<f32>, u32, u16, SampleType)> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
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
    let track = format.tracks().iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .ok_or_else(|| AudioIoError::DecodeError("No audio track found".to_string()))?;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| AudioIoError::DecodeError(format!("Codec init failed: {}", e)))?;

    let sample_rate = track.codec_params.sample_rate.unwrap_or(44100);
    let channels = track.codec_params.channels.map(|c| c.count() as u16).unwrap_or(1);
    let mut all_samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(_) => break,
        };
        let decoded = match decoder.decode(&packet) { Ok(d) => d, Err(_) => continue };
        let spec = *decoded.spec();
        let duration = decoded.capacity();
        let mut sample_buf = SampleBuffer::<f32>::new(duration as u64, spec);
        sample_buf.copy_interleaved_ref(decoded);
        all_samples.extend_from_slice(sample_buf.samples());
    }

    Ok((all_samples, sample_rate, channels, SampleType::Float32))
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

    // Debug: always log all tags and items in debug mode
    #[cfg(debug_assertions)]
    {
        eprintln!("[DEBUG] === Extracting comment from file ===");
        for tag in tagged_file.tags() {
            eprintln!("[DEBUG] Tag type: {:?}", tag.tag_type());
            for item in tag.items() {
                eprintln!("[DEBUG]   Item key: {:?}", item.key());
                match item.value() {
                    ItemValue::Text(text) => eprintln!("[DEBUG]     Text: {}", text),
                    ItemValue::Locator(text) => eprintln!("[DEBUG]     Locator: {}", text),
                    ItemValue::Binary(data) => eprintln!("[DEBUG]     Binary: {} bytes", data.len()),
                }
            }
        }
    }

    // Try all tags (not just primary/first) to find comments
    for tag in tagged_file.tags() {
        // Try standard comment keys in order of preference
        // 1. ItemKey::Comment (WAV ICMT, ID3 COMM, etc.)
        if let Some(comment) = tag.get_string(&ItemKey::Comment) {
            #[cfg(debug_assertions)]
            eprintln!("[DEBUG] Found comment via ItemKey::Comment: {}", comment);
            return Some(comment.to_string());
        }

        // 2. ItemKey::Description (FLAC Vorbis DESCRIPTION)
        if let Some(desc) = tag.get_string(&ItemKey::Description) {
            #[cfg(debug_assertions)]
            eprintln!("[DEBUG] Found description via ItemKey::Description: {}", desc);
            return Some(desc.to_string());
        }

        // 3. Try Accessor trait method
        if let Some(comment) = tag.comment() {
            #[cfg(debug_assertions)]
            eprintln!("[DEBUG] Found comment via Accessor::comment(): {}", comment);
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
                            #[cfg(debug_assertions)]
                            eprintln!("[DEBUG] Found comment via key search ({}): {}", key_lower, text);
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
                            #[cfg(debug_assertions)]
                            eprintln!("[DEBUG] Found COMM frame ({}): {}", key_str, text);
                            return Some(text.to_string());
                        }
                    }
                }
                _ => {}
            }
        }
    }

    // Debug: log all tags and items if no comment found
    #[cfg(debug_assertions)]
    for tag in tagged_file.tags() {
        eprintln!("[DEBUG] Tag type: {:?}", tag.tag_type());
        for item in tag.items() {
            eprintln!("[DEBUG]   Item key: {:?}", item.key());
            match item.value() {
                ItemValue::Text(text) => eprintln!("[DEBUG]     Text: {}", text),
                ItemValue::Locator(text) => eprintln!("[DEBUG]     Locator: {}", text),
                ItemValue::Binary(data) => eprintln!("[DEBUG]     Binary: {} bytes", data.len()),
            }
        }
    }

    None
}

fn probe_with_ffprobe(path: &Path) -> Result<AudioMeta> {
    ffmpeg::ffprobe(path)
}
