use super::{AudioIoError, Result};
use crate::AudioMeta;
use std::path::Path;
use std::process::Command;

pub fn ffmpeg_available() -> bool {
    Command::new("ffmpeg").arg("-version").output().map(|o| o.status.success()).unwrap_or(false)
}

pub fn ffmpeg_decode(input: &Path, output: &Path) -> Result<()> {
    let status = Command::new("ffmpeg").arg("-y").arg("-i").arg(input)
        .arg("-f").arg("wav").arg("-acodec").arg("pcm_f32le").arg(output)
        .status().map_err(|_| AudioIoError::FfmpegNotAvailable)?;
    if !status.success() { return Err(AudioIoError::DecodeError("ffmpeg decode failed".to_string())); }
    Ok(())
}

pub fn ffprobe(path: &Path) -> Result<AudioMeta> {
    let output = Command::new("ffprobe").arg("-v").arg("quiet")
        .arg("-show_streams").arg("-show_format").arg("-of").arg("json").arg(path)
        .output().map_err(|_| AudioIoError::FfmpegNotAvailable)?;
    if !output.status.success() { return Err(AudioIoError::DecodeError("ffprobe failed".to_string())); }
    let json_str = String::from_utf8_lossy(&output.stdout);
    let json: serde_json::Value = serde_json::from_str(&json_str)
        .map_err(|e| AudioIoError::DecodeError(format!("ffprobe JSON: {}", e)))?;
    let stream = json["streams"].as_array().and_then(|arr| arr.first())
        .ok_or_else(|| AudioIoError::DecodeError("No streams".to_string()))?;
    let sample_rate = stream["sample_rate"].as_str().and_then(|s| s.parse::<u32>().ok()).unwrap_or(44100);
    let channels = stream["channels"].as_u64().map(|c| c as u16).unwrap_or(1);
    let duration_secs = json["format"]["duration"].as_str().and_then(|s| s.parse::<f64>().ok())
        .or_else(|| stream["duration"].as_str().and_then(|s| s.parse::<f64>().ok())).unwrap_or(0.0);
    let bits_per_sample = stream["bits_per_raw_sample"].as_u64()
        .or_else(|| stream["bits_per_sample"].as_u64()).unwrap_or(16);
    let sample_type = match bits_per_sample {
        8 => crate::SampleType::UInt8, 16 => crate::SampleType::Int16,
        24 => crate::SampleType::Int24, 32 => crate::SampleType::Int32,
        _ => crate::SampleType::Float32,
    };
    Ok(AudioMeta { duration_secs, sample_rate, channels, sample_type, comment: None, status: crate::ProbeStatus::Ok })
}
