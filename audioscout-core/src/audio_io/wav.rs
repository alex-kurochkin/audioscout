use super::{AudioIoError, PcmFile, Result};
use crate::{AudioMeta, ProbeStatus, SampleType};
use std::path::Path;

pub struct WavFile;

impl PcmFile for WavFile {
    fn load(path: &Path) -> Result<(Vec<f32>, u32, u16, SampleType)> {
        let reader = hound::WavReader::open(path)
            .map_err(|e| AudioIoError::DecodeError(format!("WAV open: {}", e)))?;
        let spec = reader.spec();
        let sample_rate = spec.sample_rate;
        let channels = spec.channels;

        let (samples, sample_type) = match spec.sample_format {
            hound::SampleFormat::Int => match spec.bits_per_sample {
                8 => {
                    let raw: Vec<i16> = reader.into_samples::<i16>().filter_map(|s| s.ok()).collect();
                    (raw.iter().map(|&s| s as f32 / 128.0).collect(), SampleType::UInt8)
                }
                16 => {
                    let raw: Vec<i16> = reader.into_samples::<i16>().filter_map(|s| s.ok()).collect();
                    (raw.iter().map(|&s| s as f32 / 32768.0).collect(), SampleType::Int16)
                }
                24 => {
                    let raw: Vec<i32> = reader.into_samples::<i32>().filter_map(|s| s.ok()).collect();
                    (raw.iter().map(|&s| s as f32 / 8388608.0).collect(), SampleType::Int24)
                }
                32 => {
                    let raw: Vec<i32> = reader.into_samples::<i32>().filter_map(|s| s.ok()).collect();
                    (raw.iter().map(|&s| s as f32 / 2147483648.0).collect(), SampleType::Int32)
                }
                _ => return Err(AudioIoError::UnsupportedFormat(format!("Unsupported bits: {}", spec.bits_per_sample))),
            },
            hound::SampleFormat::Float => {
                let raw: Vec<f32> = reader.into_samples::<f32>().filter_map(|s| s.ok()).collect();
                (raw, SampleType::Float32)
            }
        };
        Ok((samples, sample_rate, channels, sample_type))
    }

    fn save(path: &Path, samples: &[f32], sample_rate: u32, channels: u16, sample_type: SampleType) -> Result<()> {
        let spec = hound::WavSpec {
            channels, sample_rate,
            bits_per_sample: match sample_type {
                SampleType::UInt8 => 8, SampleType::Int16 => 16, SampleType::Int24 => 24,
                SampleType::Int32 => 32, SampleType::Float32 | SampleType::Float64 => 32, _ => 16,
            },
            sample_format: match sample_type {
                SampleType::Float32 | SampleType::Float64 => hound::SampleFormat::Float,
                _ => hound::SampleFormat::Int,
            },
        };
        let mut writer = hound::WavWriter::create(path, spec)
            .map_err(|e| AudioIoError::SaveError(format!("WAV create: {}", e)))?;
        for &sample in samples {
            match sample_type {
                SampleType::UInt8 => writer.write_sample((sample * 128.0).clamp(-128.0, 127.0) as i16).ok(),
                SampleType::Int16 => writer.write_sample((sample * 32768.0).clamp(-32768.0, 32767.0) as i16).ok(),
                SampleType::Int24 => writer.write_sample((sample * 8388608.0).clamp(-8388608.0, 8388607.0) as i32).ok(),
                SampleType::Int32 => writer.write_sample((sample * 2147483648.0).clamp(-2147483648.0, 2147483647.0) as i32).ok(),
                _ => writer.write_sample(sample).ok(),
            };
        }
        writer.finalize().map_err(|e| AudioIoError::SaveError(format!("WAV finalize: {}", e)))?;
        Ok(())
    }

    fn probe(path: &Path) -> Result<AudioMeta> {
        let reader = hound::WavReader::open(path)
            .map_err(|e| AudioIoError::DecodeError(format!("WAV probe: {}", e)))?;
        let spec = reader.spec();
        let num_samples = reader.len() as u64;
        let duration_secs = num_samples as f64 / (spec.sample_rate as f64 * spec.channels as f64);
        let sample_type = match (spec.sample_format, spec.bits_per_sample) {
            (hound::SampleFormat::Int, 8) => SampleType::UInt8,
            (hound::SampleFormat::Int, 16) => SampleType::Int16,
            (hound::SampleFormat::Int, 24) => SampleType::Int24,
            (hound::SampleFormat::Int, 32) => SampleType::Int32,
            (hound::SampleFormat::Float, 32) => SampleType::Float32,
            (hound::SampleFormat::Float, 64) => SampleType::Float64,
            _ => SampleType::Compressed,
        };
        Ok(AudioMeta { duration_secs, sample_rate: spec.sample_rate, channels: spec.channels, sample_type, comment: None, status: ProbeStatus::Ok })
    }
}
