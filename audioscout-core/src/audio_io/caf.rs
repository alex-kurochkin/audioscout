use super::{AudioIoError, PcmFile, Result};
use crate::{AudioMeta, ProbeStatus, SampleType};
use std::path::Path;

pub struct CafFile;

impl PcmFile for CafFile {
    fn load(path: &Path) -> Result<(Vec<f32>, u32, u16, SampleType)> {
        let data = std::fs::read(path)?;
        if data.len() < 8 || &data[0..4] != b"caff" {
            return Err(AudioIoError::DecodeError("Not a CAF file".to_string()));
        }
        let mut pos = 8;
        let mut sample_rate: u32 = 44100;
        let mut channels: u16 = 1;
        let mut bits_per_sample: u16 = 16;
        let mut is_float = false;
        let mut audio_data: Vec<u8> = Vec::new();

        while pos + 12 <= data.len() {
            let chunk_type = &data[pos..pos + 4];
            let chunk_size = u64::from_be_bytes([
                data[pos+4], data[pos+5], data[pos+6], data[pos+7],
                data[pos+8], data[pos+9], data[pos+10], data[pos+11],
            ]) as usize;
            pos += 12;

            if chunk_type == b"desc" && pos + 32 <= data.len() {
                let sr_bytes = &data[pos..pos + 8];
                let sr_f64 = f64::from_be_bytes([sr_bytes[0], sr_bytes[1], sr_bytes[2], sr_bytes[3], sr_bytes[4], sr_bytes[5], sr_bytes[6], sr_bytes[7]]);
                sample_rate = sr_f64 as u32;
                let format_id = &data[pos + 8..pos + 12];
                is_float = format_id == b"lpcm" && (data[pos + 16] & 1) != 0;
                channels = u32::from_be_bytes([0, data[pos+20], data[pos+21], data[pos+22]]) as u16;
                bits_per_sample = u32::from_be_bytes([0, data[pos+24], data[pos+25], data[pos+26]]) as u16;
            } else if chunk_type == b"data" {
                let data_start = pos + 4;
                let data_end = std::cmp::min(pos + chunk_size, data.len());
                if data_start < data_end { audio_data = data[data_start..data_end].to_vec(); }
            }
            pos += chunk_size;
        }

        let sample_type = if is_float { SampleType::Float32 } else {
            match bits_per_sample { 8 => SampleType::UInt8, 16 => SampleType::Int16, 24 => SampleType::Int24, 32 => SampleType::Int32, _ => SampleType::Int16 }
        };
        let samples = decode_caf_pcm(&audio_data, bits_per_sample, is_float);
        Ok((samples, sample_rate, channels, sample_type))
    }

    fn save(path: &Path, samples: &[f32], sample_rate: u32, channels: u16, sample_type: SampleType) -> Result<()> {
        let bits_per_sample: u16 = match sample_type {
            SampleType::UInt8 => 8, SampleType::Int16 => 16, SampleType::Int24 => 24,
            SampleType::Int32 => 32, SampleType::Float32 => 32, _ => 16,
        };
        let is_float = matches!(sample_type, SampleType::Float32 | SampleType::Float64);
        let pcm_data = encode_caf_pcm(samples, bits_per_sample, is_float);
        let mut file = std::fs::File::create(path)?;
        use std::io::Write;
        file.write_all(b"caff")?;
        file.write_all(&[0, 1, 0, 0])?;
        file.write_all(b"desc")?;
        file.write_all(&32u64.to_be_bytes())?;
        file.write_all(&(sample_rate as f64).to_be_bytes())?;
        file.write_all(b"lpcm")?;
        let format_flags: u32 = if is_float { 1 } else { 0 };
        file.write_all(&format_flags.to_be_bytes())?;
        file.write_all(&0u32.to_be_bytes())?;
        file.write_all(&(channels as u32).to_be_bytes())?;
        file.write_all(&(bits_per_sample as u32).to_be_bytes())?;
        file.write_all(&0u32.to_be_bytes())?;
        let data_size = 4 + pcm_data.len();
        file.write_all(b"data")?;
        file.write_all(&(data_size as u64).to_be_bytes())?;
        file.write_all(&0u32.to_be_bytes())?;
        file.write_all(&pcm_data)?;
        Ok(())
    }

    fn probe(path: &Path) -> Result<AudioMeta> {
        let (samples, sr, ch, st) = Self::load(path)?;
        let total_frames = if ch == 0 { 0 } else { samples.len() / ch as usize };
        Ok(AudioMeta { duration_secs: total_frames as f64 / sr as f64, sample_rate: sr, channels: ch, sample_type: st, comment: None, status: ProbeStatus::Ok })
    }
}

fn decode_caf_pcm(data: &[u8], bits: u16, is_float: bool) -> Vec<f32> {
    if is_float { return data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(); }
    match bits {
        8 => data.iter().map(|&b| (b as f32 - 128.0) / 128.0).collect(),
        16 => data.chunks_exact(2).map(|c| { i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0 }).collect(),
        24 => data.chunks_exact(3).map(|c| {
            let v = (c[0] as i32) | ((c[1] as i32) << 8) | ((c[2] as i32) << 16);
            let v = if v & 0x800000 != 0 { v | !0xFFFFFF } else { v };
            v as f32 / 8388608.0
        }).collect(),
        32 => data.chunks_exact(4).map(|c| { i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32 / 2147483648.0 }).collect(),
        _ => vec![],
    }
}

fn encode_caf_pcm(samples: &[f32], bits: u16, is_float: bool) -> Vec<u8> {
    if is_float { return samples.iter().flat_map(|&s| s.to_le_bytes()).collect(); }
    match bits {
        8 => samples.iter().map(|&s| ((s * 128.0) + 128.0).clamp(0.0, 255.0) as u8).collect(),
        16 => samples.iter().flat_map(|&s| { ((s * 32768.0).clamp(-32768.0, 32767.0) as i16).to_le_bytes() }).collect(),
        24 => samples.iter().flat_map(|&s| {
            let v = (s * 8388608.0).clamp(-8388608.0, 8388607.0) as i32;
            [v as u8, (v >> 8) as u8, (v >> 16) as u8]
        }).collect(),
        32 => samples.iter().flat_map(|&s| { ((s * 2147483648.0).clamp(-2147483648.0, 2147483647.0) as i32).to_le_bytes() }).collect(),
        _ => vec![],
    }
}
