use super::{AudioIoError, PcmFile, Result};
use crate::{AudioMeta, ProbeStatus, SampleType};
use std::path::Path;
use std::io::{Read, Write, Cursor, Seek, SeekFrom};

pub struct AiffFile;
const AIFF_FORM: &[u8; 4] = b"FORM";
const AIFF_AIFF: &[u8; 4] = b"AIFF";
const AIFF_COMM: &[u8; 4] = b"COMM";
const AIFF_SSND: &[u8; 4] = b"SSND";

impl PcmFile for AiffFile {
    fn load(path: &Path) -> Result<(Vec<f32>, u32, u16, SampleType)> {
        let data = std::fs::read(path)?;
        let mut cursor = Cursor::new(&data);
        let mut form = [0u8; 4];
        cursor.read_exact(&mut form).map_err(|e| AudioIoError::DecodeError(e.to_string()))?;
        if &form != AIFF_FORM { return Err(AudioIoError::DecodeError("Not AIFF".to_string())); }
        let mut size_buf = [0u8; 4];
        cursor.read_exact(&mut size_buf).map_err(|e| AudioIoError::DecodeError(e.to_string()))?;
        let mut form_type = [0u8; 4];
        cursor.read_exact(&mut form_type).map_err(|e| AudioIoError::DecodeError(e.to_string()))?;
        if &form_type != AIFF_AIFF { return Err(AudioIoError::DecodeError("Wrong form type".to_string())); }
        let mut num_channels: u16 = 1;
        let mut bits_per_sample: u16 = 16;
        let mut sample_rate: u32 = 44100;
        let mut audio_data: Vec<u8> = Vec::new();
        loop {
            let mut chunk_id = [0u8; 4];
            if cursor.read_exact(&mut chunk_id).is_err() { break; }
            let mut chunk_size_buf = [0u8; 4];
            if cursor.read_exact(&mut chunk_size_buf).is_err() { break; }
            let chunk_size = u32::from_be_bytes(chunk_size_buf) as usize;
            let chunk_start = cursor.position() as usize;
            if &chunk_id == AIFF_COMM {
                let mut buf = [0u8; 2]; cursor.read_exact(&mut buf).ok();
                num_channels = u16::from_be_bytes(buf);
                let mut buf4 = [0u8; 4]; cursor.read_exact(&mut buf4).ok();
                let _num_frames = u32::from_be_bytes(buf4);
                let mut buf2 = [0u8; 2]; cursor.read_exact(&mut buf2).ok();
                bits_per_sample = u16::from_be_bytes(buf2);
                let mut sr_bytes = [0u8; 10]; cursor.read_exact(&mut sr_bytes).ok();
                sample_rate = extended_to_u32(&sr_bytes);
            } else if &chunk_id == AIFF_SSND {
                cursor.seek(SeekFrom::Current(8)).ok();
                let data_size = chunk_size.saturating_sub(8);
                audio_data.resize(data_size, 0);
                cursor.read_exact(&mut audio_data).ok();
            }
            let end_pos = chunk_start + chunk_size;
            let aligned = if chunk_size % 2 != 0 { end_pos + 1 } else { end_pos };
            cursor.set_position(aligned as u64);
        }
        let sample_type = match bits_per_sample {
            8 => SampleType::UInt8, 16 => SampleType::Int16, 24 => SampleType::Int24, 32 => SampleType::Int32, _ => SampleType::Int16,
        };
        let samples = decode_aiff_pcm(&audio_data, bits_per_sample);
        Ok((samples, sample_rate, num_channels, sample_type))
    }
    fn save(path: &Path, samples: &[f32], sample_rate: u32, channels: u16, sample_type: SampleType) -> Result<()> {
        let bits_per_sample: u16 = match sample_type {
            SampleType::UInt8 => 8, SampleType::Int16 => 16, SampleType::Int24 => 24, SampleType::Int32 => 32, _ => 16,
        };
        let num_frames = if channels == 0 { 0 } else { samples.len() / channels as usize };
        let pcm_data = encode_aiff_pcm(samples, bits_per_sample);
        let mut file = std::fs::File::create(path)?;
        let comm_size = 18u32;
        let ssnd_size = 8 + pcm_data.len() as u32;
        let form_size = 4 + 8 + comm_size + 8 + ssnd_size;
        file.write_all(AIFF_FORM)?;
        file.write_all(&(form_size as u32).to_be_bytes())?;
        file.write_all(AIFF_AIFF)?;
        file.write_all(AIFF_COMM)?;
        file.write_all(&comm_size.to_be_bytes())?;
        file.write_all(&channels.to_be_bytes())?;
        file.write_all(&(num_frames as u32).to_be_bytes())?;
        file.write_all(&bits_per_sample.to_be_bytes())?;
        let sr_extended = u32_to_extended(sample_rate);
        file.write_all(&sr_extended)?;
        file.write_all(AIFF_SSND)?;
        file.write_all(&ssnd_size.to_be_bytes())?;
        file.write_all(&[0u8; 4])?;
        file.write_all(&[0u8; 4])?;
        file.write_all(&pcm_data)?;
        Ok(())
    }
    fn probe(path: &Path) -> Result<AudioMeta> {
        let (samples, sr, ch, st) = Self::load(path)?;
        let total_frames = if ch == 0 { 0 } else { samples.len() / ch as usize };
        Ok(AudioMeta { duration_secs: total_frames as f64 / sr as f64, sample_rate: sr, channels: ch, sample_type: st, comment: None, status: ProbeStatus::Ok })
    }
}

fn decode_aiff_pcm(data: &[u8], bits: u16) -> Vec<f32> {
    match bits {
        8 => data.iter().map(|&b| (b as f32 - 128.0) / 128.0).collect(),
        16 => data.chunks_exact(2).map(|c| { i16::from_be_bytes([c[0], c[1]]) as f32 / 32768.0 }).collect(),
        24 => data.chunks_exact(3).map(|c| {
            let v = ((c[0] as i32) << 16) | ((c[1] as i32) << 8) | (c[2] as i32);
            let v = if v & 0x800000 != 0 { v | !0xFFFFFF } else { v };
            v as f32 / 8388608.0
        }).collect(),
        32 => data.chunks_exact(4).map(|c| { i32::from_be_bytes([c[0], c[1], c[2], c[3]]) as f32 / 2147483648.0 }).collect(),
        _ => vec![],
    }
}

fn encode_aiff_pcm(samples: &[f32], bits: u16) -> Vec<u8> {
    match bits {
        8 => samples.iter().map(|&s| ((s * 128.0) + 128.0).clamp(0.0, 255.0) as u8).collect(),
        16 => samples.iter().flat_map(|&s| { ((s * 32768.0).clamp(-32768.0, 32767.0) as i16).to_be_bytes() }).collect(),
        24 => samples.iter().flat_map(|&s| {
            let v = (s * 8388608.0).clamp(-8388608.0, 8388607.0) as i32;
            [(v >> 16) as u8, (v >> 8) as u8, v as u8]
        }).collect(),
        32 => samples.iter().flat_map(|&s| { ((s * 2147483648.0).clamp(-2147483648.0, 2147483647.0) as i32).to_be_bytes() }).collect(),
        _ => vec![],
    }
}

fn extended_to_u32(bytes: &[u8; 10]) -> u32 {
    let exponent = ((bytes[0] as u16 & 0x7F) << 8) | bytes[1] as u16;
    let mantissa_hi = u32::from_be_bytes([bytes[2], bytes[3], bytes[4], bytes[5]]);
    if exponent == 0 && mantissa_hi == 0 { return 0; }
    let exp = exponent as i32 - 16383;
    let mantissa = mantissa_hi as f64 / 4294967296.0;
    (mantissa * 2.0f64.powi(exp)) as u32
}

fn u32_to_extended(value: u32) -> [u8; 10] {
    if value == 0 { return [0; 10]; }
    let fval = value as f64;
    let exp = fval.log2().floor() as i32;
    let mantissa = fval / 2.0f64.powi(exp);
    let exponent = (exp + 16383) as u16;
    let mantissa_int = (mantissa * 4294967296.0) as u32;
    let mut result = [0u8; 10];
    result[0] = (exponent >> 8) as u8;
    result[1] = exponent as u8;
    result[2..6].copy_from_slice(&mantissa_int.to_be_bytes());
    result
}
