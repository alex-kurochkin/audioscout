use std::time::Instant;
fn main() {
    let arg = std::env::args().nth(1).unwrap_or_else(|| "/tmp/test_stereo.wav".into());
    let path = std::path::Path::new(arg.as_str());
    let t = Instant::now();
    let (env, sr) = audioscout_core::audio_io::envelope::load_audio_envelope(
        path, 340 * 4, audioscout_core::preview_worker::PREVIEW_DECODE_BUDGET).unwrap();
    let e1 = t.elapsed();
    let t2 = Instant::now();
    let h = if env.channels == 1 { 20 } else { 40 };
    let img = audioscout_core::render::waveform::render_envelope(&env, 340, h, audioscout_core::render::waveform::Style::Dark);
    let e2 = t2.elapsed();
    img.save("/tmp/env.png").unwrap();
    eprintln!("envelope={:?} render={:?} ch={} frames={} sr={}", e1, e2, env.channels, env.frames, sr);
}
