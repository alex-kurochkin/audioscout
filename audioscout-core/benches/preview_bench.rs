use std::time::Instant;
fn main() {
    let path = std::path::Path::new("/tmp/big_stereo.wav");
    if !path.exists() { eprintln!("no test file"); return; }
    for _ in 0..3 {
        let t = Instant::now();
        let (env, sr) = audioscout_core::audio_io::load_audio_envelope(path, 340*4).unwrap();
        let e1 = t.elapsed();
        let t2 = Instant::now();
        let img = audioscout_core::render::waveform::render_envelope(&env, 340, 40, audioscout_core::render::waveform::Style::Dark);
        let e2 = t2.elapsed();
        img.save("/tmp/env.png").unwrap();
        eprintln!("envelope={:?} render={:?} ch={} frames={} sr={}", e1, e2, env.channels, env.frames, sr);
    }
}
