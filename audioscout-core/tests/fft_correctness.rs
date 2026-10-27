//! Independent correctness check for the hand-rolled radix-2 FFT used by the
//! spectrogram renderer (compare against rustfft, already a dependency).

use audioscout_core::render::spectrogram::fft_radix2_naive;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

#[test]
fn fft_matches_rustfft() {
    let n = 512;
    let sig: Vec<f32> = (0..n)
        .map(|i| {
            (core::f32::consts::TAU * 7.0 * i as f32 / n as f32).sin() * 0.8
                + (core::f32::consts::TAU * 100.0 * i as f32 / n as f32).cos() * 0.3
        })
        .collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let mut ref_buf: Vec<Complex<f32>> = sig.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fft.process(&mut ref_buf);

    let ours = fft_radix2_naive(&sig);
    assert_eq!(ours.len(), ref_buf.len());
    let max_ref: f32 = ref_buf.iter().map(|c| c.norm()).fold(0.0, f32::max);
    for (a, b) in ours.iter().zip(&ref_buf) {
        let (ar, ai) = *a;
        let diff = ((ar - b.re).abs().max((ai - b.im).abs())) / max_ref;
        assert!(diff < 1e-4, "bin mismatch: rel err {diff}");
    }
}
