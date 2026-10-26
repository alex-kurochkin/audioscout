use audioscout_core::audio_io::envelope::load_audio_envelope;
use audioscout_core::preview_worker::PREVIEW_DECODE_BUDGET;
fn main(){
    for arg in std::env::args().skip(1){
        let path=std::path::Path::new(&arg);
        let (env,_sr)=load_audio_envelope(path,1360,PREVIEW_DECODE_BUDGET).unwrap();
        // print per-channel lo/hi as CSV
        for c in 0..env.channels{
            let name=format!("/tmp/pv/env_{}_c{}.csv", path.extension().unwrap().to_str().unwrap(), c);
            let mut s=String::new();
            for i in 0..env.lo[c].len(){ s.push_str(&format!("{},{}\n", env.lo[c][i], env.hi[c][i])); }
            std::fs::write(name, s).unwrap();
        }
        eprintln!("{} ch={} frames={}", arg, env.channels, env.frames);
    }
}
