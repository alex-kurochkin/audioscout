mod app;
use std::path::PathBuf;

fn main() -> eframe::Result<()> {
    let config_dir = audioscout_core::config::config_dir();
    let _ = std::fs::create_dir_all(&config_dir);
    let log_dir = config_dir.join("logs");
    let _ = std::fs::create_dir_all(&log_dir);
    let file_appender = tracing_appender::rolling::daily(&log_dir, "audioscout");
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
    tracing_subscriber::fmt().with_writer(non_blocking).with_ansi(false).with_target(false).init();
    eprintln!("AudioScout starting...");
    let start_dir = std::env::var("AUDIOSCOUT_DIR").ok().map(PathBuf::from);
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 860.0]).with_title("AudioScout"),
        ..Default::default()
    };
    eframe::run_native("AudioScout", native_options, Box::new(move |cc| {
        Ok(Box::new(app::AudioScoutApp::new(cc, config_dir, start_dir)))
    }))
}
