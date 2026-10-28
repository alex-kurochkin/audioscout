mod app;
use std::path::PathBuf;

fn main() -> eframe::Result<()> {
    let config_dir = audioscout_core::config::config_dir();
    let _ = std::fs::create_dir_all(&config_dir);
    let log_dir = config_dir.join("logs");
    let _ = std::fs::create_dir_all(&log_dir);
    // DEBUG BUILD: mirror every log line to the terminal as well, so the
    // playback diagnostics ([play] / [output] / [poll] tags) are visible in
    // the console when the app is started from a Debian shell. Release builds
    // keep the original file-only logging.
    let file_appender = tracing_appender::rolling::daily(&log_dir, "audioscout");
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
    #[cfg(debug_assertions)]
    {
        use tracing_subscriber::layer::SubscriberExt as _;
        use tracing_subscriber::util::SubscriberInitExt as _;
        let fmt_layer = tracing_subscriber::fmt::layer()
            .with_writer(non_blocking)
            .with_ansi(false)
            .with_target(false);
        let stderr_layer = tracing_subscriber::fmt::layer()
            .with_writer(std::io::stderr)
            .with_ansi(false)
            .with_target(false);
        let filter = tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
        tracing_subscriber::registry().with(filter).with(fmt_layer).with(stderr_layer).init();
    }
    #[cfg(not(debug_assertions))]
    {
        let filter = tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
        tracing_subscriber::fmt().with_writer(non_blocking).with_ansi(false).with_target(false).with_env_filter(filter).init();
    }
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
