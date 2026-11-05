//! Guandan table server for the Android app.
//!
//! The same web UI and the same HTTP / WebSocket protocol as the Python server
//! (python/guandan/server), for playing only: tables with AI and LAN players, permissions,
//! hints and reviews. Training needs PyTorch and stays on the computer version.

pub mod agent;
pub mod app;
pub mod net;
pub mod perm;
pub mod room;
pub mod util;

#[cfg(target_os = "android")]
mod android;

use std::path::PathBuf;
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

pub use guandan_core::VERSION;

pub struct Ctx {
    pub net: Arc<net::Net>,
    pub perm: perm::Perm,
    pub rooms: Mutex<Vec<room::RoomRef>>,
    pub clients: Mutex<Vec<serde_json::Value>>,
    pub home: PathBuf,
    pub port: AtomicU16,
    pub think_slots: tokio::sync::Semaphore,
    pub next_conn: std::sync::atomic::AtomicU64,
    pub own_ips: Mutex<(f64, Vec<String>)>,
    /// "android" for the app, "desktop" for the test binary
    pub platform: String,
}

pub struct Config {
    pub home: PathBuf,
    pub model: PathBuf,
    pub port: u16,
    pub platform: String,
}

static CTX: OnceLock<Arc<Ctx>> = OnceLock::new();

/// Start the server on a thread of its own; returns the port it listens on (the requested one
/// or, if that is taken, one of the next ten).
pub fn start(cfg: Config) -> Result<u16, String> {
    if let Some(c) = CTX.get() {
        return Ok(c.port.load(Ordering::SeqCst));
    }
    let bytes = std::fs::read(&cfg.model).map_err(|e| format!("读不到模型文件 {}：{e}", cfg.model.display()))?;
    let net = net::Net::load(&bytes)?;
    let _ = std::fs::create_dir_all(&cfg.home);
    let ctx = Arc::new(Ctx {
        net: Arc::new(net),
        perm: perm::Perm::new(&cfg.home),
        rooms: Mutex::new(vec![]),
        clients: Mutex::new(vec![]),
        home: cfg.home.clone(),
        port: AtomicU16::new(cfg.port),
        think_slots: tokio::sync::Semaphore::new(2),
        next_conn: std::sync::atomic::AtomicU64::new(1),
        own_ips: Mutex::new((0.0, vec![])),
        platform: cfg.platform.clone(),
    });
    let (tx, rx) = std::sync::mpsc::channel::<Result<u16, String>>();
    let c2 = ctx.clone();
    std::thread::Builder::new()
        .name("guandan-server".into())
        .spawn(move || {
            let rt = match tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build() {
                Ok(rt) => rt,
                Err(e) => {
                    let _ = tx.send(Err(format!("{e}")));
                    return;
                }
            };
            rt.block_on(async move {
                let mut listener = None;
                for p in cfg.port..cfg.port.saturating_add(10) {
                    if let Ok(l) = tokio::net::TcpListener::bind(("0.0.0.0", p)).await {
                        c2.port.store(p, Ordering::SeqCst);
                        listener = Some(l);
                        break;
                    }
                }
                let Some(l) = listener else {
                    let _ = tx.send(Err(format!("端口 {} 起都被占用了", cfg.port)));
                    return;
                };
                let _ = tx.send(Ok(c2.port.load(Ordering::SeqCst)));
                let app = app::router(c2.clone());
                let _ = axum::serve(l, app.into_make_service_with_connect_info::<std::net::SocketAddr>()).await;
            });
        })
        .map_err(|e| e.to_string())?;
    let port = rx.recv().map_err(|e| e.to_string())??;
    let _ = CTX.set(ctx);
    Ok(port)
}

/// People at tables right now (players and watchers connected), for the app to decide whether
/// it may stop the server while in the background.
pub fn connections() -> usize {
    match CTX.get() {
        Some(c) => c.rooms.lock().unwrap().iter().map(|r| r.lock().map(|r| r.conns.len()).unwrap_or(0)).sum(),
        None => 0,
    }
}

/// Open tables.
pub fn tables() -> usize {
    match CTX.get() {
        Some(c) => c.rooms.lock().unwrap().iter().filter(|r| r.lock().map(|r| !r.closed).unwrap_or(false)).count(),
        None => 0,
    }
}

/// The address other devices on the Wi-Fi open, or "" without a LAN.
pub fn lan_url() -> String {
    let port = CTX.get().map(|c| c.port.load(Ordering::SeqCst)).unwrap_or(8000);
    util::lan_ips().first().map(|ip| format!("http://{ip}:{port}/")).unwrap_or_default()
}
