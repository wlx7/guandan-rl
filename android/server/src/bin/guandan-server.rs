//! Desktop runner of the Android server, for testing:
//!     guandan-server --model guandan-medium.gdm [--home DIR] [--port 8000]
//!     guandan-server --model guandan-medium.gdm --selftest SEED   (prints Q values)
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let model = PathBuf::from(arg("--model").unwrap_or_else(|| "guandan-medium.gdm".into()));
    if let Some(seed) = arg("--selftest") {
        let bytes = std::fs::read(&model).expect("model");
        let net = guandan_server::net::Net::load(&bytes).expect("load");
        let mut g = guandan_core::game::Game::new(seed.parse().unwrap(), guandan_core::game::Rules::default());
        // play a few moves with the network to get a non-trivial history
        let n: usize = arg("--moves").and_then(|m| m.parse().ok()).unwrap_or(0);
        for _ in 0..n {
            let m = guandan_server::agent::choose(&net, &g);
            g.apply(&m).unwrap();
        }
        for (m, q) in guandan_server::agent::evaluate(&net, &g) {
            println!("{} {} {:?} {:.6}", m.kind as u8, m.key, m.cards, q);
        }
        return;
    }
    let cfg = guandan_server::Config {
        home: PathBuf::from(arg("--home").unwrap_or_else(|| "guandan-home".into())),
        model,
        port: arg("--port").and_then(|p| p.parse().ok()).unwrap_or(8000),
        platform: arg("--platform").unwrap_or_else(|| "desktop".into()),
    };
    match guandan_server::start(cfg) {
        Ok(port) => {
            println!("listening on http://127.0.0.1:{port}/  LAN: {}", guandan_server::lan_url());
            loop {
                std::thread::sleep(std::time::Duration::from_secs(3600));
            }
        }
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}
