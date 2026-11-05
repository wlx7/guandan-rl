use std::net::{IpAddr, UdpSocket};
use std::time::{SystemTime, UNIX_EPOCH};

use base64::Engine;
use rand::RngCore;

pub fn now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

pub fn random_hex(nbytes: usize) -> String {
    let mut b = vec![0u8; nbytes];
    rand::thread_rng().fill_bytes(&mut b);
    hex::encode(b)
}

/// Like Python's secrets.token_urlsafe(nbytes).
pub fn random_urlsafe(nbytes: usize) -> String {
    let mut b = vec![0u8; nbytes];
    rand::thread_rng().fill_bytes(&mut b);
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(b)
}

/// This device's LAN addresses, the one of the default route (normally Wi-Fi) first.
pub fn lan_ips() -> Vec<String> {
    let mut primary = None;
    if let Ok(s) = UdpSocket::bind("0.0.0.0:0") {
        if s.connect("8.8.8.8:80").is_ok() {
            if let Ok(a) = s.local_addr() {
                primary = Some(a.ip().to_string());
            }
        }
    }
    let mut ips: Vec<String> = vec![];
    if let Ok(ifs) = if_addrs::get_if_addrs() {
        for i in ifs {
            if let IpAddr::V4(v4) = i.ip() {
                if !v4.is_loopback() && !v4.is_link_local() {
                    let s = v4.to_string();
                    if !ips.contains(&s) {
                        ips.push(s);
                    }
                }
            }
        }
    }
    if let Some(p) = &primary {
        if !p.starts_with("127.") && p != "0.0.0.0" && !ips.contains(p) {
            ips.push(p.clone());
        }
    }
    let score = |ip: &String| -> (u8, String) {
        let r = if Some(ip) == primary.as_ref() {
            0
        } else if ip.starts_with("192.168.") {
            1
        } else if ip.starts_with("10.") {
            2
        } else {
            3
        };
        (r, ip.clone())
    };
    ips.sort_by_key(score);
    ips
}
