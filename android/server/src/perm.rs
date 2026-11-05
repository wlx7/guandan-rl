//! Who may manage what (python/guandan/server/perm.py): the host (the phone running the app,
//! or a device logged in with the host password) manages everything; other devices manage the
//! tables they opened. Same settings.json format as the Python server.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::util::{random_hex, random_urlsafe};

pub struct Perm {
    path: PathBuf,
    fails: Mutex<HashMap<String, Vec<Instant>>>,
}

#[derive(Clone)]
pub struct Settings {
    pub guest_create: bool,
    pub guest_train: bool,
    pub pin_salt: Option<String>,
    pub pin_hash: Option<String>,
    pub admin_tokens: Vec<String>,
}

fn sha(s: &str) -> String {
    hex::encode(Sha256::digest(s.as_bytes()))
}

fn same(a: &str, b: &str) -> bool {
    // constant time for equal lengths
    a.len() == b.len() && a.bytes().zip(b.bytes()).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

pub fn is_loopback(host: &str) -> bool {
    host == "127.0.0.1" || host == "::1" || host == "localhost" || host == "::ffff:127.0.0.1"
}

impl Perm {
    pub fn new(home: &std::path::Path) -> Perm {
        Perm { path: home.join("settings.json"), fails: Mutex::new(HashMap::new()) }
    }

    pub fn load(&self) -> Settings {
        let d: Value = std::fs::read(&self.path).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(Value::Null);
        let s = |k: &str| d.get(k).and_then(|v| v.as_str()).map(|x| x.to_string());
        Settings {
            guest_create: d.get("guest_create").and_then(|v| v.as_bool()).unwrap_or(true),
            guest_train: d.get("guest_train").and_then(|v| v.as_bool()).unwrap_or(false),
            pin_salt: s("pin_salt"),
            pin_hash: s("pin_hash"),
            admin_tokens: d
                .get("admin_tokens")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_str().map(|x| x.to_string())).collect())
                .unwrap_or_default(),
        }
    }

    fn save(&self, s: &Settings) {
        let v = json!({
            "guest_create": s.guest_create, "guest_train": s.guest_train,
            "pin_salt": s.pin_salt, "pin_hash": s.pin_hash, "admin_tokens": s.admin_tokens,
        });
        let tmp = self.path.with_extension("json.tmp");
        if std::fs::write(&tmp, serde_json::to_string_pretty(&v).unwrap_or_default()).is_ok() {
            let _ = std::fs::rename(&tmp, &self.path);
        }
    }

    pub fn token_ok(&self, token: Option<&str>) -> bool {
        match token {
            Some(t) if !t.is_empty() => {
                let h = sha(t);
                self.load().admin_tokens.iter().any(|x| same(x, &h))
            }
            _ => false,
        }
    }

    pub fn set_pin(&self, pin: &str) {
        let mut d = self.load();
        let pin = pin.trim();
        if pin.is_empty() {
            d.pin_salt = None;
            d.pin_hash = None;
        } else {
            let salt = random_hex(8);
            d.pin_hash = Some(sha(&format!("{salt}{pin}")));
            d.pin_salt = Some(salt);
        }
        d.admin_tokens.clear();
        self.save(&d);
    }

    pub fn login(&self, pin: &str, host: &str) -> Result<String, String> {
        let now = Instant::now();
        let mut fails = self.fails.lock().unwrap();
        let recent: Vec<Instant> = fails
            .get(host)
            .map(|v| v.iter().copied().filter(|t| now.duration_since(*t) < Duration::from_secs(300)).collect())
            .unwrap_or_default();
        if recent.len() >= 5 {
            fails.insert(host.to_string(), recent);
            return Err("密码错误次数太多，请 5 分钟后再试".into());
        }
        let mut d = self.load();
        let Some(hash) = d.pin_hash.clone() else {
            return Err("房主还没有设置房主密码".into());
        };
        let salt = d.pin_salt.clone().unwrap_or_default();
        if !same(&sha(&format!("{salt}{}", pin.trim())), &hash) {
            let mut r = recent;
            r.push(now);
            fails.insert(host.to_string(), r);
            return Err("房主密码不对".into());
        }
        fails.insert(host.to_string(), recent);
        let token = random_urlsafe(24);
        d.admin_tokens.push(sha(&token));
        let n = d.admin_tokens.len();
        if n > 20 {
            d.admin_tokens.drain(..n - 20);
        }
        self.save(&d);
        Ok(token)
    }

    pub fn logout(&self, token: Option<&str>) {
        if let Some(t) = token {
            let h = sha(t);
            let mut d = self.load();
            d.admin_tokens.retain(|x| x != &h);
            self.save(&d);
        }
    }

    pub fn update(&self, guest_create: Option<bool>, guest_train: Option<bool>) {
        let mut d = self.load();
        if let Some(v) = guest_create {
            d.guest_create = v;
        }
        if let Some(v) = guest_train {
            d.guest_train = v;
        }
        self.save(&d);
    }
}
