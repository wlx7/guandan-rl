//! HTTP API, WebSocket and the embedded web UI (python/guandan/server/app.py, play part).

use std::net::SocketAddr;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{ConnectInfo, Path, Query, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::{SinkExt, StreamExt};
use include_dir::{include_dir, Dir};
use serde_json::{json, Map, Value};
use tokio::sync::mpsc;

use crate::perm::is_loopback;
use crate::room::{self, Ident, Room, RoomOpts, RoomRef};
use crate::util::{lan_ips, now};
use crate::{Ctx, VERSION};

static WEB: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/../../python/guandan/web");
const MAX_ROOMS: usize = 8;
const MODEL_ID: &str = "builtin/guandan-medium.pt";

type C = State<Arc<Ctx>>;
type ApiResult = Result<Json<Value>, (StatusCode, Json<Value>)>;

fn fail(code: u16, msg: &str) -> (StatusCode, Json<Value>) {
    (StatusCode::from_u16(code).unwrap_or(StatusCode::BAD_REQUEST), Json(json!({"detail": msg})))
}

fn body_json(b: &Bytes) -> Value {
    serde_json::from_slice(b).unwrap_or(Value::Object(Map::new()))
}

pub fn router(ctx: Arc<Ctx>) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/static/{*path}", get(static_file))
        .route("/api/me", get(api_me))
        .route("/api/perm/login", post(api_login))
        .route("/api/perm/logout", post(api_logout))
        .route("/api/perm/settings", post(api_perm_settings))
        .route("/api/client", post(api_client))
        .route("/api/clients", get(api_clients))
        .route("/api/info", get(api_info))
        .route("/api/qr", get(api_qr))
        .route("/api/models", get(api_models))
        .route("/api/models/manage", post(no_training))
        .route("/api/train/runs", get(|| async { Json(json!([])) }))
        .route("/api/train/start", post(no_training))
        .route("/api/train/stop", post(no_training))
        .route("/api/train/export", post(no_training))
        .route("/api/rooms", get(api_rooms))
        .route("/api/room", get(api_room))
        .route("/api/room/create", post(api_room_create))
        .route("/api/room/join", post(api_room_join))
        .route("/api/room/fill_ai", post(api_room_fill_ai))
        .route("/api/room/pause", post(api_room_pause))
        .route("/api/room/close", post(api_room_close))
        .route("/ws", get(ws_endpoint))
        .with_state(ctx)
}

// ------------------------------------------------------------------ who is asking
fn own_ips(ctx: &Ctx) -> Vec<String> {
    let mut o = ctx.own_ips.lock().unwrap();
    if now() - o.0 > 60.0 {
        *o = (now(), lan_ips());
    }
    o.1.clone()
}

fn ident(ctx: &Ctx, addr: &SocketAddr, headers: &HeaderMap, q: &std::collections::HashMap<String, String>) -> Ident {
    let host = match addr.ip() {
        std::net::IpAddr::V6(v6) => v6.to_ipv4_mapped().map(|v4| v4.to_string()).unwrap_or_else(|| v6.to_string()),
        ip => ip.to_string(),
    };
    let hv = |k: &str| headers.get(k).and_then(|v| v.to_str().ok()).map(|s| s.to_string());
    let token = hv("x-guandan-admin").or_else(|| q.get("admin").cloned()).filter(|t| !t.is_empty());
    let mut device = hv("x-guandan-device").or_else(|| q.get("device").cloned()).unwrap_or_default();
    device.truncate(64);
    let local = is_loopback(&host) || own_ips(ctx).contains(&host);
    let admin = local || ctx.perm.token_ok(token.as_deref());
    Ident { admin, local, device, host, token }
}

type Q = Query<std::collections::HashMap<String, String>>;

// ------------------------------------------------------------------ web UI
fn file_response(path: &str) -> Response {
    let Some(f) = WEB.get_file(path) else {
        return (StatusCode::NOT_FOUND, "not found").into_response();
    };
    let ct = match path.rsplit('.').next().unwrap_or("") {
        "html" => "text/html; charset=utf-8",
        "js" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "json" => "application/json",
        "ico" => "image/x-icon",
        "woff2" => "font/woff2",
        _ => "application/octet-stream",
    };
    // always revalidate, so that an upgraded app shows the new UI
    ([(header::CONTENT_TYPE, ct), (header::CACHE_CONTROL, "no-cache")], f.contents()).into_response()
}

async fn index() -> Response {
    file_response("index.html")
}

async fn static_file(Path(path): Path<String>) -> Response {
    if path.contains("..") {
        return (StatusCode::NOT_FOUND, "not found").into_response();
    }
    file_response(&path)
}

// ------------------------------------------------------------------ info and permissions
fn me_json(ctx: &Ctx, i: &Ident) -> Value {
    let st = ctx.perm.load();
    json!({"admin": i.admin, "local": i.local, "pin_set": st.pin_hash.is_some(), "guest_create": st.guest_create,
           "guest_train": false, "train": false, "host_device": host_device(ctx)})
}

/// How the UI calls the device that runs the server ("这台手机" in the app).
fn host_device(ctx: &Ctx) -> &'static str {
    if ctx.platform == "android" {
        "手机"
    } else {
        "电脑"
    }
}

async fn api_me(State(ctx): C, ConnectInfo(a): ConnectInfo<SocketAddr>, h: HeaderMap, q: Q) -> Json<Value> {
    let i = ident(&ctx, &a, &h, &q);
    Json(me_json(&ctx, &i))
}

async fn api_login(State(ctx): C, ConnectInfo(a): ConnectInfo<SocketAddr>, b: Bytes) -> ApiResult {
    let body = body_json(&b);
    let pin = match &body["pin"] {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        _ => String::new(),
    };
    match ctx.perm.login(&pin, &a.ip().to_string()) {
        Ok(token) => Ok(Json(json!({"token": token}))),
        Err(e) => Err(fail(400, &e)),
    }
}

async fn api_logout(State(ctx): C, ConnectInfo(a): ConnectInfo<SocketAddr>, h: HeaderMap, q: Q) -> Json<Value> {
    let i = ident(&ctx, &a, &h, &q);
    ctx.perm.logout(i.token.as_deref());
    Json(json!({"ok": true}))
}

async fn api_perm_settings(State(ctx): C, ConnectInfo(a): ConnectInfo<SocketAddr>, h: HeaderMap, q: Q, b: Bytes) -> ApiResult {
    let i = ident(&ctx, &a, &h, &q);
    if !i.admin {
        return Err(fail(403, "只有房主可以修改权限设置"));
    }
    let body = body_json(&b);
    ctx.perm.update(body.get("guest_create").and_then(|v| v.as_bool()), None);
    if let Some(p) = body.get("pin") {
        let pin = p.as_str().map(|s| s.to_string()).unwrap_or_else(|| if p.is_null() { String::new() } else { p.to_string() });
        let pin = pin.trim().to_string();
        if !pin.is_empty() && pin.chars().count() < 4 {
            return Err(fail(400, "房主密码至少 4 位"));
        }
        ctx.perm.set_pin(&pin);
    }
    let i = ident(&ctx, &a, &h, &q);
    Ok(Json(me_json(&ctx, &i)))
}

async fn api_client(State(ctx): C, ConnectInfo(a): ConnectInfo<SocketAddr>, h: HeaderMap, q: Q, b: Bytes) -> Json<Value> {
    let i = ident(&ctx, &a, &h, &q);
    let body = body_json(&b);
    let mut row = json!({"t": format!("{:.0}", now()), "host": i.host, "device": i.device.chars().take(8).collect::<String>()});
    for k in ["w", "h", "sw", "sh", "dpr", "fullscreen", "ua"] {
        if let Some(v) = body.get(k) {
            let ok = v.is_number() || v.is_boolean() || v.as_str().map(|s| s.len() <= 200).unwrap_or(false);
            if ok {
                row[k] = v.clone();
            }
        }
    }
    let mut c = ctx.clients.lock().unwrap();
    c.push(row);
    let n = c.len();
    if n > 40 {
        c.drain(..n - 40);
    }
    Json(json!({"ok": true}))
}

async fn api_clients(State(ctx): C, ConnectInfo(a): ConnectInfo<SocketAddr>, h: HeaderMap, q: Q) -> ApiResult {
    if !ident(&ctx, &a, &h, &q).admin {
        return Err(fail(403, "只有房主可以查看设备屏幕信息"));
    }
    let c = ctx.clients.lock().unwrap();
    Ok(Json(json!({"clients": c.iter().rev().cloned().collect::<Vec<_>>()})))
}

async fn api_info(State(ctx): C) -> Json<Value> {
    let port = ctx.port.load(Ordering::SeqCst);
    let urls: Vec<String> = lan_ips().iter().map(|ip| format!("http://{ip}:{port}/")).collect();
    let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    Json(json!({
        "version": VERSION,
        "lan_urls": urls,
        "home": ctx.home.display().to_string(),
        "device": {"cuda": false, "gpu": null, "cpu_count": cpus, "torch": null},
        "presets": [],
        "can_train": false,
        "portable": true,
        "app": ctx.platform,
        "recommended_preset": "medium",
        "default_model": MODEL_ID,
        "cpu_count": cpus,
        "limits": {},
        "scoring": room::scoring_presets(),
        "active_runs": [],
    }))
}

async fn api_qr(Query(q): Q) -> Response {
    let text = q.get("text").cloned().unwrap_or_default();
    match qrcode::QrCode::new(text.as_bytes()) {
        Ok(code) => {
            let svg = code
                .render::<qrcode::render::svg::Color>()
                .min_dimensions(200, 200)
                .quiet_zone(true)
                .dark_color(qrcode::render::svg::Color("#000000"))
                .light_color(qrcode::render::svg::Color("#ffffff"))
                .build();
            ([(header::CONTENT_TYPE, "image/svg+xml")], svg).into_response()
        }
        Err(_) => (StatusCode::BAD_REQUEST, "bad text").into_response(),
    }
}

async fn api_models() -> Json<Value> {
    Json(json!([{"id": MODEL_ID, "source": "内置", "group": "内置", "label": format!("内置模型 v{VERSION}（推荐）"),
                 "hidden": false, "mtime": 0, "size": 0}]))
}

async fn no_training() -> (StatusCode, Json<Value>) {
    fail(400, "安卓版只能对战，训练请用电脑版")
}

// ------------------------------------------------------------------ rooms
fn lock(r: &RoomRef) -> std::sync::MutexGuard<'_, Room> {
    r.lock().unwrap_or_else(|e| e.into_inner())
}

/// Open tables, oldest first.
fn rooms(ctx: &Ctx) -> Vec<RoomRef> {
    let mut rs = ctx.rooms.lock().unwrap();
    rs.retain(|r| !lock(r).closed);
    rs.clone()
}

fn get_room(ctx: &Ctx, room_id: Option<&str>, token: Option<&str>) -> Option<RoomRef> {
    let rs = rooms(ctx);
    if let Some(t) = token.filter(|t| !t.is_empty()) {
        if let Some(r) = rs.iter().find(|r| lock(r).seat_of_token(t).is_some()) {
            return Some(r.clone());
        }
    }
    match room_id.filter(|s| !s.is_empty()) {
        Some(id) => rs.into_iter().find(|r| lock(r).id == id),
        None => rs.last().cloned(),
    }
}

async fn api_rooms(State(ctx): C, ConnectInfo(a): ConnectInfo<SocketAddr>, h: HeaderMap, q: Q) -> Json<Value> {
    let i = ident(&ctx, &a, &h, &q);
    let list: Vec<Value> = rooms(&ctx)
        .iter()
        .map(|r| {
            let r = lock(r);
            let mut s = r.summary();
            s.insert("can_manage".into(), json!(r.can_manage(Some(&i))));
            Value::Object(s)
        })
        .collect();
    Json(json!({"rooms": list, "max": MAX_ROOMS}))
}

async fn api_room(State(ctx): C, ConnectInfo(a): ConnectInfo<SocketAddr>, h: HeaderMap, q: Q) -> Json<Value> {
    let i = ident(&ctx, &a, &h, &q);
    match get_room(&ctx, q.get("id").map(|s| s.as_str()), None) {
        None => Json(json!({"exists": false})),
        Some(r) => {
            let r = lock(&r);
            let mut info = r.public_info();
            info.insert("exists".into(), json!(true));
            info.insert("can_manage".into(), json!(r.can_manage(Some(&i))));
            Json(Value::Object(info))
        }
    }
}

async fn api_room_create(State(ctx): C, ConnectInfo(a): ConnectInfo<SocketAddr>, h: HeaderMap, q: Q, b: Bytes) -> ApiResult {
    let me = ident(&ctx, &a, &h, &q);
    if !(me.admin || ctx.perm.load().guest_create) {
        return Err(fail(403, "房主没有开放其他设备开桌，请让房主开桌后扫码加入"));
    }
    if rooms(&ctx).len() >= MAX_ROOMS {
        return Err(fail(400, &format!("最多同时开 {MAX_ROOMS} 桌，请先在首页关掉不用的牌桌")));
    }
    let body = body_json(&b);
    let mode = body["mode"].as_str().unwrap_or("teach").to_string();
    let seats_in = body["seats"].as_array().cloned().unwrap_or_default();
    let mut seats = vec![];
    for i in 0..4 {
        let s = seats_in.get(i).cloned().unwrap_or(json!({"kind": "ai"}));
        let mut kind = s["kind"].as_str().unwrap_or("ai").to_string();
        if kind != "human" {
            kind = "ai".into();
        }
        if mode == "teach" {
            kind = if i == 0 { "human" } else { "ai" }.into();
        }
        let model = if kind == "ai" { Some(MODEL_ID.to_string()) } else { None };
        let name: String = s["name"].as_str().unwrap_or("").chars().take(20).collect();
        seats.push((kind, model, name));
    }
    let timeout = body["timeout"].as_u64().or_else(|| body["timeout"].as_f64().map(|x| x as u64)).unwrap_or(20).clamp(5, 600);
    let ai_speed = body["ai_speed"].as_f64().unwrap_or(1.0);
    let first_human = seats[0].0 == "human";
    let any_human = seats.iter().any(|s| s.0 == "human");
    let mut r = Room::new(RoomOpts {
        mode,
        seats,
        hint_model: Some(MODEL_ID.into()),
        timeout,
        ai_speed,
        length: body["length"].as_str().unwrap_or("full").to_string(),
        think: "fast".into(), // look-ahead search needs PyTorch
        scoring: body["scoring"].as_str().unwrap_or("none").to_string(),
    });
    r.owner = me.device.clone();
    let mut seat = Value::Null;
    let mut token = Value::Null;
    // host_plays = false: this device only hosts; phones on the LAN take the human seats
    if any_human && body.get("host_plays").and_then(|v| v.as_bool()).unwrap_or(true) {
        let name: String = body["name"].as_str().filter(|s| !s.is_empty()).unwrap_or("房主").chars().take(20).collect();
        if let Ok((s, t)) = r.join(if first_human { Some(0) } else { None }, &name, None) {
            seat = json!(s);
            token = json!(t);
        }
    }
    let id = r.id.clone();
    ctx.rooms.lock().unwrap().push(std::sync::Arc::new(std::sync::Mutex::new(r)));
    Ok(Json(json!({"room": id, "seat": seat, "token": token})))
}

async fn api_room_join(State(ctx): C, b: Bytes) -> ApiResult {
    let body = body_json(&b);
    let rid = body["room"].as_str().map(|s| s.to_string());
    let token = body["token"].as_str().map(|s| s.to_string());
    let Some(r) = get_room(&ctx, rid.as_deref(), token.as_deref()) else {
        if rid.is_some() && token.is_some() {
            return Err(fail(410, "这张牌桌已经关闭了"));
        }
        return Err(fail(404, "这张牌桌不存在或已经关闭"));
    };
    let mut r = lock(&r);
    let seat = body["seat"].as_u64().map(|s| s as usize);
    let name: String = body["name"].as_str().unwrap_or("").chars().take(20).collect();
    match r.join(seat, &name, token.as_deref()) {
        Ok((s, t)) => Ok(Json(json!({"room": r.id, "seat": s, "token": t}))),
        Err(e) => Err(fail(400, &e)),
    }
}

async fn api_room_fill_ai(State(ctx): C, ConnectInfo(a): ConnectInfo<SocketAddr>, h: HeaderMap, q: Q, b: Bytes) -> ApiResult {
    let i = ident(&ctx, &a, &h, &q);
    let body = body_json(&b);
    let Some(r) = get_room(&ctx, body["room"].as_str(), None) else {
        return Err(fail(400, "牌桌已经开始或不存在"));
    };
    {
        let mut g = lock(&r);
        if g.status != "waiting" {
            return Err(fail(400, "牌桌已经开始或不存在"));
        }
        if !g.can_manage(Some(&i)) {
            return Err(fail(403, "只有房主或开这张桌的人可以管理这张牌桌"));
        }
        g.fill_ai();
    }
    room::maybe_start(&r, &ctx);
    lock(&r).broadcast();
    Ok(Json(json!({"ok": true})))
}

async fn api_room_pause(State(ctx): C, ConnectInfo(a): ConnectInfo<SocketAddr>, h: HeaderMap, q: Q, b: Bytes) -> ApiResult {
    let i = ident(&ctx, &a, &h, &q);
    let body = body_json(&b);
    let Some(r) = get_room(&ctx, body["room"].as_str(), None) else {
        return Err(fail(404, "这张牌桌不存在或已经关闭"));
    };
    let mut g = lock(&r);
    if !g.can_manage(Some(&i)) {
        return Err(fail(403, "只有房主或开这张桌的人可以管理这张牌桌"));
    }
    g.set_pause(body["paused"].as_bool().unwrap_or(false), "房主");
    g.broadcast();
    Ok(Json(json!({"ok": true})))
}

async fn api_room_close(State(ctx): C, ConnectInfo(a): ConnectInfo<SocketAddr>, h: HeaderMap, q: Q, b: Bytes) -> ApiResult {
    let i = ident(&ctx, &a, &h, &q);
    let body = body_json(&b);
    let rid = body["room"].as_str().filter(|s| !s.is_empty()).map(|s| s.to_string());
    let r = rid.as_deref().and_then(|id| get_room(&ctx, Some(id), None));
    let targets: Vec<RoomRef> = match (&r, &rid) {
        (Some(r), _) => {
            if !lock(r).can_manage(Some(&i)) {
                return Err(fail(403, "只有房主或开这张桌的人可以管理这张牌桌"));
            }
            vec![r.clone()]
        }
        (None, Some(_)) => vec![],
        (None, None) => {
            if !i.admin {
                return Err(fail(403, "只有房主可以关闭所有牌桌"));
            }
            rooms(&ctx)
        }
    };
    for x in &targets {
        lock(x).end("房主关闭了这张牌桌");
    }
    ctx.rooms.lock().unwrap().retain(|x| !targets.iter().any(|t| Arc::ptr_eq(t, x)));
    Ok(Json(json!({"ok": true})))
}

// ------------------------------------------------------------------ websocket
async fn ws_endpoint(State(ctx): C, ConnectInfo(a): ConnectInfo<SocketAddr>, h: HeaderMap, q: Q, ws: WebSocketUpgrade) -> Response {
    let i = ident(&ctx, &a, &h, &q);
    ws.on_upgrade(move |socket| ws_session(ctx, socket, q.0, i))
}

async fn ws_session(ctx: Arc<Ctx>, socket: WebSocket, q: std::collections::HashMap<String, String>, ident: Ident) {
    let (mut sink, mut stream) = socket.split();
    let token = q.get("token").cloned().filter(|t| !t.is_empty());
    let rid = q.get("room").cloned().filter(|t| !t.is_empty());
    let fatal = |m: &str| Message::Text(json!({"type": "error", "message": m, "fatal": true}).to_string().into());
    let Some(r) = get_room(&ctx, rid.as_deref(), token.as_deref()) else {
        let m = if rid.is_some() || token.is_some() { "这张牌桌已经关闭了" } else { "当前没有牌桌" };
        let _ = sink.send(fatal(m)).await;
        let _ = sink.close().await;
        return;
    };
    let cid = ctx.next_conn.fetch_add(1, Ordering::SeqCst);
    let (tx, mut rx) = mpsc::unbounded_channel::<String>();
    let seat: Option<i32> = {
        let mut g = lock(&r);
        let seat = token.as_deref().and_then(|t| g.seat_of_token(t));
        match (seat, &token) {
            (None, Some(_)) => None,
            (None, None) => {
                g.spectators.insert(cid);
                g.conns.insert(cid, room::Conn { tx, ident });
                Some(-1)
            }
            (Some(s), _) => {
                g.seats[s].conns.insert(cid);
                g.conns.insert(cid, room::Conn { tx, ident });
                Some(s as i32)
            }
        }
    };
    let Some(seat) = seat else {
        let _ = sink.send(fatal("座位已失效，请重新加入")).await;
        let _ = sink.close().await;
        return;
    };
    let writer = tokio::spawn(async move {
        while let Some(m) = rx.recv().await {
            if sink.send(Message::Text(m.into())).await.is_err() {
                break;
            }
        }
        let _ = sink.close().await;
    });
    room::maybe_start(&r, &ctx);
    lock(&r).broadcast();
    while let Some(Ok(m)) = stream.next().await {
        let text = match m {
            Message::Text(t) => t.to_string(),
            Message::Close(_) => break,
            _ => continue,
        };
        let Ok(msg) = serde_json::from_str::<Value>(&text) else { continue };
        if lock(&r).closed {
            lock(&r).send(cid, &json!({"type": "error", "message": "牌桌已关闭", "fatal": true}));
            break;
        }
        room::handle(&r, &ctx, seat, cid, msg).await;
    }
    {
        let mut g = lock(&r);
        g.conns.remove(&cid);
        if seat >= 0 {
            g.seats[seat as usize].conns.remove(&cid);
        } else {
            g.spectators.remove(&cid);
        }
        g.broadcast();
    }
    // the writer ends when the room drops our sender (removed above)
    let _ = tokio::time::timeout(std::time::Duration::from_secs(2), writer).await;
}
