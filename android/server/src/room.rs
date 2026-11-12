//! A game table: seats (human / AI), turn loop, timers, hints and per-player views.
//! Port of python/guandan/server/room.py; the messages are the same, so the same web UI works
//! with either server.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde_json::{json, Map, Value};
use tokio::sync::mpsc;

use guandan_core::cards::hand_to_vec;
use guandan_core::cards::{hand_count, wild_type};
use guandan_core::game::{Game, Phase, Rules, ScoreRules};
use guandan_core::moves::{classify, Kind, Move};

use crate::agent;
use crate::util::{now, random_hex, random_urlsafe};
use crate::Ctx;

pub type ConnId = u64;
pub type RoomRef = Arc<Mutex<Room>>;

#[derive(Clone, Debug, Default)]
pub struct Ident {
    pub admin: bool,
    pub local: bool,
    pub device: String,
    pub host: String,
    pub token: Option<String>,
}

pub struct Conn {
    pub tx: mpsc::UnboundedSender<String>,
    pub ident: Ident,
}

pub struct Seat {
    pub index: usize,
    pub kind: String, // "human" | "ai"
    pub model: Option<String>,
    pub name: String,
    pub token: Option<String>,
    pub conns: HashSet<ConnId>,
    pub ready: bool,
}

impl Seat {
    pub fn taken(&self) -> bool {
        self.token.is_some()
    }
    pub fn connected(&self) -> bool {
        !self.conns.is_empty()
    }
    fn human(&self) -> bool {
        self.kind == "human"
    }
}

pub fn scoring(name: &str) -> Option<ScoreRules> {
    match name {
        "standard" => Some(ScoreRules { doubles: [false, true, true, true], cap: 16 }),
        "all" => Some(ScoreRules { doubles: [true, true, true, true], cap: 256 }),
        _ => None,
    }
}

pub fn scoring_presets() -> Value {
    json!({
        "standard": {"doubles": [false, true, true, true], "cap": 16, "label": "标准倍数",
                     "desc": "6 张及以上炸弹、同花顺、天王炸每出一个翻一倍，封顶 16 倍"},
        "all": {"doubles": [true, true, true, true], "cap": 256, "label": "全炸翻倍",
                "desc": "每出一个炸弹（含同花顺、天王炸）翻一倍，封顶 256 倍"},
    })
}

const SEAT_NAMES: [&str; 4] = ["南", "东", "北", "西"];

fn round3(x: f32) -> f64 {
    ((x as f64) * 1000.0).round() / 1000.0
}

pub struct RoomOpts {
    pub mode: String,
    pub seats: Vec<(String, Option<String>, String)>, // kind, model, name
    pub hint_model: Option<String>,
    pub timeout: u64,
    pub ai_speed: f64,
    pub length: String,
    pub think: String,
    pub scoring: String,
}

pub struct Room {
    pub id: String,
    pub mode: String,
    pub length: String,
    pub think: String,
    pub scoring: String,
    totals: [i64; 2],
    pub timeout: u64,
    ai_speed: f64,
    pub hint_model: Option<String>,
    pub seats: Vec<Seat>,
    pub spectators: HashSet<ConnId>,
    pub conns: HashMap<ConnId, Conn>,
    rng: StdRng,
    points: [i64; 2],
    deals: u32,
    pub game: Game,
    pub status: String,
    pub seq: u64,
    event_no: u64,
    event: Value,
    trick: BTreeMap<u8, Value>,
    deadline: Option<f64>,
    pub closed: bool,
    created: f64,
    match_no: u32,
    score: [u32; 2],
    round_log: Vec<Value>,
    review: Option<Value>,
    paused: bool,
    manual_pause: Option<String>,
    pub owner: String,
    last_active: f64,
    inbox: Vec<mpsc::UnboundedSender<(Value, ConnId)>>,
    inbox_rx: Option<Vec<mpsc::UnboundedReceiver<(Value, ConnId)>>>,
    task: Option<tokio::task::AbortHandle>,
}

impl Room {
    pub fn new(o: RoomOpts) -> Room {
        let length = if o.length == "quick" { "quick" } else { "full" }.to_string();
        let scoring_name = if length == "quick" && scoring(&o.scoring).is_some() { o.scoring.clone() } else { "none".into() };
        let seats = o
            .seats
            .into_iter()
            .enumerate()
            .map(|(i, (kind, model, name))| {
                let name = if kind == "ai" && name.is_empty() { format!("AI {}", SEAT_NAMES[i]) } else { name };
                Seat { index: i, kind, model, name, token: None, conns: HashSet::new(), ready: false }
            })
            .collect();
        let mut inbox = vec![];
        let mut inbox_rx = vec![];
        for _ in 0..4 {
            let (tx, rx) = mpsc::unbounded_channel();
            inbox.push(tx);
            inbox_rx.push(rx);
        }
        let mut rng = StdRng::from_entropy();
        let game = Self::make_game(&length, &mut rng);
        Room {
            id: random_hex(3),
            mode: if o.mode == "teach" { "teach" } else { "real" }.to_string(),
            length,
            think: if o.think == "deep" { "deep" } else { "fast" }.to_string(),
            scoring: scoring_name,
            totals: [0, 0],
            timeout: o.timeout,
            ai_speed: o.ai_speed,
            hint_model: o.hint_model,
            seats,
            spectators: HashSet::new(),
            conns: HashMap::new(),
            rng,
            points: [0, 0],
            deals: 1,
            game,
            status: "waiting".into(),
            seq: 0,
            event_no: 0,
            event: json!({"type": "deal", "id": 0}),
            trick: BTreeMap::new(),
            deadline: None,
            closed: false,
            created: now(),
            match_no: 1,
            score: [0, 0],
            round_log: vec![],
            review: None,
            paused: false,
            manual_pause: None,
            owner: String::new(),
            last_active: now(),
            inbox,
            inbox_rx: Some(inbox_rx),
            task: None,
        }
    }

    fn make_game(length: &str, rng: &mut StdRng) -> Game {
        let seed = rng.gen::<u64>() >> 16;
        if length == "full" {
            return Game::new(seed, Rules { a_fail_limit: 3, tribute: true });
        }
        let mut g = Game::new(seed, Rules { a_fail_limit: 3, tribute: false });
        let lv = rng.gen_range(0..13u8);
        g.set_levels([lv, lv], rng.gen_range(0..2i8));
        g.round_no = 0;
        g.prev_order = None;
        g.start_round();
        g
    }

    fn new_event(&mut self, mut ev: Value) {
        self.event_no += 1;
        ev["id"] = json!(self.event_no);
        self.event = ev;
    }

    // ------------------------------------------------------------------ seats
    fn has_humans(&self) -> bool {
        self.seats.iter().any(|s| s.human())
    }

    /// A table with human players runs while one of them is connected; an all-AI table runs
    /// while someone watches it.
    pub fn someone_here(&self) -> bool {
        if self.has_humans() {
            self.seats.iter().any(|s| s.human() && s.connected())
        } else {
            !self.spectators.is_empty()
        }
    }

    pub fn can_manage(&self, ident: Option<&Ident>) -> bool {
        match ident {
            None => false,
            Some(i) => i.admin || (!self.owner.is_empty() && i.device == self.owner),
        }
    }

    pub fn can_manage_conn(&self, cid: ConnId) -> bool {
        self.can_manage(self.conns.get(&cid).map(|c| &c.ident))
    }

    pub fn can_run(&self) -> bool {
        self.manual_pause.is_none() && self.someone_here()
    }

    pub fn set_pause(&mut self, on: bool, by: &str) {
        self.manual_pause = if on { Some(if by.is_empty() { "有人".to_string() } else { by.to_string() }) } else { None };
    }

    pub fn join(&mut self, seat_index: Option<usize>, name: &str, token: Option<&str>) -> Result<(usize, String), String> {
        if let Some(t) = token.filter(|t| !t.is_empty()) {
            for s in self.seats.iter_mut() {
                if s.token.as_deref() == Some(t) {
                    if !name.is_empty() {
                        s.name = name.to_string();
                    }
                    return Ok((s.index, t.to_string()));
                }
            }
        }
        let idx = match seat_index {
            Some(i) if i < 4 => i,
            Some(_) => return Err("座位不存在".into()),
            None => match self.seats.iter().find(|s| s.human() && !s.taken()) {
                Some(s) => s.index,
                None => return Err("没有空余的玩家座位".into()),
            },
        };
        let s = &mut self.seats[idx];
        if !s.human() {
            return Err("该座位是 AI".into());
        }
        if s.taken() {
            return Err("该座位已有人".into());
        }
        let t = random_urlsafe(12);
        s.token = Some(t.clone());
        s.name = if name.is_empty() { format!("玩家{}", idx + 1) } else { name.to_string() };
        Ok((idx, t))
    }

    pub fn seat_of_token(&self, token: &str) -> Option<usize> {
        self.seats.iter().find(|s| s.token.as_deref() == Some(token)).map(|s| s.index)
    }

    pub fn public_info(&self) -> Map<String, Value> {
        let seats: Vec<Value> = self
            .seats
            .iter()
            .map(|s| {
                json!({"index": s.index, "kind": s.kind, "name": s.name, "taken": s.taken(), "connected": s.connected(),
                       "model": s.model.as_ref().map(|m| m.rsplit(['/', '\\']).next().unwrap_or("").to_string())})
            })
            .collect();
        let v = json!({"id": self.id, "mode": self.mode, "length": self.length, "think": self.think,
                       "scoring": self.scoring, "status": self.status, "timeout": self.timeout, "seats": seats});
        v.as_object().unwrap().clone()
    }

    pub fn summary(&self) -> Map<String, Value> {
        let g = &self.game;
        let mut info = self.public_info();
        let extra = json!({
            "created": self.created,
            "last_active": self.last_active,
            "paused": self.paused || self.manual_pause.is_some(),
            "pause_by": self.manual_pause,
            "level": g.level,
            "team_level": g.team_level,
            "round_no": if self.length == "full" { g.round_no } else { self.deals },
            "match_no": self.match_no,
            "score": self.score,
            "points": self.points,
            "humans": self.seats.iter().filter(|s| s.human()).count(),
            "connected": self.seats.iter().filter(|s| s.human() && s.connected()).count(),
            "spectators": self.spectators.len(),
        });
        for (k, v) in extra.as_object().unwrap() {
            info.insert(k.clone(), v.clone());
        }
        info
    }

    pub fn fill_ai(&mut self) {
        let hm = self.hint_model.clone();
        for s in self.seats.iter_mut() {
            if s.human() && !s.taken() {
                s.kind = "ai".into();
                s.model = hm.clone();
                s.name = format!("AI {}", SEAT_NAMES[s.index]);
            }
        }
    }

    pub fn close(&mut self) {
        self.closed = true;
        if let Some(t) = self.task.take() {
            t.abort();
        }
    }

    /// Close the table for good and send everybody at it back to the home page.
    pub fn end(&mut self, message: &str) {
        self.close();
        let msg = json!({"type": "closed", "message": message}).to_string();
        for c in self.conns.values() {
            let _ = c.tx.send(msg.clone());
        }
    }

    // ------------------------------------------------------------------ views
    fn move_info(&self, m: &Move) -> Value {
        let desc = if m.kind == Kind::Pass { "不出".to_string() } else { m.describe(self.game.level) };
        json!({"kind": m.kind as u8, "key": m.key, "cards": m.cards, "desc": desc})
    }

    pub fn view(&self, v: i32) -> Value {
        let g = &self.game;
        let counts: Vec<u32> = (0..4).map(|p| hand_count(&g.hands[p])).collect();
        let finished = g.finished.clone();
        let seats: Vec<Value> = self
            .seats
            .iter()
            .map(|s| {
                let ai = s.kind == "ai";
                json!({"name": s.name, "kind": s.kind, "connected": s.connected() || ai, "taken": s.taken() || ai,
                       "ready": s.ready || ai})
            })
            .collect();
        let trick: Map<String, Value> = self.trick.iter().map(|(k, v)| (k.to_string(), v.clone())).collect();
        let tribute: Vec<Value> =
            g.tribute.iter().map(|t| json!({"from": t.from, "to": t.to, "card": t.card, "is_return": t.is_return})).collect();
        let pending: Vec<Value> = g.pending_returns.iter().map(|(a, b)| json!([a, b])).collect();
        let sc = scoring(&self.scoring);
        let mut st = json!({
            "type": "state",
            "seq": self.seq,
            "room": self.id,
            "mode": self.mode,
            "length": self.length,
            "points": self.points,
            "think": self.think,
            "scoring": self.scoring,
            "totals": self.totals,
            "status": self.status,
            "you": v,
            "phase": g.phase as u8,
            "current": g.current,
            "level": g.level,
            "wild": wild_type(g.level),
            "team_level": g.team_level,
            "banker": g.banker,
            "round_no": if self.length == "full" { g.round_no } else { self.deals },
            "match_no": self.match_no,
            "score": self.score,
            "a_fail": g.a_fail,
            "is_leading": g.is_leading(),
            "counts": counts,
            "finished": finished,
            "seats": seats,
            "paused": self.paused,
            "pause_by": self.manual_pause,
            "trick": trick,
            "tribute": tribute,
            "pending_returns": pending,
            "anti_tribute": g.anti_tribute,
            "deadline": self.deadline.map(|d| (d * 1000.0) as i64),
            "server_time": (now() * 1000.0) as i64,
            "timeout": if self.mode == "real" { Some(self.timeout) } else { None },
            "event": self.event,
            "hands": {},
            "hint_available": self.mode == "teach" && self.hint_model.is_some(),
            "bombs": g.bombs,
            "multiplier": sc.as_ref().map(|s| s.multiplier(&g.bombs)).unwrap_or(1),
        });
        st["last_move"] = match &g.last_move {
            Some(m) => {
                let mut i = self.move_info(m);
                i["player"] = json!(g.last_player);
                i
            }
            None => Value::Null,
        };
        if v >= 0 {
            st["hand"] = json!(hand_to_vec(&g.hands[v as usize]));
        }
        st["played"] = json!((0..4).map(|p| hand_to_vec(&g.played[p])).collect::<Vec<_>>());
        let mut visible: Vec<usize> = vec![];
        if (v < 0 && !self.has_humans()) || self.mode == "teach" {
            visible = vec![0, 1, 2, 3];
        } else if v >= 0 && g.finished.contains(&(v as u8)) {
            visible = vec![((v + 2) % 4) as usize];
        }
        for p in visible {
            if p as i32 != v {
                st["hands"][p.to_string()] = json!(hand_to_vec(&g.hands[p]));
            }
        }
        if matches!(g.phase, Phase::RoundOver | Phase::MatchOver) {
            if let Some(r) = &g.last_result {
                let (mut match_over, mut a_reset) = (r.match_over, r.a_reset);
                if self.length == "quick" {
                    // every quick deal stands alone: no "过 A", no match to win
                    match_over = false;
                    a_reset = false;
                }
                st["result"] = json!({
                    "order": r.order, "win_team": r.win_team, "gain": r.gain, "match_over": match_over,
                    "before": r.levels_before, "after": r.levels_after, "a_reset": a_reset,
                    "score": sc.as_ref().map(|s| g.round_score(r.win_team, s) as i64),
                });
            }
        }
        if let Some(rv) = &self.review {
            if rv["seat"].as_i64() == Some(v as i64) {
                st["review"] = rv.clone();
            }
        }
        st
    }

    pub fn send(&self, cid: ConnId, msg: &Value) {
        if let Some(c) = self.conns.get(&cid) {
            let _ = c.tx.send(msg.to_string());
        }
    }

    pub fn broadcast(&mut self) {
        self.seq += 1;
        for s in &self.seats {
            if s.conns.is_empty() {
                continue;
            }
            let m = self.view(s.index as i32).to_string();
            for c in &s.conns {
                if let Some(conn) = self.conns.get(c) {
                    let _ = conn.tx.send(m.clone());
                }
            }
        }
        if !self.spectators.is_empty() {
            let m = self.view(-1).to_string();
            for c in &self.spectators {
                if let Some(conn) = self.conns.get(c) {
                    let _ = conn.tx.send(m.clone());
                }
            }
        }
    }

    // ------------------------------------------------------------------ actions
    fn apply(&mut self, p: u8, mv: &Move, how: &str) {
        self.last_active = now();
        let before_trick = self.game.trick_id;
        let phase_before = self.game.phase;
        if self.game.apply(mv).is_err() {
            return;
        }
        let g = &self.game;
        if phase_before == Phase::Return {
            let to = g.tribute.iter().filter(|t| t.is_return).last().map(|t| t.to).unwrap_or(0);
            let card = mv.cards.first().copied().unwrap_or(0);
            let play = g.phase == Phase::Play;
            self.new_event(json!({"type": "return", "from": p, "to": to, "card": card, "how": how}));
            if play {
                self.trick.clear();
            }
            return;
        }
        let info = self.move_info(mv);
        let pass = mv.kind == Kind::Pass;
        let mut t = info.clone();
        t["pass_"] = json!(pass);
        self.trick.insert(p, t);
        let mut ev = info;
        ev["type"] = json!(if pass { "pass" } else { "play" });
        ev["player"] = json!(p);
        ev["how"] = json!(how);
        ev["bomb"] = json!(matches!(mv.kind, Kind::Bomb | Kind::StraightFlush | Kind::JokerBomb));
        ev["finished"] = json!(g.finished.contains(&p) && !pass);
        if matches!(g.phase, Phase::RoundOver | Phase::MatchOver) {
            ev["round_over"] = json!(true);
            let sc = scoring(&self.scoring);
            if let Some(r) = g.last_result.clone() {
                let w = r.win_team as usize;
                let rs = sc.as_ref().map(|s| g.round_score(r.win_team, s) as i64);
                if let Some(v) = rs {
                    self.totals[w] += v;
                    self.totals[1 - w] -= v;
                }
                if self.length == "quick" {
                    let gain = rs.unwrap_or(r.gain as i64);
                    self.points[w] += gain;
                    self.points[1 - w] -= gain;
                } else if r.match_over {
                    self.score[w] += 1;
                }
                let rn = self.game.round_no;
                self.round_log.push(json!({"round": rn, "order": r.order, "win_team": r.win_team, "gain": r.gain,
                                           "after": r.levels_after}));
            }
        } else if g.trick_id != before_trick {
            ev["trick_end"] = json!(true);
            ev["jiefeng"] = json!(g.jiefeng);
            ev["leader"] = json!(g.current);
            self.trick.clear();
        }
        self.new_event(ev);
    }

    /// A human's message during their turn: Ok(Some(move)) to play it, Ok(None) to ignore it,
    /// Err(reply) to answer with an error or a choice of interpretations.
    fn check_human(&self, msg: &Value) -> Result<Option<Move>, Value> {
        let g = &self.game;
        let act = msg["action"].as_str().unwrap_or("");
        let err = |m: &str| Err(json!({"type": "error", "message": m}));
        if g.phase == Phase::Return {
            if act != "return" {
                return err("请先选择一张牌还贡");
            }
            let c = msg["card"].as_i64().unwrap_or(-1);
            let mut found = None;
            for m in g.legal_moves() {
                if m.cards.len() == 1 && m.cards[0] as i64 == c {
                    found = Some(m);
                }
            }
            return match found {
                Some(m) => Ok(Some(m)),
                None => err("还贡牌必须是 10 或以下的牌"),
            };
        }
        match act {
            "pass" => {
                if g.is_leading() {
                    err("你是首家，必须出牌")
                } else {
                    Ok(Some(Move::pass()))
                }
            }
            "play" => {
                let raw: Vec<i64> = msg["cards"].as_array().map(|a| a.iter().filter_map(|x| x.as_i64()).collect()).unwrap_or_default();
                if raw.is_empty() || raw.iter().any(|&c| !(0..54).contains(&c)) || raw.len() > 27 {
                    return err("不是有效牌型");
                }
                let cards: Vec<u8> = raw.iter().map(|&c| c as u8).collect();
                let mut interps = g.legal_interpretations(&cards);
                if let Some(k) = msg.get("kind").and_then(|k| k.as_i64()) {
                    let key = msg.get("key").and_then(|k| k.as_i64());
                    interps.retain(|x| x.kind as i64 == k && key.map_or(true, |kk| x.key as i64 == kk));
                }
                if interps.is_empty() {
                    return err(if classify(&cards, g.level).is_empty() { "不是有效牌型" } else { "管不上上家的牌" });
                }
                let mut kinds: Vec<u8> = interps.iter().map(|x| x.kind as u8).collect();
                kinds.sort_unstable();
                kinds.dedup();
                if kinds.len() > 1 {
                    let opts: Vec<Value> = kinds
                        .iter()
                        .map(|&kd| {
                            let best = interps.iter().filter(|x| x.kind as u8 == kd).max_by_key(|x| x.key).unwrap();
                            self.move_info(best)
                        })
                        .collect();
                    return Err(json!({"type": "choose", "options": opts}));
                }
                Ok(interps.into_iter().max_by_key(|x| x.key))
            }
            _ => Ok(None),
        }
    }

    fn ai_delay(&self) -> f64 {
        let base = if self.mode == "teach" { 0.55 } else { 0.85 };
        (base + rand::random::<f64>() * 0.5) / self.ai_speed.max(0.2)
    }
}

fn lock(r: &RoomRef) -> std::sync::MutexGuard<'_, Room> {
    r.lock().unwrap_or_else(|e| e.into_inner())
}

async fn sleep(s: f64) {
    if s > 0.0 {
        tokio::time::sleep(Duration::from_secs_f64(s)).await;
    }
}

/// Start the game when every human seat is taken and connected.
pub fn maybe_start(room: &RoomRef, ctx: &Arc<Ctx>) -> bool {
    let mut r = lock(room);
    if r.status == "waiting" && r.seats.iter().filter(|s| s.human()).all(|s| s.taken() && s.connected()) {
        r.status = "playing".into();
        r.new_event(json!({"type": "deal"}));
        let rx = r.inbox_rx.take().unwrap_or_default();
        let task = tokio::spawn(run(room.clone(), ctx.clone(), rx));
        r.task = Some(task.abort_handle());
        return true;
    }
    false
}

async fn think<T: Send + 'static>(ctx: &Arc<Ctx>, f: impl FnOnce() -> T + Send + 'static) -> Option<T> {
    let _permit = ctx.think_slots.acquire().await.ok()?;
    tokio::task::spawn_blocking(f).await.ok()
}

async fn ai_move(room: &RoomRef, ctx: &Arc<Ctx>) -> Move {
    let (g, deep) = {
        let r = lock(room);
        (r.game.clone(), r.think == "deep")
    };
    let net = ctx.net.clone();
    let g2 = g.clone();
    match think(ctx, move || agent::choose(&net, &g2, deep)).await {
        Some(m) => m,
        None => agent::rule_move(&g),
    }
}

/// Pause while nobody is at the table or while paused by hand; play goes on from the same spot.
async fn hold(room: &RoomRef) {
    {
        let mut r = lock(room);
        if r.closed || r.can_run() {
            return;
        }
        r.paused = true;
        r.deadline = None;
        r.broadcast();
    }
    loop {
        {
            let r = lock(room);
            if r.closed || r.can_run() {
                break;
            }
        }
        sleep(0.3).await;
    }
    {
        let mut r = lock(room);
        r.paused = false;
        r.broadcast();
    }
    sleep(0.8).await; // let the returning player see the table before play goes on
}

async fn wait_continue(room: &RoomRef) {
    let (humans, real) = {
        let mut r = lock(room);
        for s in r.seats.iter_mut() {
            s.ready = s.kind == "ai";
        }
        r.broadcast();
        (r.has_humans(), r.mode == "real")
    };
    if !humans {
        sleep(5.0).await;
        hold(room).await;
        return;
    }
    let wait = if real { 30.0 } else { 1e9 };
    let mut t_end = now() + wait;
    loop {
        let run = {
            let r = lock(room);
            if r.closed {
                return;
            }
            r.can_run()
        };
        if !run {
            hold(room).await;
            t_end = now() + wait;
        }
        {
            let r = lock(room);
            // players who left do not hold up the others
            if r.seats.iter().filter(|s| s.human()).all(|s| s.ready || !s.connected()) {
                return;
            }
        }
        if real && now() > t_end {
            return;
        }
        sleep(0.2).await;
    }
}

/// (move, how) for a human seat, or None when the table should pause first.
async fn human_turn(room: &RoomRef, ctx: &Arc<Ctx>, p: usize, rx: &mut mpsc::UnboundedReceiver<(Value, ConnId)>) -> Option<(Move, &'static str)> {
    let connected = {
        let r = lock(room);
        if !r.can_run() {
            return None;
        }
        r.seats[p].connected()
    };
    if !connected {
        // the player left while others play on: the AI plays for them (托管)
        let t0 = now();
        let (g, delay, deep) = {
            let r = lock(room);
            (r.game.clone(), r.ai_delay(), r.think == "deep")
        };
        let net = ctx.net.clone();
        let g2 = g.clone();
        let mv = think(ctx, move || agent::choose(&net, &g2, deep)).await.unwrap_or_else(|| agent::rule_move(&g));
        sleep(delay - (now() - t0)).await;
        let r = lock(room);
        if r.seats[p].connected() || !r.can_run() {
            return None;
        }
        return Some((mv, "auto"));
    }
    let timed = {
        let mut r = lock(room);
        let timed = r.mode == "real";
        r.deadline = if timed { Some(now() + r.timeout as f64) } else { None };
        while rx.try_recv().is_ok() {}
        r.broadcast();
        timed
    };
    loop {
        match tokio::time::timeout(Duration::from_millis(300), rx.recv()).await {
            Ok(Some((msg, cid))) => {
                let r = lock(room);
                if r.closed {
                    return None;
                }
                match r.check_human(&msg) {
                    Ok(Some(mv)) => return Some((mv, "normal")),
                    Ok(None) => {}
                    Err(reply) => r.send(cid, &reply),
                }
            }
            Ok(None) => return None,
            Err(_) => {}
        }
        let mut r = lock(room);
        if r.closed {
            return None;
        }
        if !r.seats[p].connected() || r.manual_pause.is_some() {
            r.deadline = None;
            return None;
        }
        if timed && now() > r.deadline.unwrap_or(f64::MAX) + 0.3 {
            return Some((agent::random_move(&r.game), "timeout"));
        }
    }
}

/// Teaching mode: compare the human's move with the AI's preferred move.
fn make_review(net: &crate::net::Net, g: &Game, p: usize, mv: &Move, describe: impl Fn(&Move) -> Value) -> Option<Value> {
    let ev = agent::evaluate(net, g);
    if ev.len() <= 1 {
        return None;
    }
    let mut cards = mv.cards.clone();
    cards.sort_unstable();
    let mut hit: Option<(usize, f32)> = None;
    for (i, (m, q)) in ev.iter().enumerate() {
        if m.kind == mv.kind && m.key == mv.key && m.cards.len() == cards.len() {
            let mut mc = m.cards.clone();
            mc.sort_unstable();
            if hit.is_none() || mc == cards {
                hit = Some((i + 1, q * 3.0));
                if mc == cards {
                    break;
                }
            }
        }
    }
    let (best_m, best_q) = &ev[0];
    let mut best = describe(best_m);
    best["q"] = json!(round3(best_q * 3.0));
    let chosen = describe(&Move { kind: mv.kind, key: mv.key, cards });
    Some(json!({"seat": p, "best": best, "chosen": chosen,
                "chosen_q": hit.map(|h| round3(h.1)), "rank": hit.map(|h| h.0), "total": ev.len(), "searched": true}))
}

async fn run(room: RoomRef, ctx: Arc<Ctx>, mut inbox: Vec<mpsc::UnboundedReceiver<(Value, ConnId)>>) {
    lock(&room).broadcast();
    sleep(1.2).await;
    let tribute = {
        let mut r = lock(&room);
        let t = r.game.phase == Phase::Return || !r.game.tribute.is_empty();
        if t {
            r.new_event(json!({"type": "tribute"}));
            r.broadcast();
        }
        t
    };
    if tribute {
        sleep(1.0).await;
    }
    loop {
        let over = {
            let r = lock(&room);
            if r.closed {
                break;
            }
            matches!(r.game.phase, Phase::RoundOver | Phase::MatchOver)
        };
        if over {
            lock(&room).deadline = None;
            wait_continue(&room).await;
            {
                let mut r = lock(&room);
                if r.closed {
                    break;
                }
                if r.length == "quick" {
                    let len = r.length.clone();
                    r.game = Room::make_game(&len, &mut r.rng);
                    r.deals += 1;
                } else {
                    if r.game.phase == Phase::MatchOver {
                        r.match_no += 1;
                    }
                    r.game.next_round();
                }
                r.trick.clear();
                r.review = None;
                r.new_event(json!({"type": "deal"}));
                r.broadcast();
            }
            sleep(1.4).await;
            let t = {
                let mut r = lock(&room);
                let t = !r.game.tribute.is_empty();
                if t {
                    r.new_event(json!({"type": "tribute"}));
                    r.broadcast();
                }
                t
            };
            if t {
                sleep(1.2).await;
            }
            continue;
        }
        hold(&room).await;
        let (p, ai) = {
            let r = lock(&room);
            if r.closed {
                break;
            }
            let p = r.game.current as usize;
            (p, r.seats[p].kind == "ai")
        };
        if ai {
            let t0 = now();
            let delay = {
                let mut r = lock(&room);
                r.deadline = None;
                r.ai_delay()
            };
            let mv = ai_move(&room, &ctx).await;
            sleep(delay - (now() - t0)).await;
            let mut r = lock(&room);
            if !r.can_run() || r.closed {
                continue; // everybody left (or paused) while the AI was thinking
            }
            r.apply(p as u8, &mv, "normal");
        } else {
            let Some((mv, how)) = human_turn(&room, &ctx, p, &mut inbox[p]).await else {
                continue; // paused or closed: decide again from the top
            };
            let review_job = {
                let mut r = lock(&room);
                r.deadline = None;
                r.review = None;
                if how == "normal" && r.mode == "teach" && r.hint_model.is_some() && r.game.phase == Phase::Play {
                    Some(r.game.clone())
                } else {
                    None
                }
            };
            if let Some(g) = review_job {
                let net = ctx.net.clone();
                let mv2 = mv.clone();
                let level = g.level;
                let rv = think(&ctx, move || {
                    make_review(&net, &g, p, &mv2, |m| {
                        let desc = if m.kind == Kind::Pass { "不出".to_string() } else { m.describe(level) };
                        json!({"kind": m.kind as u8, "key": m.key, "cards": m.cards, "desc": desc})
                    })
                })
                .await
                .flatten();
                lock(&room).review = rv;
            }
            lock(&room).apply(p as u8, &mv, how);
        }
        lock(&room).broadcast();
    }
}

/// A message from a player or watcher at the table.
pub async fn handle(room: &RoomRef, ctx: &Arc<Ctx>, seat: i32, cid: ConnId, msg: Value) {
    let act = msg["action"].as_str().unwrap_or("").to_string();
    match act.as_str() {
        "hint" => {
            let g = {
                let r = lock(room);
                if r.mode != "teach" || r.hint_model.is_none() || seat < 0 {
                    r.send(cid, &json!({"type": "error", "message": "实战模式不能使用 AI 提示"}));
                    return;
                }
                if r.game.current as i32 != seat || !matches!(r.game.phase, Phase::Play | Phase::Return) {
                    r.send(cid, &json!({"type": "error", "message": "还没轮到你"}));
                    return;
                }
                r.game.clone()
            };
            let net = ctx.net.clone();
            let level = g.level;
            let hs = think(ctx, move || {
                agent::evaluate(&net, &g)
                    .into_iter()
                    .take(5)
                    .map(|(m, q)| {
                        let desc = if m.kind == Kind::Pass { "不出".to_string() } else { m.describe(level) };
                        json!({"kind": m.kind as u8, "key": m.key, "cards": m.cards, "desc": desc, "q": round3(q * 3.0)})
                    })
                    .collect::<Vec<_>>()
            })
            .await
            .unwrap_or_default();
            let r = lock(room);
            r.send(cid, &json!({"type": "hints", "hints": hs, "seq": r.seq}));
        }
        "end" => {
            let mut r = lock(room);
            if !r.can_manage_conn(cid) {
                r.send(cid, &json!({"type": "error", "message": "只有房主或开这张桌的人可以结束牌局"}));
                return;
            }
            let who = if seat >= 0 { r.seats[seat as usize].name.clone() } else { "房主".into() };
            r.end(&format!("{who} 结束了这张牌桌"));
        }
        "pause" | "resume" => {
            let mut r = lock(room);
            // players at the table may pause; watchers only if they manage the table
            if seat < 0 && !r.can_manage_conn(cid) {
                r.send(cid, &json!({"type": "error", "message": "观战时不能暂停牌局"}));
                return;
            }
            let who = if seat >= 0 { r.seats[seat as usize].name.clone() } else { "房主".into() };
            r.set_pause(act == "pause", &who);
            r.broadcast();
        }
        "ready" => {
            if seat >= 0 {
                let mut r = lock(room);
                r.seats[seat as usize].ready = true;
                r.broadcast();
            }
        }
        "play" | "pass" | "return" => {
            if seat < 0 {
                return;
            }
            let r = lock(room);
            if r.game.current as i32 != seat {
                r.send(cid, &json!({"type": "error", "message": "还没轮到你"}));
                return;
            }
            let _ = r.inbox[seat as usize].send((msg, cid));
        }
        _ => {}
    }
}
