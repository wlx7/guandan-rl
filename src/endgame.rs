//! Exact endgame solver for one fully known deal (every hand visible).
//!
//! Used by perfect-information Monte-Carlo (PIMC) search: the hidden hands are sampled
//! several times from the calibrated belief posterior and every sampled world is solved
//! exactly to the end of the round. Position values match `Game::round_reward` (+3/+2/+1
//! when team 0 wins the round, negative when it loses; passing A as banker counts as 3),
//! so minimax values are small integers and tight placement bounds (`nfin`), incremental
//! suit-canonical Zobrist hashing, a generation-tagged flat transposition table, and
//! zero-allocation stack move generation cut the tree by orders of magnitude.

use crate::cards::*;
use crate::game::*;
use crate::moves::*;

/// Stack-allocated move (every legal Guandan move has at most 10 cards).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct EMove {
    kind: Kind,
    key: u8,
    len: u8,
    cards: [u8; 10],
}

impl EMove {
    #[inline]
    fn pass() -> Self {
        EMove { kind: Kind::Pass, key: 0, len: 0, cards: [0; 10] }
    }

    #[inline]
    fn single(key: u8, c: u8) -> Self {
        let mut cards = [0u8; 10];
        cards[0] = c;
        EMove { kind: Kind::Single, key, len: 1, cards }
    }

    #[inline]
    fn pair(key: u8, c0: u8, c1: u8) -> Self {
        let mut cards = [0u8; 10];
        if c0 <= c1 {
            cards[0] = c0;
            cards[1] = c1;
        } else {
            cards[0] = c1;
            cards[1] = c0;
        }
        EMove { kind: Kind::Pair, key, len: 2, cards }
    }

    #[inline]
    fn from_slice(kind: Kind, key: u8, slice: &[u8]) -> Self {
        let mut cards = [0u8; 10];
        let n = slice.len();
        cards[..n].copy_from_slice(slice);
        cards[..n].sort_unstable();
        EMove { kind, key, len: n as u8, cards }
    }

    #[inline]
    fn from_move(m: &Move) -> Self {
        Self::from_slice(m.kind, m.key, &m.cards)
    }

    #[inline]
    fn is_pass(&self) -> bool {
        self.kind == Kind::Pass
    }

    #[inline]
    fn is_bomb(&self) -> bool {
        matches!(self.kind, Kind::Bomb | Kind::StraightFlush | Kind::JokerBomb)
    }

    #[inline]
    fn bomb_class(&self) -> u8 {
        match self.kind {
            Kind::Bomb => {
                if self.len <= 5 {
                    self.len - 4
                } else {
                    self.len - 3
                }
            }
            Kind::StraightFlush => 2,
            Kind::JokerBomb => 8,
            _ => 0,
        }
    }

    #[inline]
    fn power(&self) -> u16 {
        self.bomb_class() as u16 * 32 + self.key as u16
    }

    #[inline]
    fn beats_last(&self, prev: (Kind, u8, u8)) -> bool {
        if self.is_pass() || self.kind == Kind::Tribute {
            return false;
        }
        let (pk, pkey, plen) = prev;
        if pk == Kind::Pass {
            return true;
        }
        let p_bomb = matches!(pk, Kind::Bomb | Kind::StraightFlush | Kind::JokerBomb);
        if self.is_bomb() {
            if p_bomb {
                let p_class = match pk {
                    Kind::Bomb => if plen <= 5 { plen - 4 } else { plen - 3 },
                    Kind::StraightFlush => 2,
                    Kind::JokerBomb => 8,
                    _ => 0,
                };
                self.power() > (p_class as u16 * 32 + pkey as u16)
            } else {
                true
            }
        } else {
            !p_bomb && self.kind == pk && self.key > pkey && self.len == plen
        }
    }
}

impl PartialOrd for EMove {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for EMove {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.kind, self.key, &self.cards[..self.len as usize])
            .cmp(&(other.kind, other.key, &other.cards[..other.len as usize]))
    }
}

// ----------------------------------------------------------------------------- Zobrist table

struct Zobrist {
    card: [[[u64; 3]; 54]; 4],
    nonsf_rank: [[[u64; 9]; 13]; 4],
    turn: [u64; 4],
    last_player: [u64; 4],
    passes: [u64; 4],
    fin: [[u64; 4]; 4],
    nfin: [u64; 4],
    no_last: u64,
    last_kind: [u64; 12],
    last_key: [u64; 16],
    last_len: [u64; 11],
}

const fn make_zobrist() -> Zobrist {
    let mut z = 0x9E37_79B9_7F4A_7C15u64;
    macro_rules! next {
        () => {{
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut v = z;
            v = (v ^ (v >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            v = (v ^ (v >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            v ^ (v >> 31)
        }};
    }
    let mut card = [[[0u64; 3]; 54]; 4];
    let mut p = 0;
    while p < 4 {
        let mut t = 0;
        while t < 54 {
            card[p][t][0] = 0;
            card[p][t][1] = next!();
            card[p][t][2] = next!();
            t += 1;
        }
        p += 1;
    }
    let mut nonsf_rank = [[[0u64; 9]; 13]; 4];
    let mut p = 0;
    while p < 4 {
        let mut r = 0;
        while r < 13 {
            nonsf_rank[p][r][0] = 0;
            let mut c = 1;
            while c < 9 {
                nonsf_rank[p][r][c] = next!();
                c += 1;
            }
            r += 1;
        }
        p += 1;
    }
    let mut turn = [0u64; 4];
    let mut last_player = [0u64; 4];
    let mut passes = [0u64; 4];
    let mut fin = [[0u64; 4]; 4];
    let mut nfin = [0u64; 4];
    let mut i = 0;
    while i < 4 {
        turn[i] = next!();
        last_player[i] = next!();
        passes[i] = next!();
        nfin[i] = next!();
        let mut j = 0;
        while j < 4 {
            fin[i][j] = next!();
            j += 1;
        }
        i += 1;
    }
    let no_last = next!();
    let mut last_kind = [0u64; 12];
    let mut i = 0;
    while i < 12 {
        last_kind[i] = next!();
        i += 1;
    }
    let mut last_key = [0u64; 16];
    let mut i = 0;
    while i < 16 {
        last_key[i] = next!();
        i += 1;
    }
    let mut last_len = [0u64; 11];
    let mut i = 0;
    while i < 11 {
        last_len[i] = next!();
        i += 1;
    }
    Zobrist { card, nonsf_rank, turn, last_player, passes, fin, nfin, no_last, last_kind, last_key, last_len }
}

static ZOB: Zobrist = make_zobrist();

// ----------------------------------------------------------------------------- zero-alloc move gen

struct EInfo {
    level: u8,
    cnt: [u8; 13],
    sc: [[u8; 4]; 13],
    w: u8,
    sj: u8,
    bj: u8,
    sfm: [[bool; 4]; 13],
    has_sf: bool,
}

impl EInfo {
    #[inline]
    fn new(hand: &Hand, hand_len: u8, level: u8) -> EInfo {
        let wt = wild_type(level);
        let mut cnt = [0u8; 13];
        let mut sc = [[0u8; 4]; 13];
        let mut w = 0u8;
        for t in 0..52u8 {
            let c = hand[t as usize];
            if c == 0 {
                continue;
            }
            if t == wt {
                w += c;
            } else {
                let r = (t >> 2) as usize;
                sc[r][(t & 3) as usize] += c;
                cnt[r] += c;
            }
        }
        let mut sfm = [[false; 4]; 13];
        let mut has_sf = false;
        if hand_len >= 5 {
            let mut suit_tot = [0u8; 4];
            for r in 0..13 {
                for s in 0..4 {
                    suit_tot[s] += sc[r][s];
                }
            }
            for s in 0..4 {
                if suit_tot[s] + w < 5 {
                    continue;
                }
                for p in 0..=9u8 {
                    let mut missing = 0u8;
                    for i in 0..5 {
                        if sc[seq_rank(p + i) as usize][s] == 0 {
                            missing += 1;
                        }
                    }
                    if missing <= w {
                        has_sf = true;
                        for i in 0..5 {
                            let r = seq_rank(p + i) as usize;
                            if sc[r][s] > 0 {
                                sfm[r][s] = true;
                            }
                        }
                    }
                }
            }
        }
        EInfo { level, cnt, sc, w, sj: hand[52], bj: hand[53], sfm, has_sf }
    }
}

struct EPicker<'a> {
    info: &'a EInfo,
    used: [[u8; 4]; 13],
}

impl<'a> EPicker<'a> {
    #[inline]
    fn new(info: &'a EInfo) -> Self {
        EPicker { info, used: [[0; 4]; 13] }
    }

    #[inline]
    fn take(&mut self, r: u8, k: u8, out: &mut [u8; 10], len: &mut usize) {
        let ru = r as usize;
        for _ in 0..k {
            let mut best_score = i32::MAX;
            let mut best_s = 0usize;
            for s in 0..4 {
                let avail = self.info.sc[ru][s] as i32 - self.used[ru][s] as i32;
                if avail <= 0 {
                    continue;
                }
                let score = (self.info.sfm[ru][s] as i32) * 10 - avail;
                if score < best_score {
                    best_score = score;
                    best_s = s;
                }
            }
            self.used[ru][best_s] += 1;
            out[*len] = ctype(r, best_s as u8);
            *len += 1;
        }
    }
}

#[inline]
fn push_wilds(level: u8, k: u8, out: &mut [u8; 10], len: &mut usize) {
    let wt = wild_type(level);
    for _ in 0..k {
        out[*len] = wt;
        *len += 1;
    }
}

#[inline]
fn finish_emove(kind: Kind, key: u8, mut cards: [u8; 10], len: usize) -> EMove {
    cards[..len].sort_unstable();
    EMove { kind, key, len: len as u8, cards }
}

#[inline]
fn emove_same_rank(info: &EInfo, r: u8, n: u8, kind: Kind) -> Option<EMove> {
    let c = info.cnt[r as usize];
    let nat = c.min(n);
    let need = n - nat;
    if need > info.w || (nat == 0 && r != info.level) {
        return None;
    }
    let mut cards = [0u8; 10];
    let mut len = 0usize;
    EPicker::new(info).take(r, nat, &mut cards, &mut len);
    push_wilds(info.level, need, &mut cards, &mut len);
    Some(finish_emove(kind, strength(r, info.level), cards, len))
}

fn egen_singles(info: &EInfo, min_key: Option<u8>, out: &mut Vec<EMove>) {
    for r in 0..13u8 {
        let k = strength(r, info.level);
        if min_key.map_or(true, |mk| k > mk) {
            if let Some(m) = emove_same_rank(info, r, 1, Kind::Single) {
                out.push(m);
            }
        }
    }
    if info.sj > 0 && min_key.map_or(true, |mk| 14 > mk) {
        out.push(EMove::single(14, SJ));
    }
    if info.bj > 0 && min_key.map_or(true, |mk| 15 > mk) {
        out.push(EMove::single(15, BJ));
    }
}

fn egen_pairs(info: &EInfo, min_key: Option<u8>, out: &mut Vec<EMove>) {
    for r in 0..13u8 {
        let k = strength(r, info.level);
        if min_key.map_or(true, |mk| k > mk) {
            if let Some(m) = emove_same_rank(info, r, 2, Kind::Pair) {
                out.push(m);
            }
        }
    }
    if info.sj >= 2 && min_key.map_or(true, |mk| 14 > mk) {
        out.push(EMove::pair(14, SJ, SJ));
    }
    if info.bj >= 2 && min_key.map_or(true, |mk| 15 > mk) {
        out.push(EMove::pair(15, BJ, BJ));
    }
}

fn egen_triples(info: &EInfo, min_key: Option<u8>, out: &mut Vec<EMove>) {
    for r in 0..13u8 {
        let k = strength(r, info.level);
        if min_key.map_or(true, |mk| k > mk) {
            if let Some(m) = emove_same_rank(info, r, 3, Kind::Triple) {
                out.push(m);
            }
        }
    }
}

fn egen_straight_flushes(info: &EInfo, prev: Option<(Kind, u8, u8)>, out: &mut Vec<EMove>) {
    if !info.has_sf {
        return;
    }
    for s in 0..4u8 {
        for p in 0..=9u8 {
            let mut need = 0u8;
            for i in 0..5 {
                if info.sc[seq_rank(p + i) as usize][s as usize] == 0 {
                    need += 1;
                }
            }
            if need > info.w {
                continue;
            }
            let mut cards = [0u8; 10];
            let mut len = 0usize;
            for i in 0..5 {
                let r = seq_rank(p + i);
                if info.sc[r as usize][s as usize] > 0 {
                    cards[len] = ctype(r, s);
                    len += 1;
                }
            }
            push_wilds(info.level, need, &mut cards, &mut len);
            let m = finish_emove(Kind::StraightFlush, p, cards, len);
            if prev.map_or(true, |pm| m.beats_last(pm)) {
                out.push(m);
            }
        }
    }
}

fn egen_bombs(info: &EInfo, hand_len: u8, prev: Option<(Kind, u8, u8)>, out: &mut Vec<EMove>) {
    if hand_len < 4 {
        return;
    }
    for r in 0..13u8 {
        let c = info.cnt[r as usize];
        if c == 0 {
            continue;
        }
        let total = (c + info.w).min(10);
        let key = strength(r, info.level);
        for size in 4..=total {
            let nat = c.min(size);
            let need = size - nat;
            let mut cards = [0u8; 10];
            let mut len = 0usize;
            EPicker::new(info).take(r, nat, &mut cards, &mut len);
            push_wilds(info.level, need, &mut cards, &mut len);
            let m = finish_emove(Kind::Bomb, key, cards, len);
            if prev.map_or(true, |pm| m.beats_last(pm)) {
                out.push(m);
            }
        }
    }
    if hand_len >= 5 {
        egen_straight_flushes(info, prev, out);
    }
    if info.sj == 2 && info.bj == 2 {
        let m = EMove { kind: Kind::JokerBomb, key: 0, len: 4, cards: [SJ, SJ, BJ, BJ, 0, 0, 0, 0, 0, 0] };
        if prev.map_or(true, |pm| m.beats_last(pm)) {
            out.push(m);
        }
    }
}

fn egen_full_houses(info: &EInfo, min_key: Option<u8>, out: &mut Vec<EMove>) {
    let mut triples = [(0u8, 0u8, 0u8); 13];
    let mut nt = 0usize;
    for r in 0..13u8 {
        let c = info.cnt[r as usize];
        if c == 0 {
            continue;
        }
        let k = strength(r, info.level);
        if min_key.map_or(false, |mk| k <= mk) {
            continue;
        }
        let nat = c.min(3);
        let need = 3 - nat;
        if need <= info.w {
            triples[nt] = (r, nat, need);
            nt += 1;
        }
    }
    if nt == 0 {
        return;
    }
    let mut pairs = [(0u8, 0u8, 0u8); 16];
    let mut np = 0usize;
    for r in 0..13u8 {
        let c = info.cnt[r as usize];
        if c >= 2 {
            pairs[np] = (r, 2, 0);
            np += 1;
        } else if c == 1 && info.w >= 1 {
            pairs[np] = (r, 1, 1);
            np += 1;
        }
    }
    if info.w >= 2 {
        pairs[np] = (255, 0, 2);
        np += 1;
    }
    if info.sj >= 2 {
        pairs[np] = (13, 2, 0);
        np += 1;
    }
    if info.bj >= 2 {
        pairs[np] = (14, 2, 0);
        np += 1;
    }
    for &(tr, tn, tw) in &triples[..nt] {
        let key = strength(tr, info.level);
        for &(pr, pn, pw) in &pairs[..np] {
            if pr == tr || tw + pw > info.w {
                continue;
            }
            let mut cards = [0u8; 10];
            let mut len = 0usize;
            let mut pk = EPicker::new(info);
            pk.take(tr, tn, &mut cards, &mut len);
            match pr {
                13 => {
                    cards[len] = SJ;
                    cards[len + 1] = SJ;
                    len += 2;
                }
                14 => {
                    cards[len] = BJ;
                    cards[len + 1] = BJ;
                    len += 2;
                }
                255 => {}
                _ => pk.take(pr, pn, &mut cards, &mut len),
            }
            push_wilds(info.level, tw + pw, &mut cards, &mut len);
            out.push(finish_emove(Kind::FullHouse, key, cards, len));
        }
    }
}

fn egen_sequences(info: &EInfo, kind: Kind, width: u8, mult: u8, min_key: Option<u8>, out: &mut Vec<EMove>) {
    let max_start = 14 - width;
    let start_p = min_key.map_or(0, |mk| mk.saturating_add(1));
    for p in start_p..=max_start {
        let mut need = 0u8;
        for i in 0..width {
            let c = info.cnt[seq_rank(p + i) as usize];
            need += mult.saturating_sub(c);
        }
        if need > info.w {
            continue;
        }
        let mut cards = [0u8; 10];
        let mut len = 0usize;
        let mut pk = EPicker::new(info);
        for i in 0..width {
            let r = seq_rank(p + i);
            let nat = info.cnt[r as usize].min(mult);
            pk.take(r, nat, &mut cards, &mut len);
        }
        push_wilds(info.level, need, &mut cards, &mut len);
        out.push(finish_emove(kind, p, cards, len));
    }
}

/// Generate all legal moves into `out` (cleared first), matching `crate::moves::legal_moves`.
fn gen_emoves(hand: &Hand, hand_len: u8, level: u8, prev: Option<(Kind, u8, u8)>, out: &mut Vec<EMove>) {
    out.clear();
    if hand_len == 0 {
        return;
    }
    // Ultra-fast 1-card hand shortcut (the most common leaf state in endgames)
    if hand_len == 1 {
        let mut c = 0u8;
        for t in 0..54u8 {
            if hand[t as usize] > 0 {
                c = t;
                break;
            }
        }
        let key = if c == SJ {
            14
        } else if c == BJ {
            15
        } else {
            strength(c >> 2, level)
        };
        match prev {
            None => {
                out.push(EMove::single(key, c));
                return;
            }
            Some((pk, pkey, _)) => {
                out.push(EMove::pass());
                if pk == Kind::Single && key > pkey {
                    out.push(EMove::single(key, c));
                }
                return;
            }
        }
    }
    // Fast forced-pass check when hand has < 4 cards (cannot hold any bomb)
    if let Some((pk, pkey, plen)) = prev {
        if pk == Kind::JokerBomb {
            out.push(EMove::pass());
            return;
        }
        let p_bomb = matches!(pk, Kind::Bomb | Kind::StraightFlush | Kind::JokerBomb);
        if hand_len < 4 && (p_bomb || hand_len < plen || (pk == Kind::Single && pkey >= 15)) {
            out.push(EMove::pass());
            return;
        }
    }

    let info = EInfo::new(hand, hand_len, level);
    match prev {
        None => {
            egen_singles(&info, None, out);
            egen_pairs(&info, None, out);
            if hand_len >= 3 {
                egen_triples(&info, None, out);
            }
            if hand_len >= 5 {
                egen_full_houses(&info, None, out);
                egen_sequences(&info, Kind::Straight, 5, 1, None, out);
            }
            if hand_len >= 6 {
                egen_sequences(&info, Kind::Tube, 3, 2, None, out);
                egen_sequences(&info, Kind::Plate, 2, 3, None, out);
            }
            if hand_len >= 4 {
                egen_bombs(&info, hand_len, None, out);
            }
            out.sort_unstable();
            out.dedup();
        }
        Some(pm) => {
            out.push(EMove::pass());
            let (pk, pkey, _) = pm;
            let p_bomb = matches!(pk, Kind::Bomb | Kind::StraightFlush | Kind::JokerBomb);
            if !p_bomb {
                match pk {
                    Kind::Single => egen_singles(&info, Some(pkey), out),
                    Kind::Pair if hand_len >= 2 => egen_pairs(&info, Some(pkey), out),
                    Kind::Triple if hand_len >= 3 => egen_triples(&info, Some(pkey), out),
                    Kind::FullHouse if hand_len >= 5 => egen_full_houses(&info, Some(pkey), out),
                    Kind::Straight if hand_len >= 5 => egen_sequences(&info, Kind::Straight, 5, 1, Some(pkey), out),
                    Kind::Tube if hand_len >= 6 => egen_sequences(&info, Kind::Tube, 3, 2, Some(pkey), out),
                    Kind::Plate if hand_len >= 6 => egen_sequences(&info, Kind::Plate, 2, 3, Some(pkey), out),
                    _ => {}
                }
            }
            if hand_len >= 4 {
                egen_bombs(&info, hand_len, Some(pm), out);
            }
            out[1..].sort_unstable();
            // Dedup moves after Pass at index 0
            if out.len() > 2 {
                let mut w = 1usize;
                for r in 2..out.len() {
                    if out[r] != out[w] {
                        w += 1;
                        out[w] = out[r];
                    }
                }
                out.truncate(w + 1);
            }
        }
    }
}

// ----------------------------------------------------------------------------- EState

#[derive(Clone, Copy)]
pub struct EState {
    pub hands: [Hand; 4],
    pub counts: [u8; 4],
    pub nonsf_cnt: [[u8; 13]; 4],
    pub level: u8,
    pub current: u8,
    pub last: Option<(Kind, u8, u8)>,
    pub last_player: u8,
    pub passes: u8,
    pub finished: [u8; 4],
    pub nfin: u8,
    pub banker_at_a: i8,
    pub hash: u64,
}

impl EState {
    pub fn from_game(g: &Game) -> EState {
        let sf = Self::compute_sf_mask(g);
        Self::from_game_with_sf(g, &sf)
    }

    fn compute_sf_mask(g: &Game) -> [[bool; 54]; 4] {
        let mut mask = [[false; 54]; 4];
        let wt = wild_type(g.level);
        for p in 0..4 {
            let hc = hand_count(&g.hands[p]) as u8;
            let info = EInfo::new(&g.hands[p], hc, g.level);
            for t in 0..54usize {
                if t >= 52 || t as u8 == wt {
                    mask[p][t] = true;
                } else {
                    mask[p][t] = info.sfm[t >> 2][t & 3];
                }
            }
        }
        mask
    }

    fn from_game_with_sf(g: &Game, sf: &[[bool; 54]; 4]) -> EState {
        let mut finished = [0u8; 4];
        for (i, &f) in g.finished.iter().enumerate().take(4) {
            finished[i] = f;
        }
        let mut counts = [0u8; 4];
        let mut nonsf_cnt = [[0u8; 13]; 4];
        for p in 0..4 {
            counts[p] = hand_count(&g.hands[p]) as u8;
            for t in 0..52usize {
                if !sf[p][t] {
                    nonsf_cnt[p][t >> 2] += g.hands[p][t];
                }
            }
        }
        let last = g.last_move.as_ref().map(|m| (m.kind, m.key, m.cards.len() as u8));
        let banker_at_a = if g.rules.tribute && g.banker >= 0 && g.level == ACE && g.team_level[g.banker as usize] == ACE {
            g.banker
        } else {
            -1
        };
        let mut s = EState {
            hands: g.hands,
            counts,
            nonsf_cnt,
            level: g.level,
            current: g.current,
            last,
            last_player: g.last_player,
            passes: g.passes,
            finished,
            nfin: g.finished.len() as u8,
            banker_at_a,
            hash: 0,
        };
        s.hash = s.compute_hash(sf);
        s
    }

    fn compute_hash(&self, sf: &[[bool; 54]; 4]) -> u64 {
        let mut h = ZOB.turn[self.current as usize]
            ^ ZOB.passes[(self.passes as usize).min(3)]
            ^ ZOB.nfin[(self.nfin as usize).min(3)];
        for i in 0..self.nfin as usize {
            h ^= ZOB.fin[i][self.finished[i] as usize];
        }
        match self.last {
            None => h ^= ZOB.no_last,
            Some((k, key, len)) => {
                h ^= ZOB.last_player[self.last_player as usize]
                    ^ ZOB.last_kind[k as usize]
                    ^ ZOB.last_key[(key as usize).min(15)]
                    ^ ZOB.last_len[(len as usize).min(10)];
            }
        }
        for p in 0..4 {
            for t in 0..54 {
                if sf[p][t] {
                    h ^= ZOB.card[p][t][(self.hands[p][t] as usize).min(2)];
                }
            }
            for r in 0..13 {
                h ^= ZOB.nonsf_rank[p][r][(self.nonsf_cnt[p][r] as usize).min(8)];
            }
        }
        h
    }

    #[inline]
    fn is_finished(&self, p: u8) -> bool {
        let nf = self.nfin as usize;
        (nf > 0 && self.finished[0] == p)
            || (nf > 1 && self.finished[1] == p)
            || (nf > 2 && self.finished[2] == p)
    }

    #[inline]
    fn next_active(&self, p: u8) -> u8 {
        let mut q = (p + 1) & 3;
        for _ in 0..4 {
            if !self.is_finished(q) {
                return q;
            }
            q = (q + 1) & 3;
        }
        q
    }

    #[inline]
    fn set_last(&mut self, next_last: Option<(Kind, u8, u8)>, next_lp: u8) {
        match self.last {
            None => self.hash ^= ZOB.no_last,
            Some((k, key, len)) => {
                self.hash ^= ZOB.last_player[self.last_player as usize]
                    ^ ZOB.last_kind[k as usize]
                    ^ ZOB.last_key[(key as usize).min(15)]
                    ^ ZOB.last_len[(len as usize).min(10)];
            }
        }
        self.last = next_last;
        self.last_player = next_lp;
        match self.last {
            None => self.hash ^= ZOB.no_last,
            Some((k, key, len)) => {
                self.hash ^= ZOB.last_player[self.last_player as usize]
                    ^ ZOB.last_kind[k as usize]
                    ^ ZOB.last_key[(key as usize).min(15)]
                    ^ ZOB.last_len[(len as usize).min(10)];
            }
        }
    }

    #[inline]
    fn set_passes(&mut self, next_p: u8) {
        self.hash ^= ZOB.passes[(self.passes as usize).min(3)] ^ ZOB.passes[(next_p as usize).min(3)];
        self.passes = next_p;
    }

    #[inline]
    fn set_current(&mut self, next_c: u8) {
        self.hash ^= ZOB.turn[self.current as usize] ^ ZOB.turn[next_c as usize];
        self.current = next_c;
    }

    #[inline]
    fn advance(&mut self, p: u8) {
        if self.last.is_some() {
            let active = 4 - self.nfin as usize;
            let lp_active = !self.is_finished(self.last_player);
            let needed = active - lp_active as usize;
            if self.passes as usize >= needed {
                let leader = if lp_active {
                    self.last_player
                } else {
                    let pt = partner(self.last_player);
                    if self.is_finished(pt) {
                        self.next_active(self.last_player)
                    } else {
                        pt
                    }
                };
                self.set_last(None, self.last_player);
                self.set_passes(0);
                self.set_current(leader);
                return;
            }
        }
        self.set_current(self.next_active(p));
    }

    /// Team 0's exact round reward, matching `Game::round_reward(0)`.
    #[inline]
    fn result(&self) -> i8 {
        let mut order = [0u8; 4];
        let mut k = 0;
        for &f in &self.finished[..self.nfin as usize] {
            order[k] = f;
            k += 1;
        }
        for s in 0..4u8 {
            if !self.is_finished(s) {
                order[k] = s;
                k += 1;
            }
        }
        let first = order[0];
        let w = team(first);
        let pp = order.iter().position(|&x| x == partner(first)).unwrap();
        let mut gain: i8 = match pp {
            1 => 3,
            2 => 2,
            _ => 1,
        };
        if self.banker_at_a == w as i8 && gain >= 2 {
            gain = 3;
        }
        if w == 0 { gain } else { -gain }
    }

    /// Exact game-theoretic (min, max) bounds on Team 0's value from already-finished seats.
    #[inline]
    fn tight_bounds(&self) -> (i8, i8) {
        match self.nfin {
            0 => (-3, 3),
            1 => {
                if team(self.finished[0]) == 0 {
                    (1, 3)
                } else {
                    (-3, -1)
                }
            }
            _ => {
                // nfin >= 2 and round still active -> finished[0] and finished[1] are on opposite teams
                if team(self.finished[0]) == 0 {
                    let hi = if self.banker_at_a == 0 { 3 } else { 2 };
                    (1, hi)
                } else {
                    let lo = if self.banker_at_a == 1 { -3 } else { -2 };
                    (lo, -1)
                }
            }
        }
    }

    #[inline]
    fn play_pass(&mut self) {
        let p = self.current;
        self.set_passes(self.passes + 1);
        self.advance(p);
    }

    #[inline]
    fn play_emove(&mut self, mv: &EMove, sf: &[[bool; 54]; 4]) -> Option<i8> {
        if mv.is_pass() {
            self.play_pass();
            return None;
        }
        let p = self.current as usize;
        for i in 0..mv.len as usize {
            let c = mv.cards[i] as usize;
            let old_c = self.hands[p][c];
            let new_c = old_c - 1;
            self.hands[p][c] = new_c;
            if sf[p][c] {
                self.hash ^= ZOB.card[p][c][old_c as usize] ^ ZOB.card[p][c][new_c as usize];
            } else {
                let r = c >> 2;
                let old_r = self.nonsf_cnt[p][r];
                let new_r = old_r - 1;
                self.nonsf_cnt[p][r] = new_r;
                self.hash ^= ZOB.nonsf_rank[p][r][old_r as usize] ^ ZOB.nonsf_rank[p][r][new_r as usize];
            }
        }
        self.counts[p] -= mv.len;
        let pu8 = p as u8;
        self.set_last(Some((mv.kind, mv.key, mv.len)), pu8);
        self.set_passes(0);
        if self.counts[p] == 0 {
            let idx = self.nfin as usize;
            self.finished[idx] = pu8;
            self.hash ^= ZOB.nfin[idx] ^ ZOB.nfin[(idx + 1).min(3)] ^ ZOB.fin[idx][p];
            self.nfin += 1;
            let nf = self.nfin;
            if nf == 3 || (nf == 2 && team(self.finished[0]) == team(self.finished[1])) {
                return Some(self.result());
            }
        }
        self.advance(pu8);
        None
    }

    /// Public replay helper (used by unit tests).
    pub fn play(&mut self, mv: &Move) -> Option<i8> {
        let dummy_sf = [[true; 54]; 4];
        self.play_emove(&EMove::from_move(mv), &dummy_sf)
    }

    pub fn cards_left(&self) -> u32 {
        self.counts.iter().map(|&c| c as u32).sum()
    }
}

// ----------------------------------------------------------------------------- Flat TT & Solver

const TT_BITS: usize = 17; // 131,072 entries = 2 MB per thread (fits inside L3 cache!)
const TT_SIZE: usize = 1 << TT_BITS;
const TT_MASK: usize = TT_SIZE - 1;

#[derive(Clone, Copy)]
struct TTEntry {
    key: u64,
    gen: u16,
    lo: i8,
    hi: i8,
}

pub struct Solver {
    tt: Vec<TTEntry>,
    gen: u16,
    sf: [[bool; 54]; 4],
    bufs: Vec<Vec<EMove>>,
    pub nodes: u64,
    pub max_nodes: u64,
    pub aborted: bool,
}

impl Solver {
    pub fn new(max_nodes: u64) -> Solver {
        let mut bufs = Vec::with_capacity(96);
        for _ in 0..96 {
            bufs.push(Vec::with_capacity(64));
        }
        Solver {
            tt: vec![TTEntry { key: 0, gen: 0, lo: -3, hi: 3 }; TT_SIZE],
            gen: 1,
            sf: [[true; 54]; 4],
            bufs,
            nodes: 0,
            max_nodes,
            aborted: false,
        }
    }

    fn reset_for_world(&mut self, sf: [[bool; 54]; 4], max_nodes: u64) {
        self.gen = self.gen.wrapping_add(1);
        if self.gen == 0 {
            for e in &mut self.tt {
                e.gen = 0;
            }
            self.gen = 1;
        }
        self.sf = sf;
        self.nodes = 0;
        self.max_nodes = max_nodes;
        self.aborted = false;
    }

    #[inline]
    fn order_moves(s: &EState, mv: &mut [EMove]) {
        let p = s.current;
        let hand_len = s.counts[p as usize];
        let following_partner = s.last.is_some() && team(s.last_player) == team(p);
        let leading = s.last.is_none();
        mv.sort_unstable_by_key(|m| {
            let prio: i32 = if !m.is_pass() && m.len == hand_len {
                1000 // empties hand immediately -> triggers immediate placement / round-end cutoff
            } else if following_partner {
                if m.is_pass() {
                    600 // pass partner's winning trick first
                } else if !m.is_bomb() {
                    200 + (m.len as i32) * 8 - (m.key as i32)
                } else {
                    10
                }
            } else if leading {
                if !m.is_bomb() {
                    300 + (m.len as i32) * 16 + (m.key as i32)
                } else {
                    150 + (m.len as i32) * 8
                }
            } else {
                // following opponent
                if m.is_pass() {
                    50
                } else if !m.is_bomb() {
                    300 + (m.len as i32) * 8 - (m.key as i32)
                } else {
                    180 - (m.bomb_class() as i32) * 8 - (m.key as i32)
                }
            };
            std::cmp::Reverse(prio)
        });
    }

    /// Team 0's exact minimax value of `s` within `[alpha, beta]`.
    pub fn value(&mut self, s: &EState, alpha: i8, beta: i8) -> i8 {
        self.value_depth(*s, alpha, beta, 0)
    }

    fn value_depth(&mut self, mut s: EState, mut alpha: i8, mut beta: i8, depth: usize) -> i8 {
        // Fast-forward forced passes without consuming TT slots
        while let Some((pk, pkey, plen)) = s.last {
            let hl = s.counts[s.current as usize];
            let p_bomb = matches!(pk, Kind::Bomb | Kind::StraightFlush | Kind::JokerBomb);
            if pk == Kind::JokerBomb || (hl < 4 && (p_bomb || hl < plen || (pk == Kind::Single && pkey >= 15))) {
                s.play_pass();
            } else {
                break;
            }
        }

        self.nodes += 1;
        if self.nodes > self.max_nodes {
            self.aborted = true;
            return 0;
        }

        let (min_b, max_b) = s.tight_bounds();
        let mut lo = min_b;
        let mut hi = max_b;
        let idx = (s.hash as usize) & TT_MASK;
        let entry = self.tt[idx];
        if entry.gen == self.gen && entry.key == s.hash {
            lo = lo.max(entry.lo);
            hi = hi.min(entry.hi);
        }
        if lo >= beta {
            return lo;
        }
        if hi <= alpha {
            return hi;
        }
        if lo == hi {
            return lo;
        }
        alpha = alpha.max(lo);
        beta = beta.min(hi);

        if depth >= self.bufs.len() {
            self.bufs.push(Vec::with_capacity(64));
        }
        let p = s.current as usize;
        gen_emoves(&s.hands[p], s.counts[p], s.level, s.last, &mut self.bufs[depth]);
        let n_mv = self.bufs[depth].len();
        if n_mv == 1 {
            let mv = self.bufs[depth][0];
            let mut c = s;
            return match c.play_emove(&mv, &self.sf) {
                Some(r) => r,
                None => self.value_depth(c, alpha, beta, depth + 1),
            };
        }
        Self::order_moves(&s, &mut self.bufs[depth]);

        let maximize = team(s.current) == 0;
        let (a0, b0) = (alpha, beta);
        let mut best: i8 = if maximize { -4 } else { 4 };
        for mi in 0..n_mv {
            let mv = self.bufs[depth][mi];
            let mut c = s;
            let v = match c.play_emove(&mv, &self.sf) {
                Some(r) => r,
                None => self.value_depth(c, alpha, beta, depth + 1),
            };
            if self.aborted {
                return 0;
            }
            if maximize {
                if v > best {
                    best = v;
                }
                if best > alpha {
                    alpha = best;
                }
            } else {
                if v < best {
                    best = v;
                }
                if best < beta {
                    beta = best;
                }
            }
            if alpha >= beta {
                break;
            }
        }

        if best <= a0 {
            hi = hi.min(best);
        } else if best >= b0 {
            lo = lo.max(best);
        } else {
            lo = best;
            hi = best;
        }
        self.tt[idx] = TTEntry { key: s.hash, gen: self.gen, lo, hi };
        best
    }

    fn solve_world(&mut self, g: &Game, moves: &[EMove], order: &[usize], max_nodes: u64) -> Option<Vec<f32>> {
        let sf = EState::compute_sf_mask(g);
        let root = EState::from_game_with_sf(g, &sf);
        let me_team = team(root.current);
        let (min_b, max_b) = root.tight_bounds();
        self.reset_for_world(sf, max_nodes);
        let mut out = vec![0f32; moves.len()];
        for &i in order {
            let mut c = root;
            let v = match c.play_emove(&moves[i], &self.sf) {
                Some(r) => r,
                None => self.value_depth(c, min_b, max_b, 0),
            };
            if self.aborted {
                return None;
            }
            out[i] = if me_team == 0 { v as f32 } else { -(v as f32) };
        }
        Some(out)
    }
}

/// Exact level change for the team of the player to move after each of `moves`, with all hands
/// visible and perfect play to the end of the round; None when `max_nodes` was exceeded.
pub fn solve_moves(g: &Game, moves: &[Move], max_nodes: u64) -> Vec<Option<f32>> {
    let emoves: Vec<EMove> = moves.iter().map(EMove::from_move).collect();
    let order: Vec<usize> = (0..emoves.len()).collect();
    let mut solver = Solver::new(max_nodes);
    match solver.solve_world(g, &emoves, &order, max_nodes) {
        Some(vals) => vals.into_iter().map(Some).collect(),
        None => vec![None; moves.len()],
    }
}

/// Expected copies of every card type in the hands of the players 1, 2, 3 seats after the one to
/// move, from the network's belief logits (HIDDEN_DIM = 3 x 54 x 2: "at least one copy", "both
/// copies"), as used by `Game::determinize_weighted`.
pub fn belief_expectation(logits: &[f32]) -> [[f32; 54]; 3] {
    let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
    let mut e = [[0f32; 54]; 3];
    for r in 0..3 {
        for t in 0..54 {
            let p1 = sig(logits[(r * 54 + t) * 2]);
            let p2 = sig(logits[(r * 54 + t) * 2 + 1]).min(p1);
            e[r][t] = p1 + p2;
        }
    }
    e
}

/// Settings of the endgame search used when playing.
#[derive(Clone, Copy, Debug)]
pub struct EndgameCfg {
    /// search once at most this many cards are left in all hands together
    pub cards: u32,
    /// sampled worlds per decision
    pub worlds: usize,
    /// floor weight of the belief-guided world sampling (see Game::determinize_belief)
    pub floor: f32,
    /// node limit per world; a world that needs more is skipped
    pub max_nodes: u64,
    /// a move must beat the network's choice by this many levels on the blended score
    pub margin: f32,
    /// blend weight on the playing network's Q (in levels) to regularize PIMC and break ties
    pub prior: f32,
    /// max root candidate moves evaluated when `q_levels` is given (0 = all legal moves)
    pub max_cands: usize,
    /// max Q deficit (in levels) below `net_choice` for a root move to be considered (0 = off)
    pub q_gap: f32,
    pub threads: usize,
}

impl Default for EndgameCfg {
    fn default() -> Self {
        EndgameCfg {
            cards: 26,
            worlds: 24,
            floor: 0.02,
            max_nodes: 250_000,
            margin: 0.0,
            prior: 0.25,
            max_cands: 12,
            q_gap: 1.2,
            threads: 4,
        }
    }
}

/// Perfect-information Monte-Carlo endgame search with optional full 324 belief logits and
/// playing-network Q regularization (`q_levels`, in level units).
pub fn endgame_choose_ext(
    g: &Game,
    net_choice: usize,
    cfg: &EndgameCfg,
    expect: Option<&[[f32; 54]; 3]>,
    belief_logits: Option<&[f32]>,
    q_levels: Option<&[f32]>,
    seed: u64,
) -> Option<usize> {
    if g.phase != Phase::Play {
        return None;
    }
    let left: u32 = g.hands.iter().map(hand_count).sum();
    if left > cfg.cards {
        return None;
    }
    let moves = g.legal_moves();
    let n_mv = moves.len();
    if n_mv < 2 || net_choice >= n_mv {
        return None;
    }
    let ql = match q_levels {
        Some(v) if v.len() == n_mv => Some(v),
        _ => None,
    };

    // Select root candidates ordered by descending Q (so the best moves warm up the TT first)
    let mut eval_order: Vec<usize> = Vec::with_capacity(n_mv);
    let mut mask = vec![false; n_mv];
    if let Some(q) = ql {
        let mut ord: Vec<usize> = (0..n_mv).collect();
        ord.sort_unstable_by(|&a, &b| q[b].partial_cmp(&q[a]).unwrap_or(std::cmp::Ordering::Equal));
        let top_q = q[net_choice];
        let limit = if cfg.max_cands > 0 { cfg.max_cands.min(n_mv) } else { n_mv };
        eval_order.push(net_choice);
        mask[net_choice] = true;
        for i in ord {
            if eval_order.len() >= limit {
                break;
            }
            if mask[i] {
                continue;
            }
            if cfg.q_gap > 0.0 && top_q - q[i] > cfg.q_gap {
                continue;
            }
            mask[i] = true;
            eval_order.push(i);
        }
        // If no alternative move is within `q_gap` of `net_choice`, skip search immediately!
        if eval_order.len() < 2 {
            return None;
        }
    } else {
        mask.fill(true);
        eval_order.push(net_choice);
        for i in 0..n_mv {
            if i != net_choice {
                eval_order.push(i);
            }
        }
    }

    let emoves: Vec<EMove> = moves.iter().map(EMove::from_move).collect();
    let me = g.current;
    let mut rng = Rng::new(seed);
    let worlds: Vec<Game> = (0..cfg.worlds)
        .map(|_| {
            if let Some(lg) = belief_logits {
                g.determinize_belief(me, lg, cfg.floor, &mut rng)
            } else if let Some(e) = expect {
                g.determinize_weighted(me, e, cfg.floor, &mut rng)
            } else {
                g.determinize(me, &mut rng)
            }
        })
        .collect();

    let threads = cfg.threads.max(1).min(worlds.len());
    let chunk = worlds.len().div_ceil(threads);
    let results: Vec<Option<Vec<f32>>> = std::thread::scope(|s| {
        let handles: Vec<_> = worlds
            .chunks(chunk)
            .map(|ws| {
                let emv = &emoves;
                let ord = &eval_order;
                let max_n = cfg.max_nodes;
                s.spawn(move || {
                    let mut solver = Solver::new(max_n);
                    ws.iter().map(|w| solver.solve_world(w, emv, ord, max_n)).collect::<Vec<_>>()
                })
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().unwrap_or_default()).collect()
    });

    let mut sum = vec![0f64; n_mv];
    let mut n = 0usize;
    for r in &results {
        if let Some(vals) = r {
            for k in 0..n_mv {
                if mask[k] {
                    sum[k] += vals[k] as f64;
                }
            }
            n += 1;
        }
    }
    if n == 0 {
        return None;
    }

    let pr = if ql.is_some() { cfg.prior.clamp(0.0, 0.95) as f64 } else { 0.0 };
    let score_of = |k: usize| -> f64 {
        let pimc = sum[k] / n as f64;
        if let Some(q) = ql {
            (1.0 - pr) * pimc + pr * (q[k] as f64)
        } else {
            pimc
        }
    };

    let mut best = net_choice;
    let mut best_sc = score_of(net_choice);
    for k in 0..n_mv {
        if !mask[k] || k == net_choice {
            continue;
        }
        let sc = score_of(k);
        let better = if (sc - best_sc).abs() > 1e-9 {
            sc > best_sc
        } else if let Some(q) = ql {
            q[k] > q[best] + 1e-7
        } else {
            false
        };
        if better {
            best = k;
            best_sc = sc;
        }
    }

    let base_sc = score_of(net_choice);
    let pimc_gain = (sum[best] - sum[net_choice]) / n as f64;
    if best != net_choice && best_sc - base_sc > cfg.margin as f64 + 1e-9 && pimc_gain > 1e-9 {
        Some(best)
    } else {
        None
    }
}

/// Backward-compatible wrapper around `endgame_choose_ext`.
pub fn endgame_choose(
    g: &Game,
    net_choice: usize,
    cfg: &EndgameCfg,
    expect: Option<&[[f32; 54]; 3]>,
    seed: u64,
) -> Option<usize> {
    endgame_choose_ext(g, net_choice, cfg, expect, None, None, seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple unoptimized reference minimax solver to verify 100% exactness of all solver
    /// optimizations (tight bounds, Zobrist suit canonicalization, forced-pass fast-forward,
    /// and stack move generation).
    fn ref_minimax(g: &Game, alpha: i8, beta: i8) -> i8 {
        let mut alpha = alpha;
        let mut beta = beta;
        let me = g.current;
        let maximize = team(me) == 0;
        let mut best = if maximize { -4 } else { 4 };
        for mv in g.legal_moves() {
            let mut c = g.clone();
            c.apply(&mv).unwrap();
            let v = if c.phase != Phase::Play {
                c.round_reward(0) as i8
            } else {
                ref_minimax(&c, alpha, beta)
            };
            if maximize {
                if v > best {
                    best = v;
                }
                if best > alpha {
                    alpha = best;
                }
            } else {
                if v < best {
                    best = v;
                }
                if best < beta {
                    beta = best;
                }
            }
            if alpha >= beta {
                break;
            }
        }
        best
    }

    #[test]
    fn emoves_match_legal_moves_exhaustively() {
        let mut rng = Rng::new(20261118);
        let mut buf = Vec::with_capacity(256);
        for trial in 0..400 {
            let mut deck: Vec<u8> = (0..54u8).flat_map(|t| [t, t]).collect();
            rng.shuffle(&mut deck);
            let level = (trial % 13) as u8;
            let sz = 1 + (trial % 27);
            let hand = hand_from_cards(&deck[..sz]);
            let hl = hand_count(&hand) as u8;

            let want = legal_moves(&hand, level, None);
            gen_emoves(&hand, hl, level, None, &mut buf);
            let want_e: Vec<EMove> = want.iter().map(EMove::from_move).collect();
            assert_eq!(buf, want_e, "leading mismatch at trial {trial} sz={sz}");

            let other = hand_from_cards(&deck[27..27 + 1 + ((trial * 5) % 20)]);
            let other_moves = legal_moves(&other, level, None);
            let prev = &other_moves[(trial * 3) % other_moves.len()];
            let want_f = legal_moves(&hand, level, Some(prev));
            gen_emoves(&hand, hl, level, Some((prev.kind, prev.key, prev.cards.len() as u8)), &mut buf);
            let want_fe: Vec<EMove> = want_f.iter().map(EMove::from_move).collect();
            assert_eq!(buf, want_fe, "following mismatch at trial {trial} prev={prev:?}");
        }
    }

    #[test]
    fn fast_solver_matches_reference_minimax_exactly() {
        let mut rng = Rng::new(77);
        let mut checked = 0;
        for seed in 0..80u64 {
            let mut g = Game::new(seed + 100, Rules::default());
            if seed % 3 == 0 {
                g.set_levels([ACE, 8], 0);
                g.start_round();
            }
            while g.phase == Phase::Return {
                let mv = g.legal_moves();
                g.apply(&mv[0]).unwrap();
            }
            while g.phase == Phase::Play {
                let left: u32 = g.hands.iter().map(hand_count).sum();
                let mv = g.legal_moves();
                if left <= 8 && mv.len() > 1 {
                    let got = solve_moves(&g, &mv, 2_000_000);
                    let me_team = team(g.current);
                    for (m, gv) in mv.iter().zip(got) {
                        let mut c = g.clone();
                        c.apply(m).unwrap();
                        let rv0 = if c.phase != Phase::Play {
                            c.round_reward(0) as i8
                        } else {
                            ref_minimax(&c, -3, 3)
                        };
                        let want = if me_team == 0 { rv0 as f32 } else { -(rv0 as f32) };
                        assert_eq!(gv, Some(want), "seed {seed} left {left} move {m:?}: got {gv:?} vs want {want}");
                    }
                    checked += 1;
                    break;
                }
                let m = mv[(rng.next_u64() % mv.len() as u64) as usize].clone();
                g.apply(&m).unwrap();
            }
        }
        assert!(checked >= 40, "only checked {checked} endgames");
    }

    #[test]
    fn solver_matches_finished_rounds() {
        let mut rng = Rng::new(7);
        for seed in 0..30u64 {
            let mut g = Game::new(seed, Rules::default());
            while g.phase == Phase::Return {
                let mv = g.legal_moves();
                g.apply(&mv[0]).unwrap();
            }
            let mut e = EState::from_game(&g);
            loop {
                let mv = g.legal_moves();
                let m = mv[(rng.next_u64() % mv.len() as u64) as usize].clone();
                g.apply(&m).unwrap();
                let r = e.play(&m);
                if g.phase != Phase::Play {
                    let want = g.round_reward(0);
                    let got = r.expect("solver state did not see the round end") as f32;
                    assert_eq!(got, want, "seed {seed}: {got} vs {want}");
                    break;
                }
                assert!(r.is_none());
                assert_eq!(e.current, g.current, "seed {seed}: current player differs");
            }
        }
    }

    #[test]
    fn endgame_choose_runs_on_small_endgames() {
        let mut rng = Rng::new(11);
        let cfg = EndgameCfg { cards: 20, worlds: 16, ..EndgameCfg::default() };
        let mut searched = 0;
        for seed in 0..20u64 {
            let mut g = Game::new(seed, Rules::default());
            while g.phase == Phase::Play {
                let mv = g.legal_moves();
                let left: u32 = g.hands.iter().map(hand_count).sum();
                if left <= cfg.cards && mv.len() > 1 {
                    if let Some(i) = endgame_choose(&g, 0, &cfg, None, seed) {
                        assert!(i < mv.len());
                    }
                    searched += 1;
                }
                let m = mv[(rng.next_u64() % mv.len() as u64) as usize].clone();
                g.apply(&m).unwrap();
            }
        }
        assert!(searched > 0);
    }
}
