//! Card combinations: classification (all interpretations of a card set, used for human
//! input) and legal-move generation (used by the AI).

use crate::cards::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Kind {
    Pass = 0,
    Single = 1,
    Pair = 2,
    Triple = 3,
    FullHouse = 4,     // 三带二
    Straight = 5,      // 顺子 (5)
    Tube = 6,          // 三连对 (木板)
    Plate = 7,         // 二连三 (钢板)
    Bomb = 8,          // 4..10 同点炸弹
    StraightFlush = 9, // 同花顺
    JokerBomb = 10,    // 四大天王
    Tribute = 11,      // 还贡 (phase action)
}

pub const NUM_KINDS: usize = 12;

impl Kind {
    pub fn from_u8(v: u8) -> Option<Kind> {
        use Kind::*;
        Some(match v {
            0 => Pass,
            1 => Single,
            2 => Pair,
            3 => Triple,
            4 => FullHouse,
            5 => Straight,
            6 => Tube,
            7 => Plate,
            8 => Bomb,
            9 => StraightFlush,
            10 => JokerBomb,
            11 => Tribute,
            _ => return None,
        })
    }
    pub fn name(self) -> &'static str {
        use Kind::*;
        match self {
            Pass => "不出",
            Single => "单张",
            Pair => "对子",
            Triple => "三同张",
            FullHouse => "三带二",
            Straight => "顺子",
            Tube => "三连对",
            Plate => "钢板",
            Bomb => "炸弹",
            StraightFlush => "同花顺",
            JokerBomb => "天王炸",
            Tribute => "还贡",
        }
    }
}

/// A concrete move. `key` orders moves of the same kind:
/// * Single/Pair/Triple/FullHouse/Bomb: `strength()` of the (triple) rank
/// * Straight/Tube/Plate/StraightFlush: start position (0 = A low)
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Move {
    pub kind: Kind,
    pub key: u8,
    pub cards: Vec<u8>,
}

impl Move {
    pub fn pass() -> Move {
        Move { kind: Kind::Pass, key: 0, cards: vec![] }
    }

    pub fn new(kind: Kind, key: u8, mut cards: Vec<u8>) -> Move {
        cards.sort_unstable();
        Move { kind, key, cards }
    }

    #[inline]
    pub fn is_pass(&self) -> bool {
        self.kind == Kind::Pass
    }

    #[inline]
    pub fn is_bomb(&self) -> bool {
        matches!(self.kind, Kind::Bomb | Kind::StraightFlush | Kind::JokerBomb)
    }

    /// Bomb class: 4张=0, 5张=1, 同花顺=2, 6张=3, ..., 10张=7, 天王炸=8.
    pub fn bomb_class(&self) -> u8 {
        match self.kind {
            Kind::Bomb => {
                let n = self.cards.len() as u8;
                if n <= 5 {
                    n - 4
                } else {
                    n - 3
                }
            }
            Kind::StraightFlush => 2,
            Kind::JokerBomb => 8,
            _ => 0,
        }
    }

    #[inline]
    pub fn power(&self) -> u16 {
        self.bomb_class() as u16 * 32 + self.key as u16
    }

    /// Whether `self` may be played on top of `prev`.
    pub fn beats(&self, prev: &Move) -> bool {
        if self.is_pass() || self.kind == Kind::Tribute {
            return false;
        }
        if prev.is_pass() {
            return true;
        }
        if self.is_bomb() {
            if prev.is_bomb() {
                self.power() > prev.power()
            } else {
                true
            }
        } else {
            self.kind == prev.kind && self.key > prev.key && self.cards.len() == prev.cards.len()
        }
    }

    pub fn nwild(&self, level: u8) -> u8 {
        let w = wild_type(level);
        self.cards.iter().filter(|&&c| c == w).count() as u8
    }

    pub fn describe(&self, level: u8) -> String {
        match self.kind {
            Kind::Pass => "不出".into(),
            Kind::Straight | Kind::Tube | Kind::Plate | Kind::StraightFlush => {
                let len = match self.kind {
                    Kind::Tube => 3,
                    Kind::Plate => 2,
                    _ => 5,
                };
                let a = RANK_NAMES[seq_rank(self.key) as usize];
                let b = RANK_NAMES[seq_rank(self.key + len - 1) as usize];
                format!("{} {}-{}", self.kind.name(), a, b)
            }
            Kind::JokerBomb => "天王炸".into(),
            Kind::Tribute => format!("还贡 {}", card_name(self.cards[0])),
            Kind::Bomb => {
                let r = rank_from_strength(self.key, level);
                format!("{}张{}炸弹", self.cards.len(), RANK_NAMES[r as usize])
            }
            _ => {
                let r = rank_from_strength(self.key, level);
                format!("{} {}", self.kind.name(), RANK_NAMES[r as usize])
            }
        }
    }
}

pub fn rank_from_strength(key: u8, level: u8) -> u8 {
    if key >= 14 {
        key - 1
    } else if key == 13 {
        level
    } else {
        key
    }
}

/// Summary of a hand used by the generators.
#[derive(Clone, Debug)]
pub struct Info {
    pub level: u8,
    pub cnt: [u8; 13],      // natural counts per rank (wild cards excluded)
    pub sc: [[u8; 4]; 13],  // natural counts per rank and suit
    pub w: u8,              // number of wild cards
    pub sj: u8,
    pub bj: u8,
    pub sfm: [[bool; 4]; 13], // card belongs to a possible straight flush
}

impl Info {
    pub fn new(hand: &Hand, level: u8) -> Info {
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
                let r = rank_of(t) as usize;
                sc[r][suit_of(t) as usize] += c;
                cnt[r] += c;
            }
        }
        let mut info = Info { level, cnt, sc, w, sj: hand[52], bj: hand[53], sfm: [[false; 4]; 13] };
        for s in 0..4 {
            for p in 0..=9u8 {
                let mut missing = 0;
                for i in 0..5 {
                    if info.sc[seq_rank(p + i) as usize][s] == 0 {
                        missing += 1;
                    }
                }
                if missing <= info.w {
                    for i in 0..5 {
                        let r = seq_rank(p + i) as usize;
                        if info.sc[r][s] > 0 {
                            info.sfm[r][s] = true;
                        }
                    }
                }
            }
        }
        info
    }
}

/// Picks concrete cards of a rank, preferring suits that do not belong to a potential
/// straight flush (keeps straight flushes intact).
struct Picker<'a> {
    info: &'a Info,
    used: [[u8; 4]; 13],
}

impl<'a> Picker<'a> {
    fn new(info: &'a Info) -> Self {
        Picker { info, used: [[0; 4]; 13] }
    }
    fn take(&mut self, r: u8, k: u8, out: &mut Vec<u8>) {
        let ru = r as usize;
        for _ in 0..k {
            let mut best: Option<(i32, usize)> = None;
            for s in 0..4 {
                let avail = self.info.sc[ru][s] as i32 - self.used[ru][s] as i32;
                if avail <= 0 {
                    continue;
                }
                let score = (self.info.sfm[ru][s] as i32) * 10 - avail;
                if best.map_or(true, |(b, _)| score < b) {
                    best = Some((score, s));
                }
            }
            let s = best.expect("picker: no card available").1;
            self.used[ru][s] += 1;
            out.push(ctype(r, s as u8));
        }
    }
}

fn wilds(level: u8, k: u8, out: &mut Vec<u8>) {
    for _ in 0..k {
        out.push(wild_type(level));
    }
}

fn same_rank_move(info: &Info, r: u8, n: u8, kind: Kind) -> Option<Move> {
    let c = info.cnt[r as usize];
    let nat = c.min(n);
    let need = n - nat;
    if need > info.w || (nat == 0 && r != info.level) {
        return None;
    }
    let mut cards = Vec::with_capacity(n as usize);
    Picker::new(info).take(r, nat, &mut cards);
    wilds(info.level, need, &mut cards);
    Some(Move::new(kind, strength(r, info.level), cards))
}

pub fn gen_singles(info: &Info, out: &mut Vec<Move>) {
    for r in 0..13u8 {
        if let Some(m) = same_rank_move(info, r, 1, Kind::Single) {
            out.push(m);
        }
    }
    if info.sj > 0 {
        out.push(Move::new(Kind::Single, 14, vec![SJ]));
    }
    if info.bj > 0 {
        out.push(Move::new(Kind::Single, 15, vec![BJ]));
    }
}

pub fn gen_pairs(info: &Info, out: &mut Vec<Move>) {
    for r in 0..13u8 {
        if let Some(m) = same_rank_move(info, r, 2, Kind::Pair) {
            out.push(m);
        }
    }
    if info.sj >= 2 {
        out.push(Move::new(Kind::Pair, 14, vec![SJ, SJ]));
    }
    if info.bj >= 2 {
        out.push(Move::new(Kind::Pair, 15, vec![BJ, BJ]));
    }
}

pub fn gen_triples(info: &Info, out: &mut Vec<Move>) {
    for r in 0..13u8 {
        if let Some(m) = same_rank_move(info, r, 3, Kind::Triple) {
            out.push(m);
        }
    }
}

pub fn gen_bombs(info: &Info, out: &mut Vec<Move>) {
    for r in 0..13u8 {
        let c = info.cnt[r as usize];
        if c == 0 {
            continue;
        }
        let total = (c + info.w).min(10);
        for size in 4..=total {
            let nat = c.min(size);
            let need = size - nat;
            let mut cards = Vec::with_capacity(size as usize);
            Picker::new(info).take(r, nat, &mut cards);
            wilds(info.level, need, &mut cards);
            out.push(Move::new(Kind::Bomb, strength(r, info.level), cards));
        }
    }
    gen_straight_flushes(info, out);
    if info.sj == 2 && info.bj == 2 {
        out.push(Move::new(Kind::JokerBomb, 0, vec![SJ, SJ, BJ, BJ]));
    }
}

pub fn gen_straight_flushes(info: &Info, out: &mut Vec<Move>) {
    for s in 0..4u8 {
        for p in 0..=9u8 {
            let mut need = 0;
            for i in 0..5 {
                if info.sc[seq_rank(p + i) as usize][s as usize] == 0 {
                    need += 1;
                }
            }
            if need > info.w {
                continue;
            }
            let mut cards = Vec::with_capacity(5);
            for i in 0..5 {
                let r = seq_rank(p + i);
                if info.sc[r as usize][s as usize] > 0 {
                    cards.push(ctype(r, s));
                }
            }
            wilds(info.level, need, &mut cards);
            out.push(Move::new(Kind::StraightFlush, p, cards));
        }
    }
}

pub fn gen_full_houses(info: &Info, out: &mut Vec<Move>) {
    // triple options: (rank, natural used, wilds)
    let mut triples: Vec<(u8, u8, u8)> = Vec::new();
    for r in 0..13u8 {
        let c = info.cnt[r as usize];
        if c == 0 {
            continue;
        }
        let nat = c.min(3);
        let need = 3 - nat;
        if need <= info.w {
            triples.push((r, nat, need));
        }
    }
    // pair options: (rank or 13/14 for jokers or 255 for wild pair, natural used, wilds)
    let mut pairs: Vec<(u8, u8, u8)> = Vec::new();
    for r in 0..13u8 {
        let c = info.cnt[r as usize];
        if c >= 2 {
            pairs.push((r, 2, 0));
        } else if c == 1 && info.w >= 1 {
            pairs.push((r, 1, 1));
        }
    }
    if info.w >= 2 {
        pairs.push((255, 0, 2));
    }
    if info.sj >= 2 {
        pairs.push((13, 2, 0));
    }
    if info.bj >= 2 {
        pairs.push((14, 2, 0));
    }
    for &(tr, tn, tw) in &triples {
        for &(pr, pn, pw) in &pairs {
            if pr == tr || tw + pw > info.w {
                continue;
            }
            let mut cards = Vec::with_capacity(5);
            let mut pk = Picker::new(info);
            pk.take(tr, tn, &mut cards);
            match pr {
                13 => cards.extend_from_slice(&[SJ, SJ]),
                14 => cards.extend_from_slice(&[BJ, BJ]),
                255 => {}
                _ => pk.take(pr, pn, &mut cards),
            }
            wilds(info.level, tw + pw, &mut cards);
            out.push(Move::new(Kind::FullHouse, strength(tr, info.level), cards));
        }
    }
}

fn gen_sequences(info: &Info, kind: Kind, width: u8, mult: u8, out: &mut Vec<Move>) {
    let max_start = 14 - width; // positions 0..=13
    for p in 0..=max_start {
        let mut need = 0u8;
        for i in 0..width {
            let c = info.cnt[seq_rank(p + i) as usize];
            need += mult.saturating_sub(c);
        }
        if need > info.w {
            continue;
        }
        let mut cards = Vec::with_capacity((width * mult) as usize);
        let mut pk = Picker::new(info);
        for i in 0..width {
            let r = seq_rank(p + i);
            let nat = info.cnt[r as usize].min(mult);
            pk.take(r, nat, &mut cards);
        }
        wilds(info.level, need, &mut cards);
        out.push(Move::new(kind, p, cards));
    }
}

pub fn gen_straights(info: &Info, out: &mut Vec<Move>) {
    gen_sequences(info, Kind::Straight, 5, 1, out);
}
pub fn gen_tubes(info: &Info, out: &mut Vec<Move>) {
    gen_sequences(info, Kind::Tube, 3, 2, out);
}
pub fn gen_plates(info: &Info, out: &mut Vec<Move>) {
    gen_sequences(info, Kind::Plate, 2, 3, out);
}

/// All legal moves for `hand`. `prev` is the move to beat (None when leading).
/// When following, `Pass` is included as the first move.
pub fn legal_moves(hand: &Hand, level: u8, prev: Option<&Move>) -> Vec<Move> {
    let info = Info::new(hand, level);
    let mut out = Vec::with_capacity(256);
    match prev {
        None => {
            gen_singles(&info, &mut out);
            gen_pairs(&info, &mut out);
            gen_triples(&info, &mut out);
            gen_full_houses(&info, &mut out);
            gen_straights(&info, &mut out);
            gen_tubes(&info, &mut out);
            gen_plates(&info, &mut out);
            gen_bombs(&info, &mut out);
        }
        Some(pm) => {
            out.push(Move::pass());
            let mut tmp = Vec::with_capacity(128);
            if !pm.is_bomb() {
                match pm.kind {
                    Kind::Single => gen_singles(&info, &mut tmp),
                    Kind::Pair => gen_pairs(&info, &mut tmp),
                    Kind::Triple => gen_triples(&info, &mut tmp),
                    Kind::FullHouse => gen_full_houses(&info, &mut tmp),
                    Kind::Straight => gen_straights(&info, &mut tmp),
                    Kind::Tube => gen_tubes(&info, &mut tmp),
                    Kind::Plate => gen_plates(&info, &mut tmp),
                    _ => {}
                }
            }
            gen_bombs(&info, &mut tmp);
            for m in tmp {
                if m.beats(pm) {
                    out.push(m);
                }
            }
        }
    }
    // dedupe (identical card sets with identical meaning)
    let first_pass = !out.is_empty() && out[0].is_pass();
    let start = if first_pass { 1 } else { 0 };
    let mut rest: Vec<Move> = out.split_off(start);
    rest.sort();
    rest.dedup();
    out.extend(rest);
    out
}

/// All interpretations of an exact multiset of cards (used to validate human input).
pub fn classify(cards: &[u8], level: u8) -> Vec<Move> {
    let n = cards.len();
    let mut res: Vec<Move> = Vec::new();
    if n == 0 || n > 10 {
        return res;
    }
    let hand = hand_from_cards(cards);
    if hand.iter().any(|&c| c > 2) {
        return res;
    }
    let info = Info::new(&hand, level);
    let jokers = info.sj + info.bj;
    let ranks: Vec<u8> = (0..13u8).filter(|&r| info.cnt[r as usize] > 0).collect();
    let mk = |kind: Kind, key: u8| Move::new(kind, key, cards.to_vec());

    if n == 4 && info.sj == 2 && info.bj == 2 {
        res.push(mk(Kind::JokerBomb, 0));
    }
    if jokers == 0 && ranks.len() <= 1 {
        let r = ranks.first().copied().unwrap_or(level);
        let key = strength(r, level);
        match n {
            1 => res.push(mk(Kind::Single, key)),
            2 => res.push(mk(Kind::Pair, key)),
            3 => res.push(mk(Kind::Triple, key)),
            4..=10 => res.push(mk(Kind::Bomb, key)),
            _ => {}
        }
    }
    if jokers as usize == n && n <= 2 && (info.sj == 0 || info.bj == 0) {
        let key = if info.sj > 0 { 14 } else { 15 };
        res.push(mk(if n == 1 { Kind::Single } else { Kind::Pair }, key));
    }
    if n == 5 {
        // full house
        let mut keys: Vec<u8> = Vec::new();
        if !(info.sj > 0 && info.bj > 0) {
            for t in 0..13u8 {
                for pr in 0..15u8 {
                    if pr == t {
                        continue;
                    }
                    if ranks.iter().any(|&x| x != t && x != pr) {
                        continue;
                    }
                    if info.sj > 0 && !(pr == 13 && info.sj == 2) {
                        continue;
                    }
                    if info.bj > 0 && !(pr == 14 && info.bj == 2) {
                        continue;
                    }
                    let ct = info.cnt[t as usize];
                    let cp = if pr < 13 { info.cnt[pr as usize] } else if pr == 13 { info.sj } else { info.bj };
                    if pr >= 13 && cp != 2 {
                        continue;
                    }
                    if ct > 3 || cp > 2 || ct == 0 {
                        continue;
                    }
                    if (3 - ct) + (2 - cp) == info.w {
                        let k = strength(t, level);
                        if !keys.contains(&k) {
                            keys.push(k);
                        }
                    }
                }
            }
        }
        for k in keys {
            res.push(mk(Kind::FullHouse, k));
        }
        // straight / straight flush
        if jokers == 0 && ranks.iter().all(|&r| info.cnt[r as usize] == 1) {
            let suits: Vec<u8> = (0..4u8)
                .filter(|&s| ranks.iter().any(|&r| info.sc[r as usize][s as usize] > 0))
                .collect();
            for p in 0..=9u8 {
                let seq: Vec<u8> = (0..5).map(|i| seq_rank(p + i)).collect();
                if ranks.iter().all(|r| seq.contains(r)) {
                    res.push(mk(Kind::Straight, p));
                    if suits.len() == 1 {
                        res.push(mk(Kind::StraightFlush, p));
                    }
                }
            }
        }
    }
    if n == 6 && jokers == 0 {
        for (kind, width, mult) in [(Kind::Tube, 3u8, 2u8), (Kind::Plate, 2, 3)] {
            for p in 0..=(14 - width) {
                let seq: Vec<u8> = (0..width).map(|i| seq_rank(p + i)).collect();
                if !ranks.iter().all(|r| seq.contains(r)) {
                    continue;
                }
                if seq.iter().any(|&r| info.cnt[r as usize] > mult) {
                    continue;
                }
                let need: u8 = seq.iter().map(|&r| mult - info.cnt[r as usize]).sum();
                if need == info.w {
                    res.push(mk(kind, p));
                }
            }
        }
    }
    res.sort();
    res.dedup();
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(r: u8, s: u8) -> u8 {
        ctype(r, s)
    }

    #[test]
    fn classify_basic() {
        let level = 0; // playing 2
        // pair of 9s
        let m = classify(&[c(7, 0), c(7, 2)], level);
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].kind, Kind::Pair);
        // straight 3-7 mixed suits
        let m = classify(&[c(1, 0), c(2, 2), c(3, 3), c(4, 0), c(5, 2)], level);
        assert!(m.iter().any(|x| x.kind == Kind::Straight && x.key == 2));
        assert!(!m.iter().any(|x| x.kind == Kind::StraightFlush));
        // A2345 straight (A low)
        let m = classify(&[c(12, 0), c(0, 2), c(1, 3), c(2, 0), c(3, 2)], 5);
        assert!(m.iter().any(|x| x.kind == Kind::Straight && x.key == 0));
        // 10JQKA
        let m = classify(&[c(8, 0), c(9, 2), c(10, 3), c(11, 0), c(12, 2)], 5);
        assert!(m.iter().any(|x| x.kind == Kind::Straight && x.key == 9));
        // JQKA2 invalid
        let m = classify(&[c(9, 0), c(10, 2), c(11, 3), c(12, 0), c(0, 2)], 5);
        assert!(m.is_empty());
    }

    #[test]
    fn classify_wild() {
        let level = 3; // playing 5, wild = 5 of hearts
        let w = wild_type(level);
        // 7 7 7 W W -> bomb(5) or full house (777 + WW)
        let m = classify(&[c(5, 0), c(5, 2), c(5, 3), w, w], level);
        assert!(m.iter().any(|x| x.kind == Kind::Bomb));
        assert!(m.iter().any(|x| x.kind == Kind::FullHouse));
        // 3 4 W 6 7 -> straight 3-7
        let m = classify(&[c(1, 0), c(2, 2), w, c(4, 0), c(5, 2)], level);
        assert!(m.iter().any(|x| x.kind == Kind::Straight && x.key == 2));
        // hearts 3 4 W 6 7 -> straight flush
        let m = classify(&[c(1, 1), c(2, 1), w, c(4, 1), c(5, 1)], level);
        assert!(m.iter().any(|x| x.kind == Kind::StraightFlush && x.key == 2));
        // W + joker not a pair
        let m = classify(&[w, SJ], level);
        assert!(m.is_empty());
        // 3 3 4 4 W W -> tube (two choices) and plate 333444
        let m = classify(&[c(1, 0), c(1, 2), c(2, 0), c(2, 3), w, w], level);
        assert!(m.iter().any(|x| x.kind == Kind::Tube));
        assert!(m.iter().any(|x| x.kind == Kind::Plate));
        // single wild = level single
        let m = classify(&[w], level);
        assert_eq!(m[0].key, 13);
    }

    #[test]
    fn bomb_order() {
        let level = 0;
        let b4 = Move::new(Kind::Bomb, 12, vec![48, 49, 50, 51]);
        let b5 = Move::new(Kind::Bomb, 1, vec![4, 5, 6, 7, 4]);
        let sf = Move::new(Kind::StraightFlush, 0, vec![]);
        let b6 = Move::new(Kind::Bomb, 1, vec![4, 5, 6, 7, 4, 5]);
        let jb = Move::new(Kind::JokerBomb, 0, vec![52, 52, 53, 53]);
        assert!(b5.beats(&b4));
        assert!(sf.beats(&b5));
        assert!(b6.beats(&sf));
        assert!(jb.beats(&b6));
        assert!(!b4.beats(&b5));
        let s = Move::new(Kind::Single, 13, vec![1]);
        assert!(b4.beats(&s));
        let _ = level;
    }

    #[test]
    fn generator_matches_classifier() {
        let mut rng = Rng::new(7);
        for trial in 0..300 {
            let mut deck: Vec<u8> = (0..54u8).flat_map(|t| [t, t]).collect();
            rng.shuffle(&mut deck);
            let level = (trial % 13) as u8;
            let hand = hand_from_cards(&deck[..27]);
            let moves = legal_moves(&hand, level, None);
            assert!(!moves.is_empty());
            for m in &moves {
                assert!(hand_contains(&hand, &m.cards), "move not in hand");
                let cl = classify(&m.cards, level);
                assert!(
                    cl.iter().any(|x| x.kind == m.kind && x.key == m.key),
                    "generated {:?} not recognised by classifier: {:?}",
                    m,
                    cl
                );
            }
            // following: every generated move beats prev
            let prev = &moves[(trial * 7) % moves.len()];
            let foll = legal_moves(&hand, level, Some(prev));
            assert!(foll[0].is_pass());
            for m in &foll[1..] {
                assert!(m.beats(prev));
            }
        }
    }
}
