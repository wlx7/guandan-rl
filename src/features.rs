//! Observation encoding. Everything is from the acting player's point of view and
//! uses only information visible to that player (own hand + public history).
//! Relative seats: 0 = self, 1 = next (opponent), 2 = partner, 3 = previous (opponent).

use crate::cards::*;
use crate::game::*;
use crate::moves::*;

pub const STATE_DIM: usize = 1550;
pub const HIST_LEN: usize = 24;
pub const HIST_DIM: usize = 140;
pub const ACTION_DIM: usize = 413;
/// Training-only label: the hands of the other three players (108 bits each).
pub const HIDDEN_DIM: usize = 324;

struct W<'a> {
    v: &'a mut [u8],
    o: usize,
}

impl<'a> W<'a> {
    #[inline]
    fn onehot(&mut self, n: usize, i: usize) {
        if i < n {
            self.v[self.o + i] = 1;
        }
        self.o += n;
    }
    #[inline]
    fn flag(&mut self, b: bool) {
        self.v[self.o] = b as u8;
        self.o += 1;
    }
    fn cards108(&mut self, h: &Hand) {
        for t in 0..54 {
            let c = h[t];
            if c >= 1 {
                self.v[self.o + 2 * t] = 1;
            }
            if c >= 2 {
                self.v[self.o + 2 * t + 1] = 1;
            }
        }
        self.o += 108;
    }
    fn cards108_list(&mut self, cards: &[u8]) {
        let h = hand_from_cards(cards);
        self.cards108(&h);
    }
    /// 15 strength buckets x 8 thermometer + 2 bits wild count.
    fn strength_thermo(&mut self, h: &Hand, level: u8) {
        let wt = wild_type(level) as usize;
        let mut cnt = [0u8; 15];
        for t in 0..54 {
            if t == wt {
                continue;
            }
            cnt[sidx(rank_of(t as u8), level) as usize] += h[t];
        }
        for (i, &c) in cnt.iter().enumerate() {
            for k in 0..8u8 {
                if c > k {
                    self.v[self.o + i * 8 + k as usize] = 1;
                }
            }
        }
        self.o += 120;
        let w = h[wt];
        self.flag(w >= 1);
        self.flag(w >= 2);
    }
}

#[inline]
fn rel(me: u8, other: u8) -> usize {
    ((other + 4 - me) & 3) as usize
}

pub fn unseen(g: &Game, me: u8) -> Hand {
    let mut u = [2u8; 54];
    for t in 0..54 {
        let mut known = g.hands[me as usize][t];
        for p in 0..4 {
            known += g.played[p][t];
        }
        u[t] = u[t].saturating_sub(known);
    }
    u
}

pub fn encode_state(g: &Game, me: u8, out: &mut [u8]) {
    debug_assert_eq!(out.len(), STATE_DIM);
    out.fill(0);
    let lv = g.level;
    let hand = &g.hands[me as usize];
    let mut w = W { v: out, o: 0 };
    w.cards108(hand);
    w.strength_thermo(hand, lv);
    let u = unseen(g, me);
    w.cards108(&u);
    w.strength_thermo(&u, lv);
    for r in 1..4u8 {
        let p = (me + r) & 3;
        w.cards108(&g.played[p as usize]);
    }
    for r in 1..4u8 {
        let p = (me + r) & 3;
        w.onehot(28, hand_count(&g.hands[p as usize]) as usize);
    }
    w.onehot(28, hand_count(hand) as usize);
    let my_team = team(me) as usize;
    w.onehot(13, lv as usize);
    w.onehot(13, g.team_level[my_team] as usize);
    w.onehot(13, g.team_level[1 - my_team] as usize);
    let banker = if g.banker < 0 { 0 } else if g.banker as usize == my_team { 1 } else { 2 };
    w.onehot(3, banker);
    for r in 0..4u8 {
        let p = (me + r) & 3;
        let place = g.finished.iter().position(|&x| x == p).map(|i| i + 1).unwrap_or(0);
        w.onehot(4, place);
    }
    // trick
    w.flag(g.is_leading());
    match &g.last_move {
        Some(m) => {
            w.cards108_list(&m.cards);
            w.onehot(NUM_KINDS, m.kind as usize);
            w.onehot(16, m.key as usize);
            w.onehot(11, m.cards.len());
            w.onehot(4, rel(me, g.last_player));
        }
        None => {
            w.o += 108 + NUM_KINDS + 16 + 11 + 4;
        }
    }
    w.onehot(4, (g.passes as usize).min(3));
    // tribute info
    for r in 0..4u8 {
        let p = (me + r) & 3;
        let mut given = [0u8; 54];
        let mut recv = [0u8; 54];
        for t in &g.tribute {
            if t.from == p {
                given[t.card as usize] = 1;
            }
            if t.to == p {
                recv[t.card as usize] = 1;
            }
        }
        w.v[w.o..w.o + 54].copy_from_slice(&given);
        w.o += 54;
        w.v[w.o..w.o + 54].copy_from_slice(&recv);
        w.o += 54;
    }
    w.flag(g.phase == Phase::Return);
    w.flag(g.anti_tribute);
    w.onehot(3, g.a_fail[my_team].min(2) as usize);
    w.onehot(3, g.a_fail[1 - my_team].min(2) as usize);
    debug_assert_eq!(w.o, STATE_DIM);
}

/// The hands of the other three players (next, partner, previous), 108 bits each. Never an
/// input of the playing network: only the target of an auxiliary prediction during training.
pub fn encode_hidden(g: &Game, me: u8, out: &mut [u8]) {
    debug_assert_eq!(out.len(), HIDDEN_DIM);
    out.fill(0);
    let mut w = W { v: out, o: 0 };
    for r in 1..4u8 {
        let p = (me + r) & 3;
        w.cards108(&g.hands[p as usize]);
    }
}

pub fn encode_history(g: &Game, me: u8, out: &mut [u8]) {
    debug_assert_eq!(out.len(), HIST_LEN * HIST_DIM);
    out.fill(0);
    let n = g.history.len();
    let take = n.min(HIST_LEN);
    for i in 0..take {
        let (p, m) = &g.history[n - take + i];
        let slot = HIST_LEN - take + i;
        let o = slot * HIST_DIM;
        let mut w = W { v: &mut out[o..o + HIST_DIM], o: 0 };
        w.cards108_list(&m.cards);
        w.onehot(NUM_KINDS, m.kind as usize);
        w.onehot(16, m.key as usize);
        w.onehot(4, rel(me, *p));
    }
}

pub fn encode_action(g: &Game, me: u8, m: &Move, out: &mut [u8]) {
    debug_assert_eq!(out.len(), ACTION_DIM);
    out.fill(0);
    let lv = g.level;
    let hand = &g.hands[me as usize];
    let mut after = *hand;
    for &c in &m.cards {
        after[c as usize] = after[c as usize].saturating_sub(1);
    }
    let mut w = W { v: out, o: 0 };
    w.cards108_list(&m.cards);
    w.onehot(NUM_KINDS, m.kind as usize);
    w.onehot(16, m.key as usize);
    w.onehot(11, m.cards.len());
    w.onehot(3, m.nwild(lv) as usize);
    w.strength_thermo(&after, lv);
    w.cards108(&after);
    let left = hand_count(&after) as usize;
    w.onehot(28, left);
    w.flag(m.kind != Kind::Tribute && !m.is_pass() && left == 0);
    // bombs broken by a non-bomb move
    let mut broken = 0usize;
    if !m.is_bomb() && !m.is_pass() && m.kind != Kind::Tribute {
        let wild = wild_type(lv);
        let mut ranks = 0u16;
        for &card in &m.cards {
            if card >= SJ || card == wild {
                continue;
            }
            let rank = rank_of(card);
            let bit = 1 << rank;
            if ranks & bit != 0 {
                continue;
            }
            ranks |= bit;
            let types = (0..4).map(|suit| ctype(rank, suit)).filter(|&t| t != wild);
            let before: u8 = types.clone().map(|t| hand[t as usize]).sum();
            let left: u8 = types.map(|t| after[t as usize]).sum();
            if before >= 4 && left < 4 {
                broken += 1;
            }
        }
    }
    w.onehot(4, broken.min(3));
    debug_assert_eq!(w.o, ACTION_DIM);
}
