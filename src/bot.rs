//! A simple hand-written heuristic player. It is NOT used for training (training is pure
//! self-play); it only serves as a fixed yardstick when evaluating trained models.

use crate::cards::*;
use crate::game::*;
use crate::moves::*;

fn cost(m: &Move, info: &Info, level: u8) -> i32 {
    let wt = wild_type(level);
    let mut c: i32 = m.key as i32 * 2;
    if !m.is_bomb() {
        let mut seen = [false; 13];
        for &t in &m.cards {
            if t == wt {
                c += 10;
                continue;
            }
            let r = rank_of(t);
            if r < 13 && !seen[r as usize] {
                seen[r as usize] = true;
                if info.cnt[r as usize] >= 4 {
                    c += 30;
                }
            }
        }
    }
    c
}

pub fn rule_choose(g: &Game, moves: &[Move]) -> usize {
    if moves.len() <= 1 {
        return 0;
    }
    let me = g.current;
    let lv = g.level;
    let hand = &g.hands[me as usize];
    let info = Info::new(hand, lv);
    if g.phase == Phase::Return {
        // return the lowest card, preferring singletons
        let mut best = (i32::MAX, 0usize);
        for (i, m) in moves.iter().enumerate() {
            let t = m.cards[0];
            let r = rank_of(t) as usize;
            let s = m.key as i32 * 4 + if r < 13 { info.cnt[r] as i32 * 3 } else { 0 };
            if s < best.0 {
                best = (s, i);
            }
        }
        return best.1;
    }
    let total = hand_count(hand) as usize;
    for (i, m) in moves.iter().enumerate() {
        if !m.is_pass() && m.cards.len() == total {
            return i;
        }
    }
    if g.is_leading() {
        let mut best: Option<(i32, usize)> = None;
        for (i, m) in moves.iter().enumerate() {
            if m.is_bomb() {
                continue;
            }
            let c = cost(m, &info, lv) - m.cards.len() as i32 * 4;
            if best.map_or(true, |(b, _)| c < b) {
                best = Some((c, i));
            }
        }
        if let Some((_, i)) = best {
            return i;
        }
        let mut bi = 0;
        for (i, m) in moves.iter().enumerate() {
            if m.power() < moves[bi].power() {
                bi = i;
            }
        }
        return bi;
    }
    let last = g.last_player;
    if team(last) == team(me) {
        return 0; // let partner run
    }
    let mut best: Option<(i32, usize)> = None;
    for (i, m) in moves.iter().enumerate() {
        if m.is_pass() || m.is_bomb() {
            continue;
        }
        let c = cost(m, &info, lv);
        if best.map_or(true, |(b, _)| c < b) {
            best = Some((c, i));
        }
    }
    if let Some((_, i)) = best {
        return i;
    }
    let opp_left = hand_count(&g.hands[last as usize]);
    let lm_len = g.last_move.as_ref().map(|m| m.cards.len()).unwrap_or(0);
    if opp_left <= 8 || lm_len >= 5 {
        let mut bi: Option<usize> = None;
        for (i, m) in moves.iter().enumerate() {
            if m.is_bomb() && bi.map_or(true, |b| m.power() < moves[b].power()) {
                bi = Some(i);
            }
        }
        if let Some(i) = bi {
            return i;
        }
    }
    0
}
