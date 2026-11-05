//! The AI player: scores every legal move with the network (python/guandan/agents.py,
//! GameAgent without search).

use guandan_core::features::{encode_action, encode_history, encode_state, ACTION_DIM, HIST_DIM, HIST_LEN, STATE_DIM};
use guandan_core::game::Game;
use guandan_core::moves::{Kind, Move};

use crate::net::Net;

/// [(move, q)] sorted by q descending; q is in units of 3 levels (as in the Python server).
pub fn evaluate(net: &Net, g: &Game) -> Vec<(Move, f32)> {
    let me = g.current;
    let moves = g.legal_moves();
    let mut s = vec![0u8; STATE_DIM];
    let mut h = vec![0u8; HIST_LEN * HIST_DIM];
    encode_state(g, me, &mut s);
    encode_history(g, me, &mut h);
    let mut a = vec![0u8; moves.len() * ACTION_DIM];
    for (k, m) in moves.iter().enumerate() {
        encode_action(g, me, m, &mut a[k * ACTION_DIM..(k + 1) * ACTION_DIM]);
    }
    let z = net.encode_state(&s, &h);
    let q = net.score(&z, &a);
    let mut out: Vec<(Move, f32)> = moves.into_iter().zip(q).collect();
    out.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
    out
}

pub fn choose(net: &Net, g: &Game) -> Move {
    evaluate(net, g).swap_remove(0).0
}

/// The simple rule bot (used if something goes wrong with the network).
pub fn rule_move(g: &Game) -> Move {
    let moves = g.legal_moves();
    let i = guandan_core::bot::rule_choose(g, &moves);
    moves[i].clone()
}

/// A random legal move; used when a human player times out (no pass if anything can be played).
pub fn random_move(g: &Game) -> Move {
    use rand::seq::SliceRandom;
    let moves = g.legal_moves();
    let plays: Vec<&Move> = moves.iter().filter(|m| m.kind != Kind::Pass).collect();
    let mut r = rand::thread_rng();
    if let Some(m) = plays.choose(&mut r) {
        return (*m).clone();
    }
    moves.choose(&mut r).cloned().unwrap_or_else(Move::pass)
}
