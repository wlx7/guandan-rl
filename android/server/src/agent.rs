//! The AI player: scores every legal move with the network and switches to the exact endgame
//! search near the end of a round (python/guandan/agents.py, GameAgent).

use guandan_core::endgame::{belief_expectation, endgame_choose, EndgameCfg};
use guandan_core::features::{encode_action, encode_history, encode_state, ACTION_DIM, HIST_DIM, HIST_LEN, STATE_DIM};
use guandan_core::game::Game;
use guandan_core::moves::{Kind, Move};

use crate::net::Net;

/// Legal moves, their q and the state encoding z.
fn scores(net: &Net, g: &Game) -> (Vec<Move>, Vec<f32>, Vec<f32>) {
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
    (moves, q, z)
}

/// [(move, q)] sorted by q descending; q is in units of 3 levels (as in the Python server).
pub fn evaluate(net: &Net, g: &Game) -> Vec<(Move, f32)> {
    let (moves, q, _) = scores(net, g);
    let mut out: Vec<(Move, f32)> = moves.into_iter().zip(q).collect();
    out.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// Endgame search settings: the same as the Python server (agents.py ENDGAME_FAST / _DEEP).
fn endgame_cfg(deep: bool) -> EndgameCfg {
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2).clamp(1, 4);
    if deep {
        EndgameCfg { cards: 24, worlds: 32, max_nodes: 150_000, threads, ..EndgameCfg::default() }
    } else {
        EndgameCfg { cards: 20, worlds: 16, threads, ..EndgameCfg::default() }
    }
}

/// The network's move alone (no search).
pub fn choose_net(net: &Net, g: &Game) -> Move {
    evaluate(net, g).into_iter().next().map(|x| x.0).unwrap_or_else(Move::pass)
}

/// The network's move; near the end of a round the exact endgame search (worlds sampled from
/// the network's belief, every move solved exactly) replaces it when it finds a better one.
pub fn choose(net: &Net, g: &Game, deep: bool) -> Move {
    let (moves, q, z) = scores(net, g);
    if moves.is_empty() {
        return Move::pass();
    }
    let best = (0..q.len()).max_by(|&a, &b| q[a].partial_cmp(&q[b]).unwrap_or(std::cmp::Ordering::Equal)).unwrap();
    let cfg = endgame_cfg(deep);
    if moves.len() > 1 {
        let expect = net.belief_logits(&z).map(|lg| belief_expectation(&lg));
        if let Some(i) = endgame_choose(g, best, &cfg, expect.as_ref(), rand::random::<u64>()) {
            return moves[i].clone();
        }
    }
    moves[best].clone()
}

/// The simple rule bot (used if something goes wrong with the network).
pub fn rule_move(g: &Game) -> Move {
    let moves = g.legal_moves();
    if moves.is_empty() {
        return Move::pass();
    }
    let i = guandan_core::bot::rule_choose(g, &moves);
    moves.get(i).cloned().unwrap_or_else(Move::pass)
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
