//! The AI player: scores every legal move with the network and switches to the exact endgame
//! search near the end of a round (python/guandan/agents.py, GameAgent).

use guandan_core::cards::hand_count;
use guandan_core::endgame::{endgame_choose_full, EndgameCfg};
use guandan_core::fastnet::RolloutCfg;
use guandan_core::features::{encode_action, encode_history, encode_state, ACTION_DIM, HIST_DIM, HIST_LEN, STATE_DIM};
use guandan_core::game::{Game, Phase};
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
        EndgameCfg { cards: 30, worlds: 32, max_nodes: 250_000, prior: 0.25, max_cands: 12, q_gap: 1.0, threads, ..EndgameCfg::default() }
    } else {
        EndgameCfg { cards: 26, worlds: 24, max_nodes: 250_000, prior: 0.25, max_cands: 12, q_gap: 1.2, threads, ..EndgameCfg::default() }
    }
}

/// The network's move alone (no search).
pub fn choose_net(net: &Net, g: &Game) -> Move {
    evaluate(net, g).into_iter().next().map(|x| x.0).unwrap_or_else(Move::pass)
}

/// The network's move, upgraded by:
/// 1. Exact endgame search (`<= 26` / `30` cards) over calibrated + bomb-clustered + SMC worlds.
/// 2. Pure-Rust incremental-LSTM rollout search (+ 18-card exact leaf solver) on close mid-game
///    and tribute-return decisions.
pub fn choose(net: &Net, g: &Game, deep: bool) -> Move {
    let (moves, q, z) = scores(net, g);
    if moves.is_empty() {
        return Move::pass();
    }
    let best = (0..q.len()).max_by(|&a, &b| q[a].partial_cmp(&q[b]).unwrap_or(std::cmp::Ordering::Equal)).unwrap();
    let cfg = endgame_cfg(deep);
    if moves.len() > 1 {
        let logits = net.belief_logits(&z);
        let q_levels: Vec<f32> = q.iter().map(|&v| v * 3.0).collect();
        let cl: u32 = g.hands.iter().map(hand_count).sum();
        if g.phase == Phase::Play && cl <= cfg.cards {
            let smc = Some((&net.fast, 2usize, 6usize));
            if let Some(i) = endgame_choose_full(g, best, &cfg, None, logits.as_deref(), Some(&q_levels), 0.25, smc, rand::random::<u64>()) {
                return moves[i].clone();
            }
        } else {
            let ro_cfg = RolloutCfg {
                min_cards: cfg.cards + 1,
                max_cards: 80,
                early_cards: 56,
                early_close_gap: if deep { 0.08 } else { 0.06 },
                worlds: if deep { 20 } else { 16 },
                top_k: if deep { 5 } else { 4 },
                close_gap: if deep { 0.12 } else { 0.10 },
                max_cand_gap: 0.45,
                prior: 0.35,
                margin_se: 0.75,
                floor: cfg.floor,
                bomb_gamma: 0.25,
                solver_leaf_cards: 20,
                search_return: true,
                return_gap: if deep { 0.25 } else { 0.22 },
                return_top_k: 5,
                smc_oversample: 2,
                smc_last_k: 6,
                threads: cfg.threads,
                ..RolloutCfg::default()
            };
            if let Some(i) = net.fast.rollout_choose(g, best, &q_levels, logits.as_deref(), &ro_cfg, rand::random::<u64>()) {
                return moves[i].clone();
            }
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
