//! Exact endgame solver for one fully known deal (every hand visible).
//!
//! Used by perfect-information Monte-Carlo (PIMC) search: the hidden hands are sampled
//! several times and every sampled world is solved exactly to the end of the round. The value
//! of a position is the level change of team 0 (+3/+2/+1 when team 0 wins the round, the
//! negative when it loses), so the minimax values are small integers and alpha-beta with a
//! transposition table of bounds cuts the tree well.

use std::collections::HashMap;

use crate::cards::*;
use crate::game::*;
use crate::moves::*;

#[derive(Clone)]
pub struct EState {
    pub hands: [Hand; 4],
    pub level: u8,
    pub current: u8,
    pub last: Option<Move>,
    pub last_player: u8,
    pub passes: u8,
    pub finished: [u8; 4],
    pub nfin: u8,
}

impl EState {
    pub fn from_game(g: &Game) -> EState {
        let mut finished = [0u8; 4];
        for (i, &f) in g.finished.iter().enumerate().take(4) {
            finished[i] = f;
        }
        EState {
            hands: g.hands,
            level: g.level,
            current: g.current,
            last: g.last_move.clone(),
            last_player: g.last_player,
            passes: g.passes,
            finished,
            nfin: g.finished.len() as u8,
        }
    }

    fn is_finished(&self, p: u8) -> bool {
        self.finished[..self.nfin as usize].contains(&p)
    }

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

    /// Same trick logic as Game::advance.
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
                self.last = None;
                self.passes = 0;
                self.current = leader;
                return;
            }
        }
        self.current = self.next_active(p);
    }

    /// Team 0's level change of the finished round.
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
        let pp = order.iter().position(|&x| x == partner(first)).unwrap();
        let gain: i8 = match pp {
            1 => 3,
            2 => 2,
            _ => 1,
        };
        if team(first) == 0 {
            gain
        } else {
            -gain
        }
    }

    /// Play `mv` for the current player (assumed legal). Returns team 0's result when the
    /// round ends with this move.
    pub fn play(&mut self, mv: &Move) -> Option<i8> {
        let p = self.current;
        if mv.is_pass() {
            self.passes += 1;
            self.advance(p);
            return None;
        }
        for &c in &mv.cards {
            self.hands[p as usize][c as usize] -= 1;
        }
        self.last = Some(mv.clone());
        self.last_player = p;
        self.passes = 0;
        if hand_count(&self.hands[p as usize]) == 0 {
            self.finished[self.nfin as usize] = p;
            self.nfin += 1;
            let nf = self.nfin;
            if nf == 3 || (nf == 2 && team(self.finished[0]) == team(self.finished[1])) {
                return Some(self.result());
            }
        }
        self.advance(p);
        None
    }

    fn key(&self) -> u64 {
        // FNV-1a over everything that determines the rest of the round
        let mut h: u64 = 0xcbf29ce484222325;
        let mut eat = |b: u8| {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        };
        for hand in &self.hands {
            for &c in hand.iter() {
                eat(c);
            }
        }
        eat(self.current);
        eat(self.last_player);
        eat(self.passes);
        eat(self.nfin);
        for &f in &self.finished[..self.nfin as usize] {
            eat(f);
        }
        match &self.last {
            None => eat(255),
            Some(m) => {
                eat(m.kind as u8);
                eat(m.key);
                for &c in &m.cards {
                    eat(c);
                }
            }
        }
        h
    }

    pub fn cards_left(&self) -> u32 {
        self.hands.iter().map(hand_count).sum()
    }
}

pub struct Solver {
    tt: HashMap<u64, (i8, i8)>, // (lower, upper) bounds of team 0's value
    pub nodes: u64,
    pub max_nodes: u64,
    pub aborted: bool,
}

impl Solver {
    pub fn new(max_nodes: u64) -> Solver {
        Solver { tt: HashMap::new(), nodes: 0, max_nodes, aborted: false }
    }

    fn ordered_moves(&self, s: &EState) -> Vec<Move> {
        let mut mv = legal_moves(&s.hands[s.current as usize], s.level, s.last.as_ref());
        // more cards first (finishing fast matters), passes last
        mv.sort_by_key(|m| std::cmp::Reverse(if m.is_pass() { 0 } else { 1 + m.cards.len() }));
        mv
    }

    /// Team 0's exact value of `s` (fail-soft alpha-beta within [alpha, beta]).
    pub fn value(&mut self, s: &EState, mut alpha: i8, mut beta: i8) -> i8 {
        self.nodes += 1;
        if self.nodes > self.max_nodes {
            self.aborted = true;
            return 0;
        }
        let key = s.key();
        let (mut lo, mut hi) = self.tt.get(&key).copied().unwrap_or((-3, 3));
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
        let maximize = team(s.current) == 0;
        let (a0, b0) = (alpha, beta);
        let mut best: i8 = if maximize { -4 } else { 4 };
        for mv in self.ordered_moves(s) {
            let mut c = s.clone();
            let v = match c.play(&mv) {
                Some(r) => r,
                None => self.value(&c, alpha, beta),
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
        // store bounds: a fail-high/low result is only a bound
        if best <= a0 {
            hi = hi.min(best);
        } else if best >= b0 {
            lo = lo.max(best);
        } else {
            lo = best;
            hi = best;
        }
        self.tt.insert(key, (lo, hi));
        best
    }
}

/// Exact level change for the team of the player to move after each of `moves`, with all hands
/// visible and perfect play to the end of the round; None when `max_nodes` was exceeded.
pub fn solve_moves(g: &Game, moves: &[Move], max_nodes: u64) -> Vec<Option<f32>> {
    let root = EState::from_game(g);
    let me_team = team(root.current);
    let mut solver = Solver::new(max_nodes);
    moves
        .iter()
        .map(|mv| {
            let mut c = root.clone();
            let v = match c.play(mv) {
                Some(r) => r,
                None => solver.value(&c, -3, 3),
            };
            if solver.aborted {
                None
            } else {
                Some(if me_team == 0 { v as f32 } else { -(v as f32) })
            }
        })
        .collect()
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
    /// floor weight of the belief-guided world sampling (see Game::determinize_weighted)
    pub floor: f32,
    /// node limit per world; a world that needs more is skipped
    pub max_nodes: u64,
    /// a move must beat the network's choice by this many levels on average
    pub margin: f32,
    pub threads: usize,
}

impl Default for EndgameCfg {
    fn default() -> Self {
        EndgameCfg { cards: 20, worlds: 16, floor: 0.02, max_nodes: 400_000, margin: 0.0, threads: 4 }
    }
}

/// Perfect-information Monte-Carlo endgame search for the player to move: sample `cfg.worlds`
/// worlds (guided by `expect`, the network's belief, when given), solve every legal move exactly
/// in each, and return the index (into `g.legal_moves()`) of the best average if it beats the
/// network's choice `net_choice` by more than `cfg.margin`; None = keep the network's move
/// (also when the position is not an endgame or no world could be solved in time).
pub fn endgame_choose(g: &Game, net_choice: usize, cfg: &EndgameCfg, expect: Option<&[[f32; 54]; 3]>, seed: u64) -> Option<usize> {
    if g.phase != Phase::Play {
        return None;
    }
    let left: u32 = g.hands.iter().map(hand_count).sum();
    if left > cfg.cards {
        return None;
    }
    let moves = g.legal_moves();
    if moves.len() < 2 || net_choice >= moves.len() {
        return None;
    }
    let me = g.current;
    let mut rng = Rng::new(seed);
    let worlds: Vec<Game> = (0..cfg.worlds)
        .map(|_| match expect {
            Some(e) => g.determinize_weighted(me, e, cfg.floor, &mut rng),
            None => g.determinize(me, &mut rng),
        })
        .collect();
    let threads = cfg.threads.max(1).min(worlds.len());
    let chunk = worlds.len().div_ceil(threads);
    let results: Vec<Vec<Option<f32>>> = std::thread::scope(|s| {
        let handles: Vec<_> = worlds
            .chunks(chunk)
            .map(|ws| {
                let mv = &moves;
                s.spawn(move || ws.iter().map(|w| solve_moves(w, mv, cfg.max_nodes)).collect::<Vec<_>>())
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().unwrap_or_default()).collect()
    });
    let mut sum = vec![0f64; moves.len()];
    let mut n = 0usize;
    for r in &results {
        if r.iter().all(|x| x.is_some()) {
            for (k, v) in r.iter().enumerate() {
                sum[k] += v.unwrap() as f64;
            }
            n += 1;
        }
    }
    if n == 0 {
        return None;
    }
    let best = (0..moves.len()).max_by(|&a, &b| sum[a].partial_cmp(&sum[b]).unwrap()).unwrap();
    if (sum[best] - sum[net_choice]) / n as f64 > cfg.margin as f64 + 1e-9 {
        Some(best)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solver_matches_finished_rounds() {
        // play random games to the end with the engine; the solver's replay of the same moves
        // must report the same result as the engine
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
                    // the engine may report 3 for a won match; gains otherwise agree
                    assert!(got == want || want.abs() == 3.0, "seed {seed}: {got} vs {want}");
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
        let cfg = EndgameCfg::default();
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
