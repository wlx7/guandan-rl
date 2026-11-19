//! Factored Public Belief State (PBS) & R-NaD Trajectory Environment for Guandan.
//!
//! Implements the common-knowledge public belief representation required by ReBeL / Student-of-Games
//! and R-NaD (DeepNash) in large-hand 2v2 imperfect-information games:
//! - `encode_public_state`: 1,433-bit canonical Team-0 public state (zero private hand leakage).
//! - `PubBelief`: 4 x 54 x 2 = 432-float common-knowledge belief tensor (`P_pub(count >= 1)` and
//!   `P_pub(count == 2)` for all 4 seats), maintained via public tribute/anti-tribute deductions,
//!   Bayesian action-likelihood updates, and 4-player Sinkhorn-Knopp marginal projection so row
//!   sums match exact hand sizes and column sums match exact public unseen card counts.
//! - `PbsVecEnv`: fast vectorized trajectory environment exposing full legal action sets, public
//!   states, private states, and calibrated public beliefs at every step for R-NaD (V-trace +
//!   NeurRD) and Public Belief Value learning.

use crate::cards::*;
use crate::features::{encode_action, encode_history, encode_state, ACTION_DIM, HIST_DIM, HIST_LEN, STATE_DIM};
use crate::game::*;
use crate::moves::*;

pub const PUB_STATE_DIM: usize = 1433;
pub const PBS_DIM: usize = 432; // 4 seats x 54 card types x 2 (P >= 1, P == 2)

struct BitW<'a> {
    v: &'a mut [u8],
    o: usize,
}

impl<'a> BitW<'a> {
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

/// Public unseen card counts (`2 - sum_p played[p][t]`), using ONLY public played cards.
pub fn public_unseen(g: &Game) -> Hand {
    let mut u = [2u8; 54];
    for t in 0..54 {
        let mut pl = 0u8;
        for p in 0..4 {
            pl += g.played[p][t];
        }
        u[t] = u[t].saturating_sub(pl);
    }
    u
}

/// Canonical common-knowledge public state encoding (absolute seats 0..4, Team 0 perspective).
pub fn encode_public_state(g: &Game, out: &mut [u8]) {
    debug_assert_eq!(out.len(), PUB_STATE_DIM);
    out.fill(0);
    let lv = g.level;
    let mut w = BitW { v: out, o: 0 };
    let u = public_unseen(g);
    w.cards108(&u);
    w.strength_thermo(&u, lv);
    for p in 0..4 {
        w.cards108(&g.played[p]);
    }
    for p in 0..4 {
        w.onehot(28, hand_count(&g.hands[p]) as usize);
    }
    w.onehot(13, lv as usize);
    w.onehot(13, g.team_level[0] as usize);
    w.onehot(13, g.team_level[1] as usize);
    let banker_idx = (g.banker + 1).clamp(0, 2) as usize;
    w.onehot(3, banker_idx);
    w.onehot(3, g.a_fail[0].min(2) as usize);
    w.onehot(3, g.a_fail[1].min(2) as usize);
    for p in 0..4u8 {
        let place = g.finished.iter().position(|&x| x == p).map(|i| i + 1).unwrap_or(0);
        w.onehot(4, place);
    }
    w.flag(g.is_leading());
    match &g.last_move {
        Some(m) => {
            w.cards108_list(&m.cards);
            w.onehot(NUM_KINDS, m.kind as usize);
            w.onehot(16, m.key as usize);
            w.onehot(11, m.cards.len());
            w.onehot(4, g.last_player as usize);
        }
        None => {
            w.o += 108 + NUM_KINDS + 16 + 11 + 4;
        }
    }
    w.onehot(4, (g.passes as usize).min(3));
    w.flag(g.jiefeng);
    w.onehot(4, g.current as usize);
    w.flag(g.phase == Phase::Return);
    w.flag(g.anti_tribute);
    for p in 0..4u8 {
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
    debug_assert_eq!(w.o, PUB_STATE_DIM);
}

/// Canonical public history encoding (uses absolute seat index `0..4` in the last 4 bits).
pub fn encode_public_history(g: &Game, out: &mut [u8]) {
    debug_assert_eq!(out.len(), HIST_LEN * HIST_DIM);
    out.fill(0);
    let n = g.history.len();
    let take = n.min(HIST_LEN);
    for i in 0..take {
        let (p, m) = &g.history[n - take + i];
        let slot = HIST_LEN - take + i;
        let o = slot * HIST_DIM;
        let mut w = BitW { v: &mut out[o..o + HIST_DIM], o: 0 };
        w.cards108_list(&m.cards);
        w.onehot(NUM_KINDS, m.kind as usize);
        w.onehot(16, m.key as usize);
        w.onehot(4, (*p & 3) as usize);
    }
}

/// Common-knowledge Factored Public Belief State (`4 x 54 x 2`) over all 4 players' hands.
#[derive(Clone, Debug)]
pub struct PubBelief {
    pub raw_w: [[f64; 54]; 4],
    pub pair_bias: [[f64; 54]; 4],
    pub p1: [[f32; 54]; 4],
    pub p2: [[f32; 54]; 4],
    pub expect: [[f32; 54]; 4],
}

impl PubBelief {
    pub fn from_game(g: &Game) -> Self {
        let mut pb = PubBelief {
            raw_w: [[1.0; 54]; 4],
            pair_bias: [[1.0; 54]; 4],
            p1: [[0.0; 54]; 4],
            p2: [[0.0; 54]; 4],
            expect: [[0.0; 54]; 4],
        };
        pb.apply_tribute_deductions(g);
        pb.recalibrate(g);
        pb
    }

    fn apply_tribute_deductions(&mut self, g: &Game) {
        self.raw_w = [[1.0; 54]; 4];
        self.pair_bias = [[1.0; 54]; 4];
        let lv = g.level;
        let wt = wild_type(lv);
        let mut paid_bj = false;
        for t in &g.tribute {
            if t.is_return {
                continue;
            }
            if t.card == BJ {
                paid_bj = true;
            }
            let p = (t.from & 3) as usize;
            let s_trib = strength(rank_of(t.card), lv);
            for c in 0..54u8 {
                if c != wt && strength(rank_of(c), lv) > s_trib {
                    // Only possible if later received back via return (return cards are <= 10)
                    let got_ret = g.tribute.iter().any(|x| x.is_return && x.to == t.from && x.card == c);
                    if !got_ret {
                        self.raw_w[p][c as usize] = 1e-4;
                    }
                }
            }
        }
        if !g.anti_tribute && paid_bj {
            for t in &g.tribute {
                if !t.is_return {
                    self.raw_w[(t.from & 3) as usize][BJ as usize] = 1e-4;
                }
            }
        } else if g.anti_tribute {
            if let Some(ord) = g.prev_order {
                let double = team(ord[0]) == team(ord[1]);
                for &q in if double { &ord[..2] } else { &ord[..3] } {
                    self.raw_w[(q & 3) as usize][BJ as usize] = 1e-4;
                }
            }
        }
    }

    /// Observe public move `mv` played by `actor` BEFORE `g.apply(mv)` (or pass `prev_last` + `g_after`).
    pub fn observe_step(&mut self, prev_last: Option<(u8, &Move)>, actor: u8, mv: &Move, g_after: &Game) {
        let pu = (actor & 3) as usize;
        if g_after.phase == Phase::Play && g_after.history.len() <= 1 && !g_after.tribute.is_empty() {
            // Re-apply return/tribute constraints right after Return phase finishes
            self.apply_tribute_deductions(g_after);
        }
        if mv.is_pass() {
            if let Some((lp, lm)) = prev_last {
                if team(lp) != team(actor) {
                    let hc = hand_count(&g_after.hands[pu]);
                    let lv = g_after.level;
                    if lm.kind == Kind::Single && hc <= 12 {
                        let decay = if hc <= 4 { 0.42 } else { 0.74 };
                        for c in 0..54u8 {
                            if strength(rank_of(c), lv) > lm.key {
                                self.raw_w[pu][c as usize] = (self.raw_w[pu][c as usize] * decay).max(0.01);
                            }
                        }
                    } else if lm.kind == Kind::Pair && hc <= 8 {
                        for c in 0..54u8 {
                            if strength(rank_of(c), lv) > lm.key {
                                self.pair_bias[pu][c as usize] = (self.pair_bias[pu][c as usize] * 0.65).max(0.15);
                            }
                        }
                    }
                }
            }
        } else if mv.kind == Kind::Single && mv.cards.len() == 1 {
            // Playing a single slightly reduces the prior belief that the player was holding a natural pair of it
            let c = mv.cards[0] as usize;
            self.pair_bias[pu][c] = (self.pair_bias[pu][c] * 0.75).max(0.20);
        }
        self.recalibrate(g_after);
    }

    /// Bayesian PBS update for `actor` from `K` sampled hands weighted by policy probabilities
    /// `weights[k] = pi(observed_move | hand_k, beta)` (the exact ReBeL / Student-of-Games belief update!).
    pub fn bayes_policy_update(&mut self, g_after: &Game, actor: u8, hands: &[Hand], weights: &[f32], blend: f32) {
        if hands.is_empty() || hands.len() != weights.len() {
            return;
        }
        let sum_w: f64 = weights.iter().map(|&x| (x.max(0.0)) as f64).sum();
        if sum_w <= 1e-12 {
            return;
        }
        let pu = (actor & 3) as usize;
        let hc = hand_count(&g_after.hands[pu]) as f64;
        if hc <= 0.0 {
            self.recalibrate(g_after);
            return;
        }
        let mut post_e = [0.0f64; 54];
        for (h, &w) in hands.iter().zip(weights.iter()) {
            let wf = (w.max(0.0) as f64) / sum_w;
            for t in 0..54 {
                post_e[t] += wf * (h[t] as f64);
            }
        }
        let alpha = blend.clamp(0.0, 0.95) as f64;
        for t in 0..54 {
            let cur = (self.expect[pu][t] as f64).max(1e-4);
            let ratio = ((post_e[t] + 0.02) / (cur + 0.02)).clamp(0.20, 5.0);
            self.raw_w[pu][t] = (self.raw_w[pu][t] * ((1.0 - alpha) + alpha * ratio)).clamp(1e-3, 50.0);
        }
        self.recalibrate(g_after);
    }

    /// 4-player Sinkhorn-Knopp IPF calibrating `(expect, p1, p2)` to exact public unseen counts
    /// and exact player hand sizes, respecting known public tribute/return cards.
    pub fn recalibrate(&mut self, g: &Game) {
        let mut pool = public_unseen(g);
        let mut left = [0u32; 4];
        for p in 0..4 {
            left[p] = hand_count(&g.hands[p]);
        }
        let mut known = [[0u8; 54]; 4];
        for t in &g.tribute {
            let q = (t.to & 3) as usize;
            let c = t.card as usize;
            let still = (1u8).saturating_sub(g.played[q][c].min(1));
            if still > 0 && pool[c] > 0 && left[q] > 0 && known[q][c] < 2 {
                known[q][c] += 1;
                pool[c] -= 1;
                left[q] -= 1;
            }
        }

        let total_left: u32 = left.iter().sum();
        let mut e_rem = [[0.0f64; 54]; 4];
        if total_left > 0 {
            let tot_f = total_left as f64;
            for p in 0..4 {
                if left[p] == 0 {
                    continue;
                }
                let share = left[p] as f64 / tot_f;
                for t in 0..54 {
                    if pool[t] == 0 || known[p][t] >= 2 {
                        continue;
                    }
                    e_rem[p][t] = (pool[t] as f64) * share * self.raw_w[p][t].max(1e-4);
                }
            }
            for _ in 0..6 {
                for t in 0..54 {
                    if pool[t] == 0 {
                        continue;
                    }
                    let cs = e_rem[0][t] + e_rem[1][t] + e_rem[2][t] + e_rem[3][t];
                    if cs > 1e-12 {
                        let sc = pool[t] as f64 / cs;
                        for p in 0..4 {
                            let cap = (pool[t] as u32).min(left[p]).min((2 - known[p][t]) as u32) as f64;
                            e_rem[p][t] = (e_rem[p][t] * sc).min(cap);
                        }
                    }
                }
                for p in 0..4 {
                    if left[p] == 0 {
                        continue;
                    }
                    let rs: f64 = e_rem[p].iter().sum();
                    if rs > 1e-12 {
                        let sc = left[p] as f64 / rs;
                        for t in 0..54 {
                            if pool[t] > 0 {
                                e_rem[p][t] *= sc;
                            }
                        }
                    }
                }
            }
            for t in 0..54 {
                if pool[t] > 0 {
                    let cs = e_rem[0][t] + e_rem[1][t] + e_rem[2][t] + e_rem[3][t];
                    if cs > 1e-12 {
                        let sc = pool[t] as f64 / cs;
                        for p in 0..4 {
                            e_rem[p][t] *= sc;
                        }
                    }
                }
            }
        }

        for p in 0..4 {
            let unif_rho = if total_left > 1 && left[p] > 1 {
                (left[p] - 1) as f64 / (total_left - 1) as f64
            } else {
                0.0
            };
            for t in 0..54 {
                let k = known[p][t] as f64;
                let er = e_rem[p][t];
                let ex = (k + er).clamp(0.0, 2.0);
                self.expect[p][t] = ex as f32;
                let (a, b) = if known[p][t] == 2 {
                    (1.0, 1.0)
                } else if known[p][t] == 1 {
                    (1.0, er.clamp(0.0, 1.0))
                } else if pool[t] <= 1 {
                    (er.clamp(0.0, 1.0), 0.0)
                } else {
                    let rho = (unif_rho * self.pair_bias[p][t]).clamp(0.0, 0.95);
                    let p2_est = if ex >= 1.0 {
                        (rho * ex * 0.5 + (ex - 1.0) * (1.0 - rho * 0.5)).clamp(ex - 1.0, 0.5 * ex)
                    } else {
                        (0.5 * rho * ex * ex).min(0.5 * ex)
                    };
                    let p1_est = (ex - p2_est).clamp(p2_est, 1.0);
                    (p1_est, p2_est)
                };
                self.p1[p][t] = a as f32;
                self.p2[p][t] = b as f32;
            }
        }
    }

    /// Write absolute 432-float PBS tensor (`seats 0, 1, 2, 3`).
    pub fn write_abs(&self, out: &mut [f32]) {
        debug_assert_eq!(out.len(), PBS_DIM);
        for p in 0..4 {
            for t in 0..54 {
                let base = (p * 54 + t) * 2;
                out[base] = self.p1[p][t];
                out[base + 1] = self.p2[p][t];
            }
        }
    }

    /// Write seat-relative 432-float PBS tensor (`[me, me+1, me+2, me+3]`).
    pub fn write_rel(&self, me: u8, out: &mut [f32]) {
        debug_assert_eq!(out.len(), PBS_DIM);
        for r in 0..4u8 {
            let p = ((me + r) & 3) as usize;
            for t in 0..54 {
                let base = ((r as usize) * 54 + t) * 2;
                out[base] = self.p1[p][t];
                out[base + 1] = self.p2[p][t];
            }
        }
    }

    /// Sample a consistent world from this Public Belief State!
    /// - `me = Some(seat)`: conditions on `seat`'s true private hand (for seat-rooted subgame search).
    /// - `me = None`: samples all 4 hands purely from the common-knowledge public belief `self`.
    pub fn sample_world(&self, g: &Game, me: Option<u8>, rng: &mut Rng) -> Game {
        if let Some(m) = me {
            // Convert public belief for seats m+1, m+2, m+3 into 324 pseudo-logits and call
            // our Sinkhorn + pair + bomb-clustered `determinize_belief_clustered`!
            let mut lg = [0.0f32; 324];
            let logit = |p: f32| -> f32 {
                let c = p.clamp(1e-4, 1.0 - 1e-4);
                (c / (1.0 - c)).ln()
            };
            for r in 0..3usize {
                let q = ((m as usize) + r + 1) & 3;
                for t in 0..54 {
                    let base = (r * 54 + t) * 2;
                    lg[base] = logit(self.p1[q][t]);
                    lg[base + 1] = logit(self.p2[q][t]);
                }
            }
            return g.determinize_belief_clustered(m, &lg, 0.02, 0.25, rng);
        }

        // Pure 4-seat public world sampling (no private hand conditioned on)
        let mut out = g.clone();
        let mut pool = public_unseen(g);
        let mut left = [0u32; 4];
        let mut new_hands = [[0u8; 54]; 4];
        for p in 0..4 {
            left[p] = hand_count(&g.hands[p]);
        }
        for t in &g.tribute {
            let q = (t.to & 3) as usize;
            let c = t.card as usize;
            let still = (1u8).saturating_sub(g.played[q][c].min(1));
            if still > 0 && pool[c] > 0 && left[q] > 0 && new_hands[q][c] < 2 {
                new_hands[q][c] += 1;
                pool[c] -= 1;
                left[q] -= 1;
            }
        }
        let mut rem_e = [0.0f64; 4];
        let mut e = [[0.0f64; 54]; 4];
        for p in 0..4 {
            for t in 0..54 {
                e[p][t] = ((self.expect[p][t] as f64) - (new_hands[p][t] as f64)).max(0.0);
                rem_e[p] += e[p][t];
            }
        }
        let mut order: [(f64, u8); 54] = [(0.0, 0); 54];
        let mut n_ord = 0usize;
        for t in 0..54u8 {
            let pc = pool[t as usize];
            if pc == 0 {
                continue;
            }
            let pcf = pc as f64;
            let mx = (0..4).map(|p| e[p][t as usize] / pcf).fold(0.0f64, f64::max);
            order[n_ord] = (mx + rng.uniform() * 0.02, t);
            n_ord += 1;
        }
        order[..n_ord].sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let pick4 = |w: &[f64; 4], left: &[u32; 4], t: usize, nh: &[[u8; 54]; 4], rng: &mut Rng| -> usize {
            let sum: f64 = w.iter().sum();
            if sum > 0.0 {
                let mut x = rng.uniform() * sum;
                let mut last_ok = 0;
                for p in 0..4 {
                    if w[p] > 0.0 {
                        last_ok = p;
                        if x < w[p] {
                            return p;
                        }
                        x -= w[p];
                    }
                }
                return last_ok;
            }
            for p in 0..4 {
                if left[p] > 0 && nh[p][t] < 2 {
                    return p;
                }
            }
            (0..4).find(|&p| left[p] > 0).unwrap_or(0)
        };

        for &(_, t_u8) in &order[..n_ord] {
            let t = t_u8 as usize;
            let pc = pool[t];
            for _ in 0..pc {
                let mut w = [0.0f64; 4];
                for p in 0..4 {
                    if left[p] > 0 && new_hands[p][t] < 2 {
                        w[p] = e[p][t].max(1e-5) * (left[p] as f64 / rem_e[p].max(0.05));
                    }
                }
                let chosen = pick4(&w, &left, t, &new_hands, rng);
                new_hands[chosen][t] += 1;
                left[chosen] = left[chosen].saturating_sub(1);
            }
            for p in 0..4 {
                rem_e[p] = (rem_e[p] - e[p][t]).max(0.0);
            }
        }
        out.hands = new_hands;
        out.rng = Rng::new(rng.next_u64());
        out
    }
}

/// Vectorized trajectory environment with live Factored Public Belief State tracking (`PubBelief`)
/// for R-NaD (V-trace + NeurRD) and ReBeL Public Belief Value learning.
pub struct PbsVecEnv {
    pub games: Vec<Game>,
    pub beliefs: Vec<PubBelief>,
    moves: Vec<Vec<Move>>,
    rng: Rng,
    random_levels: bool,
}

impl PbsVecEnv {
    pub fn new(n: usize, seed: u64, random_levels: bool, rules: Rules) -> Self {
        let mut rng = Rng::new(seed);
        let mut games = Vec::with_capacity(n);
        let mut beliefs = Vec::with_capacity(n);
        for _ in 0..n {
            let mut g = Game::new(rng.next_u64(), rules.clone());
            if random_levels {
                let l0 = rng.below(13) as u8;
                let l1 = rng.below(13) as u8;
                g.set_levels([l0, l1], rng.below(2) as i8);
                g.start_round();
            }
            let b = PubBelief::from_game(&g);
            games.push(g);
            beliefs.push(b);
        }
        PbsVecEnv { games, beliefs, moves: vec![vec![]; n], rng, random_levels }
    }

    pub fn len(&self) -> usize {
        self.games.len()
    }

    pub fn is_empty(&self) -> bool {
        self.games.is_empty()
    }

    /// Observe all `n` games:
    /// `(players, priv_states, pub_states, pbs_rel, pbs_abs, hists, actions, offsets, hands108_abs)`
    #[allow(clippy::type_complexity)]
    pub fn observe(
        &mut self,
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<f32>, Vec<f32>, Vec<u8>, Vec<u8>, Vec<u32>, Vec<u8>) {
        let n = self.games.len();
        let mut players = Vec::with_capacity(n);
        let mut s_priv = vec![0u8; n * STATE_DIM];
        let mut s_pub = vec![0u8; n * PUB_STATE_DIM];
        let mut pbs_rel = vec![0.0f32; n * PBS_DIM];
        let mut pbs_abs = vec![0.0f32; n * PBS_DIM];
        let mut h = vec![0u8; n * HIST_LEN * HIST_DIM];
        let mut hands108 = vec![0u8; n * PBS_DIM];
        let mut a = Vec::with_capacity(n * 16 * ACTION_DIM);
        let mut off = Vec::with_capacity(n + 1);
        off.push(0);

        for i in 0..n {
            let g = &self.games[i];
            let pb = &self.beliefs[i];
            let me = g.current;
            let mv = g.legal_moves();
            encode_state(g, me, &mut s_priv[i * STATE_DIM..(i + 1) * STATE_DIM]);
            encode_public_state(g, &mut s_pub[i * PUB_STATE_DIM..(i + 1) * PUB_STATE_DIM]);
            pb.write_rel(me, &mut pbs_rel[i * PBS_DIM..(i + 1) * PBS_DIM]);
            pb.write_abs(&mut pbs_abs[i * PBS_DIM..(i + 1) * PBS_DIM]);
            encode_history(g, me, &mut h[i * HIST_LEN * HIST_DIM..(i + 1) * HIST_LEN * HIST_DIM]);

            // True 4-seat cards108 indicator (training-only target for linear counterfactual PBS critic w_cf!)
            let h_base = i * PBS_DIM;
            for p in 0..4 {
                for t in 0..54 {
                    let c = g.hands[p][t];
                    let idx = h_base + (p * 54 + t) * 2;
                    if c >= 1 {
                        hands108[idx] = 1;
                    }
                    if c >= 2 {
                        hands108[idx + 1] = 1;
                    }
                }
            }

            let base_a = a.len();
            a.resize(base_a + mv.len() * ACTION_DIM, 0);
            for (k, m) in mv.iter().enumerate() {
                let o = base_a + k * ACTION_DIM;
                encode_action(g, me, m, &mut a[o..o + ACTION_DIM]);
            }
            off.push(off[i] + mv.len() as u32);
            players.push(me);
            self.moves[i] = mv;
        }
        (players, s_priv, s_pub, pbs_rel, pbs_abs, h, a, off, hands108)
    }

    /// Step all `n` games with `choices`, returning `(done_flags, team0_round_rewards)`.
    /// Whenever a round ends, `team0_round_rewards[i]` holds Team 0's level reward (`±1/±2/±3`,
    /// or `0.0` if the round continues), and the game immediately advances to the next round
    /// with a fresh `PubBelief`!
    pub fn step(&mut self, choices: &[u32]) -> (Vec<u8>, Vec<f32>) {
        let n = self.games.len();
        let mut dones = vec![0u8; n];
        let mut r_team0 = vec![0.0f32; n];

        for i in 0..n {
            let mv_list = &self.moves[i];
            if mv_list.is_empty() {
                continue;
            }
            let c = (choices[i] as usize).min(mv_list.len() - 1);
            let mv = mv_list[c].clone();
            let g = &mut self.games[i];
            let actor = g.current;
            let prev_lp = g.last_player;
            let prev_lm = g.last_move.clone();

            let _ = g.apply(&mv);
            if g.phase == Phase::RoundOver || g.phase == Phase::MatchOver {
                dones[i] = 1;
                r_team0[i] = g.round_reward(0);
                if self.random_levels {
                    let l0 = self.rng.below(13) as u8;
                    let l1 = self.rng.below(13) as u8;
                    g.set_levels([l0, l1], self.rng.below(2) as i8);
                    g.round_no = 0;
                    g.prev_order = None;
                    g.start_round();
                } else if g.phase == Phase::MatchOver {
                    g.new_match();
                } else {
                    g.next_round();
                }
                self.beliefs[i] = PubBelief::from_game(g);
            } else {
                let prev_ref = prev_lm.as_ref().map(|m| (prev_lp, m));
                self.beliefs[i].observe_step(prev_ref, actor, &mv, g);
            }
        }
        (dones, r_team0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_belief_marginals_match_counts_and_unseen() {
        let mut rng = Rng::new(2026);
        for seed in 1..25u64 {
            let mut g = Game::new(seed, Rules::default());
            let mut pb = PubBelief::from_game(&g);
            let mut s_pub = [0u8; PUB_STATE_DIM];
            for _ in 0..60 {
                if g.phase != Phase::Play && g.phase != Phase::Return {
                    break;
                }
                encode_public_state(&g, &mut s_pub);
                // Check row sums == hand_counts and col sums == public_unseen
                let u = public_unseen(&g);
                for p in 0..4 {
                    let row_sum: f32 = pb.expect[p].iter().sum();
                    let want = hand_count(&g.hands[p]) as f32;
                    assert!((row_sum - want).abs() < 2e-2, "seat {p} sum {row_sum} vs {want}");
                }
                for t in 0..54 {
                    let col_sum: f32 = (0..4).map(|p| pb.expect[p][t]).sum();
                    let want = u[t] as f32;
                    assert!((col_sum - want).abs() < 2e-2, "card {t} sum {col_sum} vs {want}");
                }
                if g.phase == Phase::Play {
                    let w_pub = pb.sample_world(&g, None, &mut rng);
                    let w_cond = pb.sample_world(&g, Some(g.current), &mut rng);
                    for p in 0..4 {
                        assert_eq!(hand_count(&w_pub.hands[p]), hand_count(&g.hands[p]));
                        assert_eq!(hand_count(&w_cond.hands[p]), hand_count(&g.hands[p]));
                    }
                    assert_eq!(w_cond.hands[g.current as usize], g.hands[g.current as usize]);
                }
                let mv = g.legal_moves();
                let chosen = mv[rng.below(mv.len() as u64) as usize].clone();
                let actor = g.current;
                let plp = g.last_player;
                let plm = g.last_move.clone();
                g.apply(&chosen).unwrap();
                if g.phase == Phase::Play || g.phase == Phase::Return {
                    pb.observe_step(plm.as_ref().map(|m| (plp, m)), actor, &chosen, &g);
                }
            }
        }
    }

    #[test]
    fn pbs_vecenv_steps_cleanly() {
        let mut env = PbsVecEnv::new(8, 99, false, Rules::default());
        let mut rng = Rng::new(7);
        let mut finished = 0;
        for _ in 0..300 {
            let (players, sp, pub_s, pbs_r, pbs_a, _h, _a, off, _h108) = env.observe();
            assert_eq!(players.len(), 8);
            assert_eq!(sp.len(), 8 * STATE_DIM);
            assert_eq!(pub_s.len(), 8 * PUB_STATE_DIM);
            assert_eq!(pbs_r.len(), 8 * PBS_DIM);
            assert_eq!(pbs_a.len(), 8 * PBS_DIM);
            let ch: Vec<u32> = (0..8).map(|i| rng.below((off[i + 1] - off[i]) as u64) as u32).collect();
            let (dones, rews) = env.step(&ch);
            for (d, r) in dones.into_iter().zip(rews) {
                if d != 0 {
                    assert!(r.abs() >= 1.0);
                    finished += 1;
                }
            }
        }
        assert!(finished > 0);
    }
}
