//! Support for inference-time search: sampling the hidden hands consistently with
//! everything one player has observed ("determinization"), and running many such
//! sampled games forward in lock-step so that one batched network call drives all of them.

use crate::cards::*;
use crate::features::*;
use crate::game::*;
use crate::moves::*;

impl Game {
    /// A copy of the game in which the hands of the other three players are re-dealt at
    /// random from the cards `me` has not seen, keeping each player's card count and the
    /// cards known to be in their hand (received through tribute / return and not played).
    pub fn determinize(&self, me: u8, rng: &mut Rng) -> Game {
        self.determinize_with(me, None, 0.0, rng)
    }

    /// Like `determinize`, but every unseen card goes to one of the other players with
    /// probability guided by `expect[r][t]`: the expected number of copies of card type `t`
    /// in the hand of the player `r + 1` seats after `me` (e.g. predicted by the network's
    /// belief head). `floor` keeps every player possible so that a wrong guess is never
    /// ruled out completely. Card counts and known tribute constraints are respected exactly.
    pub fn determinize_weighted(&self, me: u8, expect: &[[f32; 54]; 3], floor: f32, rng: &mut Rng) -> Game {
        self.determinize_guided(me, Some(expect), None, floor, rng)
    }

    /// Sample hidden hands from the network's 324 belief logits (`3 x 54 x 2`: `P(count >= 1)`
    /// and `P(count == 2)` for seats `me + 1, me + 2, me + 3`), calibrated by Sinkhorn-Knopp
    /// scaling to match exact hand sizes and unseen card counts, preserving exact 2-copy pair
    /// correlations and hard tribute / anti-tribute deductions.
    pub fn determinize_belief(&self, me: u8, logits: &[f32], floor: f32, rng: &mut Rng) -> Game {
        debug_assert_eq!(logits.len(), HIDDEN_DIM);
        self.determinize_guided(me, None, Some(logits), floor, rng)
    }

    fn tribute_forbidden(&self, me: u8) -> [[bool; 54]; 3] {
        let mut forbidden = [[false; 54]; 3];
        let lv = self.level;
        let wt = wild_type(lv);
        let r_idx = |p: u8| -> Option<usize> {
            if p == me { None } else { Some((((p + 4 - me) & 3) - 1) as usize) }
        };
        let mut paid_bj = false;
        for t in &self.tribute {
            if t.is_return {
                continue;
            }
            if t.card == BJ {
                paid_bj = true;
            }
            if let Some(r) = r_idx(t.from) {
                let s_trib = strength(rank_of(t.card), lv);
                for c in 0..54u8 {
                    if c != wt && strength(rank_of(c), lv) > s_trib {
                        forbidden[r][c as usize] = true;
                    }
                }
            }
        }
        if !self.anti_tribute && paid_bj {
            for t in &self.tribute {
                if !t.is_return {
                    if let Some(r) = r_idx(t.from) {
                        forbidden[r][BJ as usize] = true;
                    }
                }
            }
        } else if self.anti_tribute {
            if let Some(ord) = self.prev_order {
                let double = team(ord[0]) == team(ord[1]);
                for &q in if double { &ord[..2] } else { &ord[..3] } {
                    if let Some(r) = r_idx(q) {
                        forbidden[r][BJ as usize] = true;
                    }
                }
            }
        }
        forbidden
    }

    fn determinize_with(&self, me: u8, expect: Option<&[[f32; 54]; 3]>, floor: f32, rng: &mut Rng) -> Game {
        self.determinize_guided(me, expect, None, floor, rng)
    }

    fn determinize_guided(
        &self,
        me: u8,
        expect: Option<&[[f32; 54]; 3]>,
        logits: Option<&[f32]>,
        floor: f32,
        rng: &mut Rng,
    ) -> Game {
        let mut g = self.clone();
        let mut pool = unseen(self, me);
        let q_of: [u8; 3] = [(me + 1) & 3, (me + 2) & 3, (me + 3) & 3];
        let mut new_hands = [[0u8; 54]; 4];
        let mut left = [0u32; 4];
        for &q in &q_of {
            left[q as usize] = hand_count(&self.hands[q as usize]);
        }
        // 1. Cards known to sit in a hand (received through tribute / return and not played)
        for t in &self.tribute {
            let q = t.to;
            if q == me {
                continue;
            }
            let c = t.card as usize;
            let known = (1u8).saturating_sub(self.played[q as usize][c].min(1));
            if known > 0 && pool[c] > 0 && left[q as usize] > 0 && new_hands[q as usize][c] < 2 {
                new_hands[q as usize][c] += 1;
                pool[c] -= 1;
                left[q as usize] -= 1;
            }
        }
        let total_left: u32 = q_of.iter().map(|&q| left[q as usize]).sum();
        if total_left == 0 {
            for &q in &q_of {
                g.hands[q as usize] = new_hands[q as usize];
            }
            g.rng = Rng::new(rng.next_u64());
            return g;
        }
        // Uniform fast path when neither belief logits nor expectations are provided
        if expect.is_none() && logits.is_none() {
            let mut rest: Vec<u8> = Vec::with_capacity(81);
            for c in 0..54u8 {
                for _ in 0..pool[c as usize] {
                    rest.push(c);
                }
            }
            rng.shuffle(&mut rest);
            let mut k = 0;
            for &q in &q_of {
                for _ in 0..left[q as usize] {
                    if k < rest.len() {
                        new_hands[q as usize][rest[k] as usize] += 1;
                        k += 1;
                    }
                }
            }
            for &q in &q_of {
                g.hands[q as usize] = new_hands[q as usize];
            }
            g.rng = Rng::new(rng.next_u64());
            return g;
        }

        // 2. Hard tribute / anti-tribute deductions + belief probabilities (p1, p2)
        let forbidden = self.tribute_forbidden(me);
        let sig = |x: f32| 1.0 / (1.0 + (-x.clamp(-30.0, 30.0)).exp());
        let mut e = [[0f64; 54]; 3];
        let mut rho = [[0f64; 54]; 3];
        let mut q1_only = [[0f64; 54]; 3];
        let fl = floor.max(1e-4) as f64;
        let tot_f = total_left as f64;

        for r in 0..3 {
            let q = q_of[r] as usize;
            if left[q] == 0 {
                continue;
            }
            let cap_share = left[q] as f64 / tot_f;
            let eps = fl * cap_share;
            let unif_rho = if total_left > 1 && left[q] > 1 {
                (left[q] - 1) as f64 / (total_left - 1) as f64
            } else {
                0.0
            };
            for t in 0..54 {
                let pc = pool[t];
                if pc == 0 || forbidden[r][t] || new_hands[q][t] >= 2 {
                    continue;
                }
                let (p1, p2) = if let Some(lg) = logits {
                    let base = (r * 54 + t) * 2;
                    let a = sig(lg[base]) as f64;
                    let b = (sig(lg[base + 1]) as f64).min(a);
                    (a, b)
                } else {
                    let ex = expect.unwrap()[r][t].clamp(0.0, 2.0) as f64;
                    // Reconstruct smooth (p1, p2) consistent with expected count `ex`
                    // and hypergeometric pair clustering when only `expect` was supplied
                    let p2_est = if ex >= 1.0 {
                        unif_rho * ex * 0.5 + (ex - 1.0) * (1.0 - unif_rho * 0.5)
                    } else {
                        (0.5 * unif_rho * ex * ex).min(0.5 * ex)
                    };
                    let p2_c = p2_est.clamp(0.0, 0.5 * ex);
                    let p1_c = (ex - p2_c).clamp(p2_c, 1.0);
                    (p1_c, p2_c)
                };
                if new_hands[q][t] == 1 {
                    let rem_p = if p1 > 1e-4 { (p2 / p1).min(1.0) } else { 0.0 };
                    e[r][t] = rem_p + eps;
                    q1_only[r][t] = rem_p + eps;
                    rho[r][t] = 0.0;
                } else if pc == 1 {
                    e[r][t] = p1 + eps;
                    q1_only[r][t] = p1 + eps;
                    rho[r][t] = 0.0;
                } else {
                    let w_net = p1 + p2;
                    e[r][t] = w_net + 2.0 * eps;
                    let net_rho = if w_net > 1e-5 { (2.0 * p2 / w_net).min(1.0) } else { unif_rho };
                    rho[r][t] = (w_net * net_rho + 2.0 * eps * unif_rho) / (w_net + 2.0 * eps);
                    q1_only[r][t] = (p1 - p2).max(0.0) + eps * (1.0 - unif_rho);
                }
            }
        }

        // Fallback if all active seats were marked forbidden for some card t
        for t in 0..54 {
            if pool[t] > 0 && (e[0][t] + e[1][t] + e[2][t]) <= 1e-12 {
                for r in 0..3 {
                    let q = q_of[r] as usize;
                    if left[q] > 0 && new_hands[q][t] < 2 {
                        e[r][t] = (left[q] as f64 / tot_f) * pool[t] as f64;
                        q1_only[r][t] = e[r][t];
                        rho[r][t] = if total_left > 1 && left[q] > 1 {
                            (left[q] - 1) as f64 / (total_left - 1) as f64
                        } else {
                            0.0
                        };
                    }
                }
            }
        }

        // 3. Sinkhorn-Knopp IPF (6 iterations): match column sums = pool[t] and row sums = left[q]
        for _ in 0..6 {
            for t in 0..54 {
                if pool[t] == 0 {
                    continue;
                }
                let cs = e[0][t] + e[1][t] + e[2][t];
                if cs > 1e-12 {
                    let scale = pool[t] as f64 / cs;
                    for r in 0..3 {
                        let q = q_of[r] as usize;
                        let cap = (pool[t] as u32).min(left[q]).min((2 - new_hands[q][t]) as u32) as f64;
                        e[r][t] = (e[r][t] * scale).min(cap);
                    }
                }
            }
            for r in 0..3 {
                let q = q_of[r] as usize;
                if left[q] == 0 {
                    continue;
                }
                let rs: f64 = e[r].iter().sum();
                if rs > 1e-12 {
                    let scale = left[q] as f64 / rs;
                    for t in 0..54 {
                        if pool[t] > 0 {
                            e[r][t] *= scale;
                        }
                    }
                }
            }
        }
        for t in 0..54 {
            if pool[t] > 0 {
                let cs = e[0][t] + e[1][t] + e[2][t];
                if cs > 1e-12 {
                    let scale = pool[t] as f64 / cs;
                    for r in 0..3 {
                        e[r][t] *= scale;
                    }
                }
            }
        }

        // 4. Order unseen card types by highest concentration first (with small random tie-break)
        let mut rem_e = [0f64; 3];
        for r in 0..3 {
            rem_e[r] = e[r].iter().sum();
        }
        let mut order: [(f64, u8); 54] = [(0.0, 0); 54];
        let mut n_ord = 0usize;
        for t in 0..54u8 {
            let pc = pool[t as usize];
            if pc == 0 {
                continue;
            }
            let pcf = pc as f64;
            let mx = (e[0][t as usize] / pcf)
                .max(e[1][t as usize] / pcf)
                .max(e[2][t as usize] / pcf);
            order[n_ord] = (mx + rng.uniform() * 0.02, t);
            n_ord += 1;
        }
        order[..n_ord].sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let pick_r = |w: &[f64; 3], left: &[u32; 4], t: usize, new_hands: &[[u8; 54]; 4], rng: &mut Rng| -> usize {
            let sum = w[0] + w[1] + w[2];
            if sum > 0.0 {
                let mut x = rng.uniform() * sum;
                let mut last_ok = 0;
                for r in 0..3 {
                    if w[r] > 0.0 {
                        last_ok = r;
                        if x < w[r] {
                            return r;
                        }
                        x -= w[r];
                    }
                }
                return last_ok;
            }
            let mut avail = [0usize; 3];
            let mut na = 0;
            for r in 0..3 {
                let q = q_of[r] as usize;
                if left[q] > 0 && new_hands[q][t] < 2 {
                    avail[na] = r;
                    na += 1;
                }
            }
            if na == 0 {
                for r in 0..3 {
                    if left[q_of[r] as usize] > 0 {
                        return r;
                    }
                }
                0
            } else {
                avail[rng.below(na as u64) as usize]
            }
        };

        // 5. Sample owners with capacity-ratio tracking and exact 2-copy Bayes correlation
        for &(_, t_u8) in &order[..n_ord] {
            let t = t_u8 as usize;
            let pc = pool[t];
            let mut cap_adj = [0f64; 3];
            for r in 0..3 {
                let q = q_of[r] as usize;
                if left[q] > 0 && new_hands[q][t] < 2 {
                    cap_adj[r] = left[q] as f64 / rem_e[r].max(0.05);
                }
            }
            if pc == 1 {
                let w = [e[0][t] * cap_adj[0], e[1][t] * cap_adj[1], e[2][t] * cap_adj[2]];
                let r1 = pick_r(&w, &left, t, &new_hands, rng);
                let q1 = q_of[r1] as usize;
                new_hands[q1][t] += 1;
                left[q1] = left[q1].saturating_sub(1);
            } else {
                let w1 = [0.5 * e[0][t] * cap_adj[0], 0.5 * e[1][t] * cap_adj[1], 0.5 * e[2][t] * cap_adj[2]];
                let r1 = pick_r(&w1, &left, t, &new_hands, rng);
                let q1 = q_of[r1] as usize;
                new_hands[q1][t] += 1;
                left[q1] = left[q1].saturating_sub(1);

                let mut w2 = [0f64; 3];
                let mut p_same = 0.0;
                if left[q1] > 0 && new_hands[q1][t] < 2 {
                    let rem_after = (rem_e[r1] - 0.5 * e[r1][t]).max(0.05);
                    p_same = (rho[r1][t] * (left[q1] as f64 / rem_after)).min(0.98);
                }
                let mut other_sum = 0.0;
                let mut other_cnt = 0usize;
                for r in 0..3 {
                    let q = q_of[r] as usize;
                    if r != r1 && left[q] > 0 && new_hands[q][t] < 2 {
                        let rem_after = (rem_e[r] - 0.5 * e[r][t]).max(0.05);
                        w2[r] = q1_only[r][t].max(1e-6) * (left[q] as f64 / rem_after);
                        other_sum += w2[r];
                        other_cnt += 1;
                    }
                }
                if other_cnt == 0 {
                    w2[r1] = 1.0;
                } else {
                    w2[r1] = p_same;
                    let scale = (1.0 - p_same) / other_sum.max(1e-12);
                    for r in 0..3 {
                        if r != r1 {
                            w2[r] *= scale;
                        }
                    }
                }
                let r2 = pick_r(&w2, &left, t, &new_hands, rng);
                let q2 = q_of[r2] as usize;
                new_hands[q2][t] += 1;
                left[q2] = left[q2].saturating_sub(1);
            }
            for r in 0..3 {
                rem_e[r] = (rem_e[r] - e[r][t]).max(0.0);
            }
        }

        for &q in &q_of {
            g.hands[q as usize] = new_hands[q as usize];
        }
        g.rng = Rng::new(rng.next_u64());
        g
    }
}

/// Encoded decisions of other players, replayed in one (possibly sampled) world.
pub struct ReplayObs {
    pub players: Vec<u8>,
    pub states: Vec<u8>,
    pub hists: Vec<u8>,
    pub actions: Vec<u8>,
    pub offsets: Vec<u32>,
    /// index of the move actually played among the legal moves, or u32::MAX if the hand
    /// of this world could not have produced it
    pub chosen: Vec<u32>,
}

impl Game {
    /// The position at the start of this round's play phase, rebuilt from the current hands
    /// plus every card played since (all of them are in `history`).
    pub fn play_start(&self) -> Option<Game> {
        let first = self.history.first()?.0;
        let mut g = self.clone();
        for p in 0..4 {
            for t in 0..54 {
                g.hands[p][t] += self.played[p][t];
            }
        }
        g.played = [[0; 54]; 4];
        g.history.clear();
        g.finished.clear();
        g.last_move = None;
        g.passes = 0;
        g.current = first;
        g.last_player = first;
        g.trick_id = 0;
        g.bombs = [0; 4];
        g.jiefeng = false;
        g.phase = Phase::Play;
        g.winner = -1;
        g.last_result = None;
        Some(g)
    }

    /// Replay this round's plays from `play_start` and encode the last `last` decisions of
    /// players other than `me` that had more than one legal move.
    pub fn replay_decisions(&self, me: u8, last: usize) -> ReplayObs {
        let mut out = ReplayObs {
            players: vec![],
            states: vec![],
            hists: vec![],
            actions: vec![],
            offsets: vec![0],
            chosen: vec![],
        };
        let mut g = match self.play_start() {
            Some(g) => g,
            None => return out,
        };
        let n = self.history.len();
        // which history steps to encode: the last `last` decisions of other players
        let mut keep = vec![false; n];
        let mut k = 0;
        for i in (0..n).rev() {
            if k >= last {
                break;
            }
            if self.history[i].0 != me {
                keep[i] = true;
                k += 1;
            }
        }
        for (i, (p, m)) in self.history.iter().enumerate() {
            if keep[i] {
                let mv = g.legal_moves();
                if mv.len() > 1 {
                    let idx = mv
                        .iter()
                        .position(|x| x.kind == m.kind && x.key == m.key && x.cards == m.cards)
                        .map(|x| x as u32)
                        .unwrap_or(u32::MAX);
                    let base_s = out.states.len();
                    out.states.resize(base_s + STATE_DIM, 0);
                    encode_state(&g, *p, &mut out.states[base_s..]);
                    let base_h = out.hists.len();
                    out.hists.resize(base_h + HIST_LEN * HIST_DIM, 0);
                    encode_history(&g, *p, &mut out.hists[base_h..]);
                    let base_a = out.actions.len();
                    out.actions.resize(base_a + mv.len() * ACTION_DIM, 0);
                    for (j, x) in mv.iter().enumerate() {
                        let o = base_a + j * ACTION_DIM;
                        encode_action(&g, *p, x, &mut out.actions[o..o + ACTION_DIM]);
                    }
                    let last_off = *out.offsets.last().unwrap();
                    out.offsets.push(last_off + mv.len() as u32);
                    out.players.push(*p);
                    out.chosen.push(idx);
                }
            }
            if g.current != *p || g.apply(m).is_err() {
                break; // this world cannot reproduce the history
            }
        }
        out
    }
}

/// A batch of games played forward until each one ends its round.
pub struct Rollouts {
    pub games: Vec<Game>,
    pub done: Vec<bool>,
    active: Vec<usize>,
    moves: Vec<Vec<Move>>,
}

impl Rollouts {
    pub fn new(games: Vec<Game>) -> Rollouts {
        let n = games.len();
        let done = games.iter().map(|g| g.phase == Phase::RoundOver || g.phase == Phase::MatchOver).collect();
        Rollouts { games, done, active: vec![], moves: vec![vec![]; n] }
    }

    /// Features of all unfinished games: (indices, players, states, hists, actions, offsets).
    pub fn observe(&mut self) -> (Vec<u32>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u32>) {
        self.active = (0..self.games.len()).filter(|&i| !self.done[i]).collect();
        let n = self.active.len();
        let mut idx = Vec::with_capacity(n);
        let mut players = Vec::with_capacity(n);
        let mut s = vec![0u8; n * STATE_DIM];
        let mut h = vec![0u8; n * HIST_LEN * HIST_DIM];
        let mut a: Vec<u8> = Vec::new();
        let mut off = vec![0u32];
        for (j, &i) in self.active.iter().enumerate() {
            let g = &self.games[i];
            let me = g.current;
            let mv = g.legal_moves();
            encode_state(g, me, &mut s[j * STATE_DIM..(j + 1) * STATE_DIM]);
            encode_history(g, me, &mut h[j * HIST_LEN * HIST_DIM..(j + 1) * HIST_LEN * HIST_DIM]);
            let base = a.len();
            a.resize(base + mv.len() * ACTION_DIM, 0);
            for (k, m) in mv.iter().enumerate() {
                let o = base + k * ACTION_DIM;
                encode_action(g, me, m, &mut a[o..o + ACTION_DIM]);
            }
            off.push(off[j] + mv.len() as u32);
            idx.push(i as u32);
            players.push(me);
            self.moves[i] = mv;
        }
        (idx, players, s, h, a, off)
    }

    /// Apply one move per game returned by the last `observe`.
    pub fn step(&mut self, choices: &[u32]) {
        let active = std::mem::take(&mut self.active);
        for (j, &i) in active.iter().enumerate() {
            let mv = &self.moves[i];
            if mv.is_empty() {
                self.done[i] = true;
                continue;
            }
            let c = (choices[j] as usize).min(mv.len() - 1);
            let m = mv[c].clone();
            let g = &mut self.games[i];
            if g.apply(&m).is_err() {
                self.done[i] = true;
                continue;
            }
            if g.phase == Phase::RoundOver || g.phase == Phase::MatchOver {
                self.done[i] = true;
            }
        }
    }

    /// Choices of the rule bot for the games returned by the last `observe`.
    pub fn rule_choices(&self) -> Vec<u32> {
        self.active
            .iter()
            .map(|&i| crate::bot::rule_choose(&self.games[i], &self.moves[i]) as u32)
            .collect()
    }

    /// Score of finished games for `team` under `rules` (NaN for unfinished ones).
    pub fn scores(&self, team: u8, rules: &ScoreRules) -> Vec<f32> {
        (0..self.games.len())
            .map(|i| if self.done[i] { self.games[i].round_score(team, rules) } else { f32::NAN })
            .collect()
    }

    /// Current multiplier of every game.
    pub fn multipliers(&self, rules: &ScoreRules) -> Vec<u32> {
        self.games.iter().map(|g| rules.multiplier(&g.bombs)).collect()
    }

    /// Outcome of finished games for `team` (NaN for unfinished ones).
    pub fn rewards(&self, team: u8) -> Vec<f32> {
        (0..self.games.len())
            .map(|i| if self.done[i] { self.games[i].round_reward(team) } else { f32::NAN })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn determinize_keeps_counts_and_own_hand() {
        let mut rng = Rng::new(5);
        for seed in 0..30u64 {
            let mut g = Game::new(seed, Rules::default());
            // play a few random moves
            for _ in 0..(seed as usize % 40) {
                if g.phase != Phase::Play && g.phase != Phase::Return {
                    break;
                }
                let mv = g.legal_moves();
                let m = mv[rng.below(mv.len() as u64) as usize].clone();
                g.apply(&m).unwrap();
            }
            if g.phase != Phase::Play {
                continue;
            }
            let me = g.current;
            let d = g.determinize(me, &mut rng);
            assert_eq!(d.hands[me as usize], g.hands[me as usize]);
            for p in 0..4 {
                assert_eq!(hand_count(&d.hands[p]), hand_count(&g.hands[p]));
            }
            let total: u32 = (0..4).map(|p| hand_count(&d.hands[p]) + hand_count(&d.played[p])).sum();
            assert_eq!(total, 108);
            for t in 0..54 {
                let n: u8 = (0..4).map(|p| d.hands[p][t] + d.played[p][t]).sum();
                assert_eq!(n, 2);
            }
        }
    }

    #[test]
    fn replay_reproduces_the_round() {
        let mut rng = Rng::new(9);
        for seed in 0..30u64 {
            let mut g = Game::new(seed, Rules::default());
            for _ in 0..(10 + seed as usize * 3) {
                if g.phase != Phase::Play {
                    break;
                }
                let mv = g.legal_moves();
                let m = mv[rng.below(mv.len() as u64) as usize].clone();
                g.apply(&m).unwrap();
            }
            if g.phase != Phase::Play {
                continue;
            }
            let mut r = g.play_start().unwrap();
            for (_, m) in g.history.clone() {
                r.apply(&m).unwrap();
            }
            assert_eq!(r.hands, g.hands);
            assert_eq!(r.played, g.played);
            assert_eq!(r.current, g.current);
            assert_eq!(r.finished, g.finished);
            let me = g.current;
            let obs = g.replay_decisions(me, 1000);
            assert!(obs.chosen.iter().all(|&c| c != u32::MAX));
            assert_eq!(obs.offsets.len(), obs.chosen.len() + 1);
            // a sampled world replays the same history (its legal move sets differ)
            let w = g.determinize(me, &mut rng);
            let mut rw = w.play_start().unwrap();
            for (_, m) in w.history.clone() {
                rw.apply(&m).unwrap();
            }
            assert_eq!(rw.hands, w.hands);
            let ow = w.replay_decisions(me, 1000);
            assert_eq!(ow.offsets.len(), ow.chosen.len() + 1);
        }
    }

    #[test]
    fn weighted_determinize_keeps_counts_and_follows_beliefs() {
        let mut rng = Rng::new(17);
        let g = Game::new(4, Rules::default());
        let me = g.current;
        // belief: the next player holds exactly what they really hold
        let mut e = [[0f32; 54]; 3];
        for r in 0..3usize {
            let q = (me as usize + r + 1) % 4;
            for t in 0..54 {
                e[r][t] = g.hands[q][t] as f32;
            }
        }
        let mut hits = 0u32;
        let mut total = 0u32;
        for _ in 0..20 {
            let d = g.determinize_weighted(me, &e, 0.01, &mut rng);
            assert_eq!(d.hands[me as usize], g.hands[me as usize]);
            for p in 0..4 {
                assert_eq!(hand_count(&d.hands[p]), hand_count(&g.hands[p]));
            }
            for t in 0..54 {
                let n: u8 = (0..4).map(|p| d.hands[p][t]).sum();
                let m: u8 = (0..4).map(|p| g.hands[p][t]).sum();
                assert_eq!(n, m);
            }
            for p in 0..4 {
                if p as u8 == me {
                    continue;
                }
                for t in 0..54 {
                    hits += d.hands[p][t].min(g.hands[p][t]) as u32;
                }
                total += hand_count(&g.hands[p]);
            }
        }
        // with a perfect belief almost every card lands with its real owner
        assert!(hits as f32 / total as f32 > 0.9, "{} / {}", hits, total);
    }

    #[test]
    fn belief_determinize_preserves_pairs_and_tribute_deductions() {
        let mut rng = Rng::new(99);
        // Play until round 2 so tribute / anti-tribute occurs
        for seed in 1..40u64 {
            let mut g = Game::new(seed, Rules::default());
            while g.phase != Phase::RoundOver {
                let mv = g.legal_moves();
                g.apply(&mv[rng.below(mv.len() as u64) as usize]).unwrap();
            }
            g.next_round();
            while g.phase == Phase::Return {
                let mv = g.legal_moves();
                g.apply(&mv[0]).unwrap();
            }
            if g.tribute.is_empty() {
                continue;
            }
            let me = g.current;
            let logits = vec![0.0f32; HIDDEN_DIM]; // neutral logits (p1 = 0.5, p2 = 0.5 -> clamped to p1)
            for _ in 0..15 {
                let d = g.determinize_belief(me, &logits, 0.02, &mut rng);
                for p in 0..4 {
                    assert_eq!(hand_count(&d.hands[p]), hand_count(&g.hands[p]));
                }
                // Check tribute payer upper-bound deduction
                for tr in &g.tribute {
                    if !tr.is_return && tr.from != me {
                        let s_trib = strength(rank_of(tr.card), g.level);
                        let wt = wild_type(g.level);
                        for c in 0..54u8 {
                            if c != wt && strength(rank_of(c), g.level) > s_trib {
                                // Only allowed if received in return (return cards are <= 10, except level card <= 10)
                                let got_ret = g.tribute.iter().any(|x| x.is_return && x.to == tr.from && x.card == c);
                                if !got_ret {
                                    assert_eq!(d.hands[tr.from as usize][c as usize], 0, "violated tribute upper bound");
                                }
                            }
                        }
                        if tr.card == BJ && !g.anti_tribute {
                            assert_eq!(d.hands[tr.from as usize][BJ as usize], 0, "payer held 2nd BJ without anti-tribute");
                        }
                    }
                }
            }
            return;
        }
        panic!("no tribute round tested");
    }

    #[test]
    fn rollouts_finish() {
        let games: Vec<Game> = (0..8).map(|s| Game::new(s, Rules::default())).collect();
        let mut r = Rollouts::new(games);
        let mut rng = Rng::new(1);
        for _ in 0..2000 {
            let (idx, _, _, _, _, off) = r.observe();
            if idx.is_empty() {
                break;
            }
            let ch: Vec<u32> = (0..idx.len()).map(|j| rng.below((off[j + 1] - off[j]) as u64) as u32).collect();
            r.step(&ch);
        }
        assert!(r.done.iter().all(|&d| d));
        assert!(r.rewards(0).iter().all(|x| x.abs() >= 1.0));
    }
}
