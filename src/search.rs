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
    /// ruled out completely. Card counts and known tribute cards are respected exactly.
    pub fn determinize_weighted(&self, me: u8, expect: &[[f32; 54]; 3], floor: f32, rng: &mut Rng) -> Game {
        self.determinize_with(me, Some(expect), floor, rng)
    }

    fn determinize_with(&self, me: u8, expect: Option<&[[f32; 54]; 3]>, floor: f32, rng: &mut Rng) -> Game {
        let mut g = self.clone();
        let mut pool = unseen(self, me);
        let others: Vec<u8> = (0..4u8).filter(|&p| p != me).collect();
        let mut new_hands = [[0u8; 54]; 4];
        let mut left = [0u32; 4];
        for &q in &others {
            left[q as usize] = hand_count(&self.hands[q as usize]);
        }
        // cards known to sit in a hand
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
        let mut rest: Vec<u8> = Vec::with_capacity(81);
        for c in 0..54u8 {
            for _ in 0..pool[c as usize] {
                rest.push(c);
            }
        }
        rng.shuffle(&mut rest);
        match expect {
            None => {
                let mut k = 0;
                for &q in &others {
                    for _ in 0..left[q as usize] {
                        if k < rest.len() {
                            new_hands[q as usize][rest[k] as usize] += 1;
                            k += 1;
                        }
                    }
                }
            }
            Some(e) => {
                for &c in &rest {
                    let mut w = [0f32; 4];
                    let mut total = 0f32;
                    for &q in &others {
                        if left[q as usize] == 0 {
                            continue;
                        }
                        let r = ((q + 4 - me) % 4 - 1) as usize;
                        let want = (e[r][c as usize] - new_hands[q as usize][c as usize] as f32).max(0.0);
                        w[q as usize] = want + floor.max(1e-4);
                        total += w[q as usize];
                    }
                    let mut x = rng.uniform() as f32 * total;
                    let mut pick = others[0];
                    for &q in &others {
                        if w[q as usize] <= 0.0 {
                            continue;
                        }
                        pick = q;
                        if x < w[q as usize] {
                            break;
                        }
                        x -= w[q as usize];
                    }
                    new_hands[pick as usize][c as usize] += 1;
                    left[pick as usize] -= 1;
                }
            }
        }
        for &q in &others {
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
