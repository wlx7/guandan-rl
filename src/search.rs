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
        let mut k = 0;
        for &q in &others {
            for _ in 0..left[q as usize] {
                if k < rest.len() {
                    new_hands[q as usize][rest[k] as usize] += 1;
                    k += 1;
                }
            }
            g.hands[q as usize] = new_hands[q as usize];
        }
        g.rng = Rng::new(rng.next_u64());
        g
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
