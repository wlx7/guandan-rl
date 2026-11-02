//! Vectorised environment used by self-play actors and evaluators. Many independent
//! matches run side by side so that one batched network call serves all of them.

use crate::cards::*;
use crate::features::*;
use crate::game::*;
use crate::moves::*;

struct Step {
    player: u8,
    state: Vec<u8>,
    hist: Vec<u8>,
    action: Vec<u8>,
    hidden: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct RoundInfo {
    pub env: usize,
    pub win_team: u8,
    pub gain: u8,
    pub match_over: bool,
}

pub struct VecEnv {
    pub games: Vec<Game>,
    moves: Vec<Vec<Move>>,
    st: Vec<Vec<u8>>,
    hs: Vec<Vec<u8>>,
    acts: Vec<Vec<u8>>,
    valid: Vec<bool>,
    traj: Vec<Vec<Step>>,
    pub out_s: Vec<u8>,
    pub out_h: Vec<u8>,
    pub out_a: Vec<u8>,
    pub out_t: Vec<f32>,
    pub out_x: Vec<u8>,
    record_hidden: bool,
    /// Seats whose decisions become training samples (per environment). Seats played by
    /// an older opponent model are excluded: their values belong to a different policy.
    learners: Vec<[bool; 4]>,
    rng: Rng,
    random_levels: bool,
}

impl VecEnv {
    pub fn new(n: usize, seed: u64, random_levels: bool, rules: Rules) -> VecEnv {
        let mut rng = Rng::new(seed);
        let mut games = Vec::with_capacity(n);
        for _ in 0..n {
            let mut g = Game::new(rng.next_u64(), rules.clone());
            if random_levels {
                Self::randomize_levels(&mut rng, &mut g);
            }
            games.push(g);
        }
        VecEnv {
            games,
            moves: vec![vec![]; n],
            st: vec![vec![0; STATE_DIM]; n],
            hs: vec![vec![0; HIST_LEN * HIST_DIM]; n],
            acts: vec![vec![]; n],
            valid: vec![false; n],
            traj: (0..n).map(|_| Vec::new()).collect(),
            out_s: vec![],
            out_h: vec![],
            out_a: vec![],
            out_t: vec![],
            out_x: vec![],
            record_hidden: false,
            learners: vec![[true; 4]; n],
            rng,
            random_levels,
        }
    }

    fn randomize_levels(rng: &mut Rng, g: &mut Game) {
        // half of the matches start from random levels so that late-game levels
        // (including passing A) are seen often during training
        if rng.uniform() < 0.5 {
            let a = rng.below(13) as u8;
            let b = rng.below(13) as u8;
            let banker = rng.below(2) as i8;
            g.set_levels([a, b], banker);
            g.round_no = 0;
            g.prev_order = None;
            g.start_round();
        }
    }

    pub fn len(&self) -> usize {
        self.games.len()
    }

    fn refresh(&mut self, i: usize) {
        if self.valid[i] {
            return;
        }
        let g = &self.games[i];
        let me = g.current;
        let mv = g.legal_moves();
        encode_state(g, me, &mut self.st[i]);
        encode_history(g, me, &mut self.hs[i]);
        let mut a = vec![0u8; mv.len() * ACTION_DIM];
        for (k, m) in mv.iter().enumerate() {
            encode_action(g, me, m, &mut a[k * ACTION_DIM..(k + 1) * ACTION_DIM]);
        }
        self.acts[i] = a;
        self.moves[i] = mv;
        self.valid[i] = true;
    }

    /// Returns (players, states, hists, actions, offsets).
    pub fn observe(&mut self) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u32>) {
        let n = self.len();
        for i in 0..n {
            self.refresh(i);
        }
        let mut players = Vec::with_capacity(n);
        let mut s = Vec::with_capacity(n * STATE_DIM);
        let mut h = Vec::with_capacity(n * HIST_LEN * HIST_DIM);
        let total: usize = self.moves.iter().map(|m| m.len()).sum();
        let mut a = Vec::with_capacity(total * ACTION_DIM);
        let mut off = Vec::with_capacity(n + 1);
        off.push(0u32);
        for i in 0..n {
            players.push(self.games[i].current);
            s.extend_from_slice(&self.st[i]);
            h.extend_from_slice(&self.hs[i]);
            a.extend_from_slice(&self.acts[i]);
            off.push(off[i] + self.moves[i].len() as u32);
        }
        (players, s, h, a, off)
    }

    pub fn num_moves(&self, i: usize) -> usize {
        self.moves[i].len()
    }

    pub fn rule_choices(&mut self) -> Vec<u32> {
        let n = self.len();
        (0..n)
            .map(|i| {
                self.refresh(i);
                crate::bot::rule_choose(&self.games[i], &self.moves[i]) as u32
            })
            .collect()
    }

    pub fn step(&mut self, choices: &[u32], record: bool) -> Vec<RoundInfo> {
        let mut results = Vec::new();
        for i in 0..self.len() {
            self.refresh(i);
            let c = (choices[i] as usize).min(self.moves[i].len() - 1);
            let me = self.games[i].current;
            if record && self.moves[i].len() > 1 && self.learners[i][me as usize] {
                self.traj[i].push(Step {
                    player: me,
                    state: self.st[i].clone(),
                    hist: self.hs[i].clone(),
                    action: self.acts[i][c * ACTION_DIM..(c + 1) * ACTION_DIM].to_vec(),
                    hidden: if self.record_hidden {
                        let mut x = vec![0u8; HIDDEN_DIM];
                        encode_hidden(&self.games[i], me, &mut x);
                        x
                    } else {
                        vec![]
                    },
                });
            }
            let mv = self.moves[i][c].clone();
            self.games[i].apply(&mv).expect("engine produced an illegal move");
            self.valid[i] = false;
            let ph = self.games[i].phase;
            if ph == Phase::RoundOver || ph == Phase::MatchOver {
                let g = &self.games[i];
                let r = g.last_result.as_ref().unwrap();
                results.push(RoundInfo { env: i, win_team: r.win_team, gain: r.gain, match_over: r.match_over });
                let rew = [g.round_reward(0), g.round_reward(1)];
                for st in self.traj[i].drain(..) {
                    self.out_s.extend_from_slice(&st.state);
                    self.out_h.extend_from_slice(&st.hist);
                    self.out_a.extend_from_slice(&st.action);
                    self.out_t.push(rew[team(st.player) as usize]);
                    if self.record_hidden {
                        self.out_x.extend_from_slice(&st.hidden);
                    }
                }
                if ph == Phase::MatchOver {
                    self.games[i].new_match();
                    if self.random_levels {
                        Self::randomize_levels(&mut self.rng, &mut self.games[i]);
                    }
                } else {
                    self.games[i].next_round();
                }
            }
        }
        results
    }

    pub fn num_samples(&self) -> usize {
        self.out_t.len()
    }

    /// (n, states, hists, actions, targets) of finished trajectories.
    pub fn take_samples(&mut self) -> (usize, Vec<u8>, Vec<u8>, Vec<u8>, Vec<f32>) {
        let n = self.out_t.len();
        (
            n,
            std::mem::take(&mut self.out_s),
            std::mem::take(&mut self.out_h),
            std::mem::take(&mut self.out_a),
            std::mem::take(&mut self.out_t),
        )
    }

    /// Also record the other players' hands of every sample (auxiliary training labels),
    /// returned by `take_hidden` in the same order as `take_samples`.
    pub fn set_record_hidden(&mut self, on: bool) {
        self.record_hidden = on;
    }

    pub fn take_hidden(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.out_x)
    }

    /// Choose which seats of environment `i` produce training samples from now on.
    pub fn set_learners(&mut self, i: usize, seats: [bool; 4]) {
        self.learners[i] = seats;
    }

    /// Seat to move in every environment.
    pub fn current_players(&self) -> Vec<u8> {
        self.games.iter().map(|g| g.current).collect()
    }

    /// Hand counts left for each seat (for logging).
    pub fn hand_counts(&self, i: usize) -> [u32; 4] {
        let g = &self.games[i];
        [0, 1, 2, 3].map(|p| hand_count(&g.hands[p]))
    }
}
