use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::cards::*;
use crate::features::*;
use crate::game::*;
use crate::moves::*;
use crate::search::Rollouts;
use crate::venv::VecEnv;

type PyMove = (u8, u8, Vec<u8>);

fn to_py(m: &Move) -> PyMove {
    (m.kind as u8, m.key, m.cards.clone())
}

fn from_py(kind: u8, key: u8, cards: Vec<u8>) -> PyResult<Move> {
    let k = Kind::from_u8(kind).ok_or_else(|| PyValueError::new_err("bad kind"))?;
    if cards.iter().any(|&c| c >= 54) {
        return Err(PyValueError::new_err("bad card"));
    }
    Ok(Move::new(k, key, cards))
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

#[pyclass(name = "VecEnv", module = "guandan._core")]
pub struct PyVecEnv {
    inner: VecEnv,
}

#[pymethods]
impl PyVecEnv {
    #[new]
    #[pyo3(signature = (n, seed, random_levels = true, a_fail_limit = 3, tribute = true))]
    fn new(n: usize, seed: u64, random_levels: bool, a_fail_limit: u8, tribute: bool) -> Self {
        PyVecEnv { inner: VecEnv::new(n, seed, random_levels, Rules { a_fail_limit, tribute }) }
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Returns (players, states, hists, actions, offsets); arrays are raw uint8 bytes.
    fn observe<'py>(
        &mut self,
        py: Python<'py>,
    ) -> (Bound<'py, PyBytes>, Bound<'py, PyBytes>, Bound<'py, PyBytes>, Bound<'py, PyBytes>, Vec<u32>) {
        let (p, s, h, a, o) = self.inner.observe();
        (PyBytes::new(py, &p), PyBytes::new(py, &s), PyBytes::new(py, &h), PyBytes::new(py, &a), o)
    }

    #[pyo3(signature = (choices, record = true))]
    fn step(&mut self, choices: Vec<u32>, record: bool) -> PyResult<Vec<(usize, u8, u8, bool)>> {
        if choices.len() != self.inner.len() {
            return Err(PyValueError::new_err("choices length mismatch"));
        }
        Ok(self
            .inner
            .step(&choices, record)
            .into_iter()
            .map(|r| (r.env, r.win_team, r.gain, r.match_over))
            .collect())
    }

    /// Like `step` with recording; `q[i]` is the network's value (in levels) of the chosen
    /// move. Used for TD(lambda) targets, see `set_td_lambda`.
    fn step_q(&mut self, choices: Vec<u32>, q: Vec<f32>) -> PyResult<Vec<(usize, u8, u8, bool)>> {
        if choices.len() != self.inner.len() || q.len() != self.inner.len() {
            return Err(PyValueError::new_err("choices / q length mismatch"));
        }
        Ok(self
            .inner
            .step_q(&choices, Some(&q), true)
            .into_iter()
            .map(|r| (r.env, r.win_team, r.gain, r.match_over))
            .collect())
    }

    /// 1.0 (default): training targets are round outcomes. Below 1: lambda-returns that mix
    /// the outcome with the values passed to `step_q` at the same player's later decisions.
    fn set_td_lambda(&mut self, lambda: f32) {
        self.inner.set_td_lambda(lambda);
    }

    fn rule_choices(&mut self) -> Vec<u32> {
        self.inner.rule_choices()
    }

    fn num_samples(&self) -> usize {
        self.inner.num_samples()
    }

    /// Returns (n, states, hists, actions, targets_f32_le).
    fn take_samples<'py>(
        &mut self,
        py: Python<'py>,
    ) -> (usize, Bound<'py, PyBytes>, Bound<'py, PyBytes>, Bound<'py, PyBytes>, Bound<'py, PyBytes>) {
        let (n, s, h, a, t) = self.inner.take_samples();
        (n, PyBytes::new(py, &s), PyBytes::new(py, &h), PyBytes::new(py, &a), PyBytes::new(py, &f32_bytes(&t)))
    }

    /// Record the other players' hands with every sample (auxiliary labels, see take_hidden).
    fn set_record_hidden(&mut self, on: bool) {
        self.inner.set_record_hidden(on);
    }

    /// Hidden-hand labels (324 bits per sample) of the samples returned by the last take_samples.
    fn take_hidden<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.take_hidden())
    }

    /// Only the given seats of environment `i` produce training samples from now on.
    fn set_learners(&mut self, i: usize, seats: [bool; 4]) {
        self.inner.set_learners(i, seats);
    }

    /// Seat to move in every environment.
    fn current_players(&self) -> Vec<u8> {
        self.inner.current_players()
    }

    /// A copy of the game running in environment `i`.
    fn game(&self, i: usize) -> PyGame {
        PyGame { g: self.inner.games[i].clone() }
    }

    fn levels(&self, i: usize) -> (u8, [u8; 2], i8) {
        let g = &self.inner.games[i];
        (g.level, g.team_level, g.banker)
    }
}

#[pyclass(name = "Game", module = "guandan._core")]
#[derive(Clone)]
pub struct PyGame {
    g: Game,
}

#[pymethods]
impl PyGame {
    #[new]
    #[pyo3(signature = (seed, a_fail_limit = 3, tribute = true))]
    fn new(seed: u64, a_fail_limit: u8, tribute: bool) -> Self {
        PyGame { g: Game::new(seed, Rules { a_fail_limit, tribute }) }
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    /// Copy with the hidden hands re-sampled from the point of view of `me` (default: current player).
    #[pyo3(signature = (seed, me = None))]
    fn determinize(&self, seed: u64, me: Option<u8>) -> Self {
        let mut rng = Rng::new(seed);
        let me = me.unwrap_or(self.g.current);
        PyGame { g: self.g.determinize(me, &mut rng) }
    }

    /// Like determinize, guided by `expect`: 3 x 54 expected copies of every card type in the
    /// hands of the players 1, 2 and 3 seats after `me` (flattened). See Game::determinize_weighted.
    #[pyo3(signature = (seed, expect, floor = 0.05, me = None))]
    fn determinize_weighted(&self, seed: u64, expect: Vec<f32>, floor: f32, me: Option<u8>) -> PyResult<Self> {
        if expect.len() != 3 * 54 {
            return Err(PyValueError::new_err("expect must have 162 values"));
        }
        let mut e = [[0f32; 54]; 3];
        for r in 0..3 {
            for t in 0..54 {
                e[r][t] = expect[r * 54 + t];
            }
        }
        let mut rng = Rng::new(seed);
        let me = me.unwrap_or(self.g.current);
        Ok(PyGame { g: self.g.determinize_weighted(me, &e, floor, &mut rng) })
    }

    fn round_reward(&self, team: u8) -> f32 {
        self.g.round_reward(team)
    }

    /// Bombs played this round: [4-5 cards, 6+ cards, straight flush, joker bomb].
    #[getter]
    fn bombs(&self) -> [u8; 4] {
        self.g.bombs
    }

    fn multiplier(&self, doubles: [bool; 4], cap: u32) -> u32 {
        ScoreRules { doubles, cap }.multiplier(&self.g.bombs)
    }

    fn round_score(&self, team: u8, doubles: [bool; 4], cap: u32) -> f32 {
        self.g.round_score(team, &ScoreRules { doubles, cap })
    }

    #[getter]
    fn phase(&self) -> u8 {
        self.g.phase as u8
    }
    #[getter]
    fn current(&self) -> u8 {
        self.g.current
    }
    #[getter]
    fn level(&self) -> u8 {
        self.g.level
    }
    #[getter]
    fn team_level(&self) -> [u8; 2] {
        self.g.team_level
    }
    #[getter]
    fn banker(&self) -> i8 {
        self.g.banker
    }
    #[getter]
    fn round_no(&self) -> u32 {
        self.g.round_no
    }
    #[getter]
    fn a_fail(&self) -> [u8; 2] {
        self.g.a_fail
    }
    #[getter]
    fn finished(&self) -> Vec<u8> {
        self.g.finished.clone()
    }
    #[getter]
    fn last_player(&self) -> u8 {
        self.g.last_player
    }
    #[getter]
    fn passes(&self) -> u8 {
        self.g.passes
    }
    #[getter]
    fn jiefeng(&self) -> bool {
        self.g.jiefeng
    }
    #[getter]
    fn winner(&self) -> i8 {
        self.g.winner
    }
    #[getter]
    fn anti_tribute(&self) -> bool {
        self.g.anti_tribute
    }
    #[getter]
    fn trick_id(&self) -> u32 {
        self.g.trick_id
    }
    #[getter]
    fn is_leading(&self) -> bool {
        self.g.is_leading()
    }
    #[getter]
    fn wild(&self) -> u8 {
        wild_type(self.g.level)
    }

    fn hand(&self, p: usize) -> Vec<u8> {
        hand_to_vec(&self.g.hands[p & 3])
    }
    fn dealt(&self, p: usize) -> Vec<u8> {
        hand_to_vec(&self.g.dealt[p & 3])
    }
    fn played(&self, p: usize) -> Vec<u8> {
        hand_to_vec(&self.g.played[p & 3])
    }
    fn hand_counts(&self) -> [u32; 4] {
        [0, 1, 2, 3].map(|p| hand_count(&self.g.hands[p]))
    }
    fn last_move(&self) -> Option<PyMove> {
        self.g.last_move.as_ref().map(to_py)
    }
    fn history(&self) -> Vec<(u8, u8, u8, Vec<u8>)> {
        self.g.history.iter().map(|(p, m)| (*p, m.kind as u8, m.key, m.cards.clone())).collect()
    }
    fn tribute(&self) -> Vec<(u8, u8, u8, bool)> {
        self.g.tribute.iter().map(|t| (t.from, t.to, t.card, t.is_return)).collect()
    }
    fn pending_returns(&self) -> Vec<(u8, u8)> {
        self.g.pending_returns.clone()
    }
    /// (order, win_team, gain, match_over, levels_before, levels_after, a_reset)
    fn last_result(&self) -> Option<([u8; 4], u8, u8, bool, [u8; 2], [u8; 2], bool)> {
        self.g
            .last_result
            .as_ref()
            .map(|r| (r.order, r.win_team, r.gain, r.match_over, r.levels_before, r.levels_after, r.a_reset))
    }
    fn legal_moves(&self) -> Vec<PyMove> {
        self.g.legal_moves().iter().map(to_py).collect()
    }
    fn legal_interpretations(&self, cards: Vec<u8>) -> Vec<PyMove> {
        self.g.legal_interpretations(&cards).iter().map(to_py).collect()
    }
    fn apply(&mut self, kind: u8, key: u8, cards: Vec<u8>) -> PyResult<()> {
        let m = from_py(kind, key, cards)?;
        self.g.apply(&m).map_err(PyValueError::new_err)
    }
    fn next_round(&mut self) {
        self.g.next_round();
    }
    fn new_match(&mut self) {
        self.g.new_match();
    }
    #[pyo3(signature = (team_level, banker, redeal = true))]
    fn set_levels(&mut self, team_level: [u8; 2], banker: i8, redeal: bool) {
        self.g.set_levels([team_level[0].min(12), team_level[1].min(12)], banker);
        if redeal {
            self.g.round_no = 0;
            self.g.prev_order = None;
            self.g.start_round();
        }
    }
    fn rule_choice(&self) -> usize {
        let mv = self.g.legal_moves();
        crate::bot::rule_choose(&self.g, &mv)
    }
    /// Features for the current player: (state, hist, actions, n_moves); actions follow legal_moves() order.
    fn encode<'py>(&self, py: Python<'py>) -> (Bound<'py, PyBytes>, Bound<'py, PyBytes>, Bound<'py, PyBytes>, usize) {
        let me = self.g.current;
        let mv = self.g.legal_moves();
        let mut s = vec![0u8; STATE_DIM];
        let mut h = vec![0u8; HIST_LEN * HIST_DIM];
        encode_state(&self.g, me, &mut s);
        encode_history(&self.g, me, &mut h);
        let mut a = vec![0u8; mv.len() * ACTION_DIM];
        for (k, m) in mv.iter().enumerate() {
            encode_action(&self.g, me, m, &mut a[k * ACTION_DIM..(k + 1) * ACTION_DIM]);
        }
        (PyBytes::new(py, &s), PyBytes::new(py, &h), PyBytes::new(py, &a), mv.len())
    }
    /// Replay this round's plays and encode the last `last` decisions of the other players:
    /// (players, states, hists, actions, offsets, chosen); chosen = 2**32-1 when the move
    /// actually played was not available to that player's hand in this world.
    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (me, last = 16))]
    fn replay_decisions<'py>(
        &self,
        py: Python<'py>,
        me: u8,
        last: usize,
    ) -> (Bound<'py, PyBytes>, Bound<'py, PyBytes>, Bound<'py, PyBytes>, Bound<'py, PyBytes>, Vec<u32>, Vec<u32>) {
        let r = self.g.replay_decisions(me, last);
        (
            PyBytes::new(py, &r.players),
            PyBytes::new(py, &r.states),
            PyBytes::new(py, &r.hists),
            PyBytes::new(py, &r.actions),
            r.offsets,
            r.chosen,
        )
    }
    fn describe(&self, kind: u8, key: u8, cards: Vec<u8>) -> PyResult<String> {
        Ok(from_py(kind, key, cards)?.describe(self.g.level))
    }
}

#[pyclass(name = "Rollouts", module = "guandan._core")]
pub struct PyRollouts {
    inner: Rollouts,
}

#[pymethods]
impl PyRollouts {
    #[new]
    fn new(games: Vec<PyRef<PyGame>>) -> Self {
        PyRollouts { inner: Rollouts::new(games.iter().map(|g| g.g.clone()).collect()) }
    }

    /// (indices, players, states, hists, actions, offsets) of unfinished games.
    fn observe<'py>(
        &mut self,
        py: Python<'py>,
    ) -> (Vec<u32>, Bound<'py, PyBytes>, Bound<'py, PyBytes>, Bound<'py, PyBytes>, Bound<'py, PyBytes>, Vec<u32>) {
        let (i, p, s, h, a, o) = self.inner.observe();
        (i, PyBytes::new(py, &p), PyBytes::new(py, &s), PyBytes::new(py, &h), PyBytes::new(py, &a), o)
    }

    fn step(&mut self, choices: Vec<u32>) {
        self.inner.step(&choices);
    }

    /// Rule-bot choices for the games of the last `observe`.
    fn rule_choices(&self) -> Vec<u32> {
        self.inner.rule_choices()
    }

    /// A copy of game `i`.
    fn game(&self, i: usize) -> PyGame {
        PyGame { g: self.inner.games[i].clone() }
    }

    fn __len__(&self) -> usize {
        self.inner.games.len()
    }

    fn rewards(&self, team: u8) -> Vec<f32> {
        self.inner.rewards(team)
    }

    /// Scores under a multiplier system: doubles = [4-5 bomb, 6+ bomb, straight flush, joker bomb].
    fn scores(&self, team: u8, doubles: [bool; 4], cap: u32) -> Vec<f32> {
        self.inner.scores(team, &ScoreRules { doubles, cap })
    }

    fn multipliers(&self, doubles: [bool; 4], cap: u32) -> Vec<u32> {
        self.inner.multipliers(&ScoreRules { doubles, cap })
    }

    fn done(&self) -> Vec<bool> {
        self.inner.done.clone()
    }
}

#[pyfunction]
fn dims() -> (usize, usize, usize, usize) {
    (STATE_DIM, HIST_LEN, HIST_DIM, ACTION_DIM)
}

#[pyfunction]
fn hidden_dim() -> usize {
    HIDDEN_DIM
}

#[pyfunction]
fn classify_cards(cards: Vec<u8>, level: u8) -> Vec<PyMove> {
    classify(&cards, level).iter().map(to_py).collect()
}

#[pyfunction]
fn card_label(t: u8) -> String {
    card_name(t)
}

#[pyfunction]
fn kind_name(k: u8) -> String {
    Kind::from_u8(k).map(|k| k.name().to_string()).unwrap_or_default()
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGame>()?;
    m.add_class::<PyVecEnv>()?;
    m.add_class::<PyRollouts>()?;
    m.add_function(wrap_pyfunction!(dims, m)?)?;
    m.add_function(wrap_pyfunction!(hidden_dim, m)?)?;
    m.add_function(wrap_pyfunction!(classify_cards, m)?)?;
    m.add_function(wrap_pyfunction!(card_label, m)?)?;
    m.add_function(wrap_pyfunction!(kind_name, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
