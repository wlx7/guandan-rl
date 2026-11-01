//! Full Guandan match state machine: dealing, tribute / return / anti-tribute,
//! trick play with 接风, finishing order, level upgrades and passing A.

use crate::cards::*;
use crate::moves::*;

#[derive(Clone, Debug)]
pub struct Rules {
    /// Failing to pass A this many times (as the banker at A) resets the team to 2. 0 = off.
    pub a_fail_limit: u8,
    /// Enable tribute (进贡/还贡/抗贡) from the 2nd round on.
    pub tribute: bool,
}

impl Default for Rules {
    fn default() -> Self {
        Rules { a_fail_limit: 3, tribute: true }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Phase {
    Return = 0,
    Play = 1,
    RoundOver = 2,
    MatchOver = 3,
}

#[derive(Clone, Debug)]
pub struct TributeRec {
    pub from: u8,
    pub to: u8,
    pub card: u8,
    pub is_return: bool,
}

#[derive(Clone, Debug)]
pub struct RoundResult {
    pub order: [u8; 4],
    pub win_team: u8,
    pub gain: u8,
    pub match_over: bool,
    pub levels_before: [u8; 2],
    pub levels_after: [u8; 2],
    pub a_reset: bool,
}

#[derive(Clone, Debug)]
pub struct Game {
    pub rng: Rng,
    pub rules: Rules,
    pub team_level: [u8; 2],
    pub a_fail: [u8; 2],
    pub banker: i8,
    pub level: u8,
    pub round_no: u32,
    pub hands: [Hand; 4],
    pub dealt: [Hand; 4],
    pub played: [Hand; 4],
    pub current: u8,
    pub phase: Phase,
    pub last_move: Option<Move>,
    pub last_player: u8,
    pub passes: u8,
    pub finished: Vec<u8>,
    pub history: Vec<(u8, Move)>,
    pub tribute: Vec<TributeRec>,
    pub anti_tribute: bool,
    pub pending_returns: Vec<(u8, u8)>,
    pub lead_after_tribute: u8,
    pub prev_order: Option<[u8; 4]>,
    pub last_result: Option<RoundResult>,
    pub winner: i8,
    pub jiefeng: bool,
    pub trick_id: u32,
    /// Bombs played this round: [4-5 cards, 6+ cards, straight flush, joker bomb]
    pub bombs: [u8; 4],
}

/// Optional score / multiplier system: round score = base (1/2/3 like the level gain)
/// x 2^(number of doubling bombs), capped.
#[derive(Clone, Copy, Debug)]
pub struct ScoreRules {
    pub doubles: [bool; 4],
    pub cap: u32,
}

impl ScoreRules {
    pub fn multiplier(&self, bombs: &[u8; 4]) -> u32 {
        let mut e: u32 = 0;
        for k in 0..4 {
            if self.doubles[k] {
                e += bombs[k] as u32;
            }
        }
        (1u32 << e.min(20)).min(self.cap.max(1))
    }
}

pub fn bomb_category(m: &Move) -> Option<usize> {
    match m.kind {
        Kind::Bomb => Some(if m.cards.len() >= 6 { 1 } else { 0 }),
        Kind::StraightFlush => Some(2),
        Kind::JokerBomb => Some(3),
        _ => None,
    }
}

#[inline]
pub fn partner(p: u8) -> u8 {
    (p + 2) & 3
}
#[inline]
pub fn team(p: u8) -> u8 {
    p & 1
}

impl Game {
    pub fn new(seed: u64, rules: Rules) -> Game {
        let mut g = Game {
            rng: Rng::new(seed),
            rules,
            team_level: [0, 0],
            a_fail: [0, 0],
            banker: -1,
            level: 0,
            round_no: 0,
            hands: [[0; 54]; 4],
            dealt: [[0; 54]; 4],
            played: [[0; 54]; 4],
            current: 0,
            phase: Phase::Play,
            last_move: None,
            last_player: 0,
            passes: 0,
            finished: vec![],
            history: vec![],
            tribute: vec![],
            anti_tribute: false,
            pending_returns: vec![],
            lead_after_tribute: 0,
            prev_order: None,
            last_result: None,
            winner: -1,
            jiefeng: false,
            trick_id: 0,
            bombs: [0; 4],
        };
        g.start_round();
        g
    }

    /// Start a fresh match keeping the rng stream.
    pub fn new_match(&mut self) {
        self.team_level = [0, 0];
        self.a_fail = [0, 0];
        self.banker = -1;
        self.round_no = 0;
        self.prev_order = None;
        self.last_result = None;
        self.winner = -1;
        self.start_round();
    }

    /// Set levels explicitly (used to diversify training and for custom games).
    pub fn set_levels(&mut self, team_level: [u8; 2], banker: i8) {
        self.team_level = team_level;
        self.banker = banker;
    }

    pub fn start_round(&mut self) {
        self.round_no += 1;
        self.level = if self.banker < 0 { self.team_level[0] } else { self.team_level[self.banker as usize] };
        let mut deck: Vec<u8> = (0..54u8).flat_map(|t| [t, t]).collect();
        self.rng.shuffle(&mut deck);
        self.hands = [[0; 54]; 4];
        for (i, &c) in deck.iter().enumerate() {
            self.hands[i / 27][c as usize] += 1;
        }
        self.played = [[0; 54]; 4];
        self.last_move = None;
        self.passes = 0;
        self.finished.clear();
        self.history.clear();
        self.tribute.clear();
        self.anti_tribute = false;
        self.pending_returns.clear();
        self.jiefeng = false;
        self.trick_id = 0;
        self.bombs = [0; 4];
        self.phase = Phase::Play;

        match (self.prev_order, self.rules.tribute) {
            (Some(order), true) => self.do_tribute(order),
            _ => {
                self.current = self.rng.below(4) as u8;
            }
        }
        self.dealt = self.hands;
    }

    fn max_tribute_card(&self, p: u8) -> u8 {
        let wt = wild_type(self.level);
        let h = &self.hands[p as usize];
        let mut best: Option<(u8, u8)> = None; // (strength, type)
        for t in 0..54u8 {
            if h[t as usize] == 0 || t == wt {
                continue;
            }
            let s = strength(rank_of(t), self.level);
            if best.map_or(true, |(bs, _)| s > bs) {
                best = Some((s, t));
            }
        }
        best.map(|b| b.1).unwrap_or(wt)
    }

    fn do_tribute(&mut self, order: [u8; 4]) {
        let first = order[0];
        let second = order[1];
        let double = team(first) == team(second);
        let payers: Vec<u8> = if double { vec![order[2], order[3]] } else { vec![order[3]] };
        let bj: u8 = payers.iter().map(|&p| self.hands[p as usize][BJ as usize]).sum();
        if bj >= 2 {
            self.anti_tribute = true;
            self.current = first;
            self.phase = Phase::Play;
            return;
        }
        let mut gives: Vec<(u8, u8, u8)> = Vec::new(); // (payer, receiver, card)
        let leader;
        if !double {
            let p = payers[0];
            gives.push((p, first, self.max_tribute_card(p)));
            leader = p;
        } else {
            let (a, b) = (payers[0], payers[1]);
            let ca = self.max_tribute_card(a);
            let cb = self.max_tribute_card(b);
            let sa = strength(rank_of(ca), self.level);
            let sb = strength(rank_of(cb), self.level);
            let (big, small) = if sa > sb {
                (a, b)
            } else if sb > sa {
                (b, a)
            } else {
                // equal: the player right after 头游 gives to 头游
                let nxt = (first + 1) & 3;
                if nxt == a { (a, b) } else { (b, a) }
            };
            let cbig = if big == a { ca } else { cb };
            let csmall = if small == a { ca } else { cb };
            gives.push((big, first, cbig));
            gives.push((small, second, csmall));
            leader = big;
        }
        for &(p, r, c) in &gives {
            self.hands[p as usize][c as usize] -= 1;
            self.hands[r as usize][c as usize] += 1;
            self.tribute.push(TributeRec { from: p, to: r, card: c, is_return: false });
            self.pending_returns.push((r, p));
        }
        self.lead_after_tribute = leader;
        self.phase = Phase::Return;
        self.current = self.pending_returns[0].0;
    }

    pub fn return_options(&self) -> Vec<Move> {
        let p = self.current;
        let h = &self.hands[p as usize];
        let wt = wild_type(self.level);
        let mut v: Vec<Move> = (0..52u8)
            .filter(|&t| h[t as usize] > 0 && rank_of(t) <= 8 && t != wt)
            .map(|t| Move::new(Kind::Tribute, rank_of(t), vec![t]))
            .collect();
        if v.is_empty() {
            // pathological hand without any card <= 10: allow the smallest card
            let mut best: Option<(u8, u8)> = None;
            for t in 0..54u8 {
                if h[t as usize] == 0 || t == wt {
                    continue;
                }
                let s = strength(rank_of(t), self.level);
                if best.map_or(true, |(bs, _)| s < bs) {
                    best = Some((s, t));
                }
            }
            if let Some((_, t)) = best {
                v.push(Move::new(Kind::Tribute, rank_of(t), vec![t]));
            }
        }
        v
    }

    #[inline]
    pub fn is_leading(&self) -> bool {
        self.last_move.is_none()
    }

    pub fn legal_moves(&self) -> Vec<Move> {
        match self.phase {
            Phase::Return => self.return_options(),
            Phase::Play => legal_moves(&self.hands[self.current as usize], self.level, self.last_move.as_ref()),
            _ => vec![],
        }
    }

    pub fn is_finished(&self, p: u8) -> bool {
        self.finished.contains(&p)
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

    /// Interpretations of a card set that are legal right now for the current player.
    pub fn legal_interpretations(&self, cards: &[u8]) -> Vec<Move> {
        if self.phase != Phase::Play {
            return vec![];
        }
        if !hand_contains(&self.hands[self.current as usize], cards) {
            return vec![];
        }
        classify(cards, self.level)
            .into_iter()
            .filter(|m| match &self.last_move {
                None => true,
                Some(pm) => m.beats(pm),
            })
            .collect()
    }

    /// Apply a move for the current player. Validates legality.
    pub fn apply(&mut self, mv: &Move) -> Result<(), String> {
        match self.phase {
            Phase::Return => self.apply_return(mv),
            Phase::Play => self.apply_play(mv),
            _ => Err("round is over".into()),
        }
    }

    fn apply_return(&mut self, mv: &Move) -> Result<(), String> {
        if mv.kind != Kind::Tribute || mv.cards.len() != 1 {
            return Err("expected a return-tribute card".into());
        }
        let opts = self.return_options();
        if !opts.iter().any(|o| o.cards == mv.cards) {
            return Err("illegal return card (must be 10 or lower)".into());
        }
        match self.pending_returns.first() {
            Some(&(r0, _)) if r0 == self.current => {}
            _ => return Err("not your turn".into()),
        }
        let (r, p) = self.pending_returns.remove(0);
        let c = mv.cards[0];
        self.hands[r as usize][c as usize] -= 1;
        self.hands[p as usize][c as usize] += 1;
        self.tribute.push(TributeRec { from: r, to: p, card: c, is_return: true });
        if let Some(&(nr, _)) = self.pending_returns.first() {
            self.current = nr;
        } else {
            self.phase = Phase::Play;
            self.current = self.lead_after_tribute;
            self.dealt = self.hands;
        }
        Ok(())
    }

    fn apply_play(&mut self, mv: &Move) -> Result<(), String> {
        let p = self.current;
        self.jiefeng = false;
        if mv.is_pass() {
            if self.is_leading() {
                return Err("leader cannot pass".into());
            }
            self.history.push((p, mv.clone()));
            self.passes += 1;
            self.advance(p);
            return Ok(());
        }
        if mv.kind == Kind::Tribute {
            return Err("not in tribute phase".into());
        }
        if !hand_contains(&self.hands[p as usize], &mv.cards) {
            return Err("cards not in hand".into());
        }
        let cl = classify(&mv.cards, self.level);
        if !cl.iter().any(|m| m.kind == mv.kind && m.key == mv.key) {
            return Err("invalid combination".into());
        }
        if let Some(pm) = &self.last_move {
            if !mv.beats(pm) {
                return Err("does not beat the previous play".into());
            }
        }
        for &c in &mv.cards {
            self.hands[p as usize][c as usize] -= 1;
            self.played[p as usize][c as usize] += 1;
        }
        self.history.push((p, mv.clone()));
        if let Some(k) = bomb_category(mv) {
            self.bombs[k] = self.bombs[k].saturating_add(1);
        }
        self.last_move = Some(mv.clone());
        self.last_player = p;
        self.passes = 0;
        if hand_count(&self.hands[p as usize]) == 0 {
            self.finished.push(p);
            let nf = self.finished.len();
            if nf == 3 || (nf == 2 && team(self.finished[0]) == team(self.finished[1])) {
                self.finish_round();
                return Ok(());
            }
        }
        self.advance(p);
        Ok(())
    }

    fn advance(&mut self, p: u8) {
        if self.last_move.is_some() {
            let active = 4 - self.finished.len();
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
                        self.jiefeng = true;
                        pt
                    }
                };
                self.last_move = None;
                self.passes = 0;
                self.current = leader;
                self.trick_id += 1;
                return;
            }
        }
        self.current = self.next_active(p);
    }

    fn finish_round(&mut self) {
        let mut order = [0u8; 4];
        let mut k = 0;
        for &f in &self.finished {
            order[k] = f;
            k += 1;
        }
        for s in 0..4u8 {
            if !self.finished.contains(&s) {
                order[k] = s;
                k += 1;
            }
        }
        let first = order[0];
        let w = team(first);
        let pp = order.iter().position(|&x| x == partner(first)).unwrap();
        let gain: u8 = match pp {
            1 => 3,
            2 => 2,
            _ => 1,
        };
        let before = self.team_level;
        let mut match_over = false;
        let mut a_reset = false;
        let banker_at_a = self.banker >= 0 && self.level == ACE && self.team_level[self.banker as usize] == ACE;
        if banker_at_a && self.banker as u8 == w && gain >= 2 {
            match_over = true;
        } else {
            let mut skip_gain = false;
            if banker_at_a {
                let b = self.banker as usize;
                self.a_fail[b] += 1;
                if self.rules.a_fail_limit > 0 && self.a_fail[b] >= self.rules.a_fail_limit {
                    self.team_level[b] = 0;
                    self.a_fail[b] = 0;
                    a_reset = true;
                    if b as u8 == w {
                        skip_gain = true;
                    }
                }
            }
            if !skip_gain {
                self.team_level[w as usize] = (self.team_level[w as usize] + gain).min(ACE);
            }
            self.banker = w as i8;
        }
        self.last_result = Some(RoundResult {
            order,
            win_team: w,
            gain,
            match_over,
            levels_before: before,
            levels_after: self.team_level,
            a_reset,
        });
        self.prev_order = Some(order);
        self.last_move = None;
        if match_over {
            self.winner = w as i8;
            self.phase = Phase::MatchOver;
        } else {
            self.phase = Phase::RoundOver;
        }
    }

    /// Continue after a finished round (or start a new match after the match ended).
    pub fn next_round(&mut self) {
        match self.phase {
            Phase::RoundOver => self.start_round(),
            Phase::MatchOver => self.new_match(),
            _ => {}
        }
    }

    /// Score of a finished round for `team` under a multiplier system.
    pub fn round_score(&self, t: u8, rules: &ScoreRules) -> f32 {
        match &self.last_result {
            None => 0.0,
            Some(r) => {
                let v = r.gain as f32 * rules.multiplier(&self.bombs) as f32;
                if r.win_team == t { v } else { -v }
            }
        }
    }

    /// Reward of a finished round for `team` (positive for the winning team).
    pub fn round_reward(&self, t: u8) -> f32 {
        match &self.last_result {
            None => 0.0,
            Some(r) => {
                let v = if r.match_over { 3.0 } else { r.gain as f32 };
                if r.win_team == t { v } else { -v }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_matches_complete() {
        let mut rng = Rng::new(3);
        for seed in 0..40u64 {
            let mut g = Game::new(seed, Rules::default());
            let mut rounds = 0;
            let mut steps = 0;
            while g.phase != Phase::MatchOver && rounds < 60 {
                match g.phase {
                    Phase::RoundOver => {
                        rounds += 1;
                        g.next_round();
                        continue;
                    }
                    _ => {}
                }
                let moves = g.legal_moves();
                assert!(!moves.is_empty(), "no legal moves");
                let total: u32 = (0..4).map(|p| hand_count(&g.hands[p]) + hand_count(&g.played[p])).sum();
                assert_eq!(total, 108);
                let m = &moves[rng.below(moves.len() as u64) as usize];
                g.apply(m).unwrap();
                steps += 1;
                assert!(steps < 100000);
            }
        }
    }

    fn setup_endgame(banker: i8, levels: [u8; 2], level: u8) -> Game {
        let mut g = Game::new(9, Rules::default());
        g.phase = Phase::Play;
        g.hands = [[0; 54]; 4];
        g.team_level = levels;
        g.banker = banker;
        g.level = level;
        g.finished.clear();
        g.last_move = None;
        g.current = 0;
        g.hands[0][ctype(3, 0) as usize] = 1; // one card each for the 0/2 team
        g.hands[2][ctype(4, 0) as usize] = 1;
        g.hands[1][ctype(1, 0) as usize] = 2;
        g.hands[3][ctype(1, 2) as usize] = 2;
        g
    }

    fn double_win(g: &mut Game) {
        // seat 0 goes out, everybody passes, partner (2) gets the lead via 接风 and goes out too
        g.apply(&Move::new(Kind::Single, strength(3, g.level), vec![ctype(3, 0)])).unwrap();
        for _ in 0..3 {
            g.apply(&Move::pass()).unwrap();
        }
        assert_eq!(g.current, 2);
        g.apply(&Move::new(Kind::Single, strength(4, g.level), vec![ctype(4, 0)])).unwrap();
    }

    #[test]
    fn double_win_upgrades_three_levels() {
        let mut g = setup_endgame(0, [5, 7], 5);
        double_win(&mut g);
        assert_eq!(g.phase, Phase::RoundOver);
        let r = g.last_result.clone().unwrap();
        assert_eq!((r.win_team, r.gain), (0, 3));
        assert_eq!(g.team_level, [8, 7]);
        assert_eq!(g.banker, 0);
    }

    #[test]
    fn passing_a_wins_the_match() {
        let mut g = setup_endgame(0, [ACE, 9], ACE);
        double_win(&mut g);
        assert_eq!(g.phase, Phase::MatchOver);
        assert_eq!(g.winner, 0);
    }

    #[test]
    fn not_banker_at_a_does_not_win_match() {
        // team 0 is at A but the round is played at team 1's level
        let mut g = setup_endgame(1, [ACE, 9], 9);
        double_win(&mut g);
        assert_eq!(g.phase, Phase::RoundOver);
        assert_eq!(g.banker, 0);
        assert_eq!(g.team_level, [ACE, 9]);
    }

    #[test]
    fn three_failed_a_attempts_reset_to_two() {
        for fails_before in 0..3u8 {
            // team 1 is the banker at A; team 0 wins the round -> failure for team 1
            let mut g = setup_endgame(1, [3, ACE], ACE);
            g.a_fail = [0, fails_before];
            double_win(&mut g);
            if fails_before < 2 {
                assert_eq!(g.a_fail[1], fails_before + 1);
                assert_eq!(g.team_level[1], ACE);
            } else {
                assert_eq!(g.a_fail[1], 0);
                assert_eq!(g.team_level[1], 0, "back to 2 after the third failure");
            }
        }
    }

    #[test]
    fn jiefeng() {
        // Construct: player 0 plays last card, everybody passes -> partner (2) leads.
        let mut g = Game::new(1, Rules::default());
        g.phase = Phase::Play;
        g.hands = [[0; 54]; 4];
        g.hands[0][ctype(5, 0) as usize] = 1;
        g.hands[1][ctype(1, 0) as usize] = 3;
        g.hands[2][ctype(2, 0) as usize] = 3;
        g.hands[3][ctype(3, 0) as usize] = 3;
        g.level = 0;
        g.current = 0;
        g.last_move = None;
        g.finished.clear();
        g.apply(&Move::new(Kind::Single, 5, vec![ctype(5, 0)])).unwrap();
        assert_eq!(g.finished, vec![0]);
        assert_eq!(g.current, 1);
        g.apply(&Move::pass()).unwrap();
        g.apply(&Move::pass()).unwrap();
        g.apply(&Move::pass()).unwrap();
        assert!(g.jiefeng);
        assert_eq!(g.current, 2);
        assert!(g.is_leading());
    }
}
