"""Factored Public Belief Search (PBS / ReBeL Subgame Search) for Guandan.

Unlike classical truncated rollout search (`search.py`, which lost -1.50 levels/round when
truncating at `depth=14` because leaf Q values were evaluated from whichever player happened
to act at ply `D`), `PublicBeliefSearch` tracks the common-knowledge Factored Public Belief
State `beta_t = (s_pub_t, B_t)` (`_core.PubBelief`) along every subgame branch and evaluates
depth-`D` leaves with `RNaDPBSNet.pbs_leaf_value(s_pub, B_D, hands_D)`:
  v_leaf(beta_D, hands_D) = V_PBS(beta_D) + <w_cf(beta_D), hands108(hands_D) - B_D>
which is always in canonical Team-0 coordinates (zero cross-player information bias!), followed
by Regularized Regret-Matching+ (CFR+) equilibrium aggregation at the root.
"""
from __future__ import annotations

import itertools
import numpy as np
import torch

from . import _core
from .rnad import RNaDPBSNet, segmented_log_softmax

STATE_DIM, HIST_LEN, HIST_DIM, ACTION_DIM = _core.dims()
PUB_STATE_DIM, PBS_DIM = _core.pbs_dims()

_seed = itertools.count(1)


def _hands108_of_game(g: "_core.Game") -> np.ndarray:
    out = np.zeros((4, 54, 2), dtype=np.float32)
    for p in range(4):
        cnt = np.bincount(list(g.hand(p)), minlength=54)
        out[p, :, 0] = (cnt >= 1).astype(np.float32)
        out[p, :, 1] = (cnt >= 2).astype(np.float32)
    return out.reshape(PBS_DIM)


class PublicBeliefSearch:
    def __init__(
        self,
        model: RNaDPBSNet,
        device: str = "cuda",
        top_k: int = 4,
        worlds: int = 16,
        depth: int = 14,
        cfr_iters: int = 32,
        reg_eta: float = 0.12,
        close_gap_logit: float = 0.60,
        solver_cards: int = 22,
    ):
        self.model = model
        self.device = torch.device(device)
        self.top_k = top_k
        self.worlds = worlds
        self.depth = depth
        self.cfr_iters = cfr_iters
        self.reg_eta = reg_eta
        self.close_gap_logit = close_gap_logit
        self.solver_cards = solver_cards
        self.rng = np.random.default_rng(2026)

    @torch.no_grad()
    def _step_policy_batch(self, games: list["_core.Game"], beliefs: list["_core.PubBelief"], greedy: bool = True) -> np.ndarray:
        b = len(games)
        s_list, p_list, h_list, a_list, counts = [], [], [], [], []
        for g, pb in zip(games, beliefs):
            s_b, h_b, a_b, n_mv = g.encode()
            s_list.append(np.frombuffer(s_b, np.uint8))
            h_list.append(np.frombuffer(h_b, np.uint8).reshape(HIST_LEN, HIST_DIM))
            a_list.append(np.frombuffer(a_b, np.uint8).reshape(n_mv, ACTION_DIM))
            p_list.append(pb.vec_rel(g.current))
            counts.append(n_mv)

        counts_arr = np.asarray(counts, np.int64)
        st = torch.from_numpy(np.stack(s_list)).to(self.device).float()
        pt = torch.from_numpy(np.asarray(p_list, np.float32)).to(self.device)
        ht = torch.from_numpy(np.stack(h_list)).to(self.device).float()
        at = torch.from_numpy(np.concatenate(a_list)).to(self.device).float()
        seg = torch.from_numpy(np.repeat(np.arange(b), counts_arr)).to(self.device)

        z = self.model.encode_priv(st, pt, ht)
        logits = self.model.policy_logits(z, at, seg)
        _, pi, _ = segmented_log_softmax(logits, seg, b)
        pi_np = pi.float().cpu().numpy()

        off = np.zeros(b + 1, np.int64)
        off[1:] = np.cumsum(counts_arr)
        ch = np.zeros(b, np.int64)
        for i in range(b):
            p_s = pi_np[off[i]:off[i + 1]]
            if len(p_s) <= 1 or greedy:
                ch[i] = int(np.argmax(p_s))
            else:
                ch[i] = int(self.rng.choice(len(p_s), p=p_s / p_s.sum()))
        return ch

    @torch.no_grad()
    def _evaluate_pbs_leaves(self, games: list["_core.Game"], beliefs: list["_core.PubBelief"]) -> np.ndarray:
        """Team 0's value in levels (`3.0 * pbs_leaf_value`) for unfinished depth-`D` leaf nodes."""
        if not games:
            return np.zeros(0, np.float32)
        sp_list, pa_list, hh_list = [], [], []
        for g, pb in zip(games, beliefs):
            sp_b, _ = g.encode_public()
            sp_list.append(np.frombuffer(sp_b, np.uint8))
            pa_list.append(pb.vec_abs())
            hh_list.append(_hands108_of_game(g))
        sp = torch.from_numpy(np.stack(sp_list)).to(self.device).float()
        pa = torch.from_numpy(np.asarray(pa_list, np.float32)).to(self.device)
        hh = torch.from_numpy(np.stack(hh_list)).to(self.device)
        v_t0 = self.model.pbs_leaf_value(sp, pa, hh).float().cpu().numpy() * 3.0
        return v_t0

    def _cfr_plus_root(self, payoff_kw: np.ndarray, pi_prior: np.ndarray) -> tuple[int, np.ndarray]:
        """Regularized Regret-Matching+ (CFR+) over `[K, W]` counterfactual world payoffs (in levels)."""
        K, W = payoff_kw.shape
        regrets = np.zeros(K, np.float64)
        strat_sum = np.zeros(K, np.float64)
        pi0 = np.maximum(pi_prior.astype(np.float64), 1e-6)
        pi0 /= pi0.sum()
        sigma = pi0.copy()

        for t in range(1, self.cfr_iters + 1):
            # Sample half-bag of worlds (or all worlds on final pass) to smooth Monte-Carlo variance
            if t == self.cfr_iters or W <= 4:
                q_a = payoff_kw.mean(axis=1)
            else:
                sub = self.rng.integers(0, W, size=W)
                q_a = payoff_kw[:, sub].mean(axis=1)
            # R-NaD KL-regularized payoff relative to blueprint pi0
            q_reg = q_a - self.reg_eta * np.log(np.maximum(sigma, 1e-8) / pi0)
            ev = float((sigma * q_reg).sum())
            inst_regret = q_reg - ev
            # RM+ non-negative clipping with linear iteration weighting
            regrets = np.maximum(0.0, regrets + inst_regret)
            if regrets.sum() > 1e-9:
                sigma = 0.85 * (regrets / regrets.sum()) + 0.15 * pi0
            else:
                sigma = pi0.copy()
            strat_sum += t * sigma

        avg_sigma = strat_sum / max(1e-12, strat_sum.sum())
        return int(np.argmax(avg_sigma)), avg_sigma

    @torch.no_grad()
    def choose_many(self, games: list["_core.Game"]) -> list[int]:
        if not games:
            return []
        pbs_roots = [_core.PubBelief.from_public_history(g) for g in games]

        # Score root policies for all games
        s_list, p_list, h_list, a_list, counts = [], [], [], [], []
        for g, pb in zip(games, pbs_roots):
            s_b, h_b, a_b, n_mv = g.encode()
            s_list.append(np.frombuffer(s_b, np.uint8))
            h_list.append(np.frombuffer(h_b, np.uint8).reshape(HIST_LEN, HIST_DIM))
            a_list.append(np.frombuffer(a_b, np.uint8).reshape(n_mv, ACTION_DIM))
            p_list.append(pb.vec_rel(g.current))
            counts.append(n_mv)

        b = len(games)
        counts_arr = np.asarray(counts, np.int64)
        st = torch.from_numpy(np.stack(s_list)).to(self.device).float()
        pt = torch.from_numpy(np.asarray(p_list, np.float32)).to(self.device)
        ht = torch.from_numpy(np.stack(h_list)).to(self.device).float()
        at = torch.from_numpy(np.concatenate(a_list)).to(self.device).float()
        seg = torch.from_numpy(np.repeat(np.arange(b), counts_arr)).to(self.device)

        z = self.model.encode_priv(st, pt, ht)
        logits = self.model.policy_logits(z, at, seg)
        _, pi, _ = segmented_log_softmax(logits, seg, b)
        logits_np = logits.float().cpu().numpy()
        pi_np = pi.float().cpu().numpy()

        off = np.zeros(b + 1, np.int64)
        off[1:] = np.cumsum(counts_arr)
        choices = [int(np.argmax(logits_np[off[i]:off[i + 1]])) for i in range(b)]

        # Exact endgame check first (<= solver_cards) + collect subgame search plan for mid-game!
        mid_plan = []
        for gi, (g, pb) in enumerate(zip(games, pbs_roots)):
            n_mv = int(counts_arr[gi])
            if n_mv <= 1:
                continue
            lg_i = logits_np[off[gi]:off[gi + 1]]
            pi_i = pi_np[off[gi]:off[gi + 1]]
            order = np.argsort(-lg_i)
            cl = g.cards_left()

            if g.phase == 1 and cl <= self.solver_cards:
                # Sample PBS worlds conditioned on `me` and solve exactly with Rust alpha-beta!
                cands = order[:min(8, n_mv)]
                moves = g.legal_moves()
                sub_mvs = [moves[int(c)] for c in cands]
                base_s = next(_seed) * 7919
                acc = np.zeros(len(cands), np.float64)
                ok = 0
                for w in range(self.worlds):
                    dw = pb.sample_world(g, base_s + w, g.current)
                    res = dw.solve_moves(sub_mvs, 150_000)
                    if res and res[0] is not None:
                        acc += np.asarray(res, np.float64)
                        ok += 1
                if ok > 0:
                    pimc = acc / ok
                    sc = 0.80 * pimc + 0.20 * lg_i[cands]
                    choices[gi] = int(cands[int(np.argmax(sc))])
                continue

            if float(lg_i[order[0]] - lg_i[order[1]]) <= self.close_gap_logit:
                cands = order[:min(self.top_k, n_mv)]
                mid_plan.append((gi, g, pb, cands, pi_i[cands]))

        if not mid_plan:
            return choices

        # Build all depth-D subgame branches for mid_plan
        sims_g: list[_core.Game] = []
        sims_pb: list[_core.PubBelief] = []
        sims_team: list[int] = []
        slices = []

        for gi, g, pb, cands, pi_c in mid_plan:
            me = g.current
            team = me % 2
            moves = g.legal_moves()
            base_s = next(_seed) * 7919
            worlds_g = [pb.sample_world(g, base_s + w, me) for w in range(self.worlds)]
            start = len(sims_g)
            for ci in cands:
                mv = moves[int(ci)]
                for wg in worlds_g:
                    x = wg.copy()
                    pb_x = pb.copy()
                    plp = x.last_player
                    plm = x.last_move()
                    x.apply(*mv)
                    if x.phase in (0, 1):
                        pb_x.observe_step(plp, plm, me, mv, x)
                    sims_g.append(x)
                    sims_pb.append(pb_x)
                    sims_team.append(team)
            slices.append((gi, cands, pi_c, start, len(cands)))

        # Step all active subgame branches for `self.depth` plies (updating PubBelief along every branch!)
        for _ in range(self.depth):
            active = [k for k, x in enumerate(sims_g) if x.phase in (0, 1)]
            if not active:
                break
            # Chunk to 512 to keep VRAM tiny
            for c_start in range(0, len(active), 512):
                sub_idx = active[c_start:c_start + 512]
                sub_g = [sims_g[k] for k in sub_idx]
                sub_pb = [sims_pb[k] for k in sub_idx]
                ch_sub = self._step_policy_batch(sub_g, sub_pb, greedy=True)
                for k, c_idx in zip(sub_idx, ch_sub):
                    x = sims_g[k]
                    mvs = x.legal_moves()
                    mv = mvs[int(c_idx)]
                    actor = x.current
                    plp = x.last_player
                    plm = x.last_move()
                    x.apply(*mv)
                    if x.phase in (0, 1):
                        sims_pb[k].observe_step(plp, plm, actor, mv, x)

        # Evaluate finished branches by exact round_reward and unfinished branches by V_PBS + w_cf!
        vals = np.zeros(len(sims_g), np.float32)
        leaf_indices = []
        for k, x in enumerate(sims_g):
            if x.phase not in (0, 1):
                vals[k] = x.round_reward(sims_team[k])
            else:
                leaf_indices.append(k)

        if leaf_indices:
            for c_start in range(0, len(leaf_indices), 512):
                sub_k = leaf_indices[c_start:c_start + 512]
                v_t0 = self._evaluate_pbs_leaves([sims_g[k] for k in sub_k], [sims_pb[k] for k in sub_k])
                for k, v0 in zip(sub_k, v_t0):
                    vals[k] = v0 if sims_team[k] == 0 else -v0

        for gi, cands, pi_c, start, nc in slices:
            payoff_kw = vals[start:start + nc * self.worlds].reshape(nc, self.worlds).astype(np.float64)
            best_local, _ = self._cfr_plus_root(payoff_kw, pi_c)
            choices[gi] = int(cands[best_local])

        return choices


class PBSSearchPlayer:
    """Duplicate-match player running Factored Public Belief Subgame Search (`PublicBeliefSearch`)."""
    def __init__(self, searcher: PublicBeliefSearch):
        self.searcher = searcher

    def act(self, ro, obs, rows):
        idx = obs[0]
        games = [ro.game(int(idx[r])) for r in rows]
        return np.asarray(self.searcher.choose_many(games), np.int64)
