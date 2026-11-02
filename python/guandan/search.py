"""Inference-time search on top of the trained network ("AI 深度思考").

For a decision the network first scores every legal move. The best `top_k` candidates are
then tested in `samples` sampled worlds: the hidden hands of the other three players are
re-dealt at random, consistently with everything the acting player has seen (cards
played, card counts, tribute). In every world each candidate is played, the game is
continued by the network for all four seats for `depth` moves, and the position reached
is valued by the network's own value (the best Q of the player to move), or by the actual
result if the round already ended. The candidate with the best average wins.

No hand-written knowledge is involved: the network provides candidates, play-outs and
values; search only spends more computation per decision.
"""
from __future__ import annotations

import itertools

import numpy as np
import torch

from . import _core
from .agents import BatchPolicy, segment_argmax
from .model import STATE_DIM, HIST_LEN, HIST_DIM, ACTION_DIM

_seed = itertools.count(1)


def _seg_max(q, off):
    return np.maximum.reduceat(q, off[:-1]) if len(off) > 1 else np.zeros(0, np.float32)


class SearchAgent:
    def __init__(self, model, device="cpu", top_k=5, samples=12, depth=14, prior=0.25):
        self.model = model
        self.device = torch.device(device)
        self.policy = BatchPolicy(model, device)
        self.top_k = top_k
        self.samples = samples
        self.depth = depth
        self.prior = prior  # weight of the network's own Q in the final score

    # ------------------------------------------------------------------ networks
    def _q(self, s, h, a, off):
        return self.policy.q_values(s, h, a, off) * 3.0  # in levels

    # ------------------------------------------------------------------ play-outs
    def _simulate(self, sims, teams):
        """Value (in levels, for `teams[i]`) of every simulated game after `depth` greedy moves."""
        ro = _core.Rollouts(sims)
        values = np.full(len(sims), np.nan, np.float32)
        teams = np.asarray(teams)
        for step in range(self.depth + 1):
            last = step == self.depth
            idx, players, s, h, a, off = ro.observe()
            if not idx:
                break
            idx = np.asarray(idx, np.int64)
            players = np.frombuffer(players, np.uint8)
            s = np.frombuffer(s, np.uint8).reshape(-1, STATE_DIM)
            h = np.frombuffer(h, np.uint8).reshape(-1, HIST_LEN, HIST_DIM)
            a = np.frombuffer(a, np.uint8).reshape(-1, ACTION_DIM)
            off = np.asarray(off, np.int64)
            q = self._q(s, h, a, off)
            if last:
                sign = np.where(players % 2 == teams[idx], 1.0, -1.0)
                values[idx] = _seg_max(q, off) * sign
                break
            ro.step(segment_argmax(q, off).tolist())
        r0 = np.asarray(ro.rewards(0), np.float32)
        done = ~np.isnan(r0)
        rt = np.where(teams == 0, r0, -r0)
        values[done] = rt[done]
        return np.nan_to_num(values, nan=0.0)

    # ------------------------------------------------------------------ API
    @torch.no_grad()
    def evaluate_many(self, games):
        """For every game: [(move_index, score, net_q, searched)] best first, where move_index
        refers to game.legal_moves(). Scores are expected level changes for the player to move."""
        encs = [g.encode() for g in games]
        s = np.stack([np.frombuffer(e[0], np.uint8) for e in encs]).reshape(-1, STATE_DIM)
        h = np.stack([np.frombuffer(e[1], np.uint8) for e in encs]).reshape(-1, HIST_LEN, HIST_DIM)
        a = np.concatenate([np.frombuffer(e[2], np.uint8) for e in encs]).reshape(-1, ACTION_DIM)
        off = np.zeros(len(games) + 1, np.int64)
        off[1:] = np.cumsum([e[3] for e in encs])
        q_all = self._q(s, h, a, off)
        out = [None] * len(games)
        sims, owner, teams = [], [], []
        plan = {}
        for gi, g in enumerate(games):
            q = q_all[off[gi]:off[gi + 1]]
            if len(q) == 1:
                out[gi] = [(0, float(q[0]), float(q[0]), True)]
                continue
            order = np.argsort(-q)
            cands = order[: self.top_k]
            plan[gi] = (q, order, cands)
            moves = g.legal_moves()
            team = g.current % 2
            base = next(_seed) * 7919
            worlds = [g.determinize(base + j) for j in range(self.samples)]
            for cj, ci in enumerate(cands):
                k, key, cards = moves[ci]
                for w in worlds:
                    x = w.copy()
                    x.apply(k, key, list(cards))
                    sims.append(x)
                    owner.append((gi, cj))
                    teams.append(team)
        if sims:
            values = self._simulate(sims, teams)
            sums = {}
            for v, o in zip(values, owner):
                sums[o] = sums.get(o, 0.0) + float(v)
        for gi, (q, order, cands) in plan.items():
            res = []
            for cj, ci in enumerate(cands):
                mean = sums[(gi, cj)] / self.samples
                res.append((int(ci), (1 - self.prior) * mean + self.prior * float(q[ci]), float(q[ci]), True))
            res.sort(key=lambda r: -r[1])
            floor = res[-1][1]
            # moves the search did not examine keep the network's order, below every searched move
            res += [(int(i), floor - 0.05 - r * 1e-3, float(q[i]), False) for r, i in enumerate(order[self.top_k:])]
            out[gi] = res
        return out

    def evaluate(self, game):
        """[(move, score, net_q, searched)] best first (move = (kind, key, cards))."""
        moves = game.legal_moves()
        return [(moves[i], sc, q, searched) for i, sc, q, searched in self.evaluate_many([game])[0]]

    def choose(self, game):
        return self.evaluate(game)[0][0]
