"""Inference-time search on top of the trained network ("AI 深度思考").

For a decision the network first scores every legal move (Q values). The best `top_k`
candidates are then tested in `samples` sampled worlds: the hidden hands of the other three
players are re-dealt at random, consistently with everything the acting player has seen
(cards played, card counts, tribute). In every world each candidate is played and the game
is continued by the network itself for all four seats, for `depth` moves; the position is
then scored by the network's own value (the best Q of the player to move) or, if the round
already ended, by the actual result. The candidate with the best average wins.

No hand-written knowledge is involved: the network provides candidates, the play-out
policy and the leaf values; search only spends more computation per decision.
"""
from __future__ import annotations

import itertools

import numpy as np
import torch

from . import _core
from .agents import BatchPolicy, unpack_obs
from .model import STATE_DIM, HIST_LEN, HIST_DIM, ACTION_DIM

_seed = itertools.count(1)


def _seg_max(q, off):
    return np.maximum.reduceat(q, off[:-1]) if len(off) > 1 else np.zeros(0, np.float32)


class SearchAgent:
    def __init__(self, model, device="cpu", top_k=5, samples=16, depth=16, prior=0.25, scoring=None):
        self.model = model
        self.device = torch.device(device)
        self.policy = BatchPolicy(model, device)
        self.top_k = top_k
        self.samples = samples
        # with a score / multiplier system the play-outs run to the end of the round so that
        # every bomb that is played (and every multiplier it adds) is counted exactly
        self.depth = 400 if scoring else depth
        self.prior = prior  # weight of the network's own Q in the final score (stabilises small searches)
        self.scoring = scoring  # dict(doubles=[...4 bools], cap=int) or None (= level gain)

    @torch.no_grad()
    def net_q(self, game):
        s, h, a, n = game.encode()
        d = self.device
        st = torch.frombuffer(bytearray(s), dtype=torch.uint8).view(1, STATE_DIM).to(d).float()
        ht = torch.frombuffer(bytearray(h), dtype=torch.uint8).view(1, HIST_LEN, HIST_DIM).to(d).float()
        at = torch.frombuffer(bytearray(a), dtype=torch.uint8).view(n, ACTION_DIM).to(d).float()
        z = self.model.encode_state(st, ht).expand(n, -1)
        return self.model.score(z, at).float().cpu().numpy() * 3.0  # in levels

    @torch.no_grad()
    def evaluate(self, game):
        """[(move, score, net_q, searched)] sorted best first.
        Scores are expected level changes, or expected round scores when a scoring system is set."""
        moves = game.legal_moves()
        q = self.net_q(game)
        if len(moves) == 1:
            return [(moves[0], float(q[0]), float(q[0]), True)]
        order = np.argsort(-q)
        cands = list(order[: self.top_k])
        team = game.current % 2
        sc = self.scoring
        mult0 = game.multiplier(sc["doubles"], sc["cap"]) if sc else 1
        q = q * mult0 if sc else q  # network values are in levels; scale to the current stake
        base = next(_seed) * 7919
        worlds = [game.determinize(base + j) for j in range(self.samples)]
        sims = []
        for ci in cands:
            k, key, cards = moves[ci]
            for w in worlds:
                g = w.copy()
                g.apply(k, key, list(cards))
                sims.append(g)
        ro = _core.Rollouts(sims)
        values = np.full(len(sims), np.nan, np.float32)
        for step in range(self.depth + 1):
            idx, players, s, h, a, off = ro.observe()
            if not idx:
                break
            players = np.frombuffer(players, np.uint8)
            s = np.frombuffer(s, np.uint8).reshape(-1, STATE_DIM)
            h = np.frombuffer(h, np.uint8).reshape(-1, HIST_LEN, HIST_DIM)
            a = np.frombuffer(a, np.uint8).reshape(-1, ACTION_DIM)
            off = np.asarray(off, np.int64)
            qv = self.policy.q_values(s, h, a, off)
            if step == self.depth:
                leaf = _seg_max(qv, off) * 3.0
                sign = np.where(players % 2 == team, 1.0, -1.0)
                if sc:
                    mult = np.asarray(ro.multipliers(sc["doubles"], sc["cap"]), np.float32)[np.asarray(idx)]
                    leaf = leaf * mult
                values[np.asarray(idx)] = leaf * sign
                break
            from .agents import segment_argmax
            ro.step(segment_argmax(qv, off).tolist())
        r = np.asarray(ro.scores(team, sc["doubles"], sc["cap"]) if sc else ro.rewards(team), np.float32)
        done = ~np.isnan(r)
        values[done] = r[done]
        values = np.nan_to_num(values, nan=0.0)
        mean = values.reshape(len(cands), self.samples).mean(1)
        out = []
        for j, ci in enumerate(cands):
            score = (1 - self.prior) * mean[j] + self.prior * q[ci]
            out.append((moves[ci], float(score), float(q[ci]), True))
        out.sort(key=lambda x: -x[1])
        floor = out[-1][1]
        # moves the search did not examine keep the network's order, below every searched move
        rest = [(moves[i], floor - 0.05 - r * 1e-3, float(q[i]), False) for r, i in enumerate(order[self.top_k:])]
        return out + rest

    def choose(self, game):
        return self.evaluate(game)[0][0]


def evaluate_search(agent: SearchAgent, opponent="rule", rounds=200, envs=16, seed=4242, log=None):
    """Round win rate of the search agent's team against 'rule' / 'random' / BatchPolicy."""
    import time
    env = _core.VecEnv(envs, seed, True)
    rng = np.random.default_rng(seed)
    team_a = np.arange(envs) % 2
    wins = played = 0
    score = 0.0
    t0 = time.time()
    while played < rounds:
        players, s, h, a, off = unpack_obs(env.observe())
        is_a = (players % 2) == team_a
        ch = np.zeros(envs, np.int64)
        rows_b = np.nonzero(~is_a)[0]
        if len(rows_b):
            if opponent == "rule":
                ch[rows_b] = np.asarray(env.rule_choices(), np.int64)[rows_b]
            elif opponent == "random":
                ch[rows_b] = (rng.random(len(rows_b)) * (off[rows_b + 1] - off[rows_b])).astype(np.int64)
            else:
                from .agents import subset_obs
                ss, hh, aa, oo = subset_obs(s, h, a, off, rows_b)
                ch[rows_b] = opponent.choose(ss, hh, aa, oo)
        for i in np.nonzero(is_a)[0]:
            g = env.game(int(i))
            best = agent.evaluate(g)[0][0]
            moves = g.legal_moves()
            ch[i] = next(j for j, m in enumerate(moves) if m[0] == best[0] and m[1] == best[1] and bytes(m[2]) == bytes(best[2]))
        for (i, win_team, gain, match_over) in env.step(ch.tolist(), False):
            v = 3 if match_over else gain
            played += 1
            if win_team == team_a[i]:
                wins += 1
                score += v
            else:
                score -= v
            if log and played % 20 == 0:
                log(f"{played} rounds: win {wins / played * 100:.1f}%  score {score / played:+.2f}  ({time.time() - t0:.0f}s)")
    return {"rounds": played, "win_rate": wins / played, "score": score / played, "seconds": time.time() - t0}
