"""Duplicate-deal evaluation.

Every deal is played twice with the same cards in the same seats: once with player A on
team 0 (seats 0 and 2) and once with A on team 1 (seats 1 and 3). The luck of the cards
then cancels out, so a few hundred deals separate two players far more reliably than
thousands of ordinary rounds. Deals are single rounds at random levels without tribute
(like a quick game).

Players are objects with `act(ro, obs, rows) -> choice indices`; see NetPlayer,
RulePlayer, RandomPlayer and SearchPlayer.
"""
from __future__ import annotations

import time

import numpy as np

from . import _core
from .agents import BatchPolicy, subset_obs
from .model import STATE_DIM, HIST_LEN, HIST_DIM, ACTION_DIM


class NetPlayer:
    """Plays the move with the best network value."""

    def __init__(self, model, device="cpu"):
        self.policy = BatchPolicy(model, device)

    def act(self, ro, obs, rows):
        _, _, s, h, a, off = obs
        ss, hh, aa, oo = subset_obs(s, h, a, off, rows)
        return self.policy.choose(ss, hh, aa, oo)


class RulePlayer:
    """The hand-written rule bot (only used as a yardstick)."""

    def act(self, ro, obs, rows):
        return np.asarray(ro.rule_choices(), np.int64)[rows]


class RandomPlayer:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, ro, obs, rows):
        off = obs[5]
        n = off[rows + 1] - off[rows]
        return (self.rng.random(len(rows)) * n).astype(np.int64)


class SearchPlayer:
    """Network + search (see search.SearchAgent)."""

    def __init__(self, searcher):
        self.searcher = searcher

    def act(self, ro, obs, rows):
        idx = obs[0]
        games = [ro.game(int(idx[r])) for r in rows]
        return np.asarray([res[0][0] for res in self.searcher.evaluate_many(games)], np.int64)


def make_deals(n, seed):
    rng = np.random.default_rng(seed)
    deals = []
    for _ in range(n):
        g = _core.Game(int(rng.integers(1, 1 << 62)))
        lv = rng.integers(0, 13, 2)
        g.set_levels([int(lv[0]), int(lv[1])], int(rng.integers(0, 2)), True)
        deals.append(g)
    return deals


def _observe(ro):
    idx, players, s, h, a, off = ro.observe()
    idx = np.asarray(idx, np.int64)
    players = np.frombuffer(players, np.uint8)
    s = np.frombuffer(s, np.uint8).reshape(-1, STATE_DIM)
    h = np.frombuffer(h, np.uint8).reshape(-1, HIST_LEN, HIST_DIM)
    a = np.frombuffer(a, np.uint8).reshape(-1, ACTION_DIM)
    return idx, players, s, h, a, np.asarray(off, np.int64)


def duplicate_match(player_a, player_b, deals=200, seed=1, batch=64, log=None, max_seconds=None):
    """Stats for player A: level change per round, round win rate and their standard errors."""
    per_deal = []  # A's level change summed over the two copies of a deal
    wins = rounds = 0
    t0 = time.time()
    done_deals = 0
    while done_deals < deals:
        k = min(batch, deals - done_deals)
        ds = make_deals(k, seed * 1_000_003 + done_deals)
        games = []
        for g in ds:
            games += [g.copy(), g.copy()]
        a_team = np.tile([0, 1], k)
        ro = _core.Rollouts(games)
        while True:
            obs = _observe(ro)
            idx, players = obs[0], obs[1]
            if len(idx) == 0:
                break
            is_a = (players % 2) == a_team[idx]
            ch = np.zeros(len(idx), np.int64)
            for pl, mask in ((player_a, is_a), (player_b, ~is_a)):
                rows = np.nonzero(mask)[0]
                if len(rows):
                    ch[rows] = pl.act(ro, obs, rows)
            ro.step(ch.tolist())
        r0 = np.asarray(ro.rewards(0), np.float32)
        ra = np.where(a_team == 0, r0, -r0)
        per_deal += list(ra[0::2] + ra[1::2])
        wins += int((ra > 0).sum())
        rounds += len(ra)
        done_deals += k
        if log:
            pd = np.asarray(per_deal)
            log(f"{done_deals} deals: score {pd.mean() / 2:+.3f} ± {pd.std() / 2 / np.sqrt(len(pd)):.3f} "
                f"win {wins / rounds * 100:.1f}%  ({time.time() - t0:.0f}s)")
        if max_seconds and time.time() - t0 > max_seconds:
            break
    pd = np.asarray(per_deal, np.float64)
    n = len(pd)
    return {
        "deals": n,
        "rounds": rounds,
        "score": float(pd.mean() / 2),  # per round
        "score_se": float(pd.std(ddof=1) / 2 / np.sqrt(n)) if n > 1 else 0.0,
        "win_rate": wins / max(1, rounds),
        "seconds": time.time() - t0,
    }
