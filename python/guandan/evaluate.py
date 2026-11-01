"""Head-to-head evaluation of policies on many parallel matches.

Win rate is measured per round (deal): a round is won by the team of the player who
finishes first (头游). `score` is the average level change per round for the tested
side (+3/+2/+1 when winning, negative when losing; passing A counts as +3).
"""
from __future__ import annotations

import time

import numpy as np

from . import _core
from .agents import BatchPolicy, unpack_obs, subset_obs


def _choices_for(agent, env, rows, s, h, a, off, rng, rule_cache):
    if len(rows) == 0:
        return np.zeros(0, np.int64)
    if agent == "random":
        n = (off[rows + 1] - off[rows])
        return (rng.random(len(rows)) * n).astype(np.int64)
    if agent == "rule":
        if rule_cache[0] is None:
            rule_cache[0] = np.asarray(env.rule_choices(), np.int64)
        return rule_cache[0][rows]
    ss, hh, aa, oo = subset_obs(s, h, a, off, rows)
    return agent.choose(ss, hh, aa, oo)


def evaluate(agent_a, agent_b, rounds=400, envs=64, seed=12345, max_seconds=None):
    """agent_* is a BatchPolicy, 'random' or 'rule'. Returns stats for agent_a."""
    env = _core.VecEnv(envs, seed, True)
    rng = np.random.default_rng(seed)
    team_a = np.arange(envs) % 2  # agent_a plays team (i % 2) in env i
    wins = 0
    played = 0
    score = 0.0
    double = 0
    t0 = time.time()
    while played < rounds:
        players, s, h, a, off = unpack_obs(env.observe())
        is_a = (players % 2) == team_a
        rows_a = np.nonzero(is_a)[0]
        rows_b = np.nonzero(~is_a)[0]
        cache = [None]
        ch = np.zeros(envs, np.int64)
        ch[rows_a] = _choices_for(agent_a, env, rows_a, s, h, a, off, rng, cache)
        ch[rows_b] = _choices_for(agent_b, env, rows_b, s, h, a, off, rng, cache)
        for (i, win_team, gain, match_over) in env.step(ch.tolist(), False):
            v = 3 if match_over else gain
            played += 1
            if win_team == team_a[i]:
                wins += 1
                score += v
                if gain == 3:
                    double += 1
            else:
                score -= v
        if max_seconds and time.time() - t0 > max_seconds:
            break
    played = max(played, 1)
    return {
        "rounds": played,
        "win_rate": wins / played,
        "score": score / played,
        "double_rate": double / played,
        "seconds": time.time() - t0,
    }


def _utf8_stdio():
    import sys as _s
    for st in (_s.stdout, _s.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main():
    _utf8_stdio()
    import argparse
    import torch
    from .model import load_checkpoint

    ap = argparse.ArgumentParser(description="Evaluate a Guandan model")
    ap.add_argument("model")
    ap.add_argument("--opponent", default="rule", help="rule | random | path to checkpoint")
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--search", default="", help="evaluate with search, e.g. 5,12,14 = top_k,samples,depth")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    m, _ = load_checkpoint(args.model, args.device)
    a = BatchPolicy(m, args.device)
    if args.opponent in ("rule", "random"):
        b = args.opponent
    else:
        mb, _ = load_checkpoint(args.opponent, args.device)
        b = BatchPolicy(mb, args.device)
    if args.search:
        from .search import SearchAgent, evaluate_search
        k, n, d = (int(x) for x in args.search.split(","))
        ag = SearchAgent(m, args.device, top_k=k, samples=n, depth=d)
        r = evaluate_search(ag, b, rounds=args.rounds, envs=min(args.envs, 16),
                            log=lambda t: print(t, flush=True))
        print(f"search({args.search}) vs {args.opponent}: win {r['win_rate']*100:.1f}%  score {r['score']:+.3f}  "
              f"({r['rounds']} rounds, {r['seconds']:.0f}s)")
        return
    r = evaluate(a, b, rounds=args.rounds, envs=args.envs)
    print(f"vs {args.opponent}: win {r['win_rate']*100:.1f}%  score {r['score']:+.3f}  "
          f"double {r['double_rate']*100:.1f}%  ({r['rounds']} rounds, {r['seconds']:.1f}s)")


if __name__ == "__main__":
    main()
