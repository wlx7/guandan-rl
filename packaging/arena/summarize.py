"""Summarize arena results: python summarize.py results/danzero [results/njupt ...] [--json out.json]

Per opponent: rounds (小局) won by our team, mean level change per round for our team
(+3/+2/+1 for a win, negative for a loss) with its standard error, complete games (从 2 打到过 A)
won according to the referee, and how often our client had to fall back.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def summarize(path):
    rounds, games_won, games, stats = [], 0, 0, []
    draws = 0
    for f in sorted(glob.glob(os.path.join(path, "t*", "ours.jsonl"))):
        recs = [json.loads(line) for line in open(f) if line.strip()]
        clients = sorted({r["client"] for r in recs if r.get("event") == "round"})
        if not clients:
            continue
        main = clients[0]  # both of our seats log every round; count it once
        rounds += [r for r in recs if r.get("event") == "round" and r["client"] == main]
        stats += [r for r in recs if r.get("event") == "stats"]
        res = [r for r in recs if r.get("event") == "gameResult" and r["client"] == main]
        pos = next((r["pos"] for r in recs if r.get("client") == main and "pos" in r), None)
        if res and pos is not None:
            v = res[-1]["victoryNum"]
            ours, theirs = v[pos], v[(pos + 1) % 4]
            games_won += ours
            games += ours + theirs
            draws += res[-1].get("draws", [0])[pos] if isinstance(res[-1].get("draws"), list) else 0
    n = len(rounds)
    wins = sum(r["win"] for r in rounds)
    lv = [r["levels"] for r in rounds]
    mean = sum(lv) / n if n else 0.0
    sd = math.sqrt(sum((x - mean) ** 2 for x in lv) / (n - 1)) if n > 1 else 0.0
    dec = sum(s.get("decisions", 0) for s in stats)
    return {
        "opponent": os.path.basename(path.rstrip("/")),
        "rounds": n,
        "round_win_rate": wins / n if n else 0.0,
        "round_win_ci95": wilson(wins, n),
        "levels_per_round": mean,
        "levels_se": sd / math.sqrt(n) if n else 0.0,
        "games": games,
        "games_won": games_won,
        "game_win_rate": games_won / games if games else 0.0,
        "game_win_ci95": wilson(games_won, games),
        "draws": draws,
        "decisions": dec,
        "desync": sum(s.get("desync", 0) for s in stats),
        "fallback": sum(s.get("fallback", 0) for s in stats),
        "desynced_rounds": sum(r.get("desynced", False) for r in rounds),
    }


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    if out in args:
        args.remove(out)
    res = [summarize(p) for p in args]
    for r in res:
        lo, hi = r["round_win_ci95"]
        glo, ghi = r["game_win_ci95"]
        print(f"vs {r['opponent']}: rounds {r['rounds']}, won {r['round_win_rate']:.1%} [{lo:.1%}, {hi:.1%}], "
              f"levels/round {r['levels_per_round']:+.3f} ± {r['levels_se']:.3f}; "
              f"games {r['games_won']}/{r['games']} = {r['game_win_rate']:.1%} [{glo:.1%}, {ghi:.1%}]; "
              f"decisions {r['decisions']}, desync {r['desync']}, fallback {r['fallback']}")
    if out:
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
