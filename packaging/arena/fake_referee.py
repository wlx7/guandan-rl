"""Local test for our_client.Player: our own engine plays the referee and sends the same
JSON messages as the competition platform to four in-process players. Every decision must
be reconstructed without a desync.

    python packaging/arena/fake_referee.py --rounds 20
"""
from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from guandan import _core  # noqa: E402
from guandan.agents import ENDGAME_FAST, GameAgent  # noqa: E402
from guandan.paths import default_model  # noqa: E402
from our_client import RANKS, SUITS, Player  # noqa: E402

LEVEL = [0]
NAMES = {0: "PASS", 1: "Single", 2: "Pair", 3: "Trips", 4: "ThreeWithTwo", 5: "Straight",
         6: "ThreePair", 7: "TwoTrips", 8: "Bomb", 9: "StraightFlush", 10: "Bomb"}


def cstr(t):
    if t == 52:
        return "SB"
    if t == 53:
        return "HR"
    return SUITS[t % 4] + RANKS[t // 4]


def fmt(m):
    if m[0] == 0:
        return ["PASS", "PASS", "PASS"]
    return [NAMES[m[0]], ref_rank(m), [cstr(c) for c in m[2]]]


def ref_rank(m):
    """The rank the competition referee would declare: lowest card of a sequence, else the rank."""
    kind, key = m[0], m[1]
    if kind in (5, 6, 7, 9):
        return "A" if key == 0 else RANKS[key - 1]
    if key >= 14:
        return "B" if key == 14 else "R"
    return RANKS[LEVEL[0]] if key == 13 else RANKS[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--endgame", action="store_true")
    args = ap.parse_args()
    random.seed(args.seed)
    agent = GameAgent(str(default_model()), "cpu", endgame=dict(ENDGAME_FAST) if args.endgame else None)
    log = "/tmp/claude-0/work/fake.jsonl"
    players = [Player(agent, log, f"p{i}") for i in range(4)]
    g = _core.Game(args.seed)
    decisions = 0
    mismatch, nback = [0], [0]
    for rnd in range(args.rounds):
        tl = g.team_level
        LEVEL[0] = g.level
        for p in range(4):
            players[p].on_message({"type": "notify", "stage": "beginning", "myPos": p,
                                   "handCards": [cstr(c) for c in g.hand(p)]})

        def ranks(p):
            return {"selfRank": RANKS[tl[p & 1]], "oppoRank": RANKS[tl[1 - (p & 1)]], "curRank": RANKS[g.level]}

        if g.anti_tribute:
            for pl in players:
                pl.on_message({"type": "notify", "stage": "anti-tribute", "antiNums": 1, "antiPos": []})
        trib = [[f, t, cstr(c)] for f, t, c, r in g.tribute() if not r]
        if trib:
            for pl in players:
                pl.on_message({"type": "notify", "stage": "tribute", "result": trib})
        backs = []
        while g.phase == 0:
            p = g.current
            moves = g.legal_moves()
            msg = {"type": "act", "stage": "back", "handCards": [cstr(c) for c in g.hand(p)],
                   "actionList": [["back", "back", [cstr(m[2][0])]] for m in moves], **ranks(p)}
            i = players[p].on_message(msg)
            m = moves[i]
            nback[0] += 1
            if agent.evaluate(g)[0][0][2] != m[2]:
                mismatch[0] += 1
            g.apply(*m)
            backs.append([p, g.tribute()[-1][1], cstr(m[2][0])])
        if backs:
            for pl in players:
                pl.on_message({"type": "notify", "stage": "back", "result": backs})
        while g.phase == 1:
            p = g.current
            moves = g.legal_moves()
            msg = {"type": "act", "stage": "play", "handCards": [cstr(c) for c in g.hand(p)],
                   "publicInfo": [{"rest": n, "playArea": None} for n in g.hand_counts()],
                   "actionList": [fmt(m) for m in moves], "greaterAction": fmt(g.last_move()) if g.last_move() else None,
                   "greaterPos": g.last_player if g.last_move() else -1, **ranks(p)}
            i = players[p].on_message(msg)
            decisions += 1
            m = moves[i]
            if not args.endgame and len(moves) > 1:
                true_best = agent.evaluate(g)[0][0]
                if sorted(true_best[2]) != sorted(m[2]):
                    mismatch[0] += 1
            g.apply(*m)
            for pl in players:
                pl.on_message({"type": "notify", "stage": "play", "curPos": p, "curAction": fmt(m)})
        order = list(g.last_result()[0])
        for pl in players:
            pl.on_message({"type": "notify", "stage": "episodeOver", "order": order, "curRank": RANKS[g.level]})
        if g.phase == 3:
            for pl in players:
                pl.on_message({"type": "notify", "stage": "gameOver", "curTimes": 1, "settingTimes": 1})
            g.new_match()
        else:
            g.next_round()
    st = [pl.stats for pl in players]
    print("decisions", decisions, "stats", st)
    print("returns", nback[0], "mismatches vs true state", mismatch[0])
    bad = sum(s["desync"] + s["fallback"] for s in st) + mismatch[0]
    print("OK" if bad == 0 else f"FAIL: {bad} desyncs/fallbacks")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
