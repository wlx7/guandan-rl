"""Our AI as a client of the national GuanDan AI competition platform (首届中国人工智能掼蛋算法大赛
offline referee, WebSocket + JSON on ws://127.0.0.1:23456/game/<name>).

The referee only tells each player its own hand and the public moves. The client keeps a
shadow game of our engine from this player's point of view: its own hand is exact, the other
hands are placeholders of the right size, and before an opponent's move is replayed its cards
are swapped into that placeholder hand (Game.ensure_holds). The network only ever reads the
player's own hand and public information (see src/features.rs), so the placeholders never
leak anything; the endgame search re-samples the hidden hands itself.

Every finished round is appended to --log as one JSON line.

    python our_client.py --name ours0 --log results.jsonl [--model path.pt] [--no-endgame]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import websocket  # websocket-client

from guandan import _core
from guandan.agents import ENDGAME_FAST, GameAgent

RANKS = "23456789TJQKA"
SUITS = "SHCD"
KIND = {"PASS": 0, "Single": 1, "Pair": 2, "Trips": 3, "ThreeWithTwo": 4, "Straight": 5,
        "ThreePair": 6, "TwoTrips": 7, "Bomb": 8, "StraightFlush": 9}


def card(s: str) -> int:
    if s == "SB":
        return 52
    if s == "HR":
        return 53
    return RANKS.index(s[1]) * 4 + SUITS.index(s[0])


def cards(lst) -> list[int]:
    if not lst or lst == "PASS":
        return []
    return [card(c) for c in lst]


def rank(r: str) -> int:
    return RANKS.index(r)


def move_key(kind: int, cs) -> tuple:
    return (kind, tuple(sorted(cs)))


class Player:
    def __init__(self, agent: GameAgent, log_path: str, name: str):
        self.agent = agent
        self.log_path = log_path
        self.name = name
        self.pos = None
        self.banker = -1         # team that won the last round (-1 at the start of a match)
        self.a_fail = [0, 0]     # rounds a team failed to pass A as the banker (engine semantics)
        self.view = None         # (team levels, banker, level) of the current round
        self.stats = {"decisions": 0, "desync": 0, "fallback": 0, "rounds": 0}
        self.new_round()

    def new_round(self):
        self.tribute = []        # (from, to, card, is_return)
        self.anti = False
        self.plays = []          # (pos, kind, cards) of this round, in order
        self.game = None
        self.applied = 0
        self.desynced = False
        self.view = None

    # ---------------------------------------------------------------- view of the round
    def levels(self, msg):
        sr, orr, cr = rank(msg["selfRank"]), rank(msg["oppoRank"]), rank(msg["curRank"])
        my = self.pos & 1
        tl = [0, 0]
        tl[my], tl[1 - my] = sr, orr
        if sr == cr and orr != cr:
            banker = my
        elif orr == cr and sr != cr:
            banker = 1 - my
        else:
            banker = self.banker
        for t in range(2):
            if tl[t] != 12:
                self.a_fail[t] = 0
        self.view = (tl, banker, cr)
        return tl, banker, cr

    def hand_sizes_after_tribute(self):
        n = [27, 27, 27, 27]
        for f, t, _c, _r in self.tribute:
            n[f] -= 1
            n[t] += 1
        return n

    def placeholder_hands(self, mine, sizes):
        """Four hands: our real one and the unseen cards dealt to the others by size."""
        pool = [t for t in range(54) for _ in range(2)]
        for c in mine:
            pool.remove(c)
        random.shuffle(pool)
        hands, k = [], 0
        for p in range(4):
            if p == self.pos:
                hands.append(list(mine))
            else:
                hands.append(pool[k:k + sizes[p]])
                k += sizes[p]
        return hands

    def build_play_view(self, msg):
        tl, banker, lv = self.levels(msg)
        mine = cards(msg["handCards"])
        # our hand now = hand at the start of play + nothing played by us yet (built at our
        # first decision), unless we already played: then add our played cards back
        for p, _k, cs in self.plays:
            if p == self.pos:
                mine += cs
        first = self.plays[0][0] if self.plays else self.pos
        g = _core.Game(random.getrandbits(63))
        hands = self.placeholder_hands(mine, [27, 27, 27, 27])
        g.set_view(tl, banker, lv, hands, first, tribute=self.tribute, anti_tribute=self.anti, a_fail=self.a_fail)
        self.game = g
        self.applied = 0
        self.catch_up()

    def catch_up(self):
        g = self.game
        while self.applied < len(self.plays) and not self.desynced:
            p, kind, cs = self.plays[self.applied]
            self.applied += 1
            if g.current != p:
                self.desync(f"turn: engine {g.current} referee {p}")
                return
            if kind == 0:
                if g.is_leading:
                    self.desync("pass while leading")
                    return
                g.apply(0, 0, [])
                continue
            try:
                if p != self.pos:
                    g.ensure_holds(p, self.pos, cs)
                opts = g.legal_interpretations(cs)
                pick = [m for m in opts if m[0] == kind] or opts
                if not pick:
                    self.desync(f"no interpretation of {cs} kind {kind}")
                    return
                g.apply(*pick[0])
            except Exception as e:  # noqa: BLE001
                self.desync(str(e))
                return

    def desync(self, why):
        self.desynced = True
        self.stats["desync"] += 1
        print(f"[{self.name}] desync: {why}", file=sys.stderr, flush=True)

    # ---------------------------------------------------------------- decisions
    def choose_index(self, actions, game):
        """Best index into the referee's actionList by our network (+ endgame search)."""
        best = self.agent.choose(game)  # network (+ exact endgame search near the end)
        want = tuple(sorted(best[2]))
        same = [i for i, a in enumerate(actions) if tuple(sorted(cards(a[2]))) == want]
        if same:  # the referee may name a kind differently (e.g. straight flush), cards decide
            return same[0], False
        # our move is not offered by the referee: take its best-valued move instead
        qc = {}
        for m, q in self.agent.evaluate(game):
            qc.setdefault(tuple(sorted(m[2])), q)
        scored = sorted(((qc.get(tuple(sorted(cards(a[2]))), -1e9), i) for i, a in enumerate(actions)), reverse=True)
        print(f"[{self.name}] our move {best} not offered; referee offers {len(actions)}: "
              f"{actions[:6]}; took {actions[scored[0][1]]} (q {scored[0][0]:.3f})", file=sys.stderr, flush=True)
        return scored[0][1], True

    def act_play(self, msg):
        actions = msg["actionList"]
        if len(actions) == 1:
            return 0
        if self.game is None:
            self.build_play_view(msg)
        else:
            self.catch_up()
        if self.desynced or self.game.current != self.pos:
            if not self.desynced:
                self.desync(f"not our turn in engine ({self.game.current} vs {self.pos})")
            # rebuild from scratch: lose the history but keep hand / levels / trick
            return self.fallback_play(msg)
        self.stats["decisions"] += 1
        i, fb = self.choose_index(actions, self.game)
        if fb:
            self.stats["fallback"] += 1
        return i

    def fallback_play(self, msg):
        """After a desync: score the referee's moves on a fresh view without history."""
        self.stats["fallback"] += 1
        tl, banker, lv = self.levels(msg)
        mine = cards(msg["handCards"])
        rest = [pi["rest"] for pi in msg["publicInfo"]]
        g = _core.Game(random.getrandbits(63))
        g.set_view(tl, banker, lv, self.placeholder_hands(mine, rest), self.pos, tribute=self.tribute, a_fail=self.a_fail)
        best, bq = 0, -1e9
        actions = msg["actionList"]
        ga = msg.get("greaterAction")
        if ga and ga[0] != "PASS" and msg.get("greaterPos", -1) not in (-1, self.pos):
            # make the trick's leader play the move to beat, so that legal moves match
            gp = msg["greaterPos"]
            g2 = _core.Game(random.getrandbits(63))
            hands = self.placeholder_hands(mine, rest)
            g2.set_view(tl, banker, lv, hands, gp, tribute=self.tribute, a_fail=self.a_fail)
            try:
                gcs = cards(ga[2])
                g2.ensure_holds(gp, self.pos, gcs)
                opts = g2.legal_interpretations(gcs)
                g2.apply(*opts[0])
                while g2.current != self.pos:
                    g2.apply(0, 0, [])
                g = g2
            except Exception:  # noqa: BLE001
                return 0 if actions[0][0] != "PASS" or len(actions) == 1 else 1
        ev = self.agent.evaluate(g)
        qc = {tuple(sorted(m[2])): q for m, q in reversed(ev)}
        for i, a in enumerate(actions):
            q = qc.get(tuple(sorted(cards(a[2]))), -1e9)
            if q > bq:
                best, bq = i, q
        return best

    def act_back(self, msg):
        actions = msg["actionList"]
        if len(actions) == 1:
            return 0
        tl, banker, lv = self.levels(msg)
        mine = cards(msg["handCards"])
        sizes = self.hand_sizes_after_tribute()
        for f, t, _c, r in self.tribute:
            if r:  # already returned
                sizes[f] -= 1
                sizes[t] += 1
        sizes[self.pos] = len(mine)
        payer = next((f for f, t, _c, r in self.tribute if not r and t == self.pos), None)
        if payer is None:
            return 0
        done = {(f, t) for f, t, _c, r in self.tribute if r}
        pending = [(self.pos, payer)] + [(t, f) for f, t, _c, r in self.tribute
                                         if not r and t != self.pos and (t, f) not in done]
        g = _core.Game(random.getrandbits(63))
        try:
            g.set_view(tl, banker, lv, self.placeholder_hands(mine, sizes), self.pos, tribute=self.tribute,
                       a_fail=self.a_fail, phase=0, pending_returns=pending, lead_after_tribute=payer)
            mv = self.agent.choose(g)
        except Exception as e:  # noqa: BLE001
            print(f"[{self.name}] back failed: {e}", file=sys.stderr, flush=True)
            return 0
        for i, a in enumerate(actions):
            if cards(a[2]) == list(mv[2]):
                return i
        return 0

    # ---------------------------------------------------------------- messages
    def on_message(self, msg) -> int | None:
        typ, stage = msg.get("type"), msg.get("stage")
        if typ == "notify":
            if stage == "beginning":
                self.pos = msg["myPos"]
                self.new_round()
            elif stage == "tribute":
                for f, t, c in msg["result"]:
                    self.tribute.append((f, t, card(c), False))
            elif stage == "back":
                for f, t, c in msg["result"]:
                    self.tribute.append((f, t, card(c), True))
            elif stage == "anti-tribute":
                self.anti = True
            elif stage == "play":
                a = msg["curAction"]
                kind = KIND.get(a[0], -1)
                cs = cards(a[2])
                if kind == 8 and cs and min(cs) >= 52:
                    kind = 10
                self.plays.append((msg["curPos"], kind, cs))
                if self.game is not None:
                    self.catch_up()
            elif stage == "episodeOver":
                self.end_round(msg)
            elif stage == "gameOver":
                self.banker, self.a_fail = -1, [0, 0]
                self.write({"event": stage, **{k: v for k, v in msg.items() if k not in ("type", "stage")}})
            elif stage == "gameResult":
                self.write({"event": stage, **{k: v for k, v in msg.items() if k not in ("type", "stage")}})
            return None
        if typ == "act":
            if stage == "play":
                return self.act_play(msg)
            if stage == "back":
                return self.act_back(msg)
            return 0  # tribute: the referee offers only the largest cards
        return None

    def end_round(self, msg):
        order = msg["order"]
        my = self.pos & 1
        first = order[0]
        partner_place = order.index((first + 2) % 4)
        gain = {1: 3, 2: 2, 3: 1}[partner_place]
        win = (first & 1) == my
        if self.view is not None:  # A-level bookkeeping as in our engine (game.rs finish_round)
            tl, b, lv = self.view
            if b >= 0 and lv == 12 and tl[b] == 12 and not ((first & 1) == b and gain >= 2):
                self.a_fail[b] += 1
                if self.a_fail[b] >= 3:
                    self.a_fail[b] = 0
        self.banker = first & 1
        self.stats["rounds"] += 1
        self.write({"event": "round", "pos": self.pos, "order": order, "rank": msg.get("curRank"),
                    "win": win, "levels": gain if win else -gain, "desynced": self.desynced})
        self.new_round()

    def write(self, rec):
        rec["client"] = self.name
        rec["t"] = round(time.time(), 1)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="ours")
    ap.add_argument("--url", default="ws://127.0.0.1:23456/game/")
    ap.add_argument("--log", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--no-endgame", action="store_true")
    args = ap.parse_args()
    from guandan.paths import default_model  # noqa: PLC0415
    model = args.model or default_model()
    agent = GameAgent(str(model), "cpu", endgame=None if args.no_endgame else dict(ENDGAME_FAST))
    player = Player(agent, args.log, args.name)
    ws = websocket.create_connection(args.url + args.name, timeout=None)
    try:
        while True:
            raw = ws.recv()
            if not raw:
                break
            msg = json.loads(raw)
            idx = player.on_message(msg)
            if idx is not None:
                ws.send(json.dumps({"actIndex": int(idx)}))
            if msg.get("stage") == "gameResult":
                break
    except (websocket.WebSocketConnectionClosedException, ConnectionError):
        pass
    finally:
        player.write({"event": "stats", **player.stats})
        print(f"[{args.name}] {player.stats}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
