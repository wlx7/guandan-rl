"""A single game table: seats (human / AI), turn loop, timers, hints and per-player views."""
from __future__ import annotations

import asyncio
import random
import secrets
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from .. import _core
from .. import scoring
from ..agents import GameAgent, random_move

PHASE_RETURN, PHASE_PLAY, PHASE_ROUND_OVER, PHASE_MATCH_OVER = 0, 1, 2, 3
KIND_PASS, KIND_TRIBUTE = 0, 11
BOMB_KINDS = (8, 9, 10)

_executor = ThreadPoolExecutor(max_workers=2)
_agents: dict[tuple, GameAgent] = {}

# "deep": the network proposes candidates and search tests them in sampled worlds
SEARCH_SETTINGS = {"top_k": 5, "samples": 24, "depth": 14}


def get_agent(path: str, think: str = "fast") -> GameAgent:
    key = (path, think)
    if key not in _agents:
        search = dict(SEARCH_SETTINGS) if think == "deep" else None
        _agents[key] = GameAgent(path, "cpu", search=search)
    return _agents[key]


class Seat:
    def __init__(self, index, kind, model=None, name=""):
        self.index = index
        self.kind = kind  # 'human' | 'ai'
        self.model = model  # checkpoint path for AI seats
        self.name = name
        self.token = None
        self.connections: set = set()
        self.ready = False

    @property
    def taken(self):
        return self.token is not None

    @property
    def connected(self):
        return bool(self.connections)


class Room:
    def __init__(self, mode: str, seats: list[dict], hint_model: str | None, timeout: int = 20,
                 ai_speed: float = 1.0, seed: int | None = None, length: str = "full", think: str = "fast",
                 scoring_name: str = "none"):
        assert mode in ("teach", "real")
        assert length in ("quick", "full")
        self.id = secrets.token_hex(3)
        self.mode = mode
        self.length = length  # quick: random level, independent deals; full: play from 2 to A
        self.think = "deep" if think == "deep" else "fast"
        # multipliers only apply to quick games (single deals); the AI plays the same either way
        self.scoring = scoring_name if (length == "quick" and scoring.get(scoring_name)) else "none"
        self.totals = [0, 0]  # accumulated scores per team when a scoring system is used
        self.timeout = int(timeout)
        self.ai_speed = ai_speed
        self.hint_model = hint_model
        self.seats = [Seat(i, s["kind"], s.get("model"), s.get("name") or "") for i, s in enumerate(seats)]
        for s in self.seats:
            if s.kind == "ai" and not s.name:
                s.name = f"AI {'南东北西'[s.index]}"
        self.spectators: set = set()
        self.rng = random.Random(seed)
        self.points = [0, 0]  # quick mode: accumulated points per team
        self.deals = 1
        self.game = self.new_game()
        self.status = "waiting"
        self.seq = 0
        self.event_no = 0
        self.event: dict = {"type": "deal", "id": 0}
        self.trick: dict[int, dict] = {}
        self.deadline: float | None = None
        self.inbox = [asyncio.Queue() for _ in range(4)]
        self.task: asyncio.Task | None = None
        self.closed = False
        self.created = time.time()
        self.match_no = 1
        self.score = [0, 0]  # matches won per team
        self.round_log: list[dict] = []
        self.review: dict | None = None
        self.paused = False  # play is stopped (see can_run) and goes on from the same spot later
        self.manual_pause: str | None = None  # name of who pressed "暂停", None when not paused by hand
        self.owner = ""  # device id of the browser that opened the table (may manage it like the host)
        self.idents: dict = {}  # websocket -> who it is (see app.ident)
        self.last_active = time.time()

    def new_event(self, ev: dict):
        self.event_no += 1
        ev["id"] = self.event_no
        self.event = ev

    def new_game(self):
        if self.length == "full":
            return _core.Game(self.rng.getrandbits(48))
        g = _core.Game(self.rng.getrandbits(48), tribute=False)
        lv = self.rng.randrange(13)
        g.set_levels([lv, lv], self.rng.randrange(2), True)
        return g

    # ------------------------------------------------------------------ seats
    def human_seats(self):
        return [s for s in self.seats if s.kind == "human"]

    def someone_here(self):
        """A table with human players runs while at least one of them is connected; an all-AI
        table runs while someone watches it."""
        humans = self.human_seats()
        if humans:
            return any(s.connected for s in humans)
        return bool(self.spectators)

    def can_manage(self, ident: dict | None) -> bool:
        """The host, or the device that opened this table, may pause / end / close it."""
        if not ident:
            return False
        return bool(ident.get("admin")) or bool(self.owner and ident.get("device") == self.owner)

    def can_run(self):
        """Play goes on unless someone pressed pause or nobody is at the table."""
        return self.manual_pause is None and self.someone_here()

    async def hold(self):
        """Pause while nobody is at the table (play when you like, stop when you leave) or while
        paused by hand; the game goes on from the same spot afterwards."""
        if self.closed or self.can_run():
            return
        self.paused = True
        self.deadline = None
        await self.broadcast()
        while not self.closed and not self.can_run():
            await asyncio.sleep(0.3)
        self.paused = False
        await self.broadcast()
        await asyncio.sleep(0.8)  # let the returning player see the table before play goes on

    def set_pause(self, on: bool, by: str = ""):
        self.manual_pause = (by or "有人") if on else None

    def join(self, seat_index: int | None, name: str, token: str | None = None):
        if token:
            for s in self.seats:
                if s.token == token:
                    if name:
                        s.name = name
                    return s.index, token
        if seat_index is None:
            free = [s for s in self.seats if s.kind == "human" and not s.taken]
            if not free:
                raise ValueError("没有空余的玩家座位")
            seat_index = free[0].index
        s = self.seats[seat_index]
        if s.kind != "human":
            raise ValueError("该座位是 AI")
        if s.taken:
            raise ValueError("该座位已有人")
        s.token = secrets.token_urlsafe(12)
        s.name = name or f"玩家{seat_index + 1}"
        return seat_index, s.token

    def seat_of_token(self, token):
        for s in self.seats:
            if s.token and s.token == token:
                return s.index
        return None

    def summary(self):
        """What the table list shows: progress, who is playing, paused or not."""
        g = self.game
        info = self.public_info()
        info.update({
            "created": self.created,
            "last_active": self.last_active,
            "paused": self.paused or self.manual_pause is not None,
            "pause_by": self.manual_pause,
            "level": g.level,
            "team_level": list(g.team_level),
            "round_no": g.round_no if self.length == "full" else self.deals,
            "match_no": self.match_no,
            "score": self.score,
            "points": self.points,
            "humans": sum(1 for s in self.seats if s.kind == "human"),
            "connected": sum(1 for s in self.seats if s.kind == "human" and s.connected),
            "spectators": len(self.spectators),
        })
        return info

    def public_info(self):
        return {
            "id": self.id,
            "mode": self.mode,
            "length": self.length,
            "think": self.think,
            "scoring": self.scoring,
            "status": self.status,
            "timeout": self.timeout,
            "seats": [{"index": s.index, "kind": s.kind, "name": s.name, "taken": s.taken,
                       "connected": s.connected,
                       "model": s.model.split("/")[-1].split("\\")[-1] if s.model else None}
                      for s in self.seats],
        }

    def fill_ai(self):
        for s in self.seats:
            if s.kind == "human" and not s.taken:
                s.kind = "ai"
                s.model = self.hint_model
                s.name = f"AI {'南东北西'[s.index]}"

    def maybe_start(self):
        if self.status == "waiting" and all(s.taken and s.connected for s in self.human_seats()):
            self.status = "playing"
            self.new_event({"type": "deal"})
            self.task = asyncio.get_running_loop().create_task(self.loop())
            return True
        return False

    def close(self):
        self.closed = True
        if self.task:
            self.task.cancel()

    async def end(self, message: str):
        """Close the table for good and send everybody at it back to the home page."""
        conns = [w for s in self.seats for w in s.connections] + list(self.spectators)
        self.close()
        await asyncio.gather(*(self.send(w, {"type": "closed", "message": message}) for w in conns))

    # ------------------------------------------------------------------ views
    def move_info(self, kind, key, cards):
        cards = list(cards)
        return {"kind": kind, "key": key, "cards": cards,
                "desc": self.game.describe(kind, key, cards) if kind != KIND_PASS else "不出"}

    def view(self, v: int):
        g = self.game
        counts = list(g.hand_counts())
        finished = list(g.finished)
        st = {
            "type": "state",
            "seq": self.seq,
            "room": self.id,
            "mode": self.mode,
            "length": self.length,
            "points": self.points,
            "think": self.think,
            "scoring": self.scoring,
            "totals": self.totals,
            "status": self.status,
            "you": v,
            "phase": g.phase,
            "current": g.current,
            "level": g.level,
            "wild": g.wild,
            "team_level": list(g.team_level),
            "banker": g.banker,
            "round_no": g.round_no if self.length == "full" else self.deals,
            "match_no": self.match_no,
            "score": self.score,
            "a_fail": list(g.a_fail),
            "is_leading": g.is_leading,
            "counts": counts,
            "finished": finished,
            "seats": [{"name": s.name, "kind": s.kind, "connected": s.connected or s.kind == "ai",
                       "taken": s.taken or s.kind == "ai",
                       "ready": s.ready or s.kind == "ai"} for s in self.seats],
            "paused": self.paused,
            "pause_by": self.manual_pause,
            "trick": {str(k): val for k, val in self.trick.items()},
            "tribute": [{"from": a, "to": b, "card": c, "is_return": r} for (a, b, c, r) in g.tribute()],
            "pending_returns": [list(x) for x in g.pending_returns()],
            "anti_tribute": g.anti_tribute,
            "deadline": int(self.deadline * 1000) if self.deadline else None,
            "server_time": int(time.time() * 1000),
            "timeout": self.timeout if self.mode == "real" else None,
            "event": self.event,
            "hands": {},
            "hint_available": self.mode == "teach" and bool(self.hint_model),
        }
        lm = g.last_move()
        st["last_move"] = dict(self.move_info(*lm), player=g.last_player) if lm else None
        if v >= 0:
            st["hand"] = list(g.hand(v))
        # cards played this round by each player (public; used by the card counter)
        st["played"] = [list(g.played(p)) for p in range(4)]
        visible = set()
        if (v < 0 and not self.human_seats()) or self.mode == "teach":
            visible = {0, 1, 2, 3}
        elif v in finished:
            visible = {(v + 2) % 4}
        for p in visible:
            if p != v:
                st["hands"][str(p)] = list(g.hand(p))
        sc = scoring.get(self.scoring)
        st["bombs"] = list(g.bombs)
        st["multiplier"] = g.multiplier(sc["doubles"], sc["cap"]) if sc else 1
        if g.phase in (PHASE_ROUND_OVER, PHASE_MATCH_OVER):
            r = g.last_result()
            if r:
                order, win_team, gain, match_over, before, after, a_reset = r
                if self.length == "quick":  # every quick deal stands alone: no "过 A", no match to win
                    match_over, a_reset = False, False
                st["result"] = {"order": list(order), "win_team": win_team, "gain": gain,
                                "match_over": match_over, "before": list(before), "after": list(after),
                                "a_reset": a_reset,
                                "score": int(g.round_score(win_team, sc["doubles"], sc["cap"])) if sc else None}
        if self.review and v == self.review.get("seat"):
            st["review"] = self.review
        return st

    async def send(self, ws, msg):
        try:
            await ws.send_json(msg)
        except Exception:
            pass

    async def broadcast(self):
        self.seq += 1
        jobs = []
        for s in self.seats:
            for ws in list(s.connections):
                jobs.append(self.send(ws, self.view(s.index)))
        for ws in list(self.spectators):
            jobs.append(self.send(ws, self.view(-1)))
        if jobs:
            await asyncio.gather(*jobs)

    # ------------------------------------------------------------------ actions
    def _apply(self, p, mv, how="normal"):
        self.last_active = time.time()
        g = self.game
        kind, key, cards = mv[0], mv[1], list(mv[2])
        before_trick = g.trick_id
        phase_before = g.phase
        g.apply(kind, key, cards)
        if phase_before == PHASE_RETURN:
            self.new_event({"type": "return", "from": p, "to": [t for t in g.tribute() if t[3]][-1][1],
                            "card": cards[0], "how": how})
            if g.phase == PHASE_PLAY:
                self.trick = {}
            return
        info = self.move_info(kind, key, cards)
        self.trick[p] = dict(info, pass_=kind == KIND_PASS)
        self.new_event(dict(info, type="pass" if kind == KIND_PASS else "play", player=p, how=how,
                            bomb=kind in BOMB_KINDS, finished=p in g.finished and kind != KIND_PASS))
        if g.phase in (PHASE_ROUND_OVER, PHASE_MATCH_OVER):
            r = g.last_result()
            self.event["round_over"] = True
            sc = scoring.get(self.scoring)
            if r and sc:
                v = int(g.round_score(r[1], sc["doubles"], sc["cap"]))
                self.totals[r[1]] += v
                self.totals[1 - r[1]] -= v
            if r and self.length == "quick":
                gain = int(g.round_score(r[1], sc["doubles"], sc["cap"])) if sc else r[2]
                self.points[r[1]] += gain
                self.points[1 - r[1]] -= gain
            elif r and r[3]:
                self.score[r[1]] += 1
            if r:
                self.round_log.append({"round": g.round_no, "order": list(r[0]), "win_team": r[1],
                                       "gain": r[2], "after": list(r[5])})
        elif g.trick_id != before_trick:
            self.event["trick_end"] = True
            self.event["jiefeng"] = g.jiefeng
            self.event["leader"] = g.current
            self.trick = {}

    def ai_move(self, p):
        seat = self.seats[p]
        g = self.game.copy()
        try:
            agent = get_agent(seat.model, self.think)
            return agent.choose(g)
        except Exception:
            traceback.print_exc()
            moves = g.legal_moves()
            return moves[g.rule_choice()]

    def auto_move(self, p):
        """A player who left while others play on: the AI plays for them (托管)."""
        g = self.game.copy()
        try:
            if self.hint_model:
                return get_agent(self.hint_model, self.think).choose(g)
        except Exception:
            traceback.print_exc()
        return g.legal_moves()[g.rule_choice()]

    def hints(self, k=5):
        g = self.game.copy()
        agent = get_agent(self.hint_model, self.think)
        ev = agent.evaluate(g)
        out = []
        for (kind, key, cards), q in ev[:k]:
            out.append(dict(self.move_info(kind, key, cards), q=round(q * 3, 3)))
        return out

    async def wait_human(self, p):
        g = self.game
        q = self.inbox[p]
        while True:
            msg, ws = await q.get()
            act = msg.get("action")
            err = None
            mv = None
            if g.phase == PHASE_RETURN:
                if act == "return":
                    c = msg.get("card")
                    for m in g.legal_moves():
                        if list(m[2]) == [c]:
                            mv = m
                    if mv is None:
                        err = "还贡牌必须是 10 或以下的牌"
                else:
                    err = "请先选择一张牌还贡"
            elif act == "pass":
                if g.is_leading:
                    err = "你是首家，必须出牌"
                else:
                    mv = (KIND_PASS, 0, [])
            elif act == "play":
                cards = [int(c) for c in msg.get("cards", [])]
                interps = [tuple(x) for x in g.legal_interpretations(cards)]
                if msg.get("kind") is not None:
                    interps = [x for x in interps if x[0] == msg["kind"] and x[1] == msg.get("key", x[1])]
                if not interps:
                    err = "管不上上家的牌" if _core.classify_cards(cards, g.level) else "不是有效牌型"
                else:
                    kinds = sorted({x[0] for x in interps})
                    if len(kinds) > 1:
                        opts = []
                        for kd in kinds:
                            best = max((x for x in interps if x[0] == kd), key=lambda x: x[1])
                            opts.append(self.move_info(best[0], best[1], best[2]))
                        await self.send(ws, {"type": "choose", "options": opts})
                        continue
                    mv = max(interps, key=lambda x: x[1])
            else:
                continue
            if err:
                await self.send(ws, {"type": "error", "message": err})
                continue
            return mv

    async def wait_continue(self):
        for s in self.seats:
            s.ready = s.kind == "ai"
        await self.broadcast()
        humans = self.human_seats()
        if not humans:
            await asyncio.sleep(5)
            await self.hold()
            return
        t_end = time.time() + (30 if self.mode == "real" else 1e9)
        while not self.closed:
            if not self.can_run():
                await self.hold()
                t_end = time.time() + (30 if self.mode == "real" else 1e9)
            # players who left do not hold up the others
            if all(s.ready or not s.connected for s in humans):
                return
            if self.mode == "real" and time.time() > t_end:
                return
            await asyncio.sleep(0.2)

    def ai_delay(self):
        base = 0.55 if self.mode == "teach" else 0.85
        return (base + random.random() * 0.5) / max(0.2, self.ai_speed)

    async def loop(self):
        loop = asyncio.get_event_loop()
        g = self.game
        try:
            await self.broadcast()
            await asyncio.sleep(1.2)
            if g.phase == PHASE_RETURN or g.tribute():
                self.new_event({"type": "tribute"})
                await self.broadcast()
                await asyncio.sleep(1.0)
            while not self.closed:
                if g.phase in (PHASE_ROUND_OVER, PHASE_MATCH_OVER):
                    self.deadline = None
                    await self.wait_continue()
                    if self.closed:
                        break
                    if self.length == "quick":
                        self.game = g = self.new_game()
                        self.deals += 1
                    else:
                        if g.phase == PHASE_MATCH_OVER:
                            self.match_no += 1
                        g.next_round()
                    self.trick = {}
                    self.review = None
                    self.new_event({"type": "deal"})
                    await self.broadcast()
                    await asyncio.sleep(1.4)
                    if g.tribute():
                        self.new_event({"type": "tribute"})
                        await self.broadcast()
                        await asyncio.sleep(1.2)
                    continue
                await self.hold()
                if self.closed:
                    break
                p = g.current
                seat = self.seats[p]
                if seat.kind == "ai":
                    self.deadline = None
                    t_think = time.time()
                    mv = await loop.run_in_executor(_executor, self.ai_move, p)
                    rest = self.ai_delay() - (time.time() - t_think)
                    if rest > 0:
                        await asyncio.sleep(rest)
                    if not self.can_run():  # everybody left (or paused) while the AI was thinking
                        continue
                    self._apply(p, mv)
                else:
                    mv, how = await self.human_turn(p)
                    if mv is None:  # paused or closed: decide again from the top
                        continue
                    self.deadline = None
                    self.review = None
                    if how == "normal" and self.mode == "teach" and self.hint_model and g.phase == PHASE_PLAY:
                        try:
                            self.review = await loop.run_in_executor(_executor, self.make_review, p, mv)
                        except Exception:
                            traceback.print_exc()
                    self._apply(p, mv, how)
                await self.broadcast()
        except asyncio.CancelledError:
            pass
        except Exception:
            traceback.print_exc()

    async def human_turn(self, p):
        """(move, how) for a human seat, or (None, None) when the table should pause first.
        how: "normal", "timeout" (real mode, random move) or "auto" (the player left while
        others play on, so the AI plays for them)."""
        loop = asyncio.get_event_loop()
        g = self.game
        seat = self.seats[p]
        if not self.can_run():
            return None, None
        if not seat.connected:
            t0 = time.time()
            mv = await loop.run_in_executor(_executor, self.auto_move, p)
            rest = self.ai_delay() - (time.time() - t0)
            if rest > 0:
                await asyncio.sleep(rest)
            if seat.connected or not self.can_run():  # came back meanwhile, everybody left, or paused
                return None, None
            return mv, "auto"
        timed = self.mode == "real"
        self.deadline = time.time() + self.timeout if timed else None
        while not self.inbox[p].empty():
            self.inbox[p].get_nowait()
        await self.broadcast()
        waiter = asyncio.ensure_future(self.wait_human(p))
        try:
            while not self.closed:
                done, _ = await asyncio.wait({waiter}, timeout=0.3)
                if done:
                    return waiter.result(), "normal"
                if not seat.connected or self.manual_pause:  # left or paused during the turn
                    self.deadline = None
                    return None, None
                if timed and time.time() > self.deadline + 0.3:
                    return random_move(g), "timeout"
            return None, None
        finally:
            if not waiter.done():
                waiter.cancel()

    def make_review(self, p, mv):
        """Teaching mode: compare the human's move with the AI's preferred move."""
        g = self.game.copy()
        agent = get_agent(self.hint_model, self.think)
        if agent.searcher is not None:
            ev = agent.searcher.evaluate(g)  # (move, score, net_q, searched), levels
        else:
            ev = [(m, q * 3, q * 3, True) for m, q in agent.evaluate(g)]
        if len(ev) <= 1:
            return None
        kind, key, cards = mv[0], mv[1], sorted(mv[2])
        hit = None
        for i, (m, sc, nq, searched) in enumerate(ev):
            if m[0] == kind and m[1] == key and len(m[2]) == len(cards):
                if hit is None or sorted(m[2]) == cards:
                    hit = (i + 1, sc if searched else nq, searched)
                    if sorted(m[2]) == cards:
                        break
        best_m, best_s = ev[0][0], ev[0][1]
        return {"seat": p, "best": dict(self.move_info(*best_m), q=round(best_s, 3)),
                "chosen": self.move_info(kind, key, cards),
                "chosen_q": None if hit is None else round(hit[1], 3),
                "rank": None if hit is None else hit[0], "total": len(ev),
                "searched": True if hit is None else hit[2]}

    async def handle(self, seat: int, ws, msg):
        act = msg.get("action")
        if act == "hint":
            if self.mode != "teach" or not self.hint_model or seat < 0:
                await self.send(ws, {"type": "error", "message": "实战模式不能使用 AI 提示"})
                return
            if self.game.current != seat or self.game.phase not in (PHASE_PLAY, PHASE_RETURN):
                await self.send(ws, {"type": "error", "message": "还没轮到你"})
                return
            hs = await asyncio.get_event_loop().run_in_executor(_executor, self.hints)
            await self.send(ws, {"type": "hints", "hints": hs, "seq": self.seq})
            return
        if act == "end":
            # the host, or the device that opened this table
            if not self.can_manage(self.idents.get(ws)):
                await self.send(ws, {"type": "error", "message": "只有房主或开这张桌的人可以结束牌局"})
                return
            who = self.seats[seat].name if seat >= 0 else "房主"
            await self.end(f"{who} 结束了这张牌桌")
            return
        if act in ("pause", "resume"):
            # players at the table may pause (play and stop any time); watchers only if they manage it
            if seat < 0 and not self.can_manage(self.idents.get(ws)):
                await self.send(ws, {"type": "error", "message": "观战时不能暂停牌局"})
                return
            who = self.seats[seat].name if seat >= 0 else "房主"
            self.set_pause(act == "pause", who)
            await self.broadcast()
            return
        if act == "ready":
            if seat >= 0:
                self.seats[seat].ready = True
                await self.broadcast()
            return
        if act in ("play", "pass", "return"):
            if seat < 0:
                return
            if self.game.current != seat:
                await self.send(ws, {"type": "error", "message": "还没轮到你"})
                return
            await self.inbox[seat].put((msg, ws))
