"""Endgame search ("残局搜索"): perfect-information Monte-Carlo with an exact solver.

When few cards are left, the hidden hands are sampled `worlds` times consistently with what
the player to move has seen (optionally guided by the network's belief head), and in every
sampled world every legal move is solved exactly to the end of the round by alpha-beta
(Rust, `Game.solve_moves`): all four players then play perfectly with all hands visible.
The move with the best average result is played; ties keep the network's choice.

Unlike the play-out search in search.py this never compares the network's values across
players (they disagree by far more than the gaps between good moves); the only error left
is the perfect-information assumption of the sampled worlds.
"""
from __future__ import annotations

import itertools
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from .model import STATE_DIM, HIST_LEN, HIST_DIM

_seed = itertools.count(1)


class EndgameSearch:
    def __init__(self, model, device="cpu", cards=20, worlds=16, guided=None, max_nodes=2_000_000,
                 threads=8, margin=0.0):
        self.model = model
        self.device = torch.device(device)
        self.cards = cards
        self.worlds = worlds
        self.guided = guided if (guided is not None and getattr(model.cfg, "belief", False)) else None
        self.max_nodes = max_nodes
        self.margin = margin  # levels a move must beat the network's choice by
        self.pool = ThreadPoolExecutor(max_workers=threads) if threads > 1 else None
        self.stats = {"decisions": 0, "searched": 0, "changed": 0, "aborted": 0}

    @torch.no_grad()
    def _expected_hands(self, games):
        encs = [g.encode() for g in games]
        s = np.stack([np.frombuffer(e[0], np.uint8) for e in encs]).reshape(-1, STATE_DIM)
        h = np.stack([np.frombuffer(e[1], np.uint8) for e in encs]).reshape(-1, HIST_LEN, HIST_DIM)
        st = torch.from_numpy(s).to(self.device).float()
        ht = torch.from_numpy(h).to(self.device).float()
        p = torch.sigmoid(self.model.belief_logits(self.model.encode_state(st, ht))).float().cpu().numpy()
        p = p.reshape(len(p), 3, 54, 2)
        p1 = p[..., 0]
        p2 = np.minimum(p[..., 1], p1)
        return (p1 + p2).reshape(len(p), 162)

    def wants(self, game) -> bool:
        return game.cards_left() <= self.cards and len(game.legal_moves()) > 1

    def choose_many(self, games, net_choices):
        """Moves (indices into game.legal_moves()) for games that `wants`; net_choices are the
        network's own picks, kept unless the search finds a better move. Uses the Rust
        Game.endgame_choose, i.e. exactly what the battle servers play."""
        out = list(net_choices)
        todo = [i for i, g in enumerate(games) if self.wants(g)]
        self.stats["decisions"] += len(games)
        if not todo:
            return out
        expect = self._expected_hands([games[i] for i in todo]) if self.guided is not None else None
        for j, i in enumerate(todo):
            e = expect[j].tolist() if expect is not None else None
            r = games[i].endgame_choose(int(net_choices[i]), e, cards=self.cards, worlds=self.worlds,
                                        floor=self.guided or 0.02, max_nodes=self.max_nodes, margin=self.margin,
                                        threads=8, seed=next(_seed) * 7919)
            self.stats["searched"] += 1
            if r is not None:
                out[i] = r
                self.stats["changed"] += 1
        return out

    def choose_many_py(self, games, net_choices):
        """The same search in Python (kept for reference / per-move statistics)."""
        out = list(net_choices)
        todo = [i for i, g in enumerate(games) if self.wants(g)]
        self.stats["decisions"] += len(games)
        if not todo:
            return out
        expect = self._expected_hands([games[i] for i in todo]) if self.guided is not None else None
        jobs = []
        for j, i in enumerate(todo):
            g = games[i]
            moves = g.legal_moves()
            base = next(_seed) * 7919
            for w in range(self.worlds):
                if expect is not None:
                    world = g.determinize_weighted(base + w, expect[j].tolist(), self.guided)
                else:
                    world = g.determinize(base + w)
                jobs.append((i, world, moves))
        run = (lambda f, xs: list(self.pool.map(f, xs))) if self.pool else (lambda f, xs: [f(x) for x in xs])
        vals = run(lambda job: job[1].solve_moves(job[2], self.max_nodes), jobs)
        acc = {}
        for (i, _w, moves), v in zip(jobs, vals):
            if v[0] is None:
                self.stats["aborted"] += 1
                continue
            a = acc.setdefault(i, [np.zeros(len(moves)), 0])
            a[0] += np.asarray(v, np.float64)
            a[1] += 1
        for i in todo:
            if i not in acc:
                continue
            mean = acc[i][0] / acc[i][1]
            net = int(net_choices[i])
            best = int(np.argmax(mean))
            self.stats["searched"] += 1
            if mean[best] > mean[net] + self.margin + 1e-9:
                out[i] = best
                self.stats["changed"] += 1
        return out
