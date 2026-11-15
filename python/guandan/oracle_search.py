"""Search with a perfect-information value network ("明牌估值搜索", experimental).

For a decision of the player to move: the playing network proposes its `top_k` moves; the
hidden hands are sampled `worlds` times from the network's belief about them
(Game.determinize_weighted, consistent with everything the player has seen); in every
sampled world the oracle network (train.py --oracle: same architecture, also sees the other
hands) values each candidate. The candidate with the best blend of the playing network's
value and the oracle's average over the worlds is played.

The oracle estimates the outcome of the current self-play policy given all hands, so its
average over worlds drawn from the right distribution estimates the same quantity as the
playing network, E[outcome | what the player saw, move], by another route; no player is
assumed to see cards it cannot see. Unlike the old play-out search (search.py) it never
compares values computed from different players' points of view.
"""
from __future__ import annotations

import itertools

import numpy as np
import torch

from .model import ACTION_DIM, HIDDEN_DIM, HIST_DIM, HIST_LEN, STATE_DIM

_seed = itertools.count(1)


class OracleSearch:
    def __init__(self, model, oracle, device="cpu", top_k=4, worlds=32, beta=1.0, floor=0.02):
        self.model = model
        self.oracle = oracle
        self.device = torch.device(device)
        self.top_k = top_k
        self.worlds = worlds
        self.beta = beta
        self.floor = floor
        self.stats = {"decisions": 0, "searched": 0, "changed": 0}

    def _t(self, x):
        return torch.from_numpy(np.ascontiguousarray(x)).to(self.device).float()

    @torch.no_grad()
    def choose_many(self, games, net_choices):
        """Indices into game.legal_moves() for every game (net_choices = the network's picks)."""
        out = list(net_choices)
        self.stats["decisions"] += len(games)
        todo = [i for i, g in enumerate(games) if g.phase == 1 and len(g.legal_moves()) > 1]
        if not todo:
            return out
        encs = [games[i].encode() for i in todo]
        s = np.stack([np.frombuffer(e[0], np.uint8) for e in encs]).reshape(-1, STATE_DIM)
        h = np.stack([np.frombuffer(e[1], np.uint8) for e in encs]).reshape(-1, HIST_LEN, HIST_DIM)
        acts = [np.frombuffer(e[2], np.uint8).reshape(e[3], ACTION_DIM) for e in encs]
        st, ht = self._t(s), self._t(h)
        z = self.model.encode_state(st, ht)
        p = torch.sigmoid(self.model.belief_logits(z)).float().cpu().numpy().reshape(len(todo), 3, 54, 2)
        expect = (p[..., 0] + np.minimum(p[..., 1], p[..., 0])).reshape(len(todo), 162)
        # the playing network's values and its top candidates
        cands, qnet = [], []
        for j in range(len(todo)):
            q = self.model.score(z[j:j + 1].expand(len(acts[j]), -1), self._t(acts[j])).float().cpu().numpy()
            k = min(self.top_k, len(q))
            c = np.argsort(-q)[:k]
            cands.append(c)
            qnet.append(q[c])
        # sampled worlds: the other hands, as the oracle's extra input
        rows_s, rows_h, rows_a, owner = [], [], [], []
        W = self.worlds
        for j, i in enumerate(todo):
            g = games[i]
            for _ in range(W):
                gw = g.determinize_weighted(next(_seed) * 7919, expect[j].tolist(), self.floor)
                x = np.frombuffer(gw.encode_hidden(), np.uint8)
                sx = np.concatenate([s[j], x])
                for k in cands[j]:
                    rows_s.append(sx)
                    rows_a.append(acts[j][k])
                    rows_h.append(j)
            owner.append(j)
        so = self._t(np.stack(rows_s))
        ho = ht[torch.as_tensor(rows_h, device=self.device)]
        ao = self._t(np.stack(rows_a))
        v = self.oracle(so, ho, ao).float().cpu().numpy()
        pos = 0
        for j, i in enumerate(todo):
            k = len(cands[j])
            vj = v[pos:pos + W * k].reshape(W, k).mean(axis=0)
            pos += W * k
            score = (1 - self.beta) * qnet[j] + self.beta * vj
            pick = int(cands[j][int(np.argmax(score))])
            self.stats["searched"] += 1
            if pick != out[i]:
                self.stats["changed"] += 1
            out[i] = pick
        return out
