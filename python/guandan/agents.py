"""Policies on top of the Q-network: batched (for self-play / evaluation) and single-game
(for the battle server)."""
from __future__ import annotations

import random
import warnings

import numpy as np
import torch

from . import _core
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

from .model import STATE_DIM, HIST_LEN, HIST_DIM, ACTION_DIM, load_checkpoint


def unpack_obs(obs):
    players, s, h, a, off = obs
    players = np.frombuffer(players, np.uint8)
    s = np.frombuffer(s, np.uint8).reshape(-1, STATE_DIM)
    h = np.frombuffer(h, np.uint8).reshape(-1, HIST_LEN, HIST_DIM)
    a = np.frombuffer(a, np.uint8).reshape(-1, ACTION_DIM)
    return players, s, h, a, np.asarray(off, np.int64)


def subset_obs(s, h, a, off, rows):
    """Select a subset of environments from a batched observation."""
    rows = np.asarray(rows, np.int64)
    n = off[rows + 1] - off[rows]
    a_sub = np.concatenate([a[off[r]:off[r + 1]] for r in rows]) if len(rows) else a[:0]
    off_sub = np.zeros(len(rows) + 1, np.int64)
    off_sub[1:] = np.cumsum(n)
    return s[rows], h[rows], a_sub, off_sub


def segment_argmax(q: np.ndarray, off: np.ndarray) -> np.ndarray:
    b = len(off) - 1
    n = np.diff(off)
    seg = np.repeat(np.arange(b), n)
    order = np.lexsort((-q, seg))
    return order[off[:-1]] - off[:-1]


class BatchPolicy:
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = torch.device(device)

    @torch.no_grad()
    def q_values(self, s, h, a, off) -> np.ndarray:
        d = self.device
        b = len(off) - 1
        if b == 0:
            return np.zeros(0, np.float32)
        idx = torch.from_numpy(np.repeat(np.arange(b), np.diff(off))).to(d)
        st = torch.from_numpy(np.ascontiguousarray(s)).to(d).float()
        ht = torch.from_numpy(np.ascontiguousarray(h)).to(d).float()
        at = torch.from_numpy(np.ascontiguousarray(a)).to(d).float()
        q = self.model(st, ht, at, idx)
        return q.float().cpu().numpy()

    def choose(self, s, h, a, off, epsilon=0.0, rng: np.random.Generator | None = None) -> np.ndarray:
        q = self.q_values(s, h, a, off)
        ch = segment_argmax(q, off)
        if epsilon > 0:
            rng = rng or np.random.default_rng()
            n = np.diff(off)
            explore = rng.random(len(ch)) < epsilon
            if explore.any():
                ch[explore] = (rng.random(explore.sum()) * n[explore]).astype(np.int64)
        return ch


class GameAgent:
    """Wraps a model for single-game decisions in the battle server."""

    def __init__(self, path_or_model, device="cpu"):
        if isinstance(path_or_model, str):
            self.model, self.meta = load_checkpoint(path_or_model, device)
            self.path = path_or_model
        else:
            self.model, self.meta, self.path = path_or_model, {}, None
        self.device = torch.device(device)
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, game: "_core.Game"):
        """Returns [(move, q)] sorted by q descending, where move = (kind, key, cards)."""
        moves = game.legal_moves()
        s, h, a, n = game.encode()
        d = self.device
        st = torch.frombuffer(bytearray(s), dtype=torch.uint8).view(1, STATE_DIM).to(d).float()
        ht = torch.frombuffer(bytearray(h), dtype=torch.uint8).view(1, HIST_LEN, HIST_DIM).to(d).float()
        at = torch.frombuffer(bytearray(a), dtype=torch.uint8).view(n, ACTION_DIM).to(d).float()
        z = self.model.encode_state(st, ht).expand(n, -1)
        q = self.model.score(z, at).float().cpu().numpy()
        order = np.argsort(-q)
        return [(moves[i], float(q[i])) for i in order]

    def choose(self, game):
        return self.evaluate(game)[0][0]


def random_move(game, allow_pass=False):
    """Random legal move; used when a human player times out."""
    moves = game.legal_moves()
    plays = [m for m in moves if m[0] != 0]
    if plays and not allow_pass:
        return random.choice(plays)
    return random.choice(moves)


def rule_move(game):
    moves = game.legal_moves()
    return moves[game.rule_choice()]
