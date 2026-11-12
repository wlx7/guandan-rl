"""Policies on top of the Q-network: batched (for self-play / evaluation) and single-game
(for the battle server)."""
from __future__ import annotations

import random
import warnings

import numpy as np

try:  # the portable build plays without PyTorch (see npnet.py)
    import torch
except ImportError:
    torch = None

from . import _core
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

STATE_DIM, HIST_LEN, HIST_DIM, ACTION_DIM = _core.dims()


def _no_grad(f):
    return torch.no_grad()(f) if torch is not None else f


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

    @_no_grad
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

    def choose(self, s, h, a, off, epsilon=0.0, rng: np.random.Generator | None = None,
               top_k=0, epsilon_top=0.0, return_q=False):
        """Greedy choice with exploration: with probability `epsilon_top` a random move among the
        network's `top_k` best, and with probability `epsilon` a uniformly random legal move.
        With `return_q` also returns the network's value of each chosen move."""
        q = self.q_values(s, h, a, off)
        ch = segment_argmax(q, off)
        if epsilon > 0 or (epsilon_top > 0 and top_k > 1):
            rng = rng or np.random.default_rng()
            n = np.diff(off)
            u = rng.random(len(ch))
            if epsilon_top > 0 and top_k > 1:
                near = np.nonzero((u < epsilon_top) & (n > 1))[0]
                for i in near:
                    qi = q[off[i]:off[i + 1]]
                    k = min(top_k, len(qi))
                    best = np.argpartition(-qi, k - 1)[:k]
                    ch[i] = int(best[int(rng.integers(k))])
                u = np.where(u < epsilon_top, 1.0, u - epsilon_top)
            explore = u < epsilon
            if explore.any():
                ch[explore] = (rng.random(explore.sum()) * n[explore]).astype(np.int64)
        if return_q:
            return ch, q[off[:-1] + ch]
        return ch


# exact endgame search when playing (see endgame.rs / endgame.py): once at most `cards` cards
# are left, `worlds` deals of the hidden cards are sampled from the network's belief and every
# legal move is solved exactly in each; measured +0.03 levels/round (20,16) and +0.045 (24,32)
ENDGAME_FAST = {"cards": 20, "worlds": 16, "floor": 0.02, "max_nodes": 400_000}
ENDGAME_DEEP = {"cards": 24, "worlds": 32, "floor": 0.02, "max_nodes": 150_000}


class GameAgent:
    """Wraps a model for single-game decisions in the battle server. Without PyTorch the
    network runs in numpy (portable build); the play-out search then is not available, the
    endgame search (Rust) is."""

    def __init__(self, path_or_model, device="cpu", search: dict | None = None, endgame: dict | None = None):
        self.endgame = endgame
        self.numpy = torch is None
        if isinstance(path_or_model, str):
            if self.numpy:
                from .npnet import load_numpy_model
                self.model, self.meta = load_numpy_model(path_or_model)
            else:
                from .model import load_checkpoint
                self.model, self.meta = load_checkpoint(path_or_model, device)
            self.path = path_or_model
        else:
            self.model, self.meta, self.path = path_or_model, {}, None
            self.numpy = not hasattr(self.model, "parameters")
        self.searcher = None
        if self.numpy:
            return
        self.device = torch.device(device)
        self.model.eval()
        if search:
            from .search import SearchAgent
            self.searcher = SearchAgent(self.model, device, **search)

    def evaluate(self, game: "_core.Game"):
        """Returns [(move, q)] sorted by q descending, where move = (kind, key, cards).
        q is in units of 3 levels (multiply by 3 for expected level change)."""
        if self.searcher is not None:
            return [(m, sc / 3.0) for m, sc, _, _ in self.searcher.evaluate(game)]
        moves = game.legal_moves()
        s, h, a, n = game.encode()
        if self.numpy:
            st = np.frombuffer(bytes(s), np.uint8).reshape(1, STATE_DIM)
            ht = np.frombuffer(bytes(h), np.uint8).reshape(1, HIST_LEN, HIST_DIM)
            at = np.frombuffer(bytes(a), np.uint8).reshape(n, ACTION_DIM)
            q = self.model.score(self.model.encode_state(st, ht), at)
        else:
            q = self._q_torch(s, h, a, n)
        order = np.argsort(-q)
        return [(moves[i], float(q[i])) for i in order]

    @_no_grad
    def _q_torch(self, s, h, a, n):
        d = self.device
        st = torch.frombuffer(bytearray(s), dtype=torch.uint8).view(1, STATE_DIM).to(d).float()
        ht = torch.frombuffer(bytearray(h), dtype=torch.uint8).view(1, HIST_LEN, HIST_DIM).to(d).float()
        at = torch.frombuffer(bytearray(a), dtype=torch.uint8).view(n, ACTION_DIM).to(d).float()
        z = self.model.encode_state(st, ht).expand(n, -1)
        return self.model.score(z, at).float().cpu().numpy()

    def _belief_expect(self, game):
        """The network's expected copies of every card type in the other three hands (162
        values, see Game.determinize_weighted), or None without a belief head."""
        s, h, _a, _n = game.encode()
        if self.numpy:
            if getattr(self.model, "belief", None) is None:
                return None
            st = np.frombuffer(bytes(s), np.uint8).reshape(1, STATE_DIM)
            ht = np.frombuffer(bytes(h), np.uint8).reshape(1, HIST_LEN, HIST_DIM)
            lg = np.asarray(self.model.belief_logits(self.model.encode_state(st, ht)), np.float64)
        else:
            if not getattr(self.model.cfg, "belief", False):
                return None
            with torch.no_grad():
                d = self.device
                st = torch.frombuffer(bytearray(s), dtype=torch.uint8).view(1, STATE_DIM).to(d).float()
                ht = torch.frombuffer(bytearray(h), dtype=torch.uint8).view(1, HIST_LEN, HIST_DIM).to(d).float()
                lg = self.model.belief_logits(self.model.encode_state(st, ht)).float().cpu().numpy().astype(np.float64)
        p = (1.0 / (1.0 + np.exp(-lg))).reshape(3, 54, 2)
        return (p[..., 0] + np.minimum(p[..., 1], p[..., 0])).reshape(162).tolist()

    def choose(self, game):
        ev = self.evaluate(game)
        best = ev[0][0]
        eg = self.endgame
        if eg and len(ev) > 1 and game.phase == 1 and game.cards_left() <= eg["cards"]:
            try:
                moves = game.legal_moves()
                net = next(i for i, m in enumerate(moves) if m[0] == best[0] and m[1] == best[1] and list(m[2]) == list(best[2]))
                i = game.endgame_choose(net, self._belief_expect(game), cards=eg["cards"], worlds=eg["worlds"],
                                        floor=eg.get("floor", 0.02), max_nodes=eg.get("max_nodes", 400_000),
                                        seed=random.getrandbits(63))
                if i is not None:
                    return moves[i]
            except Exception:
                pass  # the network's move is always a safe fallback
        return best


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
