"""Policies on top of the Q-network: batched (for self-play / evaluation) and single-game
(for the battle server)."""
from __future__ import annotations

import os
import random
import re
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
        ch = choose_from_q(q, off, epsilon, rng, top_k, epsilon_top)
        if return_q:
            return ch, q[off[:-1] + ch]
        return ch


def choose_from_q(q, off, epsilon=0.0, rng: np.random.Generator | None = None, top_k=0, epsilon_top=0.0):
    """BatchPolicy.choose on values already computed (q for every legal move, segmented by off)."""
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
    return ch


# Hybrid belief-calibrated search settings (v1.14+):
# 1. Mid-game & Phase::Return close-decision rollout search via pure-Rust FastNet
#    (incremental T=1 per-seat LSTM + exact Solver leaf cutoff at `solver_leaf_cards`).
# 2. Endgame search (`cards_left <= cards`): full 324 belief logits + Sinkhorn IPF + 2-copy Bayes +
#    cross-suit bomb correlation (`bomb_gamma`) + root-cached SMC action-likelihood filtering
#    (`smc_oversample x worlds`, last `smc_last_k` plays) + exact alpha-beta minimax blended with Q.
ENDGAME_FAST = {
    "cards": 26,
    "worlds": 24,
    "floor": 0.02,
    "bomb_gamma": 0.25,
    "max_nodes": 250_000,
    "prior": 0.25,
    "max_cands": 12,
    "q_gap": 1.2,
    "smc_oversample": 2,
    "smc_last_k": 6,
    "mid_rollout": True,
    "mid_worlds": 16,
    "mid_top_k": 3,
    "mid_close_gap": 0.08,
    "mid_prior": 0.35,
    "mid_margin_se": 0.75,
    "solver_leaf_cards": 18,
    "search_return": True,
    "return_gap": 0.20,
}
ENDGAME_DEEP = {
    "cards": 30,
    "worlds": 32,
    "floor": 0.02,
    "bomb_gamma": 0.25,
    "max_nodes": 250_000,
    "prior": 0.25,
    "max_cands": 12,
    "q_gap": 1.0,
    "smc_oversample": 2,
    "smc_last_k": 6,
    "mid_rollout": True,
    "mid_worlds": 20,
    "mid_top_k": 4,
    "mid_close_gap": 0.10,
    "mid_prior": 0.35,
    "mid_margin_se": 0.75,
    "solver_leaf_cards": 18,
    "search_return": True,
    "return_gap": 0.25,
}


def build_fastnet(model) -> "_core.FastNet | None":
    """Construct pure-Rust `_core.FastNet` from either PyTorch `GuandanNet` or `NumpyNet`.
    Returns `None` for non-LSTM or oracle architectures."""
    try:
        if hasattr(model, "state_dict"):
            cfg = model.cfg.to_dict() if hasattr(model.cfg, "to_dict") else dict(model.cfg)
            if cfg.get("arch", "lstm") != "lstm" or cfg.get("oracle", False):
                return None
            sd = model.state_dict()

            def arr(k):
                return np.ascontiguousarray(sd[k].detach().float().cpu().numpy(), dtype=np.float32)

            w_ih = arr("lstm.weight_ih_l0")
            w_hh = arr("lstm.weight_hh_l0")
            b_lstm = arr("lstm.bias_ih_l0") + arr("lstm.bias_hh_l0")
            l = int(w_hh.shape[1])
            hist_len = int(cfg.get("hist_len", 24))

            def collect_lin(prefix):
                keys = [k[:-len(".weight")] for k in sd if k.startswith(prefix + ".") and k.endswith(".weight")]
                keys.sort(key=lambda k: [int(x) for x in re.findall(r"\d+", k[len(prefix):])])
                out = []
                for k in keys:
                    w = arr(k + ".weight")
                    b = arr(k + ".bias")
                    out.append((w.ravel().tolist(), b.ravel().tolist(), int(w.shape[0]), int(w.shape[1])))
                return out

            s_mlp = collect_lin("state_mlp")
            head = collect_lin("head")
            belief = None
            if "belief.weight" in sd:
                bw = arr("belief.weight")
                bb = arr("belief.bias")
                belief = (bw.ravel().tolist(), bb.ravel().tolist(), int(bw.shape[0]), int(bw.shape[1]))
            return _core.FastNet(l, hist_len, w_ih.ravel().tolist(), w_hh.ravel().tolist(),
                                 b_lstm.ravel().tolist(), s_mlp, head, belief)

        # NumpyNet path (weights stored transposed as [n_in, n_out])
        cfg = getattr(model, "cfg", {})
        if cfg.get("arch", "lstm") != "lstm" or cfg.get("oracle", False):
            return None
        w_ih = np.ascontiguousarray(model.w_ih.T, dtype=np.float32)
        w_hh = np.ascontiguousarray(model.w_hh.T, dtype=np.float32)
        b_lstm = np.ascontiguousarray(model.b, dtype=np.float32)
        l = int(model.L)
        hist_len = int(cfg.get("hist_len", 24))
        s_mlp = [(np.ascontiguousarray(wt.T, np.float32).ravel().tolist(),
                  np.ascontiguousarray(b, np.float32).ravel().tolist(),
                  int(wt.shape[1]), int(wt.shape[0])) for wt, b in model.state_mlp]
        head = [(np.ascontiguousarray(wt.T, np.float32).ravel().tolist(),
                 np.ascontiguousarray(b, np.float32).ravel().tolist(),
                 int(wt.shape[1]), int(wt.shape[0])) for wt, b in model.head]
        belief = None
        if getattr(model, "belief", None) is not None:
            bwt, bb = model.belief
            bw = np.ascontiguousarray(bwt.T, np.float32)
            belief = (bw.ravel().tolist(), np.ascontiguousarray(bb, np.float32).ravel().tolist(),
                      int(bw.shape[0]), int(bw.shape[1]))
        return _core.FastNet(l, hist_len, w_ih.ravel().tolist(), w_hh.ravel().tolist(),
                             b_lstm.ravel().tolist(), s_mlp, head, belief)
    except Exception:
        return None


class GameAgent:
    """Wraps a model for single-game decisions in the battle server. Both PyTorch and portable
    (PyTorch-free) builds share the pure-Rust `_core.FastNet` engine for fast CPU inference,
    incremental-LSTM mid-game/return rollout search, SMC belief filtering, and exact endgame search."""

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
        self.fastnet = build_fastnet(self.model)
        self.threads = max(1, min(8, (os.cpu_count() or 4) // 2))
        self.searcher = None
        if self.numpy:
            return
        self.device = torch.device(device)
        self.model.eval()
        if search:
            from .search import SearchAgent
            self.searcher = SearchAgent(self.model, device, **search)

    def _forward(self, game: "_core.Game", want_belief: bool = False):
        """Single forward pass returning (moves, q, belief_logits_list_or_None)."""
        if self.fastnet is not None and (self.numpy or self.device.type == "cpu"):
            moves, q_list, lg = self.fastnet.evaluate_game(game, want_belief)
            return moves, np.asarray(q_list, np.float32), lg
        moves = game.legal_moves()
        s, h, a, n = game.encode()
        if self.numpy:
            st = np.frombuffer(bytes(s), np.uint8).reshape(1, STATE_DIM)
            ht = np.frombuffer(bytes(h), np.uint8).reshape(1, HIST_LEN, HIST_DIM)
            at = np.frombuffer(bytes(a), np.uint8).reshape(n, ACTION_DIM)
            z = self.model.encode_state(st, ht)
            q = self.model.score(z, at)
            lg = None
            if want_belief and getattr(self.model, "belief", None) is not None:
                lg = np.asarray(self.model.belief_logits(z)[0], np.float32).tolist()
            return moves, q, lg
        return moves, *self._forward_torch(s, h, a, n, want_belief)

    @_no_grad
    def _forward_torch(self, s, h, a, n, want_belief: bool):
        d = self.device
        st = torch.frombuffer(bytearray(s), dtype=torch.uint8).view(1, STATE_DIM).to(d).float()
        ht = torch.frombuffer(bytearray(h), dtype=torch.uint8).view(1, HIST_LEN, HIST_DIM).to(d).float()
        at = torch.frombuffer(bytearray(a), dtype=torch.uint8).view(n, ACTION_DIM).to(d).float()
        z = self.model.encode_state(st, ht)
        q = self.model.score(z.expand(n, -1), at).float().cpu().numpy()
        lg = None
        if want_belief and getattr(self.model.cfg, "belief", False):
            lg = self.model.belief_logits(z)[0].float().cpu().numpy().tolist()
        return q, lg

    def evaluate(self, game: "_core.Game"):
        """Returns [(move, q)] sorted by q descending, where move = (kind, key, cards).
        q is in units of 3 levels (multiply by 3 for expected level change)."""
        if self.searcher is not None:
            return [(m, sc / 3.0) for m, sc, _, _ in self.searcher.evaluate(game)]
        moves, q, _ = self._forward(game, want_belief=False)
        order = np.argsort(-q)
        return [(moves[i], float(q[i])) for i in order]

    def choose(self, game):
        if self.searcher is not None:
            return self.evaluate(game)[0][0]
        eg = self.endgame
        ph = game.phase
        cl = game.cards_left()
        in_eg = bool(eg and ph == 1 and cl <= eg["cards"])
        in_mid = bool(eg and not in_eg and self.fastnet is not None and eg.get("mid_rollout", False)
                      and (ph == 0 or (ph == 1 and cl <= eg.get("mid_max_cards", 56))))
        moves, q, lg = self._forward(game, want_belief=(in_eg or in_mid))
        if not len(moves):
            return (0, 0, [])
        net = int(np.argmax(q))
        if len(moves) > 1 and eg:
            ql = (q * 3.0).tolist()
            th = eg.get("threads", self.threads)
            try:
                if in_eg:
                    smc_ov = eg.get("smc_oversample", 1)
                    if self.fastnet is not None and smc_ov > 1:
                        i = self.fastnet.endgame_choose_smc(
                            game,
                            net,
                            ql,
                            lg,
                            cards=eg["cards"],
                            worlds=eg["worlds"],
                            floor=eg.get("floor", 0.02),
                            bomb_gamma=eg.get("bomb_gamma", 0.25),
                            max_nodes=eg.get("max_nodes", 250_000),
                            margin=eg.get("margin", 0.0),
                            prior=eg.get("prior", 0.25),
                            max_cands=eg.get("max_cands", 12),
                            q_gap=eg.get("q_gap", 1.2),
                            smc_oversample=smc_ov,
                            smc_last_k=eg.get("smc_last_k", 6),
                            threads=th,
                            seed=random.getrandbits(63),
                        )
                    else:
                        i = game.endgame_choose(
                            net,
                            None,
                            cards=eg["cards"],
                            worlds=eg["worlds"],
                            floor=eg.get("floor", 0.02),
                            max_nodes=eg.get("max_nodes", 250_000),
                            margin=eg.get("margin", 0.0),
                            threads=th,
                            seed=random.getrandbits(63),
                            belief_logits=lg,
                            q_levels=ql,
                            prior=eg.get("prior", 0.25),
                            max_cands=eg.get("max_cands", 12),
                            q_gap=eg.get("q_gap", 1.2),
                            bomb_gamma=eg.get("bomb_gamma", 0.25),
                        )
                    if i is not None:
                        return moves[i]
                elif in_mid:
                    sec_q = float(np.partition(q, -2)[-2]) * 3.0
                    gap_lim = eg.get("return_gap", 0.20) if ph == 0 else eg.get("mid_close_gap", 0.08)
                    if float(q[net]) * 3.0 - sec_q <= gap_lim:
                        i = self.fastnet.rollout_choose(
                            game,
                            net,
                            ql,
                            lg,
                            min_cards=eg["cards"] + 1,
                            max_cards=eg.get("mid_max_cards", 56),
                            worlds=eg.get("mid_worlds", 16),
                            top_k=eg.get("mid_top_k", 3),
                            close_gap=eg.get("mid_close_gap", 0.08),
                            max_cand_gap=eg.get("mid_max_cand_gap", 0.45),
                            prior=eg.get("mid_prior", 0.35),
                            margin_se=eg.get("mid_margin_se", 0.75),
                            floor=eg.get("floor", 0.02),
                            bomb_gamma=eg.get("bomb_gamma", 0.25),
                            solver_leaf_cards=eg.get("solver_leaf_cards", 18),
                            search_return=eg.get("search_return", True),
                            return_gap=eg.get("return_gap", 0.20),
                            return_top_k=eg.get("return_top_k", 4),
                            smc_oversample=1,
                            smc_last_k=6,
                            threads=th,
                            seed=random.getrandbits(63),
                        )
                        if i is not None:
                            return moves[i]
            except Exception:
                pass  # the network's move is always a safe fallback
        return moves[net]


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
