"""Q-network for Deep Monte-Carlo (DMC) self-play.

Q(s, a) = Head([StateEncoder(state, LSTM(history)), action_features])

The state encoder runs once per decision; the head scores every legal action, so a
single batched forward pass evaluates all candidate moves of many games at once.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn

from . import _core

STATE_DIM, HIST_LEN, HIST_DIM, ACTION_DIM = _core.dims()
HIDDEN_DIM = _core.hidden_dim()


@dataclass
class ModelConfig:
    hidden: int = 512
    lstm: int = 256
    head_layers: int = 3
    preset: str = "medium"
    belief: bool = False  # auxiliary head that predicts the other players' hands (training only)
    ntp: bool = False  # auxiliary head that predicts the next move of the history (training only)
    oracle_head: bool = False  # auxiliary 4-hand level-invariant residual value head for subgame search
    # perfect-information value network (training only): the state input also holds the other
    # three hands (HIDDEN_DIM bits appended to the state); see make_oracle and train.py --oracle
    oracle: bool = False
    # moves of history the network reads (the most recent ones); the engine encodes HIST_LEN
    hist_len: int = 24
    # history encoder: "lstm" (default) or "tf" (a small causal Transformer, experimental)
    arch: str = "lstm"
    tf_dim: int = 128
    tf_layers: int = 4
    tf_heads: int = 4

    def to_dict(self):
        return asdict(self)


PRESETS = {
    "tiny": ModelConfig(hidden=192, lstm=96, head_layers=2, preset="tiny"),
    "small": ModelConfig(hidden=320, lstm=160, head_layers=3, preset="small"),
    "medium": ModelConfig(hidden=512, lstm=256, head_layers=3, preset="medium"),
    "large": ModelConfig(hidden=768, lstm=384, head_layers=4, preset="large"),
    "xlarge": ModelConfig(hidden=1024, lstm=512, head_layers=4, preset="xlarge"),
}
RECOMMENDED_PRESET = "medium"

PRESET_NOTES = {
    "tiny": "最快，适合 CPU 调试",
    "small": "轻量，CPU 也能训练",
    "medium": "推荐：容量与体积的最佳平衡（约 2.5M 参数，fp16 约 5MB）",
    "large": "更大容量，需要 GPU 与更长训练",
    "xlarge": "实验用，收益递减",
}


def mlp(sizes, last_act=True):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_act:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class HistTransformer(nn.Module):
    """History encoder: one token per move (oldest first, zero rows = no move yet), plus a
    learned summary token at the end whose output encodes the whole history. Causal attention,
    so the output at every move can also predict the next one (the --ntp auxiliary task)."""

    def __init__(self, d: int, layers: int, heads: int, max_len: int):
        super().__init__()
        self.inp = nn.Linear(HIST_DIM, d)
        self.pos = nn.Parameter(torch.randn(max_len + 1, d) * 0.02)
        self.summary = nn.Parameter(torch.randn(d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, hist: torch.Tensor):
        """hist [B,T,HIST_DIM] -> (summary [B,d], per-move outputs [B,T,d])"""
        b, t, _ = hist.shape
        x = torch.cat([self.inp(hist), self.summary.expand(b, 1, -1)], dim=1) + self.pos[: t + 1]
        mask = nn.Transformer.generate_square_subsequent_mask(t + 1, device=hist.device, dtype=x.dtype)
        y = self.norm(self.enc(x, mask=mask, is_causal=True))
        return y[:, -1], y[:, :-1]


class OracleValueHead(nn.Module):
    """4-hand level-invariant residual evaluator attached to a frozen playing network for
    short-horizon opening/early-mid subgame search (`FastNet.eval_oracle`)."""

    def __init__(self, z_dim: int, h1: int = 512, h2: int = 256):
        super().__init__()
        in_dim = 122 + 408 + z_dim + ACTION_DIM + 1
        self.fc0 = nn.Linear(in_dim, h1)
        self.ln0 = nn.LayerNorm(h1)
        self.fc1 = nn.Linear(h1, h2)
        self.ln1 = nn.LayerNorm(h2)
        self.fc2 = nn.Linear(h2, 1)


class GuandanNet(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        cfg = cfg or PRESETS[RECOMMENDED_PRESET]
        self.cfg = cfg
        h = cfg.hidden
        if cfg.arch == "tf":
            self.hist_enc = HistTransformer(cfg.tf_dim, cfg.tf_layers, cfg.tf_heads, cfg.hist_len)
            hd = cfg.tf_dim
        else:
            self.lstm = nn.LSTM(HIST_DIM, cfg.lstm, batch_first=True)
            hd = cfg.lstm
        self.hist_out = hd
        self.state_mlp = mlp([STATE_DIM + (HIDDEN_DIM if cfg.oracle else 0) + hd, h, h])
        head = [h + ACTION_DIM] + [h] * (cfg.head_layers - 1) + [h // 2]
        self.head = nn.Sequential(mlp(head), nn.Linear(h // 2, 1))
        if cfg.belief:
            self.belief = nn.Linear(h, HIDDEN_DIM)
        if cfg.ntp:
            self.ntp = nn.Linear(hd, HIST_DIM)
        if cfg.oracle_head:
            self.oracle_head = OracleValueHead(h)

    def add_belief_head(self):
        """Add the auxiliary hidden-hand head to a model trained without it."""
        if not self.cfg.belief:
            self.cfg.belief = True
            self.belief = nn.Linear(self.cfg.hidden, HIDDEN_DIM)
        return self

    def add_ntp_head(self):
        """Add the auxiliary next-move head (predicts history step t+1 from the LSTM at t)."""
        if not self.cfg.ntp:
            self.cfg.ntp = True
            self.ntp = nn.Linear(self.hist_out, HIST_DIM)
        return self

    def _window(self, hist: torch.Tensor) -> torch.Tensor:
        n = getattr(self.cfg, "hist_len", 24)
        return hist[:, -n:] if hist.shape[1] > n else hist

    def _history(self, hist: torch.Tensor):
        """(summary of the history [B,d], output at every move [B,T,d])"""
        hist = self._window(hist)
        if self.cfg.arch == "tf":
            return self.hist_enc(hist)
        out, (hn, _) = self.lstm(hist)
        return hn[-1], out

    def encode_state_seq(self, state: torch.Tensor, hist: torch.Tensor):
        """State encoding plus the history encoder's output at every step (next-move task)."""
        hn, out = self._history(hist)
        return self.state_mlp(torch.cat([state, hn], dim=-1)), out

    def belief_logits(self, z: torch.Tensor) -> torch.Tensor:
        """Logits of the other three hands (108 bits each, as in the state encoding)."""
        return self.belief(z)

    def encode_state(self, state: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
        hn, _ = self._history(hist)
        return self.state_mlp(torch.cat([state, hn], dim=-1))

    def score(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([z, actions], dim=-1)).squeeze(-1)

    def forward(self, state, hist, actions, index=None):
        """state [B,S], hist [B,T,H], actions [N,A], index [N] -> q [N].
        If index is None, actions are aligned 1:1 with states."""
        z = self.encode_state(state, hist)
        if index is not None:
            z = z.index_select(0, index)
        return self.score(z, actions)


@torch.no_grad()
def make_oracle(model: "GuandanNet") -> "GuandanNet":
    """Perfect-information copy of `model`: the state input gets HIDDEN_DIM more bits (the
    other hands, appended after the state) whose weights start at zero, so that the oracle
    starts out with exactly the values of `model` and learns to use the hidden cards."""
    cfg = ModelConfig(**{**model.cfg.to_dict(), "oracle": True, "belief": False, "ntp": False, "oracle_head": False})
    o = GuandanNet(cfg)
    src = {k: v for k, v in model.state_dict().items() if not k.startswith(("belief.", "ntp.", "oracle_head."))}
    w = src["state_mlp.0.weight"]  # [h, STATE_DIM + lstm]
    z = torch.zeros(w.shape[0], HIDDEN_DIM, dtype=w.dtype)
    src["state_mlp.0.weight"] = torch.cat([w[:, :STATE_DIM], z, w[:, STATE_DIM:]], dim=1)
    o.load_state_dict(src)
    return o


@torch.no_grad()
def widen(small: "GuandanNet", cfg: ModelConfig, noise: float = 0.0, seed: int = 0) -> "GuandanNet":
    """A bigger network that computes the same function as `small` (Net2Net).

    Every layer is widened by copying random existing units and splitting their outgoing
    weights between the copies, so the larger model starts from what the small one has
    learned instead of from scratch; extra head layers are inserted as identity layers
    (exact after a ReLU). The split between copies is random (the shares still sum to one),
    which keeps the function exact but gives the copies different gradients, so they can
    learn different things. `noise` optionally perturbs all weights (relative to their scale)."""
    if small.cfg.arch != "lstm":
        raise ValueError("only LSTM models can be widened")
    s = small.cfg
    if cfg.hidden < s.hidden or cfg.lstm < s.lstm or cfg.head_layers < s.head_layers:
        raise ValueError("只能加宽到更大的模型")
    g = torch.Generator().manual_seed(seed)
    big = GuandanNet(ModelConfig(hidden=cfg.hidden, lstm=cfg.lstm, head_layers=cfg.head_layers,
                                 preset=cfg.preset, belief=s.belief, ntp=s.ntp, hist_len=s.hist_len))

    def mapping(n_old, n_new):
        """(old unit of every new unit, share of the old unit's outgoing weight it carries)"""
        idx = torch.cat([torch.arange(n_old), torch.randint(0, n_old, (n_new - n_old,), generator=g)])
        r = 0.5 + torch.rand(n_new, generator=g)
        total = torch.zeros(n_old).index_add_(0, idx, r)
        return idx, r / total[idx]

    def jitter(w):
        if noise > 0:
            w += torch.randn(w.shape, generator=g) * (noise * float(w.std()) + 1e-8)
        return w

    def linear(dst, src, out_map, in_parts):
        """in_parts: list of (old width, new width, mapping or None for unchanged inputs)."""
        w_old, b_old = src.weight.detach().float(), src.bias.detach().float()
        rows = out_map[0] if out_map is not None else torch.arange(w_old.shape[0])
        cols, scale, o_old = [], [], 0
        for n_old, n_new, m in in_parts:
            if m is None:
                cols.append(torch.arange(o_old, o_old + n_old))
                scale.append(torch.ones(n_old))
            else:
                idx, share = m
                cols.append(o_old + idx)
                scale.append(share)
            o_old += n_old
        cols, scale = torch.cat(cols), torch.cat(scale)
        w = w_old[rows][:, cols] * scale
        dst.weight.copy_(jitter(w))
        dst.bias.copy_(b_old[rows])

    # LSTM: gates (input, forget, cell, output) are stacked blocks of `lstm` rows
    L0, L1 = s.lstm, cfg.lstm
    mL = mapping(L0, L1)
    rows = torch.cat([q * L0 + mL[0] for q in range(4)])
    colsL, scaleL = mL
    lo, lb = small.lstm, big.lstm
    lb.weight_ih_l0.copy_(jitter(lo.weight_ih_l0.detach()[rows].clone()))
    lb.weight_hh_l0.copy_(jitter(lo.weight_hh_l0.detach()[rows][:, colsL] * scaleL))
    lb.bias_ih_l0.copy_(lo.bias_ih_l0.detach()[rows])
    lb.bias_hh_l0.copy_(lo.bias_hh_l0.detach()[rows])

    H0, H1 = s.hidden, cfg.hidden
    lin_s = [m for m in small.state_mlp if isinstance(m, nn.Linear)]
    lin_b = [m for m in big.state_mlp if isinstance(m, nn.Linear)]
    m1 = mapping(H0, H1)
    linear(lin_b[0], lin_s[0], m1, [(STATE_DIM, STATE_DIM, None), (L0, L1, mL)])
    mz = mapping(H0, H1)  # the state encoding z, shared by the head and the belief head
    linear(lin_b[1], lin_s[1], mz, [(H0, H1, m1)])

    head_s = [m for m in small.head[0] if isinstance(m, nn.Linear)]
    head_b = [m for m in big.head[0] if isinstance(m, nn.Linear)]
    widths_s = [H0] * (s.head_layers - 1) + [H0 // 2]
    widths_b = [H1] * (cfg.head_layers - 1) + [H1 // 2]
    extra = cfg.head_layers - s.head_layers  # identity layers inserted before the last layer
    prev, prev_old, prev_new = mz, H0, H1
    j = 0
    for i, src in enumerate(head_s):
        if i == len(head_s) - 1:
            for _ in range(extra):
                dst = head_b[j]
                dst.weight.copy_(jitter(torch.eye(prev_new)))
                dst.bias.zero_()
                j += 1
        dst = head_b[j]
        m = mapping(widths_s[i], widths_b[j])
        parts = [(prev_old, prev_new, prev)] + ([(ACTION_DIM, ACTION_DIM, None)] if i == 0 else [])
        linear(dst, src, m, parts)
        prev, prev_old, prev_new = m, widths_s[i], widths_b[j]
        j += 1
    linear(big.head[1], small.head[1], None, [(prev_old, prev_new, prev)])
    if s.belief:
        linear(big.belief, small.belief, None, [(H0, H1, mz)])
    if s.ntp:
        linear(big.ntp, small.ntp, None, [(L0, L1, mL)])
    return big


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def preset_table():
    rows = []
    for name, cfg in PRESETS.items():
        n = count_params(GuandanNet(cfg))
        rows.append({
            "name": name,
            "hidden": cfg.hidden,
            "lstm": cfg.lstm,
            "head_layers": cfg.head_layers,
            "params": n,
            "size_fp16_mb": round(n * 2 / 1e6, 2),
            "recommended": name == RECOMMENDED_PRESET,
            "note": PRESET_NOTES.get(name, ""),
        })
    return rows


def save_checkpoint(path, model: GuandanNet, meta: dict | None = None, optimizer=None, half=False):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sd = {k: (v.detach().half() if half and v.is_floating_point() else v.detach()).cpu()
          for k, v in model.state_dict().items()}
    obj = {
        "format": "guandan-rl/1",
        "config": model.cfg.to_dict(),
        "state_dict": sd,
        "meta": dict(meta or {}, saved_at=time.time()),
    }
    if optimizer is not None:
        obj["optimizer"] = optimizer.state_dict()
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    for attempt in range(50):  # Windows: the target may be open in another process for a moment
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.2)
    os.replace(tmp, path)


def load_checkpoint(path, device="cpu", with_optimizer=False):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    known = ModelConfig.__dataclass_fields__
    cfg = ModelConfig(**{k: v for k, v in obj["config"].items() if k in known})
    model = GuandanNet(cfg)
    sd = {k: v.float() if v.is_floating_point() else v for k, v in obj["state_dict"].items()}
    missing, _unexpected = model.load_state_dict(sd, strict=False)  # tolerate heads of newer versions
    if missing:
        raise RuntimeError(f"checkpoint {path} lacks weights: {missing[:5]}")
    model.to(device)
    model.eval()
    if with_optimizer:
        return model, obj.get("meta", {}), obj.get("optimizer")
    return model, obj.get("meta", {})


def read_meta(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    n = sum(v.numel() for v in obj["state_dict"].values())
    return {"config": obj.get("config", {}), "meta": obj.get("meta", {}), "params": n}
