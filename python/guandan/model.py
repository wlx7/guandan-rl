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


@dataclass
class ModelConfig:
    hidden: int = 512
    lstm: int = 256
    head_layers: int = 3
    preset: str = "medium"

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


class GuandanNet(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        cfg = cfg or PRESETS[RECOMMENDED_PRESET]
        self.cfg = cfg
        h = cfg.hidden
        self.lstm = nn.LSTM(HIST_DIM, cfg.lstm, batch_first=True)
        self.state_mlp = mlp([STATE_DIM + cfg.lstm, h, h])
        head = [h + ACTION_DIM] + [h] * (cfg.head_layers - 1) + [h // 2]
        self.head = nn.Sequential(mlp(head), nn.Linear(h // 2, 1))

    def encode_state(self, state: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
        _, (hn, _) = self.lstm(hist)
        return self.state_mlp(torch.cat([state, hn[-1]], dim=-1))

    def score(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([z, actions], dim=-1)).squeeze(-1)

    def forward(self, state, hist, actions, index=None):
        """state [B,S], hist [B,T,H], actions [N,A], index [N] -> q [N].
        If index is None, actions are aligned 1:1 with states."""
        z = self.encode_state(state, hist)
        if index is not None:
            z = z.index_select(0, index)
        return self.score(z, actions)


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
    os.replace(tmp, path)


def load_checkpoint(path, device="cpu", with_optimizer=False):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**obj["config"])
    model = GuandanNet(cfg)
    sd = {k: v.float() if v.is_floating_point() else v for k, v in obj["state_dict"].items()}
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    if with_optimizer:
        return model, obj.get("meta", {}), obj.get("optimizer")
    return model, obj.get("meta", {})


def read_meta(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    n = sum(v.numel() for v in obj["state_dict"].values())
    return {"config": obj.get("config", {}), "meta": obj.get("meta", {}), "params": n}
