"""Average the weights of several checkpoints of one run (e.g. its last hourly snapshots).
The bundled models are such averages: they play better than any single snapshot.

    python -m guandan.average out.pt runs/x/model_1000000k.pt runs/x/model_1200000k.pt ...
"""
from __future__ import annotations

import sys

import torch

from .model import load_checkpoint, save_checkpoint


def average(paths, out):
    models = [load_checkpoint(p, "cpu")[0] for p in paths]
    sds = [m.state_dict() for m in models]
    avg = {k: (sum(sd[k].float() for sd in sds) / len(sds)).to(sds[0][k].dtype) if sds[0][k].is_floating_point()
           else sds[0][k] for k in sds[0]}
    m = models[0]
    m.load_state_dict(avg)
    save_checkpoint(out, m, {"averaged": list(paths)}, half=True)


if __name__ == "__main__":
    average(sys.argv[2:], sys.argv[1])
    print(f"wrote {sys.argv[1]} (average of {len(sys.argv) - 2})")
