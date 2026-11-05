"""Convert a .pt checkpoint into the model file of the Android app (.gdm).

The format is safetensors (8-byte little-endian header length, JSON header, raw tensors) with
float16 tensors, which is how the built-in model is stored anyway, so nothing is lost. Only the
tensors used to play are kept (LSTM, state MLP, Q head); the training-only belief head is
dropped. The network config is in the header's __metadata__.

    python packaging/export_gdm.py python/guandan/models/guandan-medium.pt out.gdm

Needs only numpy (reads the checkpoint without PyTorch).
"""
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
from guandan.ptload import load_pt  # noqa: E402

KEEP = ("lstm.", "state_mlp.", "head.")


def export(src, dst):
    obj = load_pt(src)
    sd = obj["state_dict"]
    header, blobs, off = {}, [], 0
    for k, v in sd.items():
        if not k.startswith(KEEP) or not hasattr(v, "dtype") or v.dtype.kind != "f":
            continue
        a = np.ascontiguousarray(np.asarray(v, np.float16))
        b = a.tobytes()
        header[k] = {"dtype": "F16", "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    meta = obj.get("meta") or {}
    header["__metadata__"] = {"config": json.dumps(obj.get("config") or {}),
                              "run": str(meta.get("run", "")), "source": os.path.basename(src)}
    h = json.dumps(header, separators=(",", ":")).encode("utf-8")
    h += b" " * (-len(h) % 8)
    with open(dst, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)
    return off


if __name__ == "__main__":
    n = export(sys.argv[1], sys.argv[2])
    print(f"{sys.argv[2]}: {n / 1e6:.1f} MB of weights")
