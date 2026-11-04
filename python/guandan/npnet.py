"""The Q-network forward pass in plain numpy, for playing without PyTorch.

The portable Windows build (one .exe, no install) ships without PyTorch: it reads the same
.pt checkpoints with a small torch-free unpickler and evaluates the network here. Training
still needs PyTorch. Numerically this matches GuandanNet in float32 (see tests/test_npnet.py).
"""
from __future__ import annotations

import pickle
import re
import zipfile

import numpy as np

from . import _core

STATE_DIM, HIST_LEN, HIST_DIM, ACTION_DIM = _core.dims()

_DTYPES = {
    "FloatStorage": np.float32, "HalfStorage": np.float16, "DoubleStorage": np.float64,
    "BFloat16Storage": None, "LongStorage": np.int64, "IntStorage": np.int32,
    "ShortStorage": np.int16, "CharStorage": np.int8, "ByteStorage": np.uint8, "BoolStorage": np.bool_,
}


class _Storage:
    def __init__(self, name):
        self.name = name


def _rebuild_tensor(storage, offset, size, stride, *_):
    a = storage
    if not size:
        return a[offset].copy()
    item = a.itemsize
    return np.lib.stride_tricks.as_strided(a[offset:], shape=tuple(size),
                                           strides=tuple(s * item for s in stride)).copy()


class _Stub:
    """Any other torch object in a checkpoint (e.g. in optimizer state) is ignored."""

    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return None

    def __setstate__(self, state):
        pass


def _bf16_to_f32(raw: bytes) -> np.ndarray:
    u = np.frombuffer(raw, np.uint16).astype(np.uint32) << 16
    return u.view(np.float32)


def load_pt(path) -> dict:
    """Read a checkpoint written by torch.save (zip format) into numpy arrays, without torch."""
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        pkl = next(n for n in names if n.endswith("data.pkl"))
        prefix = pkl[: -len("data.pkl")]

        class U(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "torch._utils" and name == "_rebuild_tensor_v2":
                    return _rebuild_tensor
                if module == "torch" and name in _DTYPES:
                    return _Storage(name)
                if module == "torch" and name == "Size":
                    return tuple
                if module == "collections" and name == "OrderedDict":
                    import collections
                    return collections.OrderedDict
                if module.startswith("torch"):
                    return _Stub
                if module.split(".")[0] in ("numpy", "builtins", "collections", "datetime", "_codecs"):
                    return super().find_class(module, name)
                raise pickle.UnpicklingError(f"不支持的对象 {module}.{name}")

            def persistent_load(self, pid):
                # ('storage', storage_type, key, location, numel)
                _, stype, key, _loc, _n = pid
                raw = zf.read(f"{prefix}data/{key}")
                name = stype.name if isinstance(stype, _Storage) else str(stype)
                if name == "BFloat16Storage":
                    return _bf16_to_f32(raw)
                dt = _DTYPES.get(name)
                if dt is None:
                    return np.frombuffer(raw, np.uint8)
                return np.frombuffer(raw, dt)

        with zf.open(pkl) as f:
            return U(f).load()


def _sigmoid(x):
    return 0.5 * (np.tanh(0.5 * x) + 1.0)  # stable for large |x|


class NumpyNet:
    """GuandanNet's inference path: encode_state / score / belief_logits on numpy arrays."""

    def __init__(self, cfg: dict, sd: dict):
        f = {k: np.ascontiguousarray(np.asarray(v, np.float32)) for k, v in sd.items()
             if hasattr(v, "dtype") and v.dtype.kind == "f"}
        self.cfg = dict(cfg)
        self.L = f["lstm.weight_hh_l0"].shape[1]
        if f["lstm.weight_ih_l0"].shape[1] != HIST_DIM or self._lin_in(f, "state_mlp") != STATE_DIM + self.L:
            raise RuntimeError("模型与当前程序的输入格式不一致")
        self.w_ih = f["lstm.weight_ih_l0"].T.copy()
        self.w_hh = f["lstm.weight_hh_l0"].T.copy()
        self.b = f["lstm.bias_ih_l0"] + f["lstm.bias_hh_l0"]
        self.state_mlp = self._linears(f, "state_mlp")
        self.head = self._linears(f, "head")
        self.belief = (f["belief.weight"].T.copy(), f["belief.bias"]) if "belief.weight" in f else None

    @staticmethod
    def _lin_in(f, prefix):
        return NumpyNet._linears(f, prefix)[0][0].shape[0]

    @staticmethod
    def _linears(f, prefix):
        """Linear layers of an nn.Sequential (possibly nested) in forward order."""
        keys = [k[: -len(".weight")] for k in f if k.startswith(prefix + ".") and k.endswith(".weight")]
        keys.sort(key=lambda k: [int(x) for x in re.findall(r"\d+", k[len(prefix):])])
        return [(f[k + ".weight"].T.copy(), f[k + ".bias"]) for k in keys]

    @staticmethod
    def _mlp(x, layers, relu_last):
        for i, (w, b) in enumerate(layers):
            x = x @ w + b
            if relu_last or i < len(layers) - 1:
                np.maximum(x, 0, out=x)
        return x

    def encode_state(self, state, hist):
        """state [B,S], hist [B,T,H] -> z [B,hidden]"""
        state = np.asarray(state, np.float32)
        hist = np.asarray(hist, np.float32)
        B, T, _ = hist.shape
        L = self.L
        xs = hist.reshape(B * T, -1) @ self.w_ih
        xs = xs.reshape(B, T, 4 * L) + self.b
        h = np.zeros((B, L), np.float32)
        c = np.zeros((B, L), np.float32)
        for t in range(T):  # PyTorch gate order: input, forget, cell, output
            g = xs[:, t] + h @ self.w_hh
            i, fg, gg, o = _sigmoid(g[:, :L]), _sigmoid(g[:, L:2 * L]), np.tanh(g[:, 2 * L:3 * L]), _sigmoid(g[:, 3 * L:])
            c = fg * c + i * gg
            h = o * np.tanh(c)
        return self._mlp(np.concatenate([state, h], axis=-1), self.state_mlp, True)

    def score(self, z, actions):
        """z [N,hidden] (or [1,hidden], broadcast), actions [N,A] -> q [N]"""
        actions = np.asarray(actions, np.float32)
        if z.shape[0] != actions.shape[0]:
            z = np.broadcast_to(z, (actions.shape[0], z.shape[1]))
        return self._mlp(np.concatenate([z, actions], axis=-1), self.head, False)[:, 0]

    def belief_logits(self, z):
        w, b = self.belief
        return z @ w + b


def load_numpy_model(path):
    """(NumpyNet, meta) from a .pt checkpoint, without torch."""
    obj = load_pt(path)
    return NumpyNet(obj.get("config", {}), obj["state_dict"]), obj.get("meta", {}) or {}


def read_meta_numpy(path):
    obj = load_pt(path)
    n = sum(int(np.prod(v.shape)) for v in obj["state_dict"].values() if hasattr(v, "shape"))
    return {"config": obj.get("config", {}), "meta": obj.get("meta", {}), "params": n}
