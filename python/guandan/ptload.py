"""Read a checkpoint written by torch.save (zip + pickle) into numpy arrays, without PyTorch.

Used by the portable build (npnet.py) and by packaging/export_gdm.py, which converts the
built-in model for the Android app. Needs only numpy (no compiled engine)."""
from __future__ import annotations

import pickle
import zipfile

import numpy as np

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
