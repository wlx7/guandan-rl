"""Load a pickle that may only contain numpy arrays (refuses any other global)."""
import pickle, sys, numpy as np
ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy", "ndarray"), ("numpy", "dtype"),
           ("numpy._core.multiarray", "_reconstruct")}
class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in ALLOWED:
            if module.startswith("numpy._core"): module = "numpy.core.multiarray"
            import importlib
            try:
                return getattr(importlib.import_module(module), name)
            except Exception:
                return getattr(importlib.import_module("numpy._core.multiarray"), name)
        raise pickle.UnpicklingError(f"refused global {module}.{name}")
if __name__ == "__main__":
    w = SafeUnpickler(open(sys.argv[1], "rb"), encoding="latin1").load()
    print(type(w), len(w), [a.shape for a in w])
    np.savez(sys.argv[2], *w)
