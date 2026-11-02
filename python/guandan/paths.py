"""Data locations. Override the root with the GUANDAN_HOME environment variable."""
import glob
import os

PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def home():
    h = os.environ.get("GUANDAN_HOME") or os.path.join(os.path.expanduser("~"), "guandan_data")
    os.makedirs(h, exist_ok=True)
    return h


_RESERVED = {"con", "prn", "aux", "nul"} | {f"com{i}" for i in range(1, 10)} | {f"lpt{i}" for i in range(1, 10)}


def safe_run_name(name: str) -> str:
    """Folder-safe run name. Names like 'aux' or 'con' are reserved devices on Windows and
    make the folder unreadable for many tools, so they get a suffix."""
    name = "".join(c for c in name if c.isalnum() or c in "-_.").strip(".")
    if name.split(".")[0].lower() in _RESERVED:
        name += "-run"
    return name


def runs_dir():
    d = os.path.join(home(), "runs")
    os.makedirs(d, exist_ok=True)
    return d


def models_dir():
    d = os.path.join(home(), "models")
    os.makedirs(d, exist_ok=True)
    return d


def builtin_models_dir():
    return os.path.join(PKG_DIR, "models")


def list_models():
    """All usable checkpoints: built-in, user model folder, and training runs."""
    out = []
    for p in sorted(glob.glob(os.path.join(builtin_models_dir(), "*.pt"))):
        out.append({"id": "builtin/" + os.path.basename(p), "path": p, "source": "内置"})
    for p in sorted(glob.glob(os.path.join(models_dir(), "*.pt"))):
        out.append({"id": "models/" + os.path.basename(p), "path": p, "source": "模型库"})
    for run in sorted(glob.glob(os.path.join(runs_dir(), "*"))):
        if not os.path.isdir(run):
            continue
        name = os.path.basename(run)
        for p in sorted(glob.glob(os.path.join(run, "model*.pt")), key=os.path.getmtime, reverse=True):
            out.append({"id": f"runs/{name}/{os.path.basename(p)}", "path": p, "source": f"训练 {name}"})
    for m in out:
        try:
            m["mtime"] = os.path.getmtime(m["path"])
            m["size"] = os.path.getsize(m["path"])
        except OSError:
            m["mtime"], m["size"] = 0, 0
    return out


def resolve_model(model_id):
    for m in list_models():
        if m["id"] == model_id:
            return m["path"]
    if model_id and os.path.exists(model_id):
        return model_id
    return None


def default_model():
    ms = list_models()
    for m in ms:
        if m["source"] == "内置":
            return m["path"]
    return ms[0]["path"] if ms else None
