"""Data locations. Override the root with the GUANDAN_HOME environment variable."""
import glob
import json
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


LIST_FILE = "model_list.json"  # display names and hidden entries, see load_model_list()


def load_model_list():
    """{"labels": {key: display name}, "hidden": [key, ...]} where key is a model id
    ("models/x.pt", "builtin/x.pt") or "runs/<run name>" for a training run."""
    try:
        with open(os.path.join(home(), LIST_FILE), encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        d = {}
    if not isinstance(d, dict):
        d = {}
    d["labels"] = d.get("labels") if isinstance(d.get("labels"), dict) else {}
    d["hidden"] = d.get("hidden") if isinstance(d.get("hidden"), list) else []
    return d


def save_model_list(d):
    p = os.path.join(home(), LIST_FILE)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)


def set_model_entry(key, label=None, hidden=None):
    """Rename (label; "" restores the default name) and/or hide an entry of the model list."""
    d = load_model_list()
    if label is not None:
        label = str(label).strip()[:60]
        if label:
            d["labels"][key] = label
        else:
            d["labels"].pop(key, None)
    if hidden is not None:
        d["hidden"] = [k for k in d["hidden"] if k != key] + ([key] if hidden else [])
    save_model_list(d)
    return d


def list_models(all=False):
    """Usable checkpoints: built-in, user model folder ("模型库") and the latest model of every
    training run. all=True also lists hidden entries and the hourly snapshots of each run."""
    from . import __version__
    cfg = load_model_list()
    labels, hidden = cfg["labels"], set(cfg["hidden"])
    out = []
    for p in sorted(glob.glob(os.path.join(builtin_models_dir(), "*.pt"))):
        mid = "builtin/" + os.path.basename(p)
        default = f"内置模型 v{__version__}（推荐）" if os.path.basename(p) == "guandan-medium.pt" else "内置：" + os.path.basename(p)
        out.append({"id": mid, "path": p, "source": "内置", "group": "内置",
                    "label": labels.get(mid) or default, "hidden": False})
    for p in sorted(glob.glob(os.path.join(models_dir(), "*.pt"))):
        mid = "models/" + os.path.basename(p)
        out.append({"id": mid, "path": p, "source": "模型库", "group": "模型库",
                    "label": labels.get(mid) or os.path.splitext(os.path.basename(p))[0], "hidden": mid in hidden})
    runs = [r for r in glob.glob(os.path.join(runs_dir(), "*")) if os.path.isdir(r)]
    for run in sorted(runs, key=os.path.getmtime, reverse=True):
        name = os.path.basename(run)
        key = f"runs/{name}"
        base_label = labels.get(key) or f"训练 {name}"
        files = sorted(glob.glob(os.path.join(run, "model*.pt")), key=os.path.getmtime, reverse=True)
        for p in files:
            f = os.path.basename(p)
            if f != "model.pt" and not all:
                continue
            mid = f"runs/{name}/{f}"
            label = base_label if f == "model.pt" else f"{base_label} · {f[6:-4]} 样本快照"
            out.append({"id": mid, "path": p, "source": f"训练 {name}", "group": "训练",
                        "label": labels.get(mid) or label, "hidden": key in hidden or mid in hidden})
    if not all:
        out = [m for m in out if not m["hidden"]]
    for m in out:
        try:
            m["mtime"] = os.path.getmtime(m["path"])
            m["size"] = os.path.getsize(m["path"])
        except OSError:
            m["mtime"], m["size"] = 0, 0
    return out


def resolve_model(model_id):
    for m in list_models(all=True):
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
