"""Launches / monitors training runs as background processes for the web UI."""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import time

from ..paths import runs_dir, models_dir

_procs: dict[str, subprocess.Popen] = {}


def _read_jsonl(path, since=0.0):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("t", 0) > since:
                rows.append(r)
    return rows


def run_status(name):
    p = _procs.get(name)
    if p is not None and p.poll() is None:
        return "running"
    rows = _read_jsonl(os.path.join(runs_dir(), name, "metrics.jsonl"))
    if not rows:
        return "idle"
    last = rows[-1]
    if last.get("type") == "stop":
        return "stopped"
    if time.time() - last.get("t", 0) < 60:
        return "running"
    return "stopped"


def list_runs():
    out = []
    for d in sorted(glob.glob(os.path.join(runs_dir(), "*")), key=os.path.getmtime, reverse=True):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        cfg = {}
        try:
            with open(os.path.join(d, "config.json"), encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            pass
        rows = _read_jsonl(os.path.join(d, "metrics.jsonl"))
        trains = [r for r in rows if r.get("type") == "train"]
        evals = [r for r in rows if r.get("type") == "eval"]
        out.append({
            "name": name,
            "status": run_status(name),
            "preset": cfg.get("model", {}).get("preset") or cfg.get("preset"),
            "params": cfg.get("params"),
            "base": cfg.get("base", ""),
            "device": cfg.get("device"),
            "started": cfg.get("started"),
            "frames": trains[-1]["frames"] if trains else 0,
            "train_seconds": trains[-1].get("train_seconds", 0) if trains else 0,
            "last_eval": evals[-1] if evals else None,
            "checkpoints": sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, "model*.pt"))),
        })
    return out


def start_run(opts: dict):
    name = opts.get("name") or time.strftime("run-%Y%m%d-%H%M%S")
    name = "".join(c for c in name if c.isalnum() or c in "-_.") or time.strftime("run-%Y%m%d-%H%M%S")
    if run_status(name) == "running":
        raise ValueError(f"训练 {name} 已在运行")
    args = [sys.executable, "-m", "guandan.train", "--name", name]
    mapping = {
        "preset": "--preset", "base": "--base", "device": "--device", "actor_device": "--actor-device",
        "actors": "--actors", "envs": "--envs", "batch": "--batch", "lr": "--lr", "epsilon": "--epsilon",
        "eval_interval": "--eval-interval", "eval_rounds": "--eval-rounds", "max_hours": "--max-hours",
        "save_interval": "--save-interval",
    }
    for k, flag in mapping.items():
        v = opts.get(k)
        if v is None or v == "":
            continue
        args += [flag, str(v)]
    d = os.path.join(runs_dir(), name)
    os.makedirs(d, exist_ok=True)
    log = open(os.path.join(d, "train.log"), "a", encoding="utf-8")
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    p = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env, **kw)
    _procs[name] = p
    return name


def stop_run(name):
    d = os.path.join(runs_dir(), name)
    if not os.path.isdir(d):
        raise ValueError("没有这个训练")
    with open(os.path.join(d, "STOP"), "w") as f:
        f.write(str(time.time()))
    return True


def metrics(name, since=0.0):
    return _read_jsonl(os.path.join(runs_dir(), name, "metrics.jsonl"), since)


def log_tail(name, n=60):
    path = os.path.join(runs_dir(), name, "train.log")
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 20000))
        data = f.read().decode("utf-8", errors="replace")
    return "\n".join(data.splitlines()[-n:])


def export_model(run, file, new_name):
    src = os.path.join(runs_dir(), run, file)
    if not os.path.exists(src):
        raise ValueError("找不到模型文件")
    new_name = "".join(c for c in new_name if c.isalnum() or c in "-_.") or run
    if not new_name.endswith(".pt"):
        new_name += ".pt"
    dst = os.path.join(models_dir(), new_name)
    shutil.copyfile(src, dst)
    return dst
