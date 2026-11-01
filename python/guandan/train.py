"""Pure self-play reinforcement learning (Deep Monte-Carlo).

* N actor processes each run `--envs` matches in parallel. All four seats are played by
  the latest network (epsilon-greedy). Every decision (state, history, chosen action) is
  labelled with the final outcome of that round for the acting player's team.
* The learner regresses Q(s, a) onto these Monte-Carlo returns.
* An evaluator process periodically measures the latest checkpoint against a random
  player, a fixed rule-based player and (optionally) the base model.

Run:  python -m guandan.train --name myrun --preset medium
Stop: create the file <run_dir>/STOP (the UI does this) or press Ctrl+C.
"""
from __future__ import annotations

import argparse
import json
import os
import queue as queue_mod
import signal
import sys
import time

import numpy as np
import torch
import torch.multiprocessing as mp

from . import _core
from .model import (GuandanNet, PRESETS, RECOMMENDED_PRESET, ModelConfig, STATE_DIM, HIST_LEN,
                    HIST_DIM, ACTION_DIM, save_checkpoint, load_checkpoint, count_params)
from .agents import BatchPolicy, unpack_obs
from .paths import runs_dir

H_FLAT = HIST_LEN * HIST_DIM


def build_parser():
    ap = argparse.ArgumentParser(description="Guandan self-play DMC training")
    ap.add_argument("--name", default=time.strftime("run-%Y%m%d-%H%M%S"))
    ap.add_argument("--preset", default=RECOMMENDED_PRESET, choices=list(PRESETS))
    ap.add_argument("--base", default="", help="checkpoint to start from (基准模型)")
    ap.add_argument("--device", default="auto", help="learner device: auto|cuda|cpu")
    ap.add_argument("--actor-device", default="cpu", help="device for actors: cpu|cuda")
    ap.add_argument("--actors", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--envs", type=int, default=64, help="parallel matches per actor")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epsilon", type=float, default=0.02)
    ap.add_argument("--reuse", type=float, default=2.0, help="average times each sample is trained on")
    ap.add_argument("--buffer", type=int, default=200_000)
    ap.add_argument("--save-interval", type=float, default=300, help="seconds between checkpoints")
    ap.add_argument("--snapshot-interval", type=float, default=3600, help="seconds between kept snapshots")
    ap.add_argument("--eval-interval", type=float, default=600, help="seconds between evaluations")
    ap.add_argument("--eval-rounds", type=int, default=400)
    ap.add_argument("--eval-threads", type=int, default=2)
    ap.add_argument("--total-frames", type=float, default=0, help="stop after this many samples (0 = no limit)")
    ap.add_argument("--max-hours", type=float, default=0, help="stop after this many hours (0 = no limit)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-amp", action="store_true", help="disable mixed precision on GPU")
    return ap


# ----------------------------------------------------------------------------- actors

def actor_main(aid, cfg: dict, shared_model, version, q, stop, args: dict):
    try:
        torch.set_num_threads(1)
        torch.set_grad_enabled(False)
        dev = args["actor_device"]
        if dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"
        model = GuandanNet(ModelConfig(**cfg)).to(dev)
        model.eval()
        policy = BatchPolicy(model, dev)
        env = _core.VecEnv(args["envs"], args["seed"] * 1000 + aid * 7919 + int(time.time()), True)
        rng = np.random.default_rng(args["seed"] * 100 + aid + int(time.time()))
        local_v = -1
        rounds = 0
        steps = 0
        send_size = 4096
        while not stop.is_set():
            if version.value != local_v:
                local_v = version.value
                model.load_state_dict(shared_model.state_dict())
            _, s, h, a, off = unpack_obs(env.observe())
            ch = policy.choose(s, h, a, off, args["epsilon"], rng)
            res = env.step(ch.tolist(), True)
            rounds += len(res)
            steps += len(ch)
            if env.num_samples() >= send_size:
                n, sb, hb, ab, tb = env.take_samples()
                s_p = np.packbits(np.frombuffer(sb, np.uint8).reshape(n, STATE_DIM), axis=1)
                h_p = np.packbits(np.frombuffer(hb, np.uint8).reshape(n, H_FLAT), axis=1)
                a_p = np.packbits(np.frombuffer(ab, np.uint8).reshape(n, ACTION_DIM), axis=1)
                t = np.frombuffer(tb, np.float32).copy()
                while not stop.is_set():
                    try:
                        q.put((aid, s_p, h_p, a_p, t, rounds, steps), timeout=1.0)
                        break
                    except queue_mod.Full:
                        continue
                rounds = 0
                steps = 0
    except KeyboardInterrupt:
        pass


# ----------------------------------------------------------------------------- evaluator

def evaluator_main(run_dir: str, args: dict, stop):
    try:
        from .evaluate import evaluate
        torch.set_num_threads(max(1, args["eval_threads"]))
        torch.set_grad_enabled(False)
        ckpt = os.path.join(run_dir, "ckpt_latest.pt")
        metrics = os.path.join(run_dir, "metrics.jsonl")
        base_policy = None
        if args.get("base"):
            try:
                bm, _ = load_checkpoint(args["base"], "cpu")
                base_policy = BatchPolicy(bm, "cpu")
            except Exception as e:  # pragma: no cover
                print("evaluator: cannot load base model:", e, flush=True)
        last_mtime = 0.0
        last_eval = 0.0
        k = 0
        while not stop.is_set():
            time.sleep(2.0)
            if not os.path.exists(ckpt):
                continue
            mt = os.path.getmtime(ckpt)
            if mt == last_mtime or time.time() - last_eval < args["eval_interval"]:
                continue
            try:
                model, meta = load_checkpoint(ckpt, "cpu")
            except Exception:
                continue
            last_mtime = mt
            last_eval = time.time()
            pol = BatchPolicy(model, "cpu")
            opps = [("random", "random"), ("rule", "rule")]
            if base_policy is not None:
                opps.append(("base", base_policy))
            row = {"type": "eval", "t": time.time(), "frames": meta.get("frames", 0),
                   "train_seconds": meta.get("train_seconds", 0)}
            for name, opp in opps:
                if stop.is_set():
                    return
                r = evaluate(pol, opp, rounds=args["eval_rounds"], envs=64, seed=1000 + k)
                row[name] = {"win_rate": r["win_rate"], "score": r["score"], "rounds": r["rounds"]}
            k += 1
            with open(metrics, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[eval] frames={row['frames']:.0f} " + " ".join(
                f"{n}:{row[n]['win_rate']*100:.1f}%/{row[n]['score']:+.2f}" for n, _ in opps), flush=True)
    except KeyboardInterrupt:
        pass


# ----------------------------------------------------------------------------- learner

class Replay:
    def __init__(self, cap):
        self.cap = cap
        self.s = np.zeros((cap, (STATE_DIM + 7) // 8), np.uint8)
        self.h = np.zeros((cap, (H_FLAT + 7) // 8), np.uint8)
        self.a = np.zeros((cap, (ACTION_DIM + 7) // 8), np.uint8)
        self.t = np.zeros(cap, np.float32)
        self.pos = 0
        self.size = 0

    def add(self, s, h, a, t):
        n = len(t)
        idx = (np.arange(n) + self.pos) % self.cap
        self.s[idx] = s
        self.h[idx] = h
        self.a[idx] = a
        self.t[idx] = t
        self.pos = (self.pos + n) % self.cap
        self.size = min(self.cap, self.size + n)

    def sample(self, b, rng):
        idx = rng.integers(0, self.size, b)
        s = np.unpackbits(self.s[idx], axis=1, count=STATE_DIM)
        h = np.unpackbits(self.h[idx], axis=1, count=H_FLAT).reshape(b, HIST_LEN, HIST_DIM)
        a = np.unpackbits(self.a[idx], axis=1, count=ACTION_DIM)
        return s, h, a, self.t[idx]


def log_line(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def run(args):
    run_dir = os.path.join(runs_dir(), args.name)
    os.makedirs(run_dir, exist_ok=True)
    stop_file = os.path.join(run_dir, "STOP")
    if os.path.exists(stop_file):
        os.remove(stop_file)
    metrics = os.path.join(run_dir, "metrics.jsonl")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    frames = 0
    train_seconds = 0.0
    resume_opt = None
    latest = os.path.join(run_dir, "ckpt_latest.pt")
    if os.path.exists(latest):  # resume an interrupted run
        model, meta, resume_opt = load_checkpoint(latest, "cpu", with_optimizer=True)
        frames = int(meta.get("frames", 0))
        train_seconds = float(meta.get("train_seconds", 0))
        print(f"resuming {args.name} from {frames} frames", flush=True)
    elif args.base:
        model, meta = load_checkpoint(args.base, "cpu")
        print(f"starting from base model {args.base} ({model.cfg.preset})", flush=True)
    else:
        model = GuandanNet(PRESETS[args.preset])
    cfg = model.cfg
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    use_amp = device.startswith("cuda") and not args.no_amp
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if resume_opt is not None:
        try:
            opt.load_state_dict(resume_opt)
            for g in opt.param_groups:
                g["lr"] = args.lr
        except Exception:
            pass

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(dict(vars(args), model=cfg.to_dict(), params=count_params(model), device=device,
                       started=time.time()), f, ensure_ascii=False, indent=2)
    log_line(metrics, {"type": "start", "t": time.time(), "frames": frames, "preset": cfg.preset,
                       "params": count_params(model), "device": device, "actors": args.actors,
                       "base": args.base})

    ctx = mp.get_context("spawn")
    shared = GuandanNet(cfg)
    shared.load_state_dict({k: v.detach().cpu() for k, v in model.state_dict().items()})
    shared.share_memory()
    version = ctx.Value("i", 0)
    q = ctx.Queue(maxsize=max(4, args.actors * 3))
    stop = ctx.Event()
    adict = dict(envs=args.envs, epsilon=args.epsilon, actor_device=args.actor_device, seed=args.seed,
                 eval_interval=args.eval_interval, eval_rounds=args.eval_rounds, eval_threads=args.eval_threads,
                 base=args.base)
    procs = []
    for i in range(args.actors):
        p = ctx.Process(target=actor_main, args=(i, cfg.to_dict(), shared, version, q, stop, adict), daemon=True)
        p.start()
        procs.append(p)
    ev = ctx.Process(target=evaluator_main, args=(run_dir, adict, stop), daemon=True)
    ev.start()
    procs.append(ev)

    def save(snapshot=False):
        meta = {"frames": frames, "train_seconds": train_seconds, "run": args.name, "preset": cfg.preset}
        save_checkpoint(latest, model, meta, optimizer=opt)
        save_checkpoint(os.path.join(run_dir, "model.pt"), model, meta, half=True)
        if snapshot:
            save_checkpoint(os.path.join(run_dir, f"model_{frames // 1000}k.pt"), model, meta, half=True)

    buf = Replay(args.buffer)
    received = 0
    trained = 0
    rounds = 0
    steps_total = 0
    loss_acc, loss_n = 0.0, 0
    t_start = time.time()
    t_last_log = t_start
    t_last_save = t_start
    t_last_snap = t_start
    t_seg = time.time()
    frames_at_log = frames
    rounds_at_log = 0
    stop_reason = "stopped"
    sync_every = 10
    n_updates = 0
    print(f"training {args.name}: preset={cfg.preset} params={count_params(model):,} device={device} "
          f"actors={args.actors}x{args.envs}", flush=True)

    interrupted = {"flag": False}

    def on_sig(*_):
        interrupted["flag"] = True
    try:
        signal.signal(signal.SIGINT, on_sig)
        signal.signal(signal.SIGTERM, on_sig)
    except Exception:
        pass

    try:
        while True:
            # ---- collect
            got = 0
            need_data = buf.size < args.batch * 4 or trained >= received * args.reuse
            while True:
                try:
                    if got == 0 and need_data:
                        item = q.get(timeout=0.5)
                    else:
                        item = q.get_nowait()
                except queue_mod.Empty:
                    break
                _, s_p, h_p, a_p, t, r_cnt, st_cnt = item
                t = t / 3.0
                buf.add(s_p, h_p, a_p, t)
                received += len(t)
                frames += len(t)
                rounds += r_cnt
                steps_total += st_cnt
                got += 1
                if got >= args.actors * 2:
                    break
            # ---- train
            budget = 0
            while buf.size >= args.batch * 4 and trained < received * args.reuse and budget < 50:
                s, h, a, t = buf.sample(args.batch, rng)
                st = torch.from_numpy(s).to(device, non_blocking=True).float()
                ht = torch.from_numpy(h).to(device, non_blocking=True).float()
                at = torch.from_numpy(a).to(device, non_blocking=True).float()
                tt = torch.from_numpy(t).to(device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    pred = model(st, ht, at)
                loss = torch.nn.functional.mse_loss(pred.float(), tt)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 40.0)
                scaler.step(opt)
                scaler.update()
                trained += args.batch
                budget += 1
                n_updates += 1
                loss_acc += float(loss.detach())
                loss_n += 1
                if n_updates % sync_every == 0:
                    with torch.no_grad():
                        for ps, pm in zip(shared.parameters(), model.parameters()):
                            ps.copy_(pm.detach().cpu())
                    version.value += 1
            now = time.time()
            train_seconds += now - t_seg
            t_seg = now
            # ---- log
            if now - t_last_log >= 10:
                dt = now - t_last_log
                row = {"type": "train", "t": now, "frames": frames,
                       "fps": (frames - frames_at_log) / dt,
                       "rounds_per_sec": (rounds - rounds_at_log) / dt,
                       "loss": loss_acc / max(1, loss_n), "updates": n_updates,
                       "buffer": buf.size, "train_seconds": train_seconds,
                       "alive": sum(p.is_alive() for p in procs[:-1])}
                log_line(metrics, row)
                print(f"[train] frames={frames:,} fps={row['fps']:.0f} loss={row['loss']:.4f} "
                      f"rounds/s={row['rounds_per_sec']:.1f} buffer={buf.size}", flush=True)
                frames_at_log = frames
                rounds_at_log = rounds
                loss_acc, loss_n = 0.0, 0
                t_last_log = now
                if row["alive"] == 0 and args.actors > 0:
                    stop_reason = "actors died"
                    break
            if now - t_last_save >= args.save_interval:
                snap = now - t_last_snap >= args.snapshot_interval
                save(snapshot=snap)
                t_last_save = now
                if snap:
                    t_last_snap = now
            if os.path.exists(stop_file) or interrupted["flag"]:
                stop_reason = "stopped by user"
                break
            if args.total_frames and frames >= args.total_frames:
                stop_reason = "reached total frames"
                break
            if args.max_hours and now - t_start >= args.max_hours * 3600:
                stop_reason = "reached time limit"
                break
    finally:
        stop.set()
        save(snapshot=True)
        log_line(metrics, {"type": "stop", "t": time.time(), "frames": frames, "reason": stop_reason})
        print(f"stopped: {stop_reason}; checkpoint saved to {run_dir}", flush=True)
        # drain queue so that actors can exit
        t_end = time.time() + 5
        while time.time() < t_end and any(p.is_alive() for p in procs):
            try:
                q.get(timeout=0.2)
            except Exception:
                pass
        for p in procs:
            if p.is_alive():
                p.terminate()


def main(argv=None):
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
