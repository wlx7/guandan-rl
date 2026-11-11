"""Pure self-play reinforcement learning (Deep Monte-Carlo).

* N actor processes each run `--envs` matches in parallel. All four seats are played by
  the latest network (epsilon-greedy). Every decision (state, history, chosen action) is
  labelled with the final outcome of that round for the acting player's team.
* Optional opponent pool (`--league`, off by default): in a share of the rounds the
  opposing team is played by an earlier version of the network; only the latest
  network's decisions become samples. In a 2+2 hour A/B test it lost to plain self-play,
  mostly because self-play produced twice as many samples per hour.
* Auxiliary task (`--aux`, on by default): the network also predicts the other players'
  hands from its state encoding (labels exist only in training). In a 2+2 hour A/B test
  it beat plain DMC by 0.39 levels per round head-to-head.
* Optional search labels (`--search-actors`, off by default, experimental).
* The learner regresses Q(s, a) onto these Monte-Carlo returns.
* An evaluator process periodically measures the latest checkpoint on duplicate deals
  (each deal played twice with the teams swapped) against a random player, a fixed
  rule-based player (never used for training) and the base model.

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
                    HIST_DIM, ACTION_DIM, HIDDEN_DIM, save_checkpoint, load_checkpoint, count_params, widen)
from .agents import BatchPolicy, unpack_obs, subset_obs
from .paths import runs_dir, safe_run_name

H_FLAT = HIST_LEN * HIST_DIM


def build_parser():
    ap = argparse.ArgumentParser(description="Guandan self-play DMC training")
    ap.add_argument("--name", default=time.strftime("run-%Y%m%d-%H%M%S"))
    ap.add_argument("--preset", default="", choices=[""] + list(PRESETS),
                    help="model size (default medium); with --base, a bigger size widens the base model first")
    ap.add_argument("--base", default="", help="checkpoint to start from (基准模型)")
    ap.add_argument("--device", default="auto", help="learner device: auto|cuda|cpu")
    ap.add_argument("--actor-device", default="auto", help="device for self-play inference: auto|cuda|cpu")
    ap.add_argument("--actors", type=int, default=0, help="self-play processes (0 = auto)")
    ap.add_argument("--envs", type=int, default=0, help="parallel matches per actor (0 = auto)")
    ap.add_argument("--batch", type=int, default=0, help="0 = auto (4096 on GPU, 512 on CPU)")
    ap.add_argument("--lr", type=float, default=0.0,
                    help="learning rate; 0 = auto: 2e-4 from scratch, 5e-5 when continuing from a trained model")
    ap.add_argument("--epsilon", type=float, default=0.02, help="share of random exploration moves")
    ap.add_argument("--epsilon-top", type=float, default=0.0,
                    help="share of exploration moves drawn from the network's --top-k best moves")
    ap.add_argument("--top-k", type=int, default=3, help="number of best moves --epsilon-top chooses among")
    ap.add_argument("--td-lambda", type=float, default=1.0,
                    help="1 = targets are round outcomes (Monte Carlo); below 1 = TD(lambda): mix the outcome "
                         "with the network's value at the same player's later decisions")
    ap.add_argument("--league", type=float, default=0.0,
                    help="share of rounds whose opponents are an earlier model from the pool (0 = pure self-play)")
    ap.add_argument("--pool", default="", help="extra checkpoints for the opponent pool, separated by ';'")
    ap.add_argument("--aux", type=float, default=1.0,
                    help="weight of the auxiliary loss that predicts the other players' hands (0 = off)")
    ap.add_argument("--ntp", type=float, default=0.0,
                    help="weight of the auxiliary loss that predicts the next move of the history (0 = off)")
    ap.add_argument("--search-actors", type=int, default=0,
                    help="self-play processes that label the top candidate moves with search values (expert iteration)")
    ap.add_argument("--search-envs", type=int, default=64, help="matches per search actor")
    ap.add_argument("--search-cfg", default="5,12,14", help="top_k,samples,depth of the search used for labels")
    ap.add_argument("--search-share", type=float, default=0.5, help="share of each batch drawn from search labels")
    ap.add_argument("--reuse", type=float, default=1.0, help="average times each sample is trained on")
    ap.add_argument("--buffer", type=int, default=200_000)
    ap.add_argument("--save-interval", type=float, default=300, help="seconds between checkpoints")
    ap.add_argument("--snapshot-interval", type=float, default=3600, help="seconds between kept snapshots")
    ap.add_argument("--eval-interval", type=float, default=600, help="seconds between evaluations")
    ap.add_argument("--eval-rounds", type=int, default=400)
    ap.add_argument("--eval-threads", type=int, default=1)
    ap.add_argument("--total-frames", type=float, default=0, help="stop after this many samples (0 = no limit)")
    ap.add_argument("--max-hours", type=float, default=0, help="stop after this many hours (0 = no limit)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-amp", action="store_true", help="disable mixed precision on GPU")
    ap.add_argument("--force", action="store_true", help="start even if another training is running")
    return ap


def set_priority(level: str):
    """'low' for actors/evaluator so the learner always gets CPU time; 'high' for the learner."""
    try:
        if os.name == "nt":
            # only raise the learner: below-normal processes get parked on slow
            # efficiency cores by Windows 11, which makes self-play ~10x slower
            if level == "high":
                import ctypes
                k = ctypes.windll.kernel32
                k.SetPriorityClass(k.GetCurrentProcess(), 0x8000)  # ABOVE_NORMAL
        elif level == "low":
            os.nice(5)
    except Exception:
        pass


def unpack_bits(x: torch.Tensor, n: int) -> torch.Tensor:
    """Inverse of np.packbits (big bit order) on any device: uint8 [B, nb] -> uint8 [B, n]."""
    shifts = torch.arange(7, -1, -1, device=x.device, dtype=torch.uint8)
    bits = (x.unsqueeze(-1) >> shifts) & 1
    return bits.reshape(x.shape[0], -1)[:, :n]


# ----------------------------------------------------------------------------- actors

def _snapshot_frames(path):
    try:
        return int(os.path.basename(path)[6:-4].rstrip("k"))
    except ValueError:
        return -1


def pool_paths(run_dir, base, extra, keep=8):
    """Opponent pool: base model, extra checkpoints, and the latest snapshots of this run."""
    import glob
    snaps = sorted(glob.glob(os.path.join(run_dir, "model_*k.pt")), key=_snapshot_frames)
    paths = ([base] if base else []) + [p for p in extra if p] + snaps[-keep:]
    return [p for p in dict.fromkeys(paths) if os.path.exists(p)]


def actor_main(aid, cfg: dict, shared_model, version, q, stop, args: dict):
    try:
        set_priority("low")
        torch.set_num_threads(1)
        torch.set_grad_enabled(False)
        dev = args["actor_device"]
        if dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"
        model = GuandanNet(ModelConfig(**cfg)).to(dev)
        model.eval()
        policy = BatchPolicy(model, dev)
        n_env = args["envs"]
        env = _core.VecEnv(n_env, args["seed"] * 1000 + aid * 7919 + int(time.time()), True)
        aux = float(args.get("aux", 0.0)) > 0
        env.set_record_hidden(aux)
        td_lambda = float(args.get("td_lambda", 1.0))
        use_td = td_lambda < 1.0
        if use_td:
            env.set_td_lambda(td_lambda)
        eps_top, top_k = float(args.get("epsilon_top", 0.0)), int(args.get("top_k", 3))
        rng = np.random.default_rng(args["seed"] * 100 + aid + int(time.time()))
        league = float(args.get("league", 0.0))
        pool: dict[str, BatchPolicy] = {}
        pool_list: list[str] = []
        t_pool = 0.0
        opp = [None] * n_env  # pool path of the opposing team's model, None = latest network
        learn_team = np.zeros(n_env, np.int64)

        def refresh_pool():
            # every few minutes this actor plays against (at most) two models of the pool, so
            # that one step never needs more than three network calls
            nonlocal pool_list
            paths = pool_paths(args["run_dir"], args.get("base", ""), args.get("pool", []))
            if len(paths) > 2:
                paths = [paths[i] for i in rng.choice(len(paths), 2, replace=False)]
            for p in paths:
                if p not in pool:
                    try:
                        m, _ = load_checkpoint(p, dev)
                        pool[p] = BatchPolicy(m, dev)
                    except Exception:
                        continue
            for p in list(pool):
                if p not in paths:
                    del pool[p]
            pool_list = [p for p in paths if p in pool]
            for i in range(n_env):
                if opp[i] is not None and opp[i] not in pool:
                    assign(i)

        def assign(i):
            if pool_list and rng.random() < league:
                opp[i] = pool_list[int(rng.integers(len(pool_list)))]
                learn_team[i] = int(rng.integers(2))
                env.set_learners(i, [p % 2 == learn_team[i] for p in range(4)])
            else:
                opp[i] = None
                env.set_learners(i, [True] * 4)

        local_v = -1
        rounds = 0
        steps = 0
        send_size = 4096
        while not stop.is_set():
            if league > 0 and time.time() - t_pool > 300:
                t_pool = time.time()
                had = any(opp)
                refresh_pool()
                if not had:
                    for i in range(n_env):
                        assign(i)
            if version.value != local_v:
                local_v = version.value
                model.load_state_dict(shared_model.state_dict())
            players, s, h, a, off = unpack_obs(env.observe())
            if any(opp):
                old = np.asarray([o is not None for o in opp]) & ((players % 2) != learn_team)
                ch = np.zeros(n_env, np.int64)
                rows = np.nonzero(~old)[0]
                ss, hh, aa, oo = subset_obs(s, h, a, off, rows)
                ch[rows] = policy.choose(ss, hh, aa, oo, args["epsilon"], rng)
                for p in set(opp[i] for i in np.nonzero(old)[0]):
                    rows = np.asarray([i for i in np.nonzero(old)[0] if opp[i] == p], np.int64)
                    ss, hh, aa, oo = subset_obs(s, h, a, off, rows)
                    ch[rows] = pool[p].choose(ss, hh, aa, oo)
                res = env.step(ch.tolist(), True)
            elif use_td:
                # targets are in levels; the network predicts levels / 3
                ch, qc = policy.choose(s, h, a, off, args["epsilon"], rng, top_k, eps_top, return_q=True)
                res = env.step_q(ch.tolist(), (qc * 3.0).tolist())
            else:
                ch = policy.choose(s, h, a, off, args["epsilon"], rng, top_k, eps_top)
                res = env.step(ch.tolist(), True)
            for (i, _w, _g, _m) in res:
                if league > 0:
                    assign(i)
            rounds += len(res)
            steps += len(ch)
            if env.num_samples() >= send_size:
                n, sb, hb, ab, tb = env.take_samples()
                s_p = np.packbits(np.frombuffer(sb, np.uint8).reshape(n, STATE_DIM), axis=1)
                h_p = np.packbits(np.frombuffer(hb, np.uint8).reshape(n, H_FLAT), axis=1)
                a_p = np.packbits(np.frombuffer(ab, np.uint8).reshape(n, ACTION_DIM), axis=1)
                t = np.frombuffer(tb, np.float32).copy()
                x_p = np.packbits(np.frombuffer(env.take_hidden(), np.uint8).reshape(n, HIDDEN_DIM), axis=1) if aux else None
                while not stop.is_set():
                    try:
                        q.put((aid, s_p, h_p, a_p, t, rounds, steps, x_p), timeout=1.0)
                        break
                    except queue_mod.Full:
                        continue
                rounds = 0
                steps = 0
    except KeyboardInterrupt:
        pass


def search_actor_main(aid, cfg: dict, shared_model, version, q, stop, args: dict):
    """Expert iteration: play with search and label the network's top candidates of every
    decision with their search values (average play-out results in sampled worlds). These
    targets are far less noisy than single-round Monte-Carlo returns and cover exactly the
    alternatives the network has to rank, which pure DMC almost never tries."""
    try:
        from .search import SearchAgent
        set_priority("low")
        torch.set_num_threads(1)
        torch.set_grad_enabled(False)
        dev = args["actor_device"]
        if dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"
        model = GuandanNet(ModelConfig(**cfg)).to(dev)
        model.eval()
        k, n, d = (int(x) for x in args["search_cfg"].split(","))
        searcher = SearchAgent(model, dev, top_k=k, samples=n, depth=d, prior=0.0)
        n_env = args["search_envs"]
        env = _core.VecEnv(n_env, args["seed"] * 1000 + 500 + aid * 7919 + int(time.time()), True)
        rng = np.random.default_rng(args["seed"] * 100 + 50 + aid + int(time.time()))
        local_v = -1
        rounds = steps = 0
        S, H, A, T = [], [], [], []
        while not stop.is_set():
            if version.value != local_v:
                local_v = version.value
                model.load_state_dict(shared_model.state_dict())
            _, s, h, a, off = unpack_obs(env.observe())
            n_moves = np.diff(off)
            ch = np.zeros(n_env, np.int64)
            multi = np.nonzero(n_moves > 1)[0]
            if len(multi):
                # a few games per search call: positions where a player leads can have hundreds
                # of legal moves, and every simulated world encodes all of them
                chunk = max(1, 768 // (k * n))
                results = []
                for c in range(0, len(multi), chunk):
                    results += searcher.evaluate_many([env.game(int(i)) for i in multi[c:c + chunk]])
                for i, res in zip(multi, results):
                    searched = [r for r in res if r[3]]
                    ch[i] = searched[0][0] if rng.random() >= 0.1 else searched[int(rng.integers(len(searched)))][0]
                    for mi, score, _q, _ in searched:
                        S.append(s[i])
                        H.append(h[i])
                        A.append(a[off[i] + mi])
                        T.append(score)  # levels, like round outcomes
            res = env.step(ch.tolist(), False)
            rounds += len(res)
            steps += n_env
            if len(T) >= 2048:
                s_p = np.packbits(np.stack(S), axis=1)
                h_p = np.packbits(np.stack(H).reshape(len(T), H_FLAT), axis=1)
                a_p = np.packbits(np.stack(A), axis=1)
                t = np.asarray(T, np.float32)
                S, H, A, T = [], [], [], []
                while not stop.is_set():
                    try:
                        q.put((-1 - aid, s_p, h_p, a_p, t, rounds, steps, None), timeout=1.0)
                        break
                    except queue_mod.Full:
                        continue
                rounds = steps = 0
    except KeyboardInterrupt:
        pass


# ----------------------------------------------------------------------------- evaluator

def evaluator_main(run_dir: str, args: dict, stop):
    """Duplicate-deal evaluation of every new checkpoint. The deals are the same at every
    evaluation, so changes between checkpoints are not hidden by the luck of the cards."""
    try:
        from .duplicate import duplicate_match, NetPlayer, RulePlayer, RandomPlayer
        set_priority("low")
        torch.set_num_threads(max(1, args["eval_threads"]))
        torch.set_grad_enabled(False)
        ckpt = os.path.join(run_dir, "ckpt_latest.pt")
        metrics = os.path.join(run_dir, "metrics.jsonl")
        base_player = None
        if args.get("base"):
            try:
                bm, _ = load_checkpoint(args["base"], "cpu")
                base_player = NetPlayer(bm, "cpu")
            except Exception as e:  # pragma: no cover
                print("evaluator: cannot load base model:", e, flush=True)
        deals = max(16, args["eval_rounds"] // 2)
        last_mtime = 0.0
        last_eval = 0.0
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
            me = NetPlayer(model, "cpu")
            opps = [("random", RandomPlayer(7), max(16, deals // 4)), ("rule", RulePlayer(), deals)]
            if base_player is not None:
                opps.append(("base", base_player, deals))
            row = {"type": "eval", "t": time.time(), "frames": meta.get("frames", 0),
                   "train_seconds": meta.get("train_seconds", 0), "duplicate": True}
            for name, opp, n in opps:
                if stop.is_set():
                    return
                r = duplicate_match(me, opp, deals=n, seed=1000)
                row[name] = {k: r[k] for k in ("win_rate", "score", "score_se", "rounds")}
            with open(metrics, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[eval] frames={row['frames']:.0f} " + " ".join(
                f"{n}:{row[n]['win_rate']*100:.1f}%/{row[n]['score']:+.2f}±{row[n]['score_se']:.2f}"
                for n, _, _ in opps), flush=True)
    except KeyboardInterrupt:
        pass


# ----------------------------------------------------------------------------- learner

class Replay:
    """Bit-packed replay buffer that lives on the learner device (GPU memory when available)."""

    def __init__(self, cap, device, hidden=False):
        self.cap = cap
        self.device = device
        self.s = torch.zeros((cap, (STATE_DIM + 7) // 8), dtype=torch.uint8, device=device)
        self.h = torch.zeros((cap, (H_FLAT + 7) // 8), dtype=torch.uint8, device=device)
        self.a = torch.zeros((cap, (ACTION_DIM + 7) // 8), dtype=torch.uint8, device=device)
        self.t = torch.zeros(cap, dtype=torch.float32, device=device)
        # auxiliary labels: the other players' hands
        self.x = torch.zeros((cap, (HIDDEN_DIM + 7) // 8), dtype=torch.uint8, device=device) if hidden else None
        self.pos = 0
        self.size = 0

    def add(self, s, h, a, t, x=None):
        n = len(t)
        idx = (torch.arange(n, device=self.device) + self.pos) % self.cap
        self.s[idx] = torch.from_numpy(s).to(self.device)
        self.h[idx] = torch.from_numpy(h).to(self.device)
        self.a[idx] = torch.from_numpy(a).to(self.device)
        self.t[idx] = torch.from_numpy(t).to(self.device)
        if self.x is not None and x is not None:
            self.x[idx] = torch.from_numpy(x).to(self.device)
        self.pos = (self.pos + n) % self.cap
        self.size = min(self.cap, self.size + n)

    def sample(self, b):
        """(states, hists, actions, targets, hidden-hand labels or None)"""
        idx = torch.randint(0, self.size, (b,), device=self.device)
        if self.device == "cpu":  # numpy is faster than torch bit ops on CPU
            ii = idx.numpy()
            s = torch.from_numpy(np.unpackbits(self.s.numpy()[ii], axis=1, count=STATE_DIM)).float()
            h = torch.from_numpy(np.unpackbits(self.h.numpy()[ii], axis=1, count=H_FLAT)).view(b, HIST_LEN, HIST_DIM).float()
            a = torch.from_numpy(np.unpackbits(self.a.numpy()[ii], axis=1, count=ACTION_DIM)).float()
            x = (torch.from_numpy(np.unpackbits(self.x.numpy()[ii], axis=1, count=HIDDEN_DIM)).float()
                 if self.x is not None else None)
            return s, h, a, self.t[idx], x
        s = unpack_bits(self.s[idx], STATE_DIM).float()
        h = unpack_bits(self.h[idx], H_FLAT).view(b, HIST_LEN, HIST_DIM).float()
        a = unpack_bits(self.a[idx], ACTION_DIM).float()
        x = unpack_bits(self.x[idx], HIDDEN_DIM).float() if self.x is not None else None
        return s, h, a, self.t[idx], x


def log_line(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def other_active_runs(exclude=None):
    from .proc import run_alive
    out = []
    for d in sorted(os.listdir(runs_dir())):
        p = os.path.join(runs_dir(), d)
        if d != exclude and os.path.isdir(p) and run_alive(p):
            out.append(d)
    return out


def run(args):
    from .proc import run_alive, write_pid, remove_pid, resource_limits
    args.name = safe_run_name(args.name) or time.strftime("run-%Y%m%d-%H%M%S")
    if args.td_lambda < 1.0 and args.league > 0:
        print("--td-lambda below 1 cannot be combined with --league", flush=True)
        sys.exit(2)
    run_dir = os.path.join(runs_dir(), args.name)
    os.makedirs(run_dir, exist_ok=True)
    if run_alive(run_dir):
        print(f"training '{args.name}' is already running (pid file in {run_dir})", flush=True)
        sys.exit(2)
    others = other_active_runs(args.name)
    if others and not args.force:
        print(f"another training is running: {', '.join(others)}. Stop it first "
              f"(two trainings do not fit into GPU memory / RAM), or pass --force.", flush=True)
        sys.exit(3)
    write_pid(run_dir)
    stop_file = os.path.join(run_dir, "STOP")
    if os.path.exists(stop_file):
        os.remove(stop_file)
    metrics = os.path.join(run_dir, "metrics.jsonl")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.actor_device == "auto":
        args.actor_device = "cuda" if torch.cuda.is_available() else "cpu"
    cpu = os.cpu_count() or 2
    if not args.actors:
        # GPU self-play: a few processes with many matches each saturate the GPU;
        # CPU self-play: one single-threaded process per spare core
        args.actors = 3 if args.actor_device == "cuda" else max(1, min(16, cpu - (4 if cpu >= 12 else 2)))
    if not args.envs:
        args.envs = 256 if args.actor_device == "cuda" else 64
    if not args.batch:
        args.batch = 4096 if device.startswith("cuda") else 512
    lim = resource_limits(args.actor_device)
    if args.actors > lim["max_actors"]:
        print(f"actors {args.actors} -> {lim['max_actors']} (limited by GPU memory / RAM)", flush=True)
        args.actors = lim["max_actors"]
    if args.actors * args.envs > lim["max_total_envs"]:
        new_envs = max(16, lim["max_total_envs"] // args.actors)
        print(f"envs per actor {args.envs} -> {new_envs} (memory limit)", flush=True)
        args.envs = new_envs
    # avoid OpenMP oversubscription: actors already keep every core busy
    spare = (os.cpu_count() or 2) - args.actors - args.eval_threads
    torch.set_num_threads(2 if device.startswith("cuda") else max(1, min(4, spare)))
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
        want = PRESETS.get(args.preset)
        c = model.cfg
        if want and want.preset != c.preset and want.hidden >= c.hidden and want.lstm >= c.lstm and want.head_layers >= c.head_layers:
            n0 = count_params(model)
            model = widen(model, want, seed=args.seed)
            print(f"widened the base model to {want.preset}: {n0:,} -> {count_params(model):,} parameters "
                  f"(same play at the start, more capacity to learn)", flush=True)
        elif want and want.preset != model.cfg.preset:
            print(f"--preset {args.preset} ignored: a base model can only be widened, so it stays {model.cfg.preset}", flush=True)
    else:
        model = GuandanNet(PRESETS[args.preset or RECOMMENDED_PRESET])
    if args.aux > 0:
        model.add_belief_head()
    if args.ntp > 0:
        model.add_ntp_head()
    cfg = model.cfg
    model.to(device).train()
    # a trained model is fine-tuned with a smaller step (hill-climb round 4: 5e-5 beat 2e-4 by
    # +0.10 levels/round when continuing from a strong model)
    lr = args.lr or (5e-5 if args.base else 2e-4)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    use_amp = device.startswith("cuda") and not args.no_amp
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if resume_opt is not None:
        try:
            opt.load_state_dict(resume_opt)
            if args.lr:  # otherwise keep the learning rate the run was started with
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
                 base=args.base, league=args.league, run_dir=run_dir,
                 pool=[p for p in args.pool.split(";") if p.strip()],
                 search_envs=args.search_envs, search_cfg=args.search_cfg, aux=args.aux, ntp=args.ntp,
                 td_lambda=args.td_lambda, epsilon_top=args.epsilon_top, top_k=args.top_k)
    procs = []
    for i in range(args.actors):
        p = ctx.Process(target=actor_main, args=(i, cfg.to_dict(), shared, version, q, stop, adict), daemon=True)
        p.start()
        procs.append(p)
    for i in range(args.search_actors):
        p = ctx.Process(target=search_actor_main, args=(i, cfg.to_dict(), shared, version, q, stop, adict), daemon=True)
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

    if device.startswith("cuda"):
        set_priority("high")
    buf = Replay(args.buffer, device, hidden=args.aux > 0)
    sbuf = Replay(max(20_000, args.buffer // 2), device) if args.search_actors else None
    s_received = 0
    busy = 0.0
    received = 0
    trained = 0
    rounds = 0
    steps_total = 0
    loss_acc, loss_n = 0.0, 0
    aux_acc = 0.0
    ntp_acc = 0.0
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
          f"actors={args.actors}x{args.envs} on {args.actor_device} batch={args.batch} lr={opt.param_groups[0]['lr']:g}", flush=True)

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
                src, s_p, h_p, a_p, t, r_cnt, st_cnt, x_p = item
                t = t / 3.0
                if src < 0 and sbuf is not None:  # search labels
                    sbuf.add(s_p, h_p, a_p, t)
                    s_received += len(t)
                    got += 1
                    continue
                buf.add(s_p, h_p, a_p, t, x_p)
                received += len(t)
                frames += len(t)
                rounds += r_cnt
                steps_total += st_cnt
                got += 1
                if got >= args.actors * 2:
                    break
            # ---- train
            budget = 0
            tb = time.time()
            while buf.size >= args.batch * 4 and trained < received * args.reuse and budget < 50:
                if sbuf is not None and sbuf.size >= 4096:
                    ns = int(args.batch * args.search_share)
                    st, ht, at, tt, xt = buf.sample(args.batch - ns)
                    s2, h2, a2, t2, _ = sbuf.sample(ns)
                    st, ht, at, tt = torch.cat([st, s2]), torch.cat([ht, h2]), torch.cat([at, a2]), torch.cat([tt, t2])
                else:
                    st, ht, at, tt, xt = buf.sample(args.batch)
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    if args.ntp > 0:
                        z, seq = model.encode_state_seq(st, ht)
                        nl = model.ntp(seq[:, :-1])
                    else:
                        z = model.encode_state(st, ht)
                        nl = None
                    pred = model.score(z, at)
                    bl = model.belief_logits(z[: len(xt)]) if (xt is not None and args.aux > 0) else None
                loss = torch.nn.functional.mse_loss(pred.float(), tt)
                if nl is not None:
                    # next-move prediction on the history: only pairs of real (non-padding) moves
                    nxt = ht[:, 1:]
                    valid = (ht[:, :-1].sum(-1) > 0) & (nxt.sum(-1) > 0)
                    if valid.any():
                        ntp_loss = torch.nn.functional.binary_cross_entropy_with_logits(nl[valid].float(), nxt[valid])
                        ntp_acc += float(ntp_loss.detach())
                        loss = loss + args.ntp * ntp_loss
                if bl is not None:
                    aux_loss = torch.nn.functional.binary_cross_entropy_with_logits(bl.float(), xt)
                    aux_acc += float(aux_loss.detach())
                    loss = loss + args.aux * aux_loss
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
            if budget:
                busy += now - tb
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
                       "learner_busy": busy / dt,
                       "search_labels": s_received,
                       "aux_loss": aux_acc / max(1, loss_n) if args.aux > 0 else None,
                       "ntp_loss": ntp_acc / max(1, loss_n) if args.ntp > 0 else None,
                       "alive": sum(p.is_alive() for p in procs[:-1])}
                log_line(metrics, row)
                print(f"[train] frames={frames:,} fps={row['fps']:.0f} loss={row['loss']:.4f} "
                      f"rounds/s={row['rounds_per_sec']:.1f} learner={row['learner_busy']*100:.0f}% "
                      f"updates={n_updates}", flush=True)
                frames_at_log = frames
                rounds_at_log = rounds
                loss_acc, loss_n = 0.0, 0
                aux_acc = 0.0
                ntp_acc = 0.0
                busy = 0.0
                t_last_log = now
                if received == 0 and now - t_start > 240:
                    stop_reason = "no self-play data after 4 minutes (out of GPU memory or RAM?)"
                    break
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
        remove_pid(run_dir)
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


def _utf8_stdio():
    import sys as _s
    for st in (_s.stdout, _s.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main(argv=None):
    _utf8_stdio()
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
