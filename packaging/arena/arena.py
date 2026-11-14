"""Run many tables of the competition referee in parallel: our AI against an external AI.

Each table is one referee process (`danserver <games>`) on its own port (portshim.so remaps
the hard-coded 23456) with four clients. Teams are fixed by connection order (1st+3rd vs
2nd+4th); even tables seat our AI at positions 0/2, odd tables at 1/3.

    python arena.py --opp danzero --tables 12 --games 20 --out results/danzero
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def opp_cmd(opp: str, name: str, ext: str):
    if opp == "danzero":
        return [sys.executable, os.path.join(HERE, "danzero_launch.py"),
                "--src", f"{ext}/Danzero_plus/wintest/danzero", "--weights", f"{ext}/danzero_q.npz",
                "--client", "client0.py", "--name", name]
    if opp == "njupt":
        return [sys.executable, os.path.join(HERE, "njupt_launch.py"),
                "--src", f"{ext}/DanLM/baselines/fin-njupt-guandan-ai", "--name", name]
    raise SystemExit(f"unknown opponent {opp}")


def ours_cmd(name: str, log: str, extra):
    return [sys.executable, os.path.join(HERE, "our_client.py"), "--name", name, "--log", log, *extra]


def start_table(k, args):
    port = args.base_port + k
    env = dict(os.environ, ARENA_PORT=str(port), LD_PRELOAD=args.shim,
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    d = os.path.join(args.out, f"t{k:02d}")
    os.makedirs(d, exist_ok=True)
    srv = subprocess.Popen([args.server, str(args.games)], cwd=d, env=env,
                           stdout=open(os.path.join(d, "server.log"), "w"), stderr=subprocess.STDOUT)
    time.sleep(2.0)
    ours_first = k % 2 == 0
    log = os.path.join(d, "ours.jsonl")
    extra = ["--no-endgame"] if args.no_endgame else []
    names = [f"a{k}x", f"b{k}x", f"c{k}x", f"d{k}x"]
    procs = [srv]
    for seat in range(4):
        mine = (seat % 2 == 0) == ours_first
        nm = names[seat]
        cmd = ours_cmd(nm, log, extra) if mine else opp_cmd(args.opp, nm, args.ext)
        procs.append(subprocess.Popen(cmd, cwd=d, env=env, stdout=open(os.path.join(d, f"{nm}.log"), "w"),
                                      stderr=subprocess.STDOUT))
        time.sleep(1.0)
    return procs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opp", required=True)
    ap.add_argument("--tables", type=int, default=8)
    ap.add_argument("--games", type=int, default=10, help="complete games (2 -> past A) per table")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ext", default="/workspace/ext")
    ap.add_argument("--server", default="/workspace/ext/danserver")
    ap.add_argument("--shim", default=os.path.join(HERE, "portshim.so"))
    ap.add_argument("--base-port", type=int, default=24000)
    ap.add_argument("--no-endgame", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tables = [start_table(k, args) for k in range(args.tables)]
    t0 = time.time()
    while any(p[0].poll() is None for p in tables):
        time.sleep(10)
        done = sum(p[0].poll() is not None for p in tables)
        print(f"{time.time() - t0:7.0f}s  tables finished {done}/{len(tables)}", flush=True)
    time.sleep(5)
    for procs in tables:
        for p in procs[1:]:
            if p.poll() is None:
                p.kill()
    print("all done", flush=True)


if __name__ == "__main__":
    main()
