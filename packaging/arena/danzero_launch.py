"""Run the published DanZero client (github.com/submit-paper/Danzero_plus, wintest/danzero)
without its TensorFlow-1 actor process: the client's ZeroMQ request to the actor is answered
in-process by the same network (567 -> 5 x 512 tanh -> 1) in numpy. The client code itself
(state features, tribute / return logic, message handling) runs unchanged.

    python danzero_launch.py --src <wintest/danzero> --weights danzero_q.npz --client client0.py --name dz0

The weights come from the repository's q_network.ckpt (a pickle of 12 numpy arrays), converted
with safe_unpickle.py, which refuses anything but plain arrays.
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys
import types

import numpy as np


class DanZeroNet:
    def __init__(self, path):
        z = np.load(path)
        self.w = [z[f"arr_{i}"].astype(np.float32) for i in range(len(z.files))]

    def __call__(self, x):
        h = np.asarray(x, np.float32)
        n = len(self.w) // 2
        for i in range(n):
            h = h @ self.w[2 * i] + self.w[2 * i + 1]
            if i < n - 1:
                h = np.tanh(h)
        return h[:, 0]


def install_shims(net):
    """Fake `zmq` and `pyarrow` modules: send(state) -> recv() returns argmax_a Q(state, a),
    exactly what the original actor process (actor.py: np.argmax(model.forward(x_batch))) did."""
    class Buf:
        def __init__(self, obj):
            self.obj = obj

        def to_buffer(self):
            return self.obj

    pa = types.ModuleType("pyarrow")
    pa.serialize = Buf
    pa.deserialize = lambda b: b
    sys.modules["pyarrow"] = pa

    class Sock:
        def connect(self, *_a):
            pass

        def send(self, state):
            self.state = state

        def recv(self):
            return int(np.argmax(net(self.state["x_batch"])))

    class Ctx:
        linger = 0

        def socket(self, *_a):
            return Sock()

    zmq = types.ModuleType("zmq")
    zmq.Context = Ctx
    zmq.REQ = 3
    zmq.REP = 4
    sys.modules["zmq"] = zmq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--client", default="client0.py")
    ap.add_argument("--name", default=None, help="user_info in the URL (default: the client's own)")
    args = ap.parse_args()
    install_shims(DanZeroNet(args.weights))
    sys.path.insert(0, os.path.abspath(args.src))
    path = os.path.join(args.src, args.client)
    if args.name:
        src = open(path, encoding="utf-8").read()
        import re  # noqa: PLC0415
        src = re.sub(r"ws://127\.0\.0\.1:23456/game/\w+", f"ws://127.0.0.1:23456/game/{args.name}", src)
        tmp = os.path.join(os.path.dirname(os.path.abspath(path)), f"_arena_{args.name}.py")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(src)
        path = tmp
    sys.argv = [path]
    runpy.run_path(path, run_name="__main__")


if __name__ == "__main__":
    main()
