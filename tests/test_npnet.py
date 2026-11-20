"""The torch-free forward pass (portable build) must match the PyTorch network."""
import os
import random

import numpy as np
import pytest

from guandan import _core
from guandan.npnet import load_numpy_model, load_pt
from guandan.paths import builtin_models_dir

MODEL = os.path.join(builtin_models_dir(), "guandan-medium.pt")


def positions(n=40, seed=3):
    rng = random.Random(seed)
    out = []
    for g_i in range(10):
        g = _core.Game(seed + g_i)
        while g.phase not in (2, 3) and len(out) < n:
            moves = g.legal_moves()
            if rng.random() < 0.2:
                out.append(g.encode())
            k, key, cards = moves[rng.randrange(len(moves))]
            g.apply(k, key, list(cards))
        if len(out) >= n:
            break
    return out


def test_numpy_matches_torch():
    torch = pytest.importorskip("torch")
    from guandan.model import load_checkpoint, STATE_DIM, HIST_LEN, HIST_DIM, ACTION_DIM
    tm, _ = load_checkpoint(MODEL)
    nm, meta = load_numpy_model(MODEL)
    ps = positions()
    assert len(ps) >= 5
    worst = 0.0
    for s, h, a, n in ps:
        st = np.frombuffer(bytes(s), np.uint8).reshape(1, STATE_DIM).astype(np.float32)
        ht = np.frombuffer(bytes(h), np.uint8).reshape(1, HIST_LEN, HIST_DIM).astype(np.float32)
        at = np.frombuffer(bytes(a), np.uint8).reshape(n, ACTION_DIM).astype(np.float32)
        with torch.no_grad():
            z = tm.encode_state(torch.from_numpy(st), torch.from_numpy(ht))
            q_t = tm.score(z.expand(n, -1), torch.from_numpy(at)).numpy()
        q_n = nm.score(nm.encode_state(st, ht), at)
        worst = max(worst, float(np.abs(q_t - q_n).max()))
        assert np.argmax(q_t) == np.argmax(q_n) or abs(q_t.max() - q_n[np.argmax(q_t)]) < 1e-5
    assert worst < 1e-4, worst


def test_load_pt_reads_config_and_meta():
    obj = load_pt(MODEL)
    assert obj["config"]["preset"]
    assert "lstm.weight_ih_l0" in obj["state_dict"]


def test_fastnet_matches_torch_and_numpy():
    torch = pytest.importorskip("torch")
    from guandan.agents import build_fastnet
    from guandan.model import load_checkpoint, STATE_DIM, HIST_LEN, HIST_DIM, ACTION_DIM
    tm, _ = load_checkpoint(MODEL)
    nm, _ = load_numpy_model(MODEL)
    fn_t = build_fastnet(tm)
    fn_n = build_fastnet(nm)
    assert fn_t is not None and fn_n is not None
    assert fn_t.has_oracle() and fn_n.has_oracle()
    g = _core.Game(42)
    assert abs(fn_t.eval_oracle_game(g, 0) - fn_n.eval_oracle_game(g, 0)) < 1e-6
    for step in range(30):
        if g.phase not in (0, 1):
            break
        mvs_t, q_ft, lg_ft = fn_t.evaluate_game(g, True)
        mvs_n, q_fn, lg_fn = fn_n.evaluate_game(g, True)
        assert np.allclose(q_ft, q_fn, atol=1e-6)
        assert np.allclose(lg_ft, lg_fn, atol=1e-6)
        if step in (0, 10, 22):
            s, h, a, n = g.encode()
            st = torch.frombuffer(bytearray(s), dtype=torch.uint8).view(1, STATE_DIM).float()
            ht = torch.frombuffer(bytearray(h), dtype=torch.uint8).view(1, HIST_LEN, HIST_DIM).float()
            at = torch.frombuffer(bytearray(a), dtype=torch.uint8).view(n, ACTION_DIM).float()
            with torch.no_grad():
                z = tm.encode_state(st, ht)
                q_ref = tm.score(z.expand(n, -1), at).numpy()
            assert float(np.max(np.abs(np.asarray(q_ft) - q_ref))) < 1e-4
        g.apply(*mvs_t[int(np.argmax(q_ft))])
