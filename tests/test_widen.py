"""Widening a trained model (Net2Net) keeps its play while adding capacity."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from guandan.model import PRESETS, GuandanNet, count_params, widen, STATE_DIM, HIST_LEN, HIST_DIM, ACTION_DIM
from test_npnet import positions


def medium_model():
    """A medium network (the built-in model is large since v1.10, so widen a seeded medium one)."""
    torch.manual_seed(0)
    m = GuandanNet(PRESETS["medium"])
    m.add_belief_head()
    return m.eval()


def q_values(model, ps):
    out = []
    with torch.no_grad():
        for s, h, a, n in ps:
            st = torch.from_numpy(np.frombuffer(bytes(s), np.uint8).reshape(1, STATE_DIM).astype(np.float32))
            ht = torch.from_numpy(np.frombuffer(bytes(h), np.uint8).reshape(1, HIST_LEN, HIST_DIM).astype(np.float32))
            at = torch.from_numpy(np.frombuffer(bytes(a), np.uint8).reshape(n, ACTION_DIM).astype(np.float32))
            z = model.encode_state(st, ht)
            out.append((model.score(z.expand(n, -1), at).numpy(), torch.sigmoid(model.belief_logits(z)).numpy() if model.cfg.belief else None))
    return out


@pytest.mark.parametrize("target", ["large", "xlarge"])
def test_widen_preserves_function(target):
    small = medium_model()
    ps = positions(30)
    exact = widen(small, PRESETS[target])
    assert count_params(exact) > 1.9 * count_params(small)
    for (q0, b0), (q1, b1) in zip(q_values(small, ps), q_values(exact, ps)):
        assert np.abs(q0 - q1).max() < 1e-4
        assert np.abs(b0 - b1).max() < 1e-4
    noisy = widen(small, PRESETS[target], noise=0.001)  # optional extra noise: still nearly the same play
    same = 0
    for (q0, _), (q1, _) in zip(q_values(small, ps), q_values(noisy, ps)):
        assert np.abs(q0 - q1).max() < 0.05
        same += int(np.argmax(q0) == np.argmax(q1))
    assert same >= 0.9 * len(ps)
