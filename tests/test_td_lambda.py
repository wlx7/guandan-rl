"""TD(lambda) training targets of the vectorised environment."""
import numpy as np

from guandan import _core


def play(lam, seed=5, n_env=4, steps=3000):
    """Play with a fixed pseudo-random policy; returns targets and the q values given per sample."""
    env = _core.VecEnv(n_env, seed, True)
    if lam is not None:
        env.set_td_lambda(lam)
    rng = np.random.default_rng(1)
    for _ in range(steps):
        players, s, h, a, off = env.observe()
        off = np.asarray(off, np.int64)
        n = np.diff(off)
        ch = (rng.random(n_env) * n).astype(np.int64)
        q = rng.normal(size=n_env).astype(np.float32)
        if lam is None:
            env.step(ch.tolist(), True)
        else:
            env.step_q(ch.tolist(), q.tolist())
    n, sb, hb, ab, tb = env.take_samples()
    return n, np.frombuffer(tb, np.float32).copy(), sb


def test_lambda_one_is_monte_carlo():
    n0, t0, s0 = play(None)
    n1, t1, s1 = play(1.0)
    assert n0 == n1 > 1000 and s0 == s1
    assert np.array_equal(t0, t1)
    assert set(np.unique(np.abs(t0))) <= {1.0, 2.0, 3.0}


def test_lambda_mixes_values_of_later_decisions():
    n0, mc, _ = play(1.0)
    n5, t5, _ = play(0.5)
    nz, tz, _ = play(0.0)
    assert n0 == n5 == nz
    # the last decision of every player keeps the round outcome, so some targets are unchanged
    same = np.isclose(t5, mc)
    assert 0.01 < same.mean() < 0.5
    # lambda 0: all other targets are exactly the (random) value given at the next decision
    assert np.abs(tz[~np.isclose(tz, mc)]).mean() < np.abs(mc).mean()
    # targets stay between the extremes of outcomes and values
    assert np.all(np.abs(t5) <= max(3.0, np.abs(tz).max()) + 1e-5)
    # lambda 0.5 lies between the Monte-Carlo target and what lambda 0 gives, on average
    assert np.abs(t5 - mc).mean() < np.abs(tz - mc).mean()
