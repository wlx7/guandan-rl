import random

import numpy as np

from guandan import _core

S, T, H, A = _core.dims()


def c(r, s):
    return r * 4 + s


def kinds(cards, level):
    return {m[0] for m in _core.classify_cards(cards, level)}


def test_classify_examples():
    lv = 3  # playing 5, wild = 5 of hearts
    w = c(lv, 1)
    assert kinds([c(7, 0), c(7, 2)], lv) == {2}                     # pair
    assert 5 in kinds([c(1, 0), c(2, 2), w, c(4, 0), c(5, 2)], lv)  # straight 3-7 with wild
    assert 9 in kinds([c(1, 1), c(2, 1), w, c(4, 1), c(5, 1)], lv)  # straight flush
    assert kinds([c(5, 0), c(5, 2), c(5, 3), w, w], lv) >= {4, 8}  # full house or 5-bomb
    assert kinds([52, 52, 53, 53], lv) == {10}                      # joker bomb
    assert kinds([w, 52], lv) == set()                              # wild can't be a joker
    assert kinds([c(9, 0), c(10, 1), c(11, 2), c(12, 3), c(0, 0)], lv) == set()  # JQKA2 invalid


def play_random_match(seed, max_rounds=40):
    g = _core.Game(seed)
    rng = random.Random(seed)
    rounds = 0
    saw_tribute = False
    while g.phase != 3 and rounds < max_rounds:
        if g.phase == 2:
            rounds += 1
            g.next_round()
            if g.tribute():
                saw_tribute = True
            continue
        moves = g.legal_moves()
        assert moves
        k, key, cards = rng.choice(moves)
        g.apply(k, key, list(cards))
        assert sum(g.hand_counts()) + sum(len(g.played(p)) for p in range(4)) == 108
    return g, rounds, saw_tribute


def test_random_matches():
    tribute_seen = False
    for seed in range(10):
        g, rounds, t = play_random_match(seed)
        tribute_seen |= t
        assert rounds > 0
    assert tribute_seen


def test_tribute_returns_restore_27_cards():
    for seed in range(30):
        g = _core.Game(seed)
        rng = random.Random(seed)
        while g.phase != 2:
            k, key, cards = rng.choice(g.legal_moves())
            g.apply(k, key, list(cards))
        g.next_round()
        if g.anti_tribute or not g.tribute():
            continue
        while g.phase == 0:
            k, key, cards = g.legal_moves()[0]
            assert k == 11
            g.apply(k, key, list(cards))
        assert g.hand_counts() == [27, 27, 27, 27]
        assert g.phase == 1
        return
    raise AssertionError("no tribute round found")


def test_illegal_moves_rejected():
    g = _core.Game(5)
    if g.phase == 1 and g.is_leading:
        try:
            g.apply(0, 0, [])
            raise AssertionError("leader passed")
        except ValueError:
            pass
    hand = g.hand(g.current)
    other = [t for t in range(54) if t not in hand]
    if other:
        try:
            g.apply(1, 0, [other[0]])
            raise AssertionError("played card not in hand")
        except ValueError:
            pass


def test_vecenv_shapes_and_samples():
    env = _core.VecEnv(8, 1)
    rng = np.random.default_rng(0)
    for _ in range(400):
        players, s, h, a, off = env.observe()
        assert len(s) == 8 * S and len(h) == 8 * T * H
        assert len(a) == off[-1] * A
        n = np.diff(off)
        ch = (rng.random(8) * n).astype(int).tolist()
        env.step(ch, True)
    n, s, h, a, t = env.take_samples()
    assert n > 0
    assert len(s) == n * S and len(a) == n * A
    tv = np.frombuffer(t, np.float32)
    assert set(np.abs(tv).tolist()) <= {1.0, 2.0, 3.0}


def test_rule_bot_beats_random():
    from guandan.evaluate import evaluate
    r = evaluate("rule", "random", rounds=200, envs=32)
    assert r["win_rate"] > 0.8
