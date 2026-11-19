"""Tests for R-NaD (Regularized Nash Dynamics) and Factored Public Belief Search (PBS)."""
import numpy as np
import pytest

from guandan import _core


def test_pub_belief_and_pbs_vecenv():
    g = _core.Game(123)
    pb = _core.PubBelief(g)
    va = np.asarray(pb.vec_abs(), np.float32).reshape(4, 54, 2)
    vr = np.asarray(pb.vec_rel(2), np.float32).reshape(4, 54, 2)
    assert va.shape == (4, 54, 2)
    assert np.allclose(vr[0], va[2], atol=1e-6)
    w = pb.sample_world(g, 999, g.current)
    assert list(w.hand(g.current)) == list(g.hand(g.current))
    assert w.hand_counts() == g.hand_counts()

    env = _core.PbsVecEnv(4, 77)
    obs = env.observe()
    assert len(obs[0]) == 4
    dones, rews = env.step([0, 0, 0, 0])
    assert len(dones) == 4 and len(np.frombuffer(rews, np.float32)) == 4


def test_rnad_step_and_pbs_search():
    torch = pytest.importorskip("torch")
    from guandan.rnad import RNaDConfig, RNaDTrainer, segmented_log_softmax
    from guandan.pbs_search import PublicBeliefSearch

    cfg = RNaDConfig(hidden=128, lstm=64, hist_len=24, preset="tiny_test")
    trainer = RNaDTrainer(cfg, device="cpu", envs=8, unroll=8, stage_updates=2, anneal_updates=2, seed=11)

    # Verify zero-init gives exact uniform initial policy
    logits = torch.zeros(7)
    seg = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    _, pi, _ = segmented_log_softmax(logits, seg, 2)
    assert np.allclose(pi[:3].numpy(), 1.0 / 3.0, atol=1e-6)
    assert np.allclose(pi[3:].numpy(), 1.0 / 4.0, atol=1e-6)

    # Run 3 R-NaD updates so stage advances 0 -> 1 and exercises anchor annealing
    for _ in range(3):
        info = trainer.step_update()
    assert trainer.stage >= 1
    assert np.isfinite(info["loss_val"]) and np.isfinite(info["loss_neurd"])

    # Verify PublicBeliefSearch returns valid legal move indices
    g1 = _core.Game(5)
    g2 = _core.Game(6)
    pbs_s = PublicBeliefSearch(trainer.target_model, device="cpu", top_k=3, worlds=4, depth=4, cfr_iters=8, solver_cards=10)
    picks = pbs_s.choose_many([g1, g2])
    assert len(picks) == 2
    assert 0 <= picks[0] < len(g1.legal_moves())
    assert 0 <= picks[1] < len(g2.legal_moves())
