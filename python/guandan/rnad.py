"""R-NaD (Regularized Nash Dynamics, DeepNash) + Factored Public Belief State (ReBeL) for Guandan.

Implements:
1. `RNaDPBSNet`:
   - Private + Seat-Relative Public-Belief Actor-Critic:
     `[state_priv (1550), pbs_rel (432), LSTM(hist)] -> z_priv`
     -> Policy logits `f(z_priv, a)`, team value `V(z_priv)`, and auxiliary hand belief `324`.
   - Common-Knowledge Public Belief Critic (ReBeL / Student-of-Games leaf evaluator):
     `[state_pub (1433), pbs_abs (432)] -> z_pub`
     -> Scalar public value `V_PBS(beta)` + 432-dim zero-mean linear counterfactual hand-value
        gradient `w_cf(beta)` (`v_cf(beta, hands) = V_PBS(beta) + <w_cf(beta), hands108 - pbs_abs>`).
2. `RNaDTrainer`:
   - Alternating-team signed V-trace over KL-regularized zero-sum 2v2 trajectories.
   - Exact analytical-KL + importance-weighted NeurRD (Neural Replicator Dynamics) logit update
     with threshold clipping `[-neurd_clip, +neurd_clip]`.
   - Multi-stage fixed-point regularization anchor iteration (`pi_{reg,0} -> pi_{reg,1} -> ...`)
     with piecewise-linear annealing and EMA target network (`target_model`).
   - Supports both **100% from-scratch training** (`--mode scratch`) and **warm-start from an
     existing DMC checkpoint** (`--mode warmstart --teacher guandan-medium.pt`).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import _core
from .duplicate import duplicate_match, RulePlayer, RandomPlayer, NetPlayer

STATE_DIM, HIST_LEN, HIST_DIM, ACTION_DIM = _core.dims()
HIDDEN_DIM = _core.hidden_dim()
PUB_STATE_DIM, PBS_DIM = _core.pbs_dims()


@dataclass
class RNaDConfig:
    hidden: int = 512
    lstm: int = 256
    hist_len: int = 64
    preset: str = "medium"
    eta: float = 0.08          # R-NaD KL friction temperature
    neurd_clip: float = 2.5    # NeurRD centered-logit clipping threshold

    def to_dict(self):
        return asdict(self)


RNAD_PRESETS = {
    "small": RNaDConfig(hidden=320, lstm=160, hist_len=48, preset="small"),
    "medium": RNaDConfig(hidden=512, lstm=256, hist_len=64, preset="medium"),
    "large": RNaDConfig(hidden=768, lstm=384, hist_len=96, preset="large"),
}


class RNaDPBSNet(nn.Module):
    def __init__(self, cfg: RNaDConfig | None = None):
        super().__init__()
        self.cfg = cfg or RNAD_PRESETS["medium"]
        h = self.cfg.hidden
        l = self.cfg.lstm

        # 1. Private + seat-relative PBS state encoder
        self.lstm = nn.LSTM(HIST_DIM, l, batch_first=True)
        self.priv_mlp = nn.Sequential(
            nn.Linear(STATE_DIM + PBS_DIM + l, h),
            nn.LayerNorm(h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.SiLU(),
        )

        # Fast factored bilinear-residual policy head: scores all legal moves in parallel
        self.pol_z_add = nn.Linear(h, h)
        self.pol_z_mul = nn.Linear(h, h)
        self.pol_act = nn.Sequential(
            nn.Linear(ACTION_DIM, h),
            nn.SiLU(),
            nn.Linear(h, h),
        )
        self.pol_trunk = nn.Sequential(
            nn.LayerNorm(h),
            nn.SiLU(),
            nn.Linear(h, h // 2),
            nn.SiLU(),
        )
        self.pol_out = nn.Linear(h // 2, 1)
        self.q_out = nn.Linear(h // 2, 1)  # Doubly-robust action-value baseline Q_game(z, a)
        # Zero-init final layers so initial policy from scratch is exact uniform and Q=0!
        nn.init.zeros_(self.pol_out.weight)
        nn.init.zeros_(self.pol_out.bias)
        nn.init.zeros_(self.q_out.weight)
        nn.init.zeros_(self.q_out.bias)

        # Acting-team state value head V(z_priv) in units of (levels / 3)
        self.val_head = nn.Sequential(
            nn.Linear(h, h // 2),
            nn.SiLU(),
            nn.Linear(h // 2, 1),
        )
        nn.init.zeros_(self.val_head[-1].weight)
        nn.init.zeros_(self.val_head[-1].bias)

        # Auxiliary private belief head (other 3 seats' hands, 324 bits)
        self.belief_head = nn.Linear(h, HIDDEN_DIM)

        # 2. Common-Knowledge Public Belief Critic (ReBeL leaf evaluator: zero private inputs!)
        self.pub_mlp = nn.Sequential(
            nn.Linear(PUB_STATE_DIM + PBS_DIM, h),
            nn.LayerNorm(h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.SiLU(),
        )
        self.pbs_val = nn.Linear(h, 1)          # Team 0 expected outcome V_PBS(beta)
        self.pbs_w_cf = nn.Linear(h, PBS_DIM)   # Linear counterfactual hand-value weights w_cf(beta)
        nn.init.zeros_(self.pbs_val.weight)
        nn.init.zeros_(self.pbs_val.bias)
        nn.init.zeros_(self.pbs_w_cf.weight)
        nn.init.zeros_(self.pbs_w_cf.bias)

    def encode_priv(self, s_priv: torch.Tensor, pbs_rel: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
        n = self.cfg.hist_len
        hw = hist[:, -n:] if hist.shape[1] > n else hist
        if hw.is_cuda:
            self.lstm.flatten_parameters()
        _, (hn, _) = self.lstm(hw)
        return self.priv_mlp(torch.cat([s_priv, pbs_rel, hn[-1]], dim=-1))

    def policy_and_q(self, z_priv: torch.Tensor, actions: torch.Tensor, seg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ap = self.pol_act(actions)
        za = self.pol_z_add(z_priv).index_select(0, seg)
        zm = torch.tanh(self.pol_z_mul(z_priv)).index_select(0, seg)
        feat = self.pol_trunk(F.silu(za + ap) + zm * ap)
        return self.pol_out(feat).squeeze(-1), self.q_out(feat).squeeze(-1)

    def policy_logits(self, z_priv: torch.Tensor, actions: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        return self.policy_and_q(z_priv, actions, seg)[0]

    def state_value(self, z_priv: torch.Tensor) -> torch.Tensor:
        return self.val_head(z_priv).squeeze(-1)

    def encode_pub(self, s_pub: torch.Tensor, pbs_abs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        zp = self.pub_mlp(torch.cat([s_pub, pbs_abs], dim=-1))
        return self.pbs_val(zp).squeeze(-1), self.pbs_w_cf(zp)

    def pbs_leaf_value(self, s_pub: torch.Tensor, pbs_abs: torch.Tensor, hands108_abs: torch.Tensor | None = None) -> torch.Tensor:
        """Team 0's value (in levels / 3) at public belief state `(s_pub, pbs_abs)`.
        When `hands108_abs` (`[B, 432]`) is given, adds the zero-mean counterfactual private hand shift."""
        v0, w_cf = self.encode_pub(s_pub, pbs_abs)
        if hands108_abs is None:
            return v0
        return v0 + (w_cf * (hands108_abs - pbs_abs)).sum(dim=-1)


def segmented_log_softmax(logits: torch.Tensor, seg: torch.Tensor, n_states: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns `(log_pi, pi, z_centered)` for variable-length action segments `[0..n_states)`."""
    mx = torch.full((n_states,), -1e9, dtype=logits.dtype, device=logits.device)
    mx.scatter_reduce_(0, seg, logits, reduce="amax", include_self=True)
    shifted = logits - mx.index_select(0, seg)
    exp_s = shifted.exp()
    sum_exp = torch.zeros(n_states, dtype=logits.dtype, device=logits.device)
    sum_exp.scatter_add_(0, seg, exp_s)
    log_pi = shifted - sum_exp.clamp_min(1e-20).log().index_select(0, seg)
    pi = log_pi.exp()

    cnt = torch.bincount(seg, minlength=n_states).clamp_min(1).to(logits.dtype)
    sum_l = torch.zeros(n_states, dtype=logits.dtype, device=logits.device)
    sum_l.scatter_add_(0, seg, logits)
    z_centered = logits - (sum_l / cnt).index_select(0, seg)
    return log_pi, pi, z_centered


class RNaDPlayer:
    """Duplicate-evaluation player for `RNaDPBSNet` (computes live public belief from `ro.game(i)`)."""
    def __init__(self, model: RNaDPBSNet, device="cuda", greedy: bool = True, seed: int = 1):
        self.model = model
        self.device = torch.device(device)
        self.greedy = greedy
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def act(self, ro, obs, rows):
        idx, _, s, h, a, off = obs
        rows = np.asarray(rows, np.int64)
        b = len(rows)
        if b == 0:
            return np.zeros(0, np.int64)
        n = off[rows + 1] - off[rows]
        s_sub = s[rows]
        h_sub = h[rows]
        a_sub = np.concatenate([a[off[r]:off[r + 1]] for r in rows])
        off_sub = np.zeros(b + 1, np.int64)
        off_sub[1:] = np.cumsum(n)

        pbs_rel_list = []
        for r in rows:
            g = ro.game(int(idx[r]))
            pb = _core.PubBelief.from_public_history(g)
            pbs_rel_list.append(pb.vec_rel(g.current))
        pbs_rel = np.asarray(pbs_rel_list, np.float32)

        st = torch.from_numpy(np.ascontiguousarray(s_sub)).to(self.device).float()
        pt = torch.from_numpy(pbs_rel).to(self.device)
        ht = torch.from_numpy(np.ascontiguousarray(h_sub)).to(self.device).float()
        at = torch.from_numpy(np.ascontiguousarray(a_sub)).to(self.device).float()
        seg = torch.from_numpy(np.repeat(np.arange(b), n)).to(self.device)

        z = self.model.encode_priv(st, pt, ht)
        logits = self.model.policy_logits(z, at, seg).float().cpu().numpy()

        out = np.zeros(b, np.int64)
        for j in range(b):
            lg = logits[off_sub[j]:off_sub[j + 1]]
            if len(lg) <= 1 or self.greedy:
                out[j] = int(np.argmax(lg))
            else:
                p = np.exp(lg - lg.max())
                p /= p.sum()
                out[j] = int(self.rng.choice(len(lg), p=p))
        return out


def save_rnad_checkpoint(path: str, model: RNaDPBSNet, meta: dict | None = None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"format": "guandan-rnad-pbs/1", "config": model.cfg.to_dict(), "state_dict": sd, "meta": meta or {}}, path)


def load_rnad_checkpoint(path: str, device="cpu") -> tuple[RNaDPBSNet, dict]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    cfg = RNaDConfig(**obj["config"])
    m = RNaDPBSNet(cfg).to(device)
    m.load_state_dict(obj["state_dict"])
    m.eval()
    return m, obj.get("meta", {})


def _unpack_pbs_obs(raw_obs):
    pl_b, sp_b, pub_b, pr_b, pa_b, h_b, a_b, off, h108_b = raw_obs
    players = np.frombuffer(pl_b, np.uint8)
    b = len(players)
    s_priv = np.frombuffer(sp_b, np.uint8).reshape(b, STATE_DIM)
    s_pub = np.frombuffer(pub_b, np.uint8).reshape(b, PUB_STATE_DIM)
    pbs_rel = np.frombuffer(pr_b, np.float32).reshape(b, PBS_DIM)
    pbs_abs = np.frombuffer(pa_b, np.float32).reshape(b, PBS_DIM)
    h = np.frombuffer(h_b, np.uint8).reshape(b, HIST_LEN, HIST_DIM)
    a = np.frombuffer(a_b, np.uint8).reshape(-1, ACTION_DIM)
    off_arr = np.asarray(off, np.int64)
    h108 = np.frombuffer(h108_b, np.uint8).reshape(b, PBS_DIM)
    return players, s_priv, s_pub, pbs_rel, pbs_abs, h, a, off_arr, h108


class RNaDTrainer:
    def __init__(
        self,
        cfg: RNaDConfig,
        device: str = "cuda",
        envs: int = 192,
        unroll: int = 32,
        lr: float = 3e-4,
        eta: float = 0.08,
        vtrace_lambda: float = 0.95,
        stage_updates: int = 250,
        anneal_updates: int = 120,
        ema_decay: float = 0.995,
        aux_weight: float = 0.5,
        pbs_weight: float = 1.0,
        seed: int = 42,
    ):
        self.cfg = cfg
        self.device = torch.device(device)
        self.envs = envs
        self.unroll = unroll
        self.eta = eta
        self.vtrace_lambda = vtrace_lambda
        self.stage_updates = stage_updates
        self.anneal_updates = anneal_updates
        self.ema_decay = ema_decay
        self.aux_weight = aux_weight
        self.pbs_weight = pbs_weight
        self.rng = np.random.default_rng(seed)

        self.model = RNaDPBSNet(cfg).to(self.device)
        self.target_model = copy.deepcopy(self.model).to(self.device)
        self.target_model.requires_grad_(False)
        self.target_model.eval()

        # Regularization anchors: None = exact uniform policy pi_0(a|x) = 1 / |A(x)|
        self.reg_prev: RNaDPBSNet | None = None
        self.reg_curr: RNaDPBSNet | None = None
        self.stage = 0
        self.stage_step = 0
        self.total_updates = 0
        self.total_frames = 0

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.vec_env = _core.PbsVecEnv(envs, seed, True, 3, True)

        # Rolling buffer of unfinished round steps per env for exact Monte-Carlo PBS return targets!
        self.ep_pub_s = [[] for _ in range(envs)]
        self.ep_pbs_a = [[] for _ in range(envs)]
        self.ep_h108 = [[] for _ in range(envs)]
        self.pbs_replay_s = []
        self.pbs_replay_b = []
        self.pbs_replay_h = []
        self.pbs_replay_r = []

    @torch.no_grad()
    def warmstart_from_dmc(self, dmc_path: str, bc_steps: int = 200, temp_levels: float = 0.25):
        """Distill initial `(policy_logits, state_value, pbs_critic)` from an existing DMC checkpoint,
        then set `reg_curr = copy(target_model)` so R-NaD starts anchored on the strong policy!"""
        from .model import load_checkpoint
        teacher, _ = load_checkpoint(dmc_path, self.device)
        teacher.eval()
        print(f"[R-NaD Warmstart] Distilling {bc_steps} batches from {dmc_path} (temp={temp_levels})...", flush=True)
        torch.set_grad_enabled(True)
        bc_opt = torch.optim.AdamW(self.model.parameters(), lr=5e-4, weight_decay=1e-4)
        inv_t = 3.0 / temp_levels

        for step in range(1, bc_steps + 1):
            players, s_priv, s_pub, pbs_rel, pbs_abs, h, a, off, h108 = _unpack_pbs_obs(self.vec_env.observe())
            b = len(players)
            n_mv = np.diff(off)
            seg_np = np.repeat(np.arange(b), n_mv)

            st = torch.from_numpy(s_priv.copy()).to(self.device).float()
            pt = torch.from_numpy(pbs_rel.copy()).to(self.device)
            ht = torch.from_numpy(h.copy()).to(self.device).float()
            at = torch.from_numpy(a.copy()).to(self.device).float()
            seg = torch.from_numpy(seg_np).to(self.device)

            with torch.no_grad():
                zt = teacher.encode_state(st, ht)
                qt = teacher.score(zt.index_select(0, seg), at)  # in levels / 3
                log_pi_t, pi_t, _ = segmented_log_softmax(qt * inv_t, seg, b)
                v_t = torch.zeros(b, device=self.device).scatter_add_(0, seg, pi_t * qt)
                # Hidden hand target for belief head: seats me+1, me+2, me+3 from h108
                hid_t = self._extract_rel_hidden(h108, players)

            z = self.model.encode_priv(st, pt, ht)
            logits_s, q_s = self.model.policy_and_q(z, at, seg)
            log_pi_s, _, _ = segmented_log_softmax(logits_s, seg, b)
            v_s = self.model.state_value(z)
            bel_s = self.model.belief_head(z)

            # KL(teacher || student) + action-Q MSE + value MSE + belief BCE
            kl_bc = (pi_t * (log_pi_t - log_pi_s)).sum() / max(1, int((n_mv > 1).sum()))
            q_loss = F.mse_loss(q_s, qt)
            v_loss = F.mse_loss(v_s, v_t)
            b_loss = F.binary_cross_entropy_with_logits(bel_s, hid_t)
            loss = kl_bc + q_loss + 0.5 * v_loss + 0.5 * b_loss

            bc_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
            bc_opt.step()

            # Step env following teacher
            ch = []
            pi_np = pi_t.cpu().numpy()
            for i in range(b):
                p_slice = pi_np[off[i]:off[i + 1]]
                ch.append(int(np.argmax(p_slice)) if self.rng.random() < 0.85 else int(self.rng.choice(len(p_slice), p=p_slice / p_slice.sum())))
            dones_b, rews_b = self.vec_env.step(ch)
            self._record_pbs_steps(s_pub, pbs_abs, h108, np.frombuffer(dones_b, np.uint8), np.frombuffer(rews_b, np.float32))
            if len(self.pbs_replay_r) >= 1024:
                self._train_pbs_critic_step(bc_opt)

        self.target_model.load_state_dict(self.model.state_dict())
        self.reg_curr = copy.deepcopy(self.target_model)
        self.reg_curr.requires_grad_(False)
        self.reg_curr.eval()
        self.stage = 1
        self.stage_step = self.anneal_updates  # fully anchored on distilled policy at start!
        print("[R-NaD Warmstart] Completed! Anchor pi_{reg,1} locked.", flush=True)

    def _extract_rel_hidden(self, h108: np.ndarray, players: np.ndarray) -> torch.Tensor:
        b = len(players)
        h4 = h108.reshape(b, 4, 108)
        out = np.zeros((b, 3, 108), np.float32)
        for r in range(3):
            seat_idx = (players + r + 1) & 3
            out[:, r, :] = h4[np.arange(b), seat_idx, :]
        return torch.from_numpy(out.reshape(b, 324)).to(self.device)

    def _record_pbs_steps(self, s_pub, pbs_abs, h108, dones, rews_t0):
        for i in range(len(dones)):
            if self.rng.random() < 0.25:  # subsample 25% of steps to decorrelate PBS buffer
                self.ep_pub_s[i].append(s_pub[i].copy())
                self.ep_pbs_a[i].append(pbs_abs[i].copy())
                self.ep_h108[i].append(h108[i].copy())
            if dones[i] != 0:
                r0 = float(rews_t0[i]) / 3.0
                for sp, pa, hh in zip(self.ep_pub_s[i], self.ep_pbs_a[i], self.ep_h108[i]):
                    self.pbs_replay_s.append(sp)
                    self.pbs_replay_b.append(pa)
                    self.pbs_replay_h.append(hh)
                    self.pbs_replay_r.append(r0)
                self.ep_pub_s[i].clear()
                self.ep_pbs_a[i].clear()
                self.ep_h108[i].clear()
        if len(self.pbs_replay_r) > 24000:
            cut = len(self.pbs_replay_r) - 24000
            del self.pbs_replay_s[:cut]
            del self.pbs_replay_b[:cut]
            del self.pbs_replay_h[:cut]
            del self.pbs_replay_r[:cut]

    def _train_pbs_critic_step(self, opt):
        n_buf = len(self.pbs_replay_r)
        if n_buf < 512:
            return 0.0
        idx = self.rng.integers(0, n_buf, size=512)
        sp = torch.from_numpy(np.stack([self.pbs_replay_s[i] for i in idx])).to(self.device).float()
        pa = torch.from_numpy(np.stack([self.pbs_replay_b[i] for i in idx])).to(self.device)
        hh = torch.from_numpy(np.stack([self.pbs_replay_h[i] for i in idx])).to(self.device).float()
        rt = torch.tensor([self.pbs_replay_r[i] for i in idx], dtype=torch.float32, device=self.device)

        v0, w_cf = self.model.encode_pub(sp, pa)
        v_cf = v0 + (w_cf * (hh - pa)).sum(dim=-1)
        loss_pbs = 0.5 * F.mse_loss(v0, rt) + 0.5 * F.mse_loss(v_cf, rt) + 1e-4 * w_cf.square().mean()
        opt.zero_grad()
        (self.pbs_weight * loss_pbs).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
        opt.step()
        return float(loss_pbs.detach())

    @torch.no_grad()
    def _reg_log_probs(self, model_reg: RNaDPBSNet | None, st, pt, ht, at, seg, n_states: int, counts_t: torch.Tensor) -> torch.Tensor:
        if model_reg is None:
            return -counts_t.log().index_select(0, seg)
        zr = model_reg.encode_priv(st, pt, ht)
        lgr = model_reg.policy_logits(zr, at, seg)
        log_pi_r, _, _ = segmented_log_softmax(lgr, seg, n_states)
        return log_pi_r

    def step_update(self) -> dict:
        """Collect `T = self.unroll` steps across `B = self.envs` games and execute one exact
        R-NaD (V-trace + NeurRD + Auxiliary Belief + Public Belief Critic) update!"""
        T = self.unroll
        B = self.envs

        traj_players = []
        traj_s_priv = []
        traj_pbs_rel = []
        traj_h = []
        traj_a = []
        traj_off = []
        traj_chosen = []
        traj_mu = []
        traj_dones = []
        traj_rews_team = []
        traj_h108 = []

        self.model.eval()
        with torch.no_grad():
            for _ in range(T):
                players, s_priv, s_pub, pbs_rel, pbs_abs, h, a, off, h108 = _unpack_pbs_obs(self.vec_env.observe())
                n_mv = np.diff(off)
                seg_np = np.repeat(np.arange(B), n_mv)

                st = torch.from_numpy(s_priv.copy()).to(self.device).float()
                pt = torch.from_numpy(pbs_rel.copy()).to(self.device)
                hl = self.cfg.hist_len
                h_win = np.ascontiguousarray(h[:, -hl:])
                ht = torch.from_numpy(h_win).to(self.device).float()
                at = torch.from_numpy(a.copy()).to(self.device).float()
                seg = torch.from_numpy(seg_np).to(self.device)

                z = self.model.encode_priv(st, pt, ht)
                logits = self.model.policy_logits(z, at, seg)
                _, pi_t, _ = segmented_log_softmax(logits, seg, B)
                pi_np = pi_t.float().cpu().numpy()

                chosen = np.zeros(B, np.int64)
                mu = np.ones(B, np.float32)
                for i in range(B):
                    k = int(n_mv[i])
                    if k <= 1:
                        chosen[i] = 0
                        mu[i] = 1.0
                    else:
                        p_s = pi_np[off[i]:off[i + 1]].astype(np.float64)
                        # Mix 2% uniform exploration so behavior policy mu >= 0.02 / k everywhere!
                        p_s = 0.98 * (p_s / p_s.sum()) + 0.02 / k
                        c = int(self.rng.choice(k, p=p_s))
                        chosen[i] = c
                        mu[i] = float(p_s[c])

                dones_b, rews_b = self.vec_env.step(chosen.tolist())
                dones = np.frombuffer(dones_b, np.uint8).copy()
                rews_t0 = np.frombuffer(rews_b, np.float32).copy()
                self._record_pbs_steps(s_pub, pbs_abs, h108, dones, rews_t0)

                # Convert terminal round reward (in levels / 3) to acting player's team sign!
                teams = (players % 2).astype(np.int64)
                rews_acting = np.where(teams == 0, rews_t0, -rews_t0) / 3.0

                traj_players.append(players.copy())
                traj_s_priv.append(s_priv.copy())
                traj_pbs_rel.append(pbs_rel.copy())
                traj_h.append(h_win)
                traj_a.append(a.copy())
                traj_off.append(off.copy())
                traj_chosen.append(chosen)
                traj_mu.append(mu)
                traj_dones.append(dones)
                traj_rews_team.append(rews_acting.astype(np.float32))
                traj_h108.append(h108.copy())

            # Bootstrap value V(x_T) at step T
            pl_T, sp_T, _, pr_T, _, h_T, _, _, _ = _unpack_pbs_obs(self.vec_env.observe())
            z_boot = self.target_model.encode_priv(
                torch.from_numpy(sp_T.copy()).to(self.device).float(),
                torch.from_numpy(pr_T.copy()).to(self.device),
                torch.from_numpy(h_T.copy()).to(self.device).float(),
            )
            v_boot = self.target_model.state_value(z_boot)  # [B]
            teams_boot = torch.from_numpy((pl_T % 2).astype(np.int64)).to(self.device)

        # Pack all S = T * B states into one big GPU forward pass!
        S = T * B
        all_players = np.concatenate(traj_players)
        all_s = torch.from_numpy(np.concatenate(traj_s_priv)).to(self.device).float()
        all_p = torch.from_numpy(np.concatenate(traj_pbs_rel)).to(self.device)
        all_h = torch.from_numpy(np.concatenate(traj_h)).to(self.device).float()
        all_a = torch.from_numpy(np.concatenate(traj_a)).to(self.device).float()
        all_h108 = np.concatenate(traj_h108)
        hid_targets = self._extract_rel_hidden(all_h108, all_players)

        counts_list = [np.diff(o) for o in traj_off]
        counts_flat = np.concatenate(counts_list)  # [S]
        seg_flat = torch.from_numpy(np.repeat(np.arange(S), counts_flat)).to(self.device)
        counts_t = torch.from_numpy(counts_flat).to(self.device).clamp_min(1).float()

        # Global action index of chosen action at each state 0..S-1
        state_starts = np.zeros(S, np.int64)
        state_starts[1:] = np.cumsum(counts_flat[:-1])
        chosen_flat = np.concatenate(traj_chosen)
        chosen_global = torch.from_numpy(state_starts + chosen_flat).to(self.device)
        mu_flat = torch.from_numpy(np.concatenate(traj_mu)).to(self.device)

        # Compute annealed log pi_reg(b | x)
        with torch.no_grad():
            alpha = min(1.0, self.stage_step / max(1, self.anneal_updates))
            lp_curr = self._reg_log_probs(self.reg_curr, all_s, all_p, all_h, all_a, seg_flat, S, counts_t)
            if alpha >= 1.0 or (self.reg_prev is None and self.stage == 0):
                log_pi_reg = lp_curr
            else:
                lp_prev = self._reg_log_probs(self.reg_prev, all_s, all_p, all_h, all_a, seg_flat, S, counts_t)
                # Interpolate in probability space as in DeepNash R-NaD
                pi_mix = (1.0 - alpha) * lp_prev.exp() + alpha * lp_curr.exp()
                log_pi_reg = pi_mix.clamp_min(1e-12).log()

        torch.set_grad_enabled(True)
        self.model.train()
        z_all = self.model.encode_priv(all_s, all_p, all_h)
        logits_all, q_pred_all = self.model.policy_and_q(z_all, all_a, seg_flat)
        log_pi, pi, z_centered = segmented_log_softmax(logits_all, seg_flat, S)
        v_pred = self.model.state_value(z_all)  # [S]
        bel_logits = self.model.belief_head(z_all)

        # Compute V-trace targets backward in time t = T-1 .. 0 (no grad through targets!)
        with torch.no_grad():
            log_ratio_reg = (log_pi - log_pi_reg)  # log(pi / pi_reg) for all actions
            multi_state = (counts_t > 1.5).float()
            # Chosen action log(pi(a_t) / pi_reg(a_t))
            chosen_kl_term = log_ratio_reg.index_select(0, chosen_global) * multi_state  # [S]
            pi_chosen = pi.index_select(0, chosen_global)
            is_ratio = (pi_chosen / mu_flat.clamp_min(1e-6))

            v_mat = v_pred.detach().view(T, B)
            kl_mat = chosen_kl_term.view(T, B)
            rho_mat = is_ratio.clamp_max(1.0).view(T, B)
            c_mat = is_ratio.clamp_max(1.0).view(T, B)
            dones_mat = torch.from_numpy(np.stack(traj_dones)).to(self.device).float()
            rews_env_mat = torch.from_numpy(np.stack(traj_rews_team)).to(self.device)
            teams_mat = torch.from_numpy(np.stack([p % 2 for p in traj_players]).astype(np.int64)).to(self.device)

            v_vs = torch.zeros((T, B), dtype=torch.float32, device=self.device)
            g_env_mat = torch.zeros((T, B), dtype=torch.float32, device=self.device)

            next_v_pred = v_boot
            next_v_vs = v_boot
            next_team = teams_boot

            for t in range(T - 1, -1, -1):
                not_done = 1.0 - dones_mat[t]
                sigma = torch.where(teams_mat[t] == next_team, 1.0, -1.0)
                r_reg = rews_env_mat[t] - self.eta * kl_mat[t]
                delta = rho_mat[t] * (r_reg + not_done * sigma * next_v_pred - v_mat[t])
                v_t = v_mat[t] + delta + not_done * self.vtrace_lambda * c_mat[t] * sigma * (next_v_vs - next_v_pred)
                v_vs[t] = v_t
                g_env_mat[t] = rews_env_mat[t] + not_done * sigma * next_v_vs

                next_v_pred = v_mat[t]
                next_v_vs = v_t
                next_team = teams_mat[t]

            v_targ_flat = v_vs.reshape(S)
            g_env_flat = g_env_mat.reshape(S)

            # Doubly-Robust R-NaD action-value estimator:
            # Q_DR(x, b) = Q_hat(x, b) + 1[b == a_t] / max(mu(a_t), 0.05) * (G_env - Q_hat(x, a_t)) - eta * log(pi / pi_reg)
            q_base = q_pred_all.detach()
            dr_res = (g_env_flat - q_base.index_select(0, chosen_global)) / mu_flat.clamp_min(0.05)
            q_total = q_base - self.eta * log_ratio_reg
            q_total.scatter_add_(0, chosen_global, dr_res)

            # Subtract expected value sum_b pi(b) * q_total(b) per state
            ev_q = torch.zeros(S, dtype=torch.float32, device=self.device)
            ev_q.scatter_add_(0, seg_flat, pi * q_total)
            adv_reg = q_total - ev_q.index_select(0, seg_flat)

            # NeurRD threshold clipping on centered logits z_centered!
            beta = self.cfg.neurd_clip
            can_push = ~(((adv_reg > 0) & (z_centered >= beta)) | ((adv_reg < 0) & (z_centered <= -beta)))
            multi_act = (counts_t > 1.5).index_select(0, seg_flat)
            neurd_grad = torch.where(can_push & multi_act, adv_reg, torch.zeros_like(adv_reg)).clamp(-5.0, 5.0)

            mean_kl = float(
                (torch.zeros(S, device=self.device).scatter_add_(0, seg_flat, pi * log_ratio_reg) * multi_state).sum()
                / multi_state.sum().clamp_min(1.0)
            )
            mean_ent = float(
                (torch.zeros(S, device=self.device).scatter_add_(0, seg_flat, -pi * log_pi) * multi_state).sum()
                / multi_state.sum().clamp_min(1.0)
            )

        n_multi_states = (counts_t > 1.5).sum().clamp_min(1).float()
        loss_neurd = -(neurd_grad * z_centered).sum() / n_multi_states
        loss_q = 0.5 * F.mse_loss(q_pred_all.index_select(0, chosen_global), g_env_flat)
        loss_val = 0.5 * F.mse_loss(v_pred, v_targ_flat)
        loss_bel = F.binary_cross_entropy_with_logits(bel_logits, hid_targets)
        total_loss = loss_neurd + loss_q + loss_val + self.aux_weight * loss_bel

        self.opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
        self.opt.step()

        # Also update the ReBeL Public Belief Value + Counterfactual Critic!
        loss_pbs = self._train_pbs_critic_step(self.opt)

        # Update EMA target network & R-NaD stage anchor schedule
        with torch.no_grad():
            for pt_p, p in zip(self.target_model.parameters(), self.model.parameters()):
                pt_p.mul_(self.ema_decay).add_(p, alpha=1.0 - self.ema_decay)

        self.total_updates += 1
        self.stage_step += 1
        self.total_frames += S

        if self.stage_step >= self.stage_updates:
            self.reg_prev = self.reg_curr
            self.reg_curr = copy.deepcopy(self.target_model)
            self.reg_curr.requires_grad_(False)
            self.reg_curr.eval()
            self.stage += 1
            self.stage_step = 0

        return {
            "updates": self.total_updates,
            "frames": self.total_frames,
            "stage": self.stage,
            "loss_neurd": float(loss_neurd.detach()),
            "loss_val": float(loss_val.detach()),
            "loss_bel": float(loss_bel.detach()),
            "loss_pbs": loss_pbs,
            "kl_reg": mean_kl,
            "entropy": mean_ent,
        }


def main():
    ap = argparse.ArgumentParser(description="Train Guandan AI with R-NaD + Factored Public Belief State")
    ap.add_argument("--mode", choices=["scratch", "warmstart"], default="scratch")
    ap.add_argument("--preset", choices=list(RNAD_PRESETS), default="small")
    ap.add_argument("--teacher", default="", help="DMC checkpoint for --mode warmstart")
    ap.add_argument("--out", required=True, help="Output directory for checkpoints & metrics")
    ap.add_argument("--updates", type=int, default=600)
    ap.add_argument("--envs", type=int, default=192)
    ap.add_argument("--unroll", type=int, default=32)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--eta", type=float, default=0.08)
    ap.add_argument("--stage-updates", type=int, default=200)
    ap.add_argument("--anneal-updates", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=150)
    ap.add_argument("--eval-deals", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = copy.deepcopy(RNAD_PRESETS[args.preset])
    cfg.eta = args.eta
    trainer = RNaDTrainer(
        cfg,
        device=args.device,
        envs=args.envs,
        unroll=args.unroll,
        lr=args.lr,
        eta=args.eta,
        stage_updates=args.stage_updates,
        anneal_updates=args.anneal_updates,
        seed=args.seed,
    )
    if args.mode == "warmstart":
        if not args.teacher:
            raise ValueError("--mode warmstart requires --teacher <path.pt>")
        trainer.warmstart_from_dmc(args.teacher, bc_steps=200)

    t0 = time.time()
    metrics_path = os.path.join(args.out, "rnad_metrics.jsonl")
    for u in range(1, args.updates + 1):
        info = trainer.step_update()
        if u % 25 == 0 or u == 1:
            elapsed = time.time() - t0
            fps = info["frames"] / max(1e-3, elapsed)
            print(f"[R-NaD u={u:4d}/{args.updates} | stage={info['stage']} | {info['frames']/1e6:.2f}M frames | {fps:,.0f} fps] "
                  f"val={info['loss_val']:.4f} bel={info['loss_bel']:.4f} pbs={info['loss_pbs']:.4f} "
                  f"KL={info['kl_reg']:.4f} H={info['entropy']:.3f}", flush=True)
        if u % args.eval_every == 0 or u == args.updates:
            torch.set_grad_enabled(False)
            p_ema = RNaDPlayer(trainer.target_model, args.device, greedy=True)
            r_rand = duplicate_match(p_ema, RandomPlayer(7), deals=min(100, args.eval_deals), seed=7001, batch=64)
            r_rule = duplicate_match(p_ema, RulePlayer(), deals=args.eval_deals, seed=7002, batch=64)
            row = {
                **info,
                "seconds": time.time() - t0,
                "vs_random": r_rand["score"],
                "vs_random_win": r_rand["win_rate"],
                "vs_rule": r_rule["score"],
                "vs_rule_se": r_rule["score_se"],
                "vs_rule_win": r_rule["win_rate"],
            }
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(f"  >>> EVAL (EMA target): vs Random {r_rand['score']:+.3f} ({r_rand['win_rate']*100:.1f}%) | "
                  f"vs Rule {r_rule['score']:+.3f} ± {r_rule['score_se']:.3f} ({r_rule['win_rate']*100:.1f}%)",
                  flush=True)
            save_rnad_checkpoint(os.path.join(args.out, "rnad_latest.pt"), trainer.target_model, row)


if __name__ == "__main__":
    main()
