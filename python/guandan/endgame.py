"""Endgame search ("残局搜索"): perfect-information Monte-Carlo with an exact solver.

When few cards are left, the hidden hands are sampled `worlds` times from the calibrated
belief posterior (`Game.determinize_belief`: full 324 belief logits, Sinkhorn-Knopp scaling
to exact hand sizes and unseen counts, exact 2-copy Bayes correlation, and hard tribute
deductions), and top candidate moves within `q_gap` levels of the network's choice are solved
exactly to the end of the round by alpha-beta (`Game.endgame_choose`), blended with weight
`prior` on the playing network's Q value to regularize imperfect-information risk and break
ties cleanly.
"""
from __future__ import annotations

import itertools
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from .agents import build_fastnet
from .model import STATE_DIM, HIST_LEN, HIST_DIM, ACTION_DIM

_seed = itertools.count(1)


class EndgameSearch:
    def __init__(self, model, device="cpu", cards=26, worlds=24, guided=0.02, max_nodes=250_000,
                 threads=8, margin=0.0, prior=0.25, max_cands=12, q_gap=1.2,
                 bomb_gamma=0.25, smc_oversample=2, smc_last_k=6,
                 mid_rollout=False, mid_worlds=16, mid_top_k=3, mid_close_gap=0.08,
                 mid_prior=0.35, mid_margin_se=0.75, solver_leaf_cards=18,
                 search_return=True, return_gap=0.20):
        self.model = model
        self.device = torch.device(device)
        self.cards = cards
        self.worlds = worlds
        self.guided = guided if (guided is not None and getattr(model.cfg, "belief", False)) else None
        self.max_nodes = max_nodes
        self.margin = margin
        self.prior = prior
        self.max_cands = max_cands
        self.q_gap = q_gap
        self.bomb_gamma = bomb_gamma
        self.smc_oversample = smc_oversample
        self.smc_last_k = smc_last_k
        self.mid_rollout = mid_rollout
        self.mid_worlds = mid_worlds
        self.mid_top_k = mid_top_k
        self.mid_close_gap = mid_close_gap
        self.mid_prior = mid_prior
        self.mid_margin_se = mid_margin_se
        self.solver_leaf_cards = solver_leaf_cards
        self.search_return = search_return
        self.return_gap = return_gap
        self.threads = threads
        self.fastnet = build_fastnet(model)
        self.pool = ThreadPoolExecutor(max_workers=threads) if threads > 1 else None
        self.stats = {"decisions": 0, "searched": 0, "changed": 0, "mid_searched": 0, "mid_changed": 0, "aborted": 0}

    @torch.no_grad()
    def _forward_games(self, games):
        encs = [g.encode() for g in games]
        s = np.stack([np.frombuffer(e[0], np.uint8) for e in encs]).reshape(-1, STATE_DIM)
        h = np.stack([np.frombuffer(e[1], np.uint8) for e in encs]).reshape(-1, HIST_LEN, HIST_DIM)
        acts = [np.frombuffer(e[2], np.uint8).reshape(e[3], ACTION_DIM) for e in encs]
        counts = np.array([e[3] for e in encs], np.int64)
        a_cat = np.concatenate(acts) if len(acts) else np.zeros((0, ACTION_DIM), np.uint8)
        st = torch.from_numpy(s).to(self.device).float()
        ht = torch.from_numpy(h).to(self.device).float()
        at = torch.from_numpy(a_cat).to(self.device).float()
        seg = torch.from_numpy(np.repeat(np.arange(len(games)), counts)).to(self.device)
        z = self.model.encode_state(st, ht)
        q = self.model.score(z.index_select(0, seg), at).float().cpu().numpy() * 3.0
        lg = self.model.belief_logits(z).float().cpu().numpy() if self.guided is not None else None
        off = np.zeros(len(games) + 1, np.int64)
        off[1:] = np.cumsum(counts)
        q_list = [q[off[i]:off[i + 1]].tolist() for i in range(len(games))]
        lg_list = [lg[i].tolist() for i in range(len(games))] if lg is not None else [None] * len(games)
        return q_list, lg_list

    def wants(self, game) -> bool:
        if len(game.legal_moves()) <= 1:
            return False
        ph = game.phase
        cl = game.cards_left()
        if ph == 1 and cl <= self.cards:
            return True
        if self.mid_rollout and self.fastnet is not None:
            if ph == 0 and self.search_return:
                return True
            if ph == 1 and self.cards < cl <= 56:
                return True
        return False

    def choose_many(self, games, net_choices, q_levels_list=None, belief_logits_list=None):
        """Moves (indices into game.legal_moves()) for games that `wants`; net_choices are the
        network's own picks, kept unless the search finds a better move. Uses the Rust
        FastNet + Game.endgame_choose, i.e. exactly what the battle servers play."""
        out = list(net_choices)
        todo = [i for i, g in enumerate(games) if self.wants(g)]
        self.stats["decisions"] += len(games)
        if not todo:
            return out
        if q_levels_list is None or (self.guided is not None and belief_logits_list is None):
            ql_sub, lg_sub = self._forward_games([games[i] for i in todo])
        else:
            ql_sub = [q_levels_list[i] for i in todo]
            lg_sub = [belief_logits_list[i] if (self.guided is not None and belief_logits_list is not None) else None for i in todo]
        for j, i in enumerate(todo):
            g = games[i]
            ph = g.phase
            cl = g.cards_left()
            if ph == 1 and cl <= self.cards:
                if self.fastnet is not None and self.smc_oversample > 1:
                    r = self.fastnet.endgame_choose_smc(
                        g,
                        int(net_choices[i]),
                        ql_sub[j],
                        lg_sub[j],
                        cards=self.cards,
                        worlds=self.worlds,
                        floor=self.guided or 0.02,
                        bomb_gamma=self.bomb_gamma,
                        max_nodes=self.max_nodes,
                        margin=self.margin,
                        prior=self.prior,
                        max_cands=self.max_cands,
                        q_gap=self.q_gap,
                        smc_oversample=self.smc_oversample,
                        smc_last_k=self.smc_last_k,
                        threads=self.threads,
                        seed=next(_seed) * 7919,
                    )
                else:
                    r = g.endgame_choose(
                        int(net_choices[i]),
                        None,
                        cards=self.cards,
                        worlds=self.worlds,
                        floor=self.guided or 0.02,
                        max_nodes=self.max_nodes,
                        margin=self.margin,
                        threads=self.threads,
                        seed=next(_seed) * 7919,
                        belief_logits=lg_sub[j],
                        q_levels=ql_sub[j],
                        prior=self.prior,
                        max_cands=self.max_cands,
                        q_gap=self.q_gap,
                        bomb_gamma=self.bomb_gamma,
                    )
                self.stats["searched"] += 1
                if r is not None:
                    out[i] = r
                    self.stats["changed"] += 1
            elif self.mid_rollout and self.fastnet is not None and ql_sub[j] is not None:
                qlj = ql_sub[j]
                top_q = qlj[int(net_choices[i])]
                sec_q = max(v for k_idx, v in enumerate(qlj) if k_idx != int(net_choices[i]))
                gap_lim = self.return_gap if ph == 0 else self.mid_close_gap
                if top_q - sec_q > gap_lim:
                    continue
                r = self.fastnet.rollout_choose(
                    g,
                    int(net_choices[i]),
                    ql_sub[j],
                    lg_sub[j],
                    min_cards=self.cards + 1,
                    max_cards=56,
                    worlds=self.mid_worlds,
                    top_k=self.mid_top_k,
                    close_gap=self.mid_close_gap,
                    max_cand_gap=0.45,
                    prior=self.mid_prior,
                    margin_se=self.mid_margin_se,
                    floor=self.guided or 0.02,
                    bomb_gamma=self.bomb_gamma,
                    solver_leaf_cards=self.solver_leaf_cards,
                    search_return=self.search_return,
                    return_gap=self.return_gap,
                    return_top_k=4,
                    smc_oversample=1,
                    smc_last_k=6,
                    threads=self.threads,
                    seed=next(_seed) * 7919,
                )
                self.stats["mid_searched"] += 1
                if r is not None:
                    out[i] = r
                    self.stats["mid_changed"] += 1
        return out
