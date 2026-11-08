"""Fine-tune a Q head on paired counterfactual round outcomes.

Each branch starts with exactly the same deal. Only the first action differs.
All four seats then follow the frozen base policy, using their own observation.
The student sees the acting seat's public observation and candidate actions.
"""
import argparse
import copy
from collections import deque
from pathlib import Path

import numpy as np
import torch

from . import _core
from .agents import BatchPolicy, segment_argmax
from .duplicate import _observe, make_deals
from .model import load_checkpoint, save_checkpoint


@torch.no_grad()
def collect_pairs(base, device, seed, rng, deals, win_weight):
    def tensor(x):
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    policy = BatchPolicy(base, device)
    table = _core.Rollouts(make_deals(deals, seed))
    samples, branches = [], []
    while True:
        indices, players, state, history, actions, offsets = _observe(table)
        if not len(indices):
            break
        z = base.encode_state(tensor(state), tensor(history))
        actions = tensor(actions)
        counts = np.diff(offsets)
        q = base.score(z[np.repeat(np.arange(len(state)), counts)], actions).cpu().numpy()
        best = segment_argmax(q, offsets)
        for row in np.flatnonzero((counts > 1) & (rng.random(len(state)) < 0.09)):
            start, end = offsets[row:row + 2]
            first = start + best[row]
            noisy = q[start:end] + rng.gumbel(0, 0.04, counts[row])
            noisy[best[row]] = -np.inf
            second = start + np.argmax(noisy)
            if q[first] - q[second] > 0.35:
                continue
            game = table.game(int(indices[row]))
            moves = game.legal_moves()
            for index in (first, second):
                branch = game.copy()
                branch.apply(*moves[index - start])
                branches.append(branch)
            samples.append((z[row].clone(), actions[first].clone(), actions[second].clone(), players[row] % 2,
                            q[first], q[second]))
        table.step(best.tolist())

    if not samples:
        return None
    rewards = []
    for start in range(0, len(branches), 384):
        games = _core.Rollouts(branches[start:start + 384])
        while True:
            idx, _, s, h, a, off = _observe(games)
            if not len(idx):
                break
            games.step(policy.choose(s, h, a, off).tolist())
        rewards.extend(games.rewards(0))
    z, first, second, team, q_first, q_second = zip(*samples)
    returns = np.asarray(rewards).reshape(-1, 2)
    returns = (1 - win_weight) * returns + win_weight * 3 * np.sign(returns)
    delta = (returns[:, 0] - returns[:, 1]) * np.where(np.asarray(team) == 0, 1, -1) / 3
    return (torch.stack(z), torch.stack(first), torch.stack(second),
            tensor(delta), tensor(np.asarray(q_first)), tensor(np.asarray(q_second)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", help="frozen opponent and initial student checkpoint")
    parser.add_argument("output", type=Path, help="directory for candidate checkpoints")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--iterations", type=int, default=160)
    parser.add_argument("--deals", type=int, default=256)
    parser.add_argument("--seed", type=int, default=61477)
    parser.add_argument("--deal-seed", type=int, default=471000)
    parser.add_argument("--initial-head", help="start the Q head from an earlier candidate for this base")
    parser.add_argument("--win-weight", type=float, default=0.0,
                        help="weight of round win/loss in the target (0 = level change, 1 = win/loss)")
    args = parser.parse_args()
    if min(args.deals, args.iterations) < 1 or min(args.seed, args.deal_seed) < 0:
        parser.error("deals and iterations must be positive and seed nonnegative")
    if not 0 <= args.win_weight <= 1:
        parser.error("win-weight must be between 0 and 1")
    torch.set_num_threads(1)
    rng = np.random.default_rng(args.seed)
    base, meta = load_checkpoint(args.base, args.device)
    base.requires_grad_(False)
    student = copy.deepcopy(base)
    if args.initial_head:
        initial, _ = load_checkpoint(args.initial_head, args.device)
        student.head.load_state_dict(initial.head.state_dict())
    student.head.requires_grad_(True)
    optimizer = torch.optim.AdamW(student.head.parameters(), lr=3e-5, weight_decay=0)
    replay = deque(maxlen=24)
    total_pairs = 0
    for iteration in range(1, args.iterations + 1):
        batch = collect_pairs(base, args.device, args.deal_seed + iteration, rng, args.deals, args.win_weight)
        if batch is not None:
            replay.append(batch)
            total_pairs += len(batch[0])
        if replay and iteration >= 4:
            z, first, second, target, old_first, old_second = map(torch.cat, zip(*replay))
            for _ in range(3):
                rows = rng.integers(len(z), size=1024)
                a = student.score(z[rows], first[rows])
                b = student.score(z[rows], second[rows])
                # The paired loss learns action differences. The anchor retains the Q scale.
                anchor = (a - old_first[rows]).square() + (b - old_second[rows]).square()
                loss = ((a - b - target[rows]).square() + 0.30 * anchor).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        if iteration % 16 == 0 or iteration == args.iterations:
            path = args.output / f"pair-{iteration}.pt"
            metadata = dict(meta, run="paired-counterfactual", paired_iterations=iteration,
                            paired_samples=total_pairs, deal_seed=args.deal_seed, seed=args.seed,
                            win_weight=args.win_weight)
            save_checkpoint(str(path), student, metadata, half=True)
            print(f"{path} ({total_pairs} pairs)", flush=True)


if __name__ == "__main__":
    main()
