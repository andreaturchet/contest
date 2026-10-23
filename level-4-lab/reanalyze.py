"""One expert-iteration step on top of candidate v2.

Anthony, Tian, and Barber, "Thinking Fast and Slow with Deep Learning and
Tree Search" (2017), train the network on the visit distribution of a search
stronger than the raw policy. Schrittwieser et al., MuZero, reanalyse old
positions with the current network instead of trusting the outcome of a game
that was later misplayed. Danihelka et al. (ICLR 2022) warn that visit counts
are a weak target when many root actions are barely visited; with 200
simulations and roughly 15 legal moves that problem is small, so the teacher
here is the visit distribution of a noise-free search, and the value target
mixes the recorded outcome with the search's visit-weighted root value.

The search is only a teacher. The uploaded model is still candidate v2's
architecture, and the referee still runs its own 100-simulation PUCT.
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import time
from pathlib import Path

import numpy as np

from arena import ROOT, match
from oracle import faithful, fingerprint, from_observation
from train_v3 import _batch_loss
from workers import augment


def teacher_config(simulations: int):
    from colony_one import mcts
    return mcts.MCTSConfig(simulations=simulations, add_root_noise=False, warmup_temperatures=())


def relabel_one(sample, evaluator, config, rng):
    from colony_one import encoding, mcts

    state = from_observation(sample[0])
    if not faithful(sample[0], state) or not state.legal():
        return None
    before = fingerprint(state)
    root = mcts.run(state.board, state.team, state.actions_left, state.round_number,
                    evaluator, config, rng=rng)
    if fingerprint(state) != before or not root.N:
        return None
    legal_actions = state.legal()
    legal = np.array([encoding.action_index(a, state.team) for a in legal_actions], np.int64)
    pi = np.zeros(len(legal), np.float32)
    index = {int(a): i for i, a in enumerate(legal)}
    visits = 0.0
    value_sum = 0.0
    for action, count in root.N.items():
        slot = index.get(int(action))
        if slot is None:
            return None
        pi[slot] = count
        visits += count
        value_sum += root.W[action]
    if visits <= 0:
        return None
    pi /= visits
    q = float(np.clip(value_sum / visits, -1.0, 1.0))
    # The outcome remembers who eventually won. The search value forgets a
    # blunder that happened after this position. Keep both.
    z = 0.5 * float(sample[3]) + 0.5 * q
    order = np.argsort(legal)
    return (
        np.ascontiguousarray(np.asarray(sample[0], np.float16), np.float16),
        legal[order].astype(np.int16),
        pi[order].astype(np.float16),
        np.float32(z),
    )


def main():
    import torch
    from colony_one.onnx_runner import OnnxEvaluator
    from model import Model, export

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=int, default=2500)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--fraction", type=float, default=0.40)
    parser.add_argument("--final-games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/reanalyze_v1")
    args = parser.parse_args()
    args.run.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    with gzip.open(ROOT / "runs/recent_tournament/data/tournament_training.pkl.gz", "rb") as handle:
        training = pickle.load(handle)
    with gzip.open(ROOT / "runs/league_v2/replay.pkl.gz", "rb") as handle:
        replay = pickle.load(handle)[-12_000:]
    evaluator = OnnxEvaluator(ROOT / "candidate_v2.onnx")
    config = teacher_config(args.simulations)
    started = time.monotonic()
    probe = [relabel_one(replay[i], evaluator, config, rng) for i in range(8)]
    rate = 8 / max(1e-6, time.monotonic() - started)
    print(f"Teacher {rate:.2f} positions/s at {args.simulations} simulations", flush=True)
    if any(row is None for row in probe):
        raise RuntimeError("teacher could not relabel the probe positions")
    budget_seconds = 150
    limit = min(args.positions, max(400, int(rate * budget_seconds)))
    order = rng.permutation(len(replay))
    chosen = [int(i) for i in order[:limit]]
    fresh = []
    t0 = time.monotonic()
    for n, index in enumerate(chosen, start=1):
        row = relabel_one(replay[index], evaluator, config, rng)
        if row is not None:
            fresh.append(row)
        if n % 250 == 0:
            print(f"  relabeled {n}/{len(chosen)} kept {len(fresh)} ({time.monotonic() - t0:.0f}s)", flush=True)
    print(f"Kept {len(fresh)} noise-free search targets", flush=True)
    held = set(chosen)
    replay_rest = [replay[i] for i in range(len(replay)) if i not in held]

    state = torch.load(ROOT / "runs/recent_auxiliary/adapted_weights.pt", map_location="cpu", weights_only=False)
    net = Model(**state["architecture"])
    net.load_state_dict(state["model"])
    device = torch.device("cpu")
    net.to(device)
    ref = Model(**state["architecture"]).to(device)
    ref.load_state_dict(state["model"])
    ref.eval()
    for parameter in ref.parameters():
        parameter.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    torch.set_num_threads(4)
    history = []
    t1 = time.monotonic()
    for step in range(1, args.steps + 1):
        batch, anchor = [], []
        for _ in range(128):
            roll = rng.random()
            if roll < args.fraction:
                batch.append(augment(fresh[int(rng.integers(len(fresh)))], rng))
            elif roll < args.fraction + 0.25:
                batch.append(augment(training[int(rng.integers(len(training)))], rng))
            else:
                batch.append(augment(replay_rest[int(rng.integers(len(replay_rest)))], rng))
            anchor.append(0.15)
        net.train()
        opt.zero_grad(set_to_none=True)
        policy, value, anchored = _batch_loss(net, ref, batch, anchor, device)
        (policy + value + anchored).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step == args.steps or step % 50 == 0:
            row = dict(step=step, policy=float(policy.detach()), value=float(value.detach()),
                       anchor=float(anchored.detach()), seconds=time.monotonic() - t1)
            history.append(row)
            print(json.dumps(row), flush=True)
    export_path = export(net, args.run / "candidate.onnx")
    (args.run / "history.json").write_text(json.dumps(dict(
        config=dict(positions=len(fresh), simulations=args.simulations, steps=args.steps,
                    lr=args.lr, fraction=args.fraction, seed=args.seed),
        history=history), indent=2))
    torch.set_num_threads(1)
    final = match(export_path, ROOT / "candidate_v2.onnx", args.final_games,
                  seed=410_000_033, output=args.run / "final_match", verbose=False)
    heuristic = match(export_path, ROOT / "baselines/current.onnx", 80, seed=420_000_041, verbose=False)
    final["heuristic"] = heuristic
    final["promotion_passed"] = final["wilson_95"][0] > 0.5
    (args.run / "final_result.json").write_text(json.dumps(final, indent=2))
    print(json.dumps(final, indent=2), flush=True)
    if final["promotion_passed"]:
        import shutil
        shutil.copyfile(export_path, args.run / "submission.onnx")
        shutil.copyfile(export_path, ROOT / "candidate_v3.onnx")
        print("Promoted candidate_v3.onnx", flush=True)
    else:
        print("No established win over candidate v2. candidate_v2.onnx was left unchanged.", flush=True)


if __name__ == "__main__":
    main()
