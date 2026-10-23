#!/usr/bin/env python3
"""
AlphaZero-style self-play trainer for Colony: One (Level 4).

Loop:
  1. Self-play N games in parallel, evaluating with the current best ONNX model
     wrapped in the referee's own PUCT MCTS. Record (obs, visit-policy, outcome).
  2. Train a small ResNet (policy + value) on the replay buffer.
  3. Export the net to ONNX, validate the submission contract.
  4. A/B the candidate vs the current best at the referee's 100 sims. Promote
     only if it wins >= PROMOTE_WINRATE, so we never regress. On promotion,
     copy to contest/model.onnx so the best model is always ready to upload.

Bootstrapped from the hand-crafted heuristic model (strong starting evaluator),
following AlphaZero (Silver et al. 2018); training on the search's visit-count
policy is what keeps improvement stable at low simulation budgets
(Danihelka et al., ICLR 2022).

Usage:
  python train_az.py --iters 30 --games 60 --selfplay-sims 64 --workers 5
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CONTEST_ROOT = HERE.parents[1]           # /Users/ciwrl/Downloads/contest
BEST_ONNX = HERE / "best.onnx"
CKPT = HERE / "az_ckpt.pt"
LOG = HERE / "train_az.log"

# ----------------------------------------------------------------- self-play
def _selfplay_chunk(args):
    """Worker: play a chunk of games with OnnxEvaluator(model_path). No torch."""
    model_path, seeds, sims, warmup_temps, warmup_step = args
    import numpy as np
    from colony_one import mcts
    from colony_one.env import ColonyOneEnv
    from colony_one.onnx_runner import OnnxEvaluator

    ev = OnnxEvaluator(model_path)
    cfg = mcts.MCTSConfig(
        simulations=sims,
        warmup_temperatures=tuple(warmup_temps),
        warmup_step_actions=warmup_step,
    )
    obs_list, pi_list, team_list, game_id = [], [], [], []
    for gi, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        env = ColonyOneEnv(seed=seed)
        recs = []
        while not env.terminated:
            team = env.current_team
            obs = env.observation()
            root = mcts.run(env.board, team, env.actions_left, env.round_number,
                            ev, cfg, rng=rng)
            pi = mcts.policy_target(root)                 # (4096,)
            move = mcts.move_number(team, env.actions_left, env.round_number)
            temp = mcts.warmup_temperature(move, cfg.warmup_temperatures,
                                           cfg.warmup_step_actions)
            idx = mcts.select_action_index(root, temperature=temp, rng=rng)
            recs.append((obs.astype(np.float32), pi.astype(np.float32), team))
            env.step(idx)
        winner = env.winner
        for obs, pi, team in recs:
            obs_list.append(obs)
            pi_list.append(pi)
            team_list.append(1.0 if team == winner else -1.0)
            game_id.append(seed)
    if not obs_list:
        return (np.zeros((0, 16, 8, 8), np.float32),
                np.zeros((0, 4096), np.float32), np.zeros((0,), np.float32))
    return (np.stack(obs_list), np.stack(pi_list),
            np.asarray(team_list, np.float32))


def self_play(model_path, n_games, sims, workers, base_seed, warmup_temps, warmup_step):
    seeds = [base_seed + i for i in range(n_games)]
    workers = max(1, workers)
    chunks = [seeds[i::workers] for i in range(workers)]
    chunks = [c for c in chunks if c]
    tasks = [(str(model_path), c, sims, warmup_temps, warmup_step) for c in chunks]
    obs_all, pi_all, z_all = [], [], []

    results = None
    if workers > 1:
        try:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=workers) as ex:
                results = list(ex.map(_selfplay_chunk, tasks))
        except (PermissionError, OSError, NotImplementedError):
            results = None  # sandbox blocks process pools -> run inline
    if results is None:
        results = [_selfplay_chunk(t) for t in tasks]

    for obs, pi, z in results:
        if len(obs):
            obs_all.append(obs); pi_all.append(pi); z_all.append(z)
    if not obs_all:
        return None
    return (np.concatenate(obs_all), np.concatenate(pi_all), np.concatenate(z_all))


# --------------------------------------------------------------------- model
def make_net(channels=64, blocks=4, pol_ch=2):
    import torch
    import torch.nn as nn

    class Res(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
            self.b1 = nn.BatchNorm2d(c)
            self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
            self.b2 = nn.BatchNorm2d(c)

        def forward(self, x):
            y = torch.relu(self.b1(self.c1(x)))
            y = self.b2(self.c2(y))
            return torch.relu(x + y)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(16, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels), nn.ReLU())
            self.blocks = nn.Sequential(*[Res(channels) for _ in range(blocks)])
            self.p_conv = nn.Sequential(
                nn.Conv2d(channels, pol_ch, 1, bias=False),
                nn.BatchNorm2d(pol_ch), nn.ReLU())
            self.p_fc = nn.Linear(pol_ch * 64, 4096)
            self.v_conv = nn.Sequential(
                nn.Conv2d(channels, 1, 1, bias=False),
                nn.BatchNorm2d(1), nn.ReLU())
            self.v_fc = nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh())

        def forward(self, x):
            h = self.blocks(self.stem(x))
            p = self.p_fc(self.p_conv(h).flatten(1))
            v = self.v_fc(self.v_conv(h).flatten(1)).squeeze(-1)
            return p, v

    return Net()


def _constants_to_initializers(model):
    """Rewrite every `Constant` node as an initializer (the validator forbids
    the `Constant` op but allows initializers)."""
    import onnx
    g = model.graph
    keep = []
    for node in g.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    t = onnx.TensorProto()
                    t.CopyFrom(attr.t)
                    t.name = node.output[0]
                    g.initializer.append(t)
                    break
        else:
            keep.append(node)
    del g.node[:]
    g.node.extend(keep)
    return model


def export_onnx(net, path):
    import onnx
    import torch
    net.eval()
    dummy = torch.zeros(1, 16, 8, 8)
    tmp = str(path) + ".tmp.onnx"
    torch.onnx.export(
        net, dummy, tmp,
        input_names=["observation"], output_names=["policy", "value"],
        dynamic_axes={"observation": {0: "batch"},
                      "policy": {0: "batch"}, "value": {0: "batch"}},
        opset_version=18, do_constant_folding=True, dynamo=False)
    model = onnx.load(tmp)
    model = _constants_to_initializers(model)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    os.remove(tmp)


def train_net(net, buf, epochs, batch, lr, device, log):
    import torch
    import torch.nn.functional as F
    obs, pi, z = buf
    n = len(obs)
    obs_t = torch.from_numpy(obs).to(device)
    pi_t = torch.from_numpy(pi).to(device)
    z_t = torch.from_numpy(z).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    net.train()
    last = (0.0, 0.0)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        tot_p = tot_v = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            ob, pt, zt = obs_t[idx], pi_t[idx], z_t[idx]
            logits, v = net(ob)
            logp = F.log_softmax(logits, dim=1)
            loss_p = -(pt * logp).sum(1).mean()
            loss_v = F.mse_loss(v, zt)
            loss = loss_p + loss_v
            opt.zero_grad(); loss.backward(); opt.step()
            tot_p += float(loss_p.detach()) * len(idx)
            tot_v += float(loss_v.detach()) * len(idx)
        last = (tot_p / n, tot_v / n)
    log(f"    train: policy_ce={last[0]:.3f} value_mse={last[1]:.3f} "
        f"(samples={n}, epochs={epochs})")


# ----------------------------------------------------------------- A/B gate
def play_match(cand_path, cur_path, games, sims):
    from colony_one import constants, mcts
    from colony_one.env import ColonyOneEnv
    from colony_one.onnx_runner import OnnxEvaluator
    ec, eo = OnnxEvaluator(cand_path), OnnxEvaluator(cur_path)
    cfg = mcts.MCTSConfig(simulations=sims)

    def play(ev_cyan, ev_mag, seed):
        rng = np.random.default_rng(seed)
        env = ColonyOneEnv(seed=seed)
        while not env.terminated:
            team = env.current_team
            ev = ev_cyan if team == constants.CYAN else ev_mag
            a = mcts.search_action(env.board, team, env.actions_left,
                                   env.round_number, ev, cfg, rng=rng)
            if a is None:
                break
            env.step(a)
        return env.winner

    wins = 0
    for g in range(games):
        if g % 2 == 0:
            wins += (play(ec, eo, 9000 + g) == constants.CYAN)
        else:
            wins += (play(eo, ec, 9000 + g) == constants.MAGENTA)
    return wins / games


def validate(path):
    import subprocess
    r = subprocess.run([sys.executable, str(HERE / "tools" / "validate_submission.py"),
                        str(path)], capture_output=True, text=True)
    return "ACCEPTED" in r.stdout, r.stdout


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--selfplay-sims", type=int, default=64)
    ap.add_argument("--ab-sims", type=int, default=100)
    ap.add_argument("--ab-games", type=int, default=40)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--buffer-iters", type=int, default=4)
    ap.add_argument("--promote-winrate", type=float, default=0.55)
    ap.add_argument("--init-onnx", default=str(HERE / "model.onnx"))
    args = ap.parse_args()

    import torch
    torch.set_num_threads(2)  # leave cores for self-play workers
    device = "cpu"

    logf = open(LOG, "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    log("=" * 60)
    log(f"START iters={args.iters} games={args.games} sp_sims={args.selfplay_sims} "
        f"workers={args.workers} net={args.channels}x{args.blocks}")

    # current best = bootstrap heuristic
    shutil.copy(args.init_onnx, BEST_ONNX)
    net = make_net(args.channels, args.blocks).to(device)
    if CKPT.exists():
        net.load_state_dict(torch.load(CKPT, map_location=device))
        log("resumed net from checkpoint")

    replay = []  # list of (obs, pi, z) per iteration

    for it in range(1, args.iters + 1):
        t0 = time.time()
        log(f"--- iter {it} : self-play {args.games} games @ {args.selfplay_sims} sims ---")
        data = self_play(BEST_ONNX, args.games, args.selfplay_sims, args.workers,
                         base_seed=10_000 * it,
                         warmup_temps=(1.0, 1.0, 0.5, 0.3), warmup_step=4)
        if data is None:
            log("  no self-play data; stopping"); break
        replay.append(data)
        replay = replay[-args.buffer_iters:]
        buf = (np.concatenate([d[0] for d in replay]),
               np.concatenate([d[1] for d in replay]),
               np.concatenate([d[2] for d in replay]))
        log(f"  self-play done in {time.time()-t0:.0f}s; buffer={len(buf[0])} samples")

        train_net(net, buf, args.epochs, args.batch, args.lr, device, log)
        torch.save(net.state_dict(), CKPT)

        cand = HERE / f"cand_{it}.onnx"
        export_onnx(net, cand)
        ok, out = validate(cand)
        size = cand.stat().st_size
        if not ok:
            log(f"  candidate REJECTED by validator (size={size}B):")
            log("    " + out.strip().splitlines()[-1])
            cand.unlink(missing_ok=True)
            continue
        log(f"  candidate ACCEPTED ({size/1024:.0f} KiB); A/B vs best "
            f"@ {args.ab_sims} sims x{args.ab_games}")
        wr = play_match(cand, BEST_ONNX, args.ab_games, args.ab_sims)
        log(f"  candidate win-rate vs best = {wr*100:.0f}%")
        if wr >= args.promote_winrate:
            shutil.copy(cand, BEST_ONNX)
            shutil.copy(cand, CONTEST_ROOT / "model.onnx")
            log(f"  >> PROMOTED iter {it} as new best (copied to contest/model.onnx)")
        else:
            log("  kept previous best")
        cand.unlink(missing_ok=True)
        log(f"  iter {it} total {time.time()-t0:.0f}s")

    log("DONE")
    logf.close()


if __name__ == "__main__":
    main()
