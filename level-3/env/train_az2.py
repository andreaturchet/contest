"""AlphaZero self-play trainer, v2 (literature-informed changes over train_az.py).

Changes vs. train_az.py
  * Playout-cap randomization (KataGo, Wu 2019): on a fraction `full_frac` of
    moves run a full search with Dirichlet root noise and RECORD the sample;
    all other moves use a cheap, noise-free search and are not recorded.  Gives
    many more finished games per unit compute (better value targets) while
    keeping high-quality policy targets.
  * Large replay window with low sample reuse (ELF OpenGo / Polygames: a sample
    should be reused <= ~8 times): default buffer 12 iterations x 1 epoch.
  * Two-stage promotion gate: a candidate that passes the first A/B must pass a
    second, independent A/B (different seeds) before replacing the best model,
    so a noisy 20-game match cannot promote a weaker net by chance.
  * Draws get value target 0.

Shared pieces (network, ONNX export, training step, A/B match, validator) are
imported from train_az.py.

Usage (resume from the v1 checkpoint):
  python train_az2.py --iters 80 --games 160 --full-sims 96 --fast-sims 16
"""
from __future__ import annotations

import argparse
import shutil
import time

import numpy as np

from train_az import (BEST_ONNX, CKPT, CONTEST_ROOT, HERE, LOG, export_onnx,
                      make_net, play_match, train_net, validate)


# ----------------------------------------------------------------- self-play
def _selfplay_chunk(args):
    """Worker: play games with playout-cap randomization.  No torch."""
    (model_path, seeds, full_sims, fast_sims, full_frac,
     warmup_temps, warmup_step) = args
    from colony_one import mcts
    from colony_one.env import ColonyOneEnv
    from colony_one.onnx_runner import OnnxEvaluator

    ev = OnnxEvaluator(model_path)
    cfg_full = mcts.MCTSConfig(simulations=full_sims, add_root_noise=True,
                               warmup_temperatures=tuple(warmup_temps),
                               warmup_step_actions=warmup_step)
    cfg_fast = mcts.MCTSConfig(simulations=fast_sims, add_root_noise=False,
                               warmup_temperatures=tuple(warmup_temps),
                               warmup_step_actions=warmup_step)
    obs_list, pi_list, z_list = [], [], []
    n_games = 0
    for seed in seeds:
        rng = np.random.default_rng(seed)
        env = ColonyOneEnv(seed=seed)
        recs = []
        while not env.terminated:
            team = env.current_team
            full = rng.random() < full_frac
            cfg = cfg_full if full else cfg_fast
            root = mcts.run(env.board, team, env.actions_left, env.round_number,
                            ev, cfg, rng=rng)
            move = mcts.move_number(team, env.actions_left, env.round_number)
            temp = mcts.warmup_temperature(move, cfg.warmup_temperatures,
                                           cfg.warmup_step_actions)
            idx = mcts.select_action_index(root, temperature=temp, rng=rng)
            if full:
                recs.append((env.observation().astype(np.float32),
                             mcts.policy_target(root).astype(np.float32), team))
            env.step(idx)
        winner = env.winner
        n_games += 1
        for obs, pi, team in recs:
            obs_list.append(obs)
            pi_list.append(pi)
            z_list.append(0.0 if winner is None else
                          (1.0 if team == winner else -1.0))
    if not obs_list:
        return (np.zeros((0, 16, 8, 8), np.float32),
                np.zeros((0, 4096), np.float32), np.zeros((0,), np.float32), n_games)
    return (np.stack(obs_list), np.stack(pi_list),
            np.asarray(z_list, np.float32), n_games)


def self_play(model_path, n_games, full_sims, fast_sims, full_frac, workers,
              base_seed, warmup_temps, warmup_step):
    seeds = [base_seed + i for i in range(n_games)]
    workers = max(1, workers)
    chunks = [c for c in (seeds[i::workers] for i in range(workers)) if c]
    tasks = [(str(model_path), c, full_sims, fast_sims, full_frac,
              warmup_temps, warmup_step) for c in chunks]

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

    obs_all, pi_all, z_all = [], [], []
    games = 0
    for obs, pi, z, g in results:
        games += g
        if len(obs):
            obs_all.append(obs); pi_all.append(pi); z_all.append(z)
    if not obs_all:
        return None, games
    return (np.concatenate(obs_all), np.concatenate(pi_all),
            np.concatenate(z_all)), games


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--games", type=int, default=160)
    ap.add_argument("--full-sims", type=int, default=96)
    ap.add_argument("--fast-sims", type=int, default=16)
    ap.add_argument("--full-frac", type=float, default=0.25)
    ap.add_argument("--ab-sims", type=int, default=50)
    ap.add_argument("--ab-games", type=int, default=20)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--channels", type=int, default=48)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--buffer-iters", type=int, default=12)
    ap.add_argument("--promote-winrate", type=float, default=0.55)
    ap.add_argument("--init-onnx", default=str(HERE / "model.onnx"))
    ap.add_argument("--keep-best", action="store_true",
                    help="do not reset best.onnx from --init-onnx")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(2)
    device = "cpu"

    logf = open(LOG, "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    log("=" * 60)
    log(f"START v2 iters={args.iters} games={args.games} "
        f"sims full/fast={args.full_sims}/{args.fast_sims} frac={args.full_frac} "
        f"buffer={args.buffer_iters}it epochs={args.epochs} "
        f"workers={args.workers} net={args.channels}x{args.blocks}")

    if not (args.keep_best and BEST_ONNX.exists()):
        shutil.copy(args.init_onnx, BEST_ONNX)
    net = make_net(args.channels, args.blocks).to(device)
    if CKPT.exists():
        net.load_state_dict(torch.load(CKPT, map_location=device))
        log("resumed net from checkpoint")

    replay = []
    for it in range(1, args.iters + 1):
        t0 = time.time()
        log(f"--- iter {it} : self-play {args.games} games ---")
        data, games = self_play(BEST_ONNX, args.games, args.full_sims, args.fast_sims,
                                args.full_frac, args.workers,
                                base_seed=1_000_000 + 10_000 * it,
                                warmup_temps=(1.0, 1.0, 0.5, 0.3), warmup_step=4)
        if data is None:
            log("  no self-play data; stopping"); break
        replay.append(data)
        replay = replay[-args.buffer_iters:]
        buf = tuple(np.concatenate([d[k] for d in replay]) for k in range(3))
        log(f"  self-play done in {time.time()-t0:.0f}s; games={games} "
            f"new={len(data[0])} buffer={len(buf[0])} samples")

        train_net(net, buf, args.epochs, args.batch, args.lr, device, log)
        torch.save(net.state_dict(), CKPT)

        cand = HERE / f"cand2_{it}.onnx"
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
            # second, independent confirmation match (different seeds)
            wr2 = _confirm_match(cand, BEST_ONNX, args.ab_games, args.ab_sims, it)
            comb = (wr + wr2) / 2
            log(f"  confirmation win-rate = {wr2*100:.0f}%  (combined {comb*100:.0f}%)")
            if comb >= args.promote_winrate:
                shutil.copy(cand, BEST_ONNX)
                shutil.copy(cand, CONTEST_ROOT / "model.onnx")
                shutil.copy(cand, HERE / f"promoted_v2_iter{it}.onnx")
                log(f"  >> PROMOTED iter {it} as new best (copied to contest/model.onnx)")
            else:
                log("  confirmation failed; kept previous best")
        else:
            log("  kept previous best")
        cand.unlink(missing_ok=True)
        log(f"  iter {it} total {time.time()-t0:.0f}s")

    log("DONE v2")
    logf.close()


def _confirm_match(cand_path, cur_path, games, sims, it):
    """Same as train_az.play_match but with a different seed block."""
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
        seed = 500_000 + 1000 * it + g
        if g % 2 == 0:
            wins += (play(ec, eo, seed) == constants.CYAN)
        else:
            wins += (play(eo, ec, seed) == constants.MAGENTA)
    return wins / games


if __name__ == "__main__":
    main()
