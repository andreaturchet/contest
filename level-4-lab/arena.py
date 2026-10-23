"""Level 4 matches using the frozen engine and the actual referee settings."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "engine"))
from colony_one import constants, mcts
from colony_one.env import ColonyOneEnv
from colony_one.onnx_runner import OnnxEvaluator


def scene():
    return np.asarray(json.loads((ROOT / "scene.json").read_text()), dtype=np.float32)


def snapshot(env, label="Start", acting=None, move=None):
    pieces = []
    for r in range(8):
        for c in range(8):
            p = env.board.piece_at(r, c)
            if p is not None:
                pieces.append([r, c, p.team, p.piece_type.value])
    return dict(pieces=pieces, locked=[[r, c] for r in range(8) for c in range(8)
                if env.board.is_locked(r, c)], round=env.round_number,
                team=env.current_team, left=env.actions_left, label=label,
                acting=acting, move=move, winner=env.winner)


def play(evaluate_a, evaluate_b, a_team, seed, simulations=100, record=False):
    env = ColonyOneEnv(scene=scene(), seed=seed)
    rngs = {constants.CYAN: np.random.default_rng(seed),
            constants.MAGENTA: np.random.default_rng(seed + 1)}
    cfg = mcts.MCTSConfig(simulations=simulations)
    frames = [snapshot(env)] if record else []
    moves = 0
    while not env.terminated:
        team = env.current_team
        ev = evaluate_a if team == a_team else evaluate_b
        action = mcts.search_action(env.board, team, env.actions_left,
                                   env.round_number, ev, cfg, rng=rngs[team])
        if action is None:
            raise RuntimeError("No legal action in a nonterminal game")
        label = f"{team}: {env.format_action(action)} ({action.kind})"
        move = [list(action.from_pos), list(action.to_pos)]
        env.step(action)
        moves += 1
        if record:
            frames.append(snapshot(env, label, team, move))
        if moves > 104:
            raise RuntimeError("Game exceeded firewall termination at round 26")
    return dict(won=env.winner == a_team, winner=env.winner, a_team=a_team,
                seed=seed, moves=moves, frames=frames)


def wilson(wins, n):
    p, z = wins / n, 1.959963984540054
    den = 1 + z*z/n
    mid = (p + z*z/(2*n)) / den
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return [mid-half, mid+half]


def match(a, b, games=20, seed=100_000, simulations=100, output=None, verbose=True):
    if games < 2 or games % 2 or simulations < 1:
        raise ValueError("Use an even number of games >=2 and simulations >=1")
    ev_a, ev_b = OnnxEvaluator(a), OnnxEvaluator(b)
    results, started = [], time.perf_counter()
    for g in range(games):
        team = constants.CYAN if g % 2 == 0 else constants.MAGENTA
        result = play(ev_a, ev_b, team, seed + 1009*g, simulations, record=output is not None)
        results.append(result)
        if verbose:
            print(f"Game {g+1}/{games}: A {'wins' if result['won'] else 'loses'} as "
                  f"{team}, {result['moves']} actions", flush=True)
    wins = sum(r["won"] for r in results)
    report = dict(model_a=str(a), model_b=str(b), games=games, wins=wins,
        win_rate=wins/games, wilson_95=wilson(wins, games), seed=seed,
        simulations=simulations, seconds=time.perf_counter()-started,
        sha256_a=hashlib.sha256(Path(a).read_bytes()).hexdigest(),
        sha256_b=hashlib.sha256(Path(b).read_bytes()).hexdigest(),
        scene_sha256=hashlib.sha256((ROOT/"scene.json").read_bytes()).hexdigest(),
        cyan_wins=sum(r["won"] for r in results if r["a_team"] == constants.CYAN),
        magenta_wins=sum(r["won"] for r in results if r["a_team"] == constants.MAGENTA))
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.with_suffix(".json").write_text(json.dumps(dict(report=report, games=results)))
        template = (ROOT/"replay.html").read_text()
        output.with_suffix(".html").write_text(template.replace("/* MATCH_DATA */null",
            json.dumps(dict(report=report, games=results)).replace("</", "<\\/")))
    if verbose:
        print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", type=Path, required=True)
    p.add_argument("--b", type=Path, default=ROOT/"baselines/current.onnx")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--seed", type=int, default=100_000)
    p.add_argument("--simulations", type=int, default=100)
    p.add_argument("--output", type=Path, default=ROOT/"replays/match")
    args = p.parse_args()
    match(args.a, args.b, args.games, args.seed, args.simulations, args.output)
