#!/usr/bin/env python3
"""
Head-to-head: model A vs model B, both wrapped in referee MCTS.
Alternates sides each game. Usage: python a2b.py A.onnx B.onnx [games] [sims]
"""
import sys

import numpy as np

from colony_one import constants, mcts
from colony_one.env import ColonyOneEnv
from colony_one.onnx_runner import OnnxEvaluator


def play(ev_cyan, ev_mag, sims, seed):
    rng = np.random.default_rng(seed)
    env = ColonyOneEnv(seed=seed)
    cfg = mcts.MCTSConfig(simulations=sims)
    while not env.terminated:
        team = env.current_team
        ev = ev_cyan if team == constants.CYAN else ev_mag
        a = mcts.search_action(env.board, team, env.actions_left,
                               env.round_number, ev, cfg, rng=rng)
        if a is None:
            break
        env.step(a)
    return env.winner


def main():
    A, B = sys.argv[1], sys.argv[2]
    games = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    sims = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    ea, eb = OnnxEvaluator(A), OnnxEvaluator(B)
    wins_a = 0
    for g in range(games):
        if g % 2 == 0:
            w = play(ea, eb, sims, 2000 + g)   # A is CYAN
            wins_a += (w == constants.CYAN)
        else:
            w = play(eb, ea, sims, 2000 + g)   # A is MAGENTA
            wins_a += (w == constants.MAGENTA)
    print(f"A={A} vs B={B}: A won {wins_a}/{games} = {100*wins_a/games:.0f}% (sims={sims})")


if __name__ == "__main__":
    main()
