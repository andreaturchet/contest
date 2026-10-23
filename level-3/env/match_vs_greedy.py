#!/usr/bin/env python3
"""
Measure win rate of an ONNX model (wrapped in referee MCTS, 60 sims) vs GreedyAgent.
Alternates which side the model plays. Usage: python match_vs_greedy.py model.onnx [games]
"""
import sys

import numpy as np

from colony_one import constants, mcts
from colony_one.baselines import GreedyAgent
from colony_one.env import ColonyOneEnv
from colony_one.onnx_runner import OnnxEvaluator


def play_game(evaluator, model_team, seed):
    rng = np.random.default_rng(seed)
    env = ColonyOneEnv(seed=seed)
    greedy = GreedyAgent(seed=seed + 12345)
    cfg = mcts.MCTSConfig(simulations=60)
    while not env.terminated:
        team = env.current_team
        if team == model_team:
            action = mcts.search_action(
                env.board, team, env.actions_left, env.round_number,
                evaluator, cfg, rng=rng)
        else:
            action = greedy.select(
                env.board, team, env.actions_left, env.round_number)
        if action is None:
            break
        env.step(action)
    return env.winner == model_team


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "model.onnx"
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    ev = OnnxEvaluator(path)
    wins = 0
    for g in range(games):
        team = constants.CYAN if g % 2 == 0 else constants.MAGENTA
        if play_game(ev, team, seed=1000 + g):
            wins += 1
    print(f"model won {wins}/{games} = {100*wins/games:.0f}% vs GreedyAgent")


if __name__ == "__main__":
    main()
