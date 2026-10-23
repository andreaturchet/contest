"""CPU self-play for Colony: One. No PyTorch import.

Workers run the published referee MCTS and ship observations to a GPU
inference process. Keeping torch out of this module matters: a Colab worker
that imported torch would copy a multi-gigabyte runtime per core.
"""

from __future__ import annotations

import traceback

import numpy as np

N = 8
SQUARES = 64
NUM_ACTIONS = 4096


def transform_square(r: int, c: int, k: int, mirror: bool) -> tuple[int, int]:
    """Map a square by an optional left-right flip, then ``k`` counterclockwise turns.

    Matches ``np.flip`` on the last axis followed by ``np.rot90`` on axes (1, 2).
    """
    if mirror:
        c = N - 1 - c
    for _ in range(k % 4):
        r, c = N - 1 - c, r
    return r, c


def transform_index(idx: int, k: int, mirror: bool) -> int:
    fr, to = divmod(int(idx), SQUARES)
    r0, c0 = divmod(fr, N)
    r1, c1 = divmod(to, N)
    r0, c0 = transform_square(r0, c0, k, mirror)
    r1, c1 = transform_square(r1, c1, k, mirror)
    return (r0 * N + c0) * SQUARES + (r1 * N + c1)


def transform_obs(obs: np.ndarray, k: int, mirror: bool) -> np.ndarray:
    """D4 action on a ``(16, 8, 8)`` observation. Scalar planes are constant, so they commute."""
    x = np.asarray(obs)
    if mirror:
        x = x[:, :, ::-1]
    k = k % 4
    if k:
        x = np.rot90(x, k, axes=(1, 2))
    return np.ascontiguousarray(x)


def apply_symmetry(sample, k: int, mirror: bool):
    """Return a symmetry of ``(obs, legal, pi, z)``. Identity returns the same objects."""
    obs, legal, pi, z = sample
    if k % 4 == 0 and not mirror:
        return sample
    obs_t = transform_obs(obs, k, mirror)
    legal_t = np.empty_like(legal)
    for i, idx in enumerate(legal):
        legal_t[i] = transform_index(int(idx), k, mirror)
    return obs_t, legal_t, pi, z


def play_one_game(evaluator, scene: np.ndarray, mcts_config, rng: np.random.Generator):
    """Play one Level-4 game. Returns samples ``(obs f16, legal i16, pi f16, z f32)``."""
    from colony_one import mcts
    from colony_one.env import ColonyOneEnv

    env = ColonyOneEnv(seed=int(rng.integers(0, 2**31 - 1)), scene=scene)
    pending = []
    for _ in range(240):
        if env.terminated:
            break
        root = mcts.run(
            env.board,
            env.current_team,
            env.actions_left,
            env.round_number,
            evaluator,
            mcts_config,
            rng=rng,
        )
        if not root.N:
            break
        legal = np.fromiter(sorted(root.N.keys()), dtype=np.int16, count=len(root.N))
        dist = mcts.policy_target(root)
        pi = dist[legal.astype(np.int64)].astype(np.float32)
        total = float(pi.sum())
        if total <= 0:
            pi[:] = 1.0 / len(pi)
        else:
            pi /= total
        move = mcts.move_number(env.current_team, env.actions_left, env.round_number)
        temperature = mcts.warmup_temperature(
            move, mcts_config.warmup_temperatures, mcts_config.warmup_step_actions
        )
        action = mcts.select_action_index(root, temperature, rng=rng)
        pending.append(
            (
                env.observation().astype(np.float16, copy=True),
                legal,
                pi.astype(np.float16),
                env.current_team,
            )
        )
        env.step(int(action))

    if not env.terminated or env.winner is None:
        return []

    winner = env.winner
    samples = []
    for obs, legal, pi, team in pending:
        z = np.float32(1.0 if team == winner else -1.0)
        samples.append((obs, legal, pi, z))
    return samples


def mcts_config_from_kwargs(kwargs: dict):
    from colony_one import mcts

    return mcts.MCTSConfig(**kwargs)


def worker_entry(worker_id, n_games, scene, mcts_kwargs, env_root, req_q, resp_q, out_q, seed):
    """Spawn target. Sends eval requests, returns finished games on ``out_q``."""
    import sys

    try:
        sys.path.insert(0, env_root)
        rng = np.random.default_rng(seed)
        cfg = mcts_config_from_kwargs(mcts_kwargs)

        class Proxy:
            def __init__(self):
                self.n = 0

            def __call__(self, planes):
                rid = self.n
                self.n += 1
                req_q.put(
                    (
                        "eval",
                        worker_id,
                        rid,
                        np.ascontiguousarray(planes, dtype=np.float32),
                    )
                )
                got = resp_q.get()
                got_id, logits, values = got
                if got_id != rid:
                    raise RuntimeError(f"worker {worker_id} expected response {rid}, got {got_id}")
                return logits, values

        proxy = Proxy()
        for game_i in range(n_games):
            samples = play_one_game(proxy, scene, cfg, rng)
            out_q.put(("game", worker_id, game_i, samples))
        out_q.put(("done", worker_id))
        req_q.put(("bye", worker_id))
    except Exception:
        tb = traceback.format_exc()
        try:
            out_q.put(("error", worker_id, tb))
            req_q.put(("bye", worker_id))
        except Exception:
            pass
