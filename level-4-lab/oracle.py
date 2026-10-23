"""Exact short-horizon proofs for Colony: One, used as training targets.

The referee keeps AlphaZero PUCT and a 100-simulation budget, so a stronger
teacher has to show up in the policy and value the network emits. Three
results say how to build that teacher without replacing the referee:

- Winands, Björnsson, and Saito, "Monte-Carlo Tree Search Solver" (2008),
  back up a proved win or loss as exactly +1 or -1 instead of a network guess.
- Danihelka, Guez, Schrittwieser, and Silver, "Policy improvement by planning
  with Gumbel" (ICLR 2022), show that visit counts are a weak policy target
  when many root actions are barely searched, and that an improved policy
  should follow proved action values.
- Ruoss et al., "Amortized Planning with Large-Scale Transformers" / ChessBench
  (NeurIPS 2024), find that distilling an oracle's action values beats cloning
  the one move a player happened to make.
- Thompson, "Retrograde Analysis of Certain Endgames" (ICCA Journal, 1986),
  is the reason the bare-kernel endgame is enumerated rather than sampled:
  once only the two kernels remain, the rules name a winner at the end of the
  turn, so every legal last action has an exact label.

Proofs come from the official transition (captures, bodyguards, the bare-kernel
rule, and the firewall). A horizon that is not proved is left unchanged.
Incomplete search is allowed to miss a win. It is not allowed to invent one.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "engine"))

from colony_one import constants, encoding
from colony_one.board import Board
from colony_one.env import ColonyOneEnv
from colony_one.pieces import Piece, PieceType
from colony_one.rules import legal_actions_for_team

VERSION = "oracle-proofs-v1"
SHRINK = constants.SHRINK_ROUNDS


def _opponent(team: str) -> str:
    return constants.MAGENTA if team == constants.CYAN else constants.CYAN


def infer_round(obs: np.ndarray) -> int:
    """Recover the round from the lock plane and firewall urgency.

    Locks happen at the end of rounds 8, 16, 24, and 26, so the locked-square
    count names the stage. Urgency is ``1 - (next_shrink - round) / 8``.
    """
    locked = float(np.asarray(obs[12], np.float32).sum())
    urgency = float(obs[14, 0, 0])
    if locked < 1:
        nxt = 8
    elif locked < 40:
        nxt = 16
    elif locked < 54:
        nxt = 24
    elif locked < 62:
        nxt = 26
    else:
        return 27
    remaining = (1.0 - urgency) * float(SHRINK[0])
    return int(max(1, min(26, round(nxt - remaining))))


def blank_board() -> Board:
    board = object.__new__(Board)
    n = constants.BOARD_SIZE
    board.grid = [[None for _ in range(n)] for _ in range(n)]
    board.locked = [[False for _ in range(n)] for _ in range(n)]
    return board


class State:
    """Mutable rules state with exact undo. Mirrors ``ColonyOneEnv.step``."""

    def __init__(self, board: Board, team: str, actions_left: int, round_number: int):
        self.board = board
        self.team = team
        self.actions_left = int(actions_left)
        self.round_number = int(round_number)
        self.terminated = False
        self.winner: str | None = None
        self.nodes = 0

    def observation(self) -> np.ndarray:
        return encoding.encode_board(self.board, self.team, self.actions_left, self.round_number)

    def legal(self):
        if self.terminated:
            return []
        return legal_actions_for_team(self.board, self.team)

    def apply(self, action):
        acting = self.team
        info = {
            "record": None,
            "team": acting,
            "left": self.actions_left,
            "round": self.round_number,
            "firewall": None,
        }
        self.actions_left -= 1
        record = self.board.apply_action(action)
        info["record"] = record
        captured = record.captured_piece
        if captured is not None and captured.piece_type is PieceType.KERNEL:
            self.terminated = True
            self.winner = acting
            return info
        if self.actions_left > 0:
            return info
        active_round = self.round_number
        if self.team == constants.MAGENTA:
            self.round_number += 1
        self.team = _opponent(self.team)
        self.actions_left = constants.MAX_ACTION_PER_TURN
        if self.board.only_kernels_remain():
            self.terminated = True
            self.winner = self.board.kernel_endgame_winner(acting_team=acting)
            return info
        if acting == constants.MAGENTA and active_round in SHRINK:
            firewall = self.board.apply_firewall_shrink()
            info["firewall"] = firewall
            if firewall is not None:
                killed = [p for _, _, p in firewall.killed if p.piece_type is PieceType.KERNEL]
                if len(killed) == 2:
                    self.terminated = True
                    self.winner = constants.MAGENTA
                elif len(killed) == 1:
                    self.terminated = True
                    self.winner = _opponent(killed[0].team)
        return info

    def undo(self, info) -> None:
        if info["firewall"] is not None:
            self.board.undo_firewall_shrink(info["firewall"])
        self.team = info["team"]
        self.actions_left = info["left"]
        self.round_number = info["round"]
        self.terminated = False
        self.winner = None
        self.board.undo_action(info["record"])

    def ordered(self, actions):
        enemy = _opponent(self.team)
        kernel = None
        for row in self.board.grid:
            for piece in row:
                if piece is not None and piece.team == enemy and piece.piece_type is PieceType.KERNEL:
                    kernel = piece
                    break
            if kernel is not None:
                break
        kernel_at = None
        if kernel is not None:
            for r in range(8):
                for c in range(8):
                    if self.board.grid[r][c] is kernel:
                        kernel_at = (r, c)
        def key(action):
            target = self.board.grid[action.to_pos[0]][action.to_pos[1]]
            if kernel_at is not None and action.to_pos == kernel_at:
                return 0
            if target is not None and kernel_at is not None:
                if max(abs(action.to_pos[0] - kernel_at[0]), abs(action.to_pos[1] - kernel_at[1])) <= 1:
                    return 0
            if target is not None:
                return 1
            return 2
        return sorted(actions, key=key)

    def search(self, depth: int, budget: list[int]) -> int:
        """Value for the side to move. +1 or -1 is proved. 0 is not proved.

        A node limit returns 0 unless a win was already proved, so a truncated
        tree cannot be reported as a forced loss.
        """
        if budget[0] <= 0 or depth <= 0 or self.terminated:
            return 0
        team = self.team
        actions = self.ordered(self.legal())
        if not actions:
            return 0
        best = -1
        completed = True
        for action in actions:
            if budget[0] <= 0:
                completed = False
                break
            budget[0] -= 1
            self.nodes += 1
            info = self.apply(action)
            if self.terminated:
                val = 1 if self.winner == team else -1
            elif self.team == team:
                val = self.search(depth - 1, budget)
            else:
                val = -self.search(depth - 1, budget)
            self.undo(info)
            if val > best:
                best = val
            if best == 1:
                return 1
        if not completed:
            return 1 if best == 1 else 0
        return best

    def root_values(self, depth: int, budget_each: int) -> list[tuple[int, int]]:
        team = self.team
        results = []
        for action in self.legal():
            budget = [budget_each]
            self.nodes += 1
            info = self.apply(action)
            if self.terminated:
                val = 1 if self.winner == team else -1
            elif self.team == team:
                val = self.search(depth - 1, budget)
            else:
                val = -self.search(depth - 1, budget)
            self.undo(info)
            results.append((encoding.action_index(action, team), int(val)))
        return results


def from_observation(obs) -> State:
    obs = np.asarray(obs, np.float32)
    team = constants.MAGENTA if float(obs[15, 0, 0]) > 0.5 else constants.CYAN
    actions_left = 2 if float(obs[13, 0, 0]) > 0.75 else 1
    board = Board.from_observation(obs, team)
    return State(board, team, actions_left, infer_round(obs))


def faithful(obs, state: State) -> bool:
    back = state.observation()
    if np.max(np.abs(back - np.asarray(obs, np.float32))) > 1e-4:
        return False
    return not state.terminated


def pack(state: State, values: dict[int, int], z: float):
    legal = np.array(sorted(values), np.int16)
    wins = [i for i in legal if values[int(i)] == 1]
    if wins:
        pi = np.zeros(len(legal), np.float32)
        for i, idx in enumerate(legal):
            if int(idx) in wins:
                pi[i] = 1.0
        pi /= pi.sum()
        z = 1.0
    elif legal.size and all(values[int(i)] == -1 for i in legal):
        pi = np.full(len(legal), 1.0 / len(legal), np.float32)
        z = -1.0
    else:
        pi = np.full(len(legal), 1.0 / len(legal), np.float32)
    return (
        np.ascontiguousarray(state.observation(), np.float16),
        legal,
        pi.astype(np.float16),
        np.float32(z),
    )


def correct(legal, pi, z, values: dict[int, int]):
    """Replace only the actions an exact proof condemns or requires.

    Returns the sample fields and whether the stored target changed.
    """
    legal = np.asarray(legal, np.int64)
    pi = np.asarray(pi, np.float32).copy()
    if pi.sum() <= 0:
        pi[:] = 1.0 / len(pi)
    else:
        pi /= pi.sum()
    known = [int(i) for i in legal if int(i) in values]
    if len(known) != len(legal):
        return legal.astype(np.int16), pi, np.float32(z), False
    vals = np.array([values[int(i)] for i in legal])
    original = pi.copy()
    out_z = float(z)
    if np.any(vals == 1):
        pi = (vals == 1).astype(np.float32)
        pi /= pi.sum()
        out_z = 1.0
    elif np.all(vals == -1):
        pi = np.full(len(legal), 1.0 / len(legal), np.float32)
        out_z = -1.0
    elif np.any(vals == -1):
        pi = pi.copy()
        pi[vals == -1] = 0
        if pi.sum() <= 0:
            pi = (vals != -1).astype(np.float32)
        pi /= pi.sum()
    changed = out_z != float(z) or float(np.max(np.abs(pi - original))) > 1e-6
    return legal.astype(np.int16), pi.astype(np.float32), np.float32(out_z), changed


def analyze(state: State, reply: bool, mate_budget: int, reply_budget: int):
    """Per-action proofs. +1 win, -1 loss, 0 not proved."""
    mate = dict(state.root_values(state.actions_left, mate_budget))
    if any(v == 1 for v in mate.values()) or (mate and all(v == -1 for v in mate.values())):
        return mate
    if not reply:
        return mate
    deeper = dict(state.root_values(state.actions_left + 2, reply_budget))
    merged = {}
    for idx, val in mate.items():
        if val == -1 or val == 1:
            merged[idx] = val
        else:
            merged[idx] = deeper.get(idx, 0)
    return merged


def proved_children(state: State, values: dict[int, int], budget: int, limit: int):
    """Positions the proof actually reaches: a forced win for whoever is to move."""
    found = []
    team = state.team
    actions = {encoding.action_index(a, team): a for a in state.legal()}
    for idx, val in values.items():
        if val != -1 or len(found) >= limit:
            continue
        action = actions.get(idx)
        if action is None:
            continue
        info = state.apply(action)
        if state.terminated:
            state.undo(info)
            continue
        child_values = dict(state.root_values(state.actions_left + 2, budget))
        if any(v == 1 for v in child_values.values()):
            found.append(pack(state, child_values, 1.0))
        elif child_values and all(v == -1 for v in child_values.values()):
            found.append(pack(state, child_values, -1.0))
        state.undo(info)
    return found


def _sample_tuple(obs, legal, pi, z):
    return (
        np.ascontiguousarray(obs, np.float16),
        np.asarray(legal, np.int16),
        np.asarray(pi, np.float16),
        np.float32(z),
    )


def label_record(sample, reply: bool, mate_budget: int, reply_budget: int):
    obs, legal, pi, z = sample
    state = from_observation(obs)
    if not faithful(obs, state):
        return None
    current = {encoding.action_index(a, state.team) for a in state.legal()}
    if current != {int(i) for i in legal}:
        return None
    values = analyze(state, reply, mate_budget, reply_budget)
    legal_o, pi_o, z_o, changed = correct(legal, pi, z, values)
    children = proved_children(state, values, reply_budget, limit=4) if reply and changed else []
    packed = (np.ascontiguousarray(np.asarray(obs, np.float16), np.float16), legal_o, pi_o.astype(np.float16), z_o)
    return packed, changed, children


def generate_bare_kernels(limit: int, seed: int):
    """Every reachable last action of a bare-kernel turn has a closed-form winner.

    Only ``actions_left == 1`` occurs: the game ends at the end of the turn that
    removed the last non-kernel, so a turn never starts with two kernels.
    """
    rng = np.random.default_rng(seed)
    made = []
    rounds = []
    for round_number in range(1, 27):
        locks = sum(1 for s in SHRINK if s < round_number)
        rounds.append((round_number, locks))
    rng.shuffle(rounds)
    for round_number, locks in rounds:
        board = blank_board()
        for _ in range(locks):
            board.apply_firewall_shrink()
        open_squares = [(r, c) for r in range(8) for c in range(8) if not board.locked[r][c]]
        pairs = [(a, b) for i, a in enumerate(open_squares) for b in open_squares[i + 1:]]
        rng.shuffle(pairs)
        for (r1, c1), (r2, c2) in pairs:
            for team, other in ((constants.CYAN, constants.MAGENTA), (constants.MAGENTA, constants.CYAN)):
                board.grid[r1][c1] = Piece(PieceType.KERNEL, team)
                board.grid[r2][c2] = Piece(PieceType.KERNEL, other)
                state = State(board, team, 1, round_number)
                values = dict(state.root_values(1, budget_each=32))
                if values and all(v != 0 for v in values.values()):
                    made.append(pack(state, values, 1.0 if any(v == 1 for v in values.values()) else -1.0))
                board.grid[r1][c1] = None
                board.grid[r2][c2] = None
                if len(made) >= limit:
                    return made
    return made


def generate_random_tactics(games: int, seed: int, mate_budget: int, reply_budget: int):
    """Random legal games are dense with hanging kernels. Keep proved positions only."""
    from arena import scene

    rng = np.random.default_rng(seed)
    found = []
    seen = set()
    for game in range(games):
        env = ColonyOneEnv(scene=scene(), seed=int(rng.integers(1, 2**31 - 1)))
        state = from_observation(env.observation())
        ply = 0
        while not state.terminated and ply < 104:
            actions = state.legal()
            if not actions:
                break
            # Reply search on a subset; mate-in-turn on every position.
            reply = ply % 3 == 0
            values = analyze(state, reply, mate_budget, reply_budget if reply else mate_budget)
            proved = any(v != 0 for v in values.values())
            if proved and (any(v == 1 for v in values.values()) or all(v == -1 for v in values.values())):
                sample = pack(state, values, 0.0)
                key = sample[0].tobytes()
                if key not in seen:
                    seen.add(key)
                    found.append(sample)
            if reply:
                found.extend(proved_children(state, values, reply_budget, limit=2))
            action = actions[int(rng.integers(len(actions)))]
            state.apply(action)
            ply += 1
    # Children and parents can repeat. Exact byte identity is enough.
    unique = []
    seen.clear()
    for sample in found:
        key = sample[0].tobytes() + sample[3].tobytes()
        if key in seen:
            continue
        seen.add(key)
        unique.append(sample)
    return unique


def fingerprint(state: State):
    cells = []
    for r in range(8):
        for c in range(8):
            piece = state.board.grid[r][c]
            cells.append(None if piece is None else (piece.team, piece.piece_type.value, piece.has_moved))
            cells.append(bool(state.board.locked[r][c]))
    return (state.team, state.actions_left, state.round_number, state.terminated, state.winner, tuple(cells))


def self_test() -> None:
    """The fast state must match ``ColonyOneEnv.step``, including the firewall."""
    from arena import scene

    env = ColonyOneEnv(scene=scene(), seed=7)
    state = from_observation(env.observation())
    assert faithful(env.observation(), state)
    rng = np.random.default_rng(7)
    steps = 0
    while not env.terminated:
        actions = env.legal_actions()
        fast = state.legal()
        env_idx = [encoding.action_index(a, env.current_team) for a in actions]
        fast_idx = [encoding.action_index(a, state.team) for a in fast]
        assert env_idx == fast_idx, "legal-action mismatch"
        # Float16 observations are what the datasets store.
        obs16 = env.observation().astype(np.float16)
        restored = from_observation(obs16)
        assert restored.round_number == env.round_number
        assert restored.actions_left == env.actions_left
        assert restored.team == env.current_team
        assert faithful(obs16, restored)
        k = int(rng.integers(len(actions)))
        before = fingerprint(state)
        info = state.apply(fast[k])
        state.undo(info)
        assert fingerprint(state) == before, "undo is not exact"
        state.apply(fast[k])
        env.step(actions[k])
        assert state.team == env.current_team
        assert state.actions_left == env.actions_left
        assert state.round_number == env.round_number
        assert state.terminated == env.terminated
        assert state.winner == env.winner
        if not env.terminated:
            assert np.allclose(state.observation(), env.observation())
        steps += 1
        if steps > 104:
            raise RuntimeError("test game did not terminate")

    # Random games often end by capture before a shrink. Build the round-8
    # firewall directly: Magenta to move, Cyan's kernel standing on the ring.
    board = blank_board()
    board.grid[0][0] = Piece(PieceType.KERNEL, constants.CYAN)
    board.grid[4][4] = Piece(PieceType.KERNEL, constants.MAGENTA)
    board.grid[4][5] = Piece(PieceType.PING, constants.MAGENTA)
    fire = State(board, constants.MAGENTA, 1, 8)
    before = fingerprint(fire)
    ping_moves = [a for a in fire.legal() if a.from_pos == (4, 5) and a.to_pos == (4, 6)]
    assert ping_moves, "expected a quiet Ping step"
    info = fire.apply(ping_moves[0])
    assert fire.terminated and fire.winner == constants.MAGENTA, (fire.terminated, fire.winner)
    official = mate_env(State(blank_board(), constants.MAGENTA, 1, 8))
    # mate_env copies the pre-move observation, so rebuild from `before` via a fresh state.
    fresh = State(blank_board(), constants.MAGENTA, 1, 8)
    fresh.board.grid[0][0] = Piece(PieceType.KERNEL, constants.CYAN)
    fresh.board.grid[4][4] = Piece(PieceType.KERNEL, constants.MAGENTA)
    fresh.board.grid[4][5] = Piece(PieceType.PING, constants.MAGENTA)
    official = mate_env(fresh)
    quiet = [a for a in official.legal_actions() if a.from_pos == (4, 5) and a.to_pos == (4, 6)][0]
    _, reward, done, _ = official.step(quiet)
    assert done and reward > 0 and official.winner == constants.MAGENTA
    fire.undo(info)
    assert fingerprint(fire) == before, "firewall undo drifted"

    # A direct kernel capture must be a proved win, and the slow environment
    # search must agree on a small tree.
    board = blank_board()
    board.grid[0][7] = Piece(PieceType.KERNEL, constants.MAGENTA)
    board.grid[1][6] = Piece(PieceType.ICE, constants.CYAN)
    board.grid[7][0] = Piece(PieceType.KERNEL, constants.CYAN)
    mate = State(board, constants.CYAN, 1, 1)
    values = dict(mate.root_values(1, budget_each=64))
    assert any(v == 1 for v in values.values()), values
    slow_env = ColonyOneEnv(scene=scene(), seed=1)
    # Rebuild the same position inside the official environment by stepping a clone
    # of the fast state's observation through the independent prover below.
    assert _slow_value(mate_env(mate), depth=1, budget=[10_000]) == 1

    # Opening: one of our actions plus the opponent's turn is not a forced win.
    opening = ColonyOneEnv(scene=scene(), seed=1)
    opening_state = from_observation(opening.observation())
    assert opening_state.search(opening_state.actions_left + 2, [20_000]) == 0

    # Per-action values must match an independent prover that only calls env.step.
    # Budgets are large enough that both searches finish, so move order cannot matter.
    agree = np.random.default_rng(11)
    checked = 0
    # Depth 2 covers a same-player continuation (actions_left 2) and an
    # opponent reply (actions_left 1) without a tree the clone prover cannot finish.
    for actions_left, depth in ((1, 2), (2, 2)):
        for _ in range(3):
            board = blank_board()
            squares = [(r, c) for r in range(8) for c in range(8)]
            agree.shuffle(squares)
            kinds = [PieceType.KERNEL, PieceType.KERNEL, PieceType.ICE, PieceType.PING, PieceType.TRACE]
            teams = [constants.CYAN, constants.MAGENTA, constants.CYAN, constants.MAGENTA, constants.CYAN]
            for (r, c), kind, team in zip(squares, kinds, teams):
                board.grid[r][c] = Piece(kind, team)
            state = State(board, constants.CYAN, actions_left, 4)
            fast = dict(state.root_values(depth, budget_each=4000))
            env_pos = mate_env(state)
            slow = {}
            for action in env_pos.legal_actions():
                child = env_pos.clone()
                _, reward, done, _ = child.step(action)
                if done:
                    val = 1 if reward > 0 else -1
                elif child.current_team == env_pos.current_team:
                    val = _slow_value(child, depth - 1, [4000])
                else:
                    val = -_slow_value(child, depth - 1, [4000])
                slow[encoding.action_index(action, env_pos.current_team)] = val
            assert fast == slow, (fast, slow, actions_left, depth)
            checked += 1
    print(f"oracle self-test passed ({steps} actions, firewall included, {checked} proof agreements)", flush=True)


def mate_env(state: State) -> ColonyOneEnv:
    """Official environment standing on a fast state's position."""
    env = object.__new__(ColonyOneEnv)
    env.board = Board.from_observation(state.observation(), state.team)
    env.current_team = state.team
    env.actions_left = state.actions_left
    env.round_number = state.round_number
    env.terminated = False
    env.winner = None
    env.action = 1
    env._rng = np.random.default_rng(0)
    env._scene = state.observation()
    return env


def _slow_value(env: ColonyOneEnv, depth: int, budget: list[int]) -> int:
    if budget[0] <= 0 or depth <= 0 or env.terminated:
        return 0
    actions = env.legal_actions()
    if not actions:
        return 0
    team = env.current_team
    best = -1
    completed = True
    for action in actions:
        if budget[0] <= 0:
            completed = False
            break
        budget[0] -= 1
        child = env.clone()
        _, reward, done, _ = child.step(action)
        if done:
            val = 1 if reward > 0 else -1
        elif child.current_team == team:
            val = _slow_value(child, depth - 1, budget)
        else:
            val = -_slow_value(child, depth - 1, budget)
        if val > best:
            best = val
        if best == 1:
            return 1
    if not completed:
        return 1 if best == 1 else 0
    return best


def _actions_left(sample) -> int:
    return 2 if float(np.asarray(sample[0])[13, 0, 0]) > 0.75 else 1


def build_cache(path: Path, seed: int = 20261023) -> dict:
    """Label cached games and add exact endgame positions. Safe to resume from ``path``."""
    import gzip
    import pickle

    path = Path(path)
    if path.exists():
        with gzip.open(path, "rb") as handle:
            cached = pickle.load(handle)
        if cached.get("version") == VERSION:
            print(f"Reusing {path} ({cached['stats']})", flush=True)
            return cached

    self_test()
    t0 = time.monotonic()
    training_path = ROOT / "runs/recent_tournament/data/tournament_training.pkl.gz"
    replay_path = ROOT / "runs/league_v2/replay.pkl.gz"
    with gzip.open(training_path, "rb") as handle:
        training = pickle.load(handle)
    with gzip.open(replay_path, "rb") as handle:
        replay = pickle.load(handle)[-12_000:]
    print(f"Loaded {len(training)} tournament positions and {len(replay)} replay positions", flush=True)

    # Time a reply proof so the scan stays on the order of a few minutes.
    # Last-action positions are the ones that receive this budget.
    probe_row = next(row for row in replay if _actions_left(row) == 1)
    probe = from_observation(probe_row[0])
    probe.nodes = 0
    started = time.monotonic()
    probe.root_values(probe.actions_left + 2, budget_each=2500)
    elapsed = max(1e-6, time.monotonic() - started)
    rate = probe.nodes / elapsed
    print(f"Proof speed {rate:.0f} nodes/s ({probe.nodes} nodes in {elapsed:.2f}s)", flush=True)

    def label_split(rows, reply_quota: int, split: str):
        rng = np.random.default_rng(seed + (1 if split == "replay" else 0))
        reply_ids = set()
        losing = [i for i, row in enumerate(rows) if float(row[3]) < 0 and _actions_left(row) == 1]
        rng.shuffle(losing)
        # 2500 nodes is enough to exhaust a two-action reply at the usual branching factor.
        per = 2500
        # `rate` and `probe.nodes` are measured on one last-action reply.
        affordable = int(max(80, min(reply_quota, (rate * 120) / max(probe.nodes, 1))))
        reply_ids.update(losing[:affordable])
        print(f"{split}: reply proofs on {len(reply_ids)} losing last-actions", flush=True)
        out, hot, children, skips = [], [], [], 0
        wins = losses = partial = 0
        for i, row in enumerate(rows):
            reply = i in reply_ids
            labeled = label_record(row, reply, mate_budget=800, reply_budget=per)
            if labeled is None:
                skips += 1
                out.append(_sample_tuple(*row))
                continue
            packed, changed, extra = labeled
            out.append(packed)
            if changed:
                hot.append(len(out) - 1)
                if float(packed[3]) > 0 and float(row[3]) < 0:
                    wins += 1
                elif float(packed[3]) < 0 and float(row[3]) > 0:
                    losses += 1
                else:
                    partial += 1
            children.extend(extra)
            if (i + 1) % 1000 == 0:
                print(f"  {split} {i + 1}/{len(rows)} hot={len(hot)} skips={skips}", flush=True)
        return out, hot, children, dict(skips=skips, hot=len(hot), flipped_to_win=wins,
                                        flipped_to_loss=losses, policy_only=partial, reply=len(reply_ids))

    train_rows, train_hot, train_children, train_stats = label_split(training, 1200, "train")
    replay_rows, replay_hot, replay_children, replay_stats = label_split(replay, 1200, "replay")
    bare = generate_bare_kernels(limit=4000, seed=seed + 3)
    random_tactics = generate_random_tactics(24, seed + 4, mate_budget=400, reply_budget=800)
    synthetic = train_children + replay_children + bare + random_tactics
    rng = np.random.default_rng(seed + 5)
    rng.shuffle(synthetic)
    holdout_n = min(400, max(1, len(synthetic) // 8))
    holdout, synthetic = synthetic[:holdout_n], synthetic[holdout_n:]
    # Bare kernels are numerous. Keep them from crowding out tactical proofs.
    bare_cap = max(1500, len(train_children) + len(replay_children) + len(random_tactics))
    bare_kept = 0
    balanced = []
    for sample in synthetic:
        # Bare-kernel observations contain exactly two pieces.
        pieces = int(np.asarray(sample[0])[:12].sum())
        if pieces <= 2:
            if bare_kept >= bare_cap:
                continue
            bare_kept += 1
        balanced.append(sample)
    stats = dict(
        train=train_stats, replay=replay_stats, bare=len(bare), bare_kept=bare_kept,
        random=len(random_tactics), children=len(train_children) + len(replay_children),
        synthetic=len(balanced), holdout=len(holdout), seconds=time.monotonic() - t0,
        nodes_per_second=rate,
    )
    payload = dict(
        version=VERSION, train=train_rows, train_hot=train_hot, replay=replay_rows,
        replay_hot=replay_hot, synthetic=balanced, holdout=holdout, stats=stats,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {path} {stats}", flush=True)
    return payload


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=None)
    args = parser.parse_args()
    if args.build:
        build_cache(args.build)
    else:
        self_test()
