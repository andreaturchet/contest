"""AlphaZero trainer for Colony: One Level 4.

The tournament wraps this network in the published MCTS (100 simulations in
Level 4) and accepts one ONNX file of at most 5 MiB. Self-play therefore uses
that same search, and the exported graph is checked against the submission
contract before it is written down as a candidate.
"""

from __future__ import annotations

import json
import math
import os
import queue
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from az_selfplay import apply_symmetry, play_one_game

NUM_ACTIONS = 4096
PRESETS = {
    # All three stay under the 5 MiB upload cap. "large" spends the budget.
    "small": {"channels": 48, "blocks": 4, "policy_hidden": 48},
    "medium": {"channels": 64, "blocks": 6, "policy_hidden": 64},
    "large": {"channels": 80, "blocks": 6, "policy_hidden": 64},
}


@dataclass
class TrainConfig:
    env_root: str
    scene_path: str
    run_dir: str
    drive_dir: str | None = None
    preset: str = "medium"
    channels: int | None = None
    blocks: int | None = None
    policy_hidden: int | None = None
    iterations: int = 80
    games_per_iteration: int = 16
    workers: int = 0
    simulations: int = 100
    sim_warmup: int = 40
    sim_ramp_iterations: int = 10
    mcts_batch_size: int = 8
    replay_size: int = 80_000
    batch_size: int = 1024
    train_steps: int = 0
    lr: float = 1e-3
    weight_decay: float = 1e-4
    value_weight: float = 1.0
    eval_every: int = 5
    eval_games: int = 4
    eval_simulations: int = 100
    export_every: int = 5
    sync_every: int = 5
    seed: int = 0
    resume: bool = True
    max_infer_batch: int = 256
    explore_temperatures: tuple = (1.0, 1.0, 0.5, 0.25)
    explore_step_actions: int = 8


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class PolicyValueNet(nn.Module):
    """Convolutional policy/value net whose ONNX export fits the 5 MiB cap.

    The policy tower is a 1x1 projection, a narrow hidden layer, then a dense
    map onto all 4096 ``(from, to)`` logits. A full 512-by-4096 layer alone
    would blow the upload limit, so the hidden width is the size knob.
    """

    def __init__(self, channels: int = 64, blocks: int = 6, policy_hidden: int = 64) -> None:
        super().__init__()
        self.channels = channels
        self.blocks = blocks
        self.policy_hidden = policy_hidden
        self.stem = nn.Sequential(
            nn.Conv2d(16, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.tower = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        self.policy_conv = nn.Conv2d(channels, 8, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(8)
        self.policy_fc1 = nn.Linear(8 * 64, policy_hidden)
        self.policy_fc2 = nn.Linear(policy_hidden, NUM_ACTIONS)
        self.value_conv = nn.Conv2d(channels, 4, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(4)
        self.value_fc1 = nn.Linear(4 * 64, 64)
        self.value_fc2 = nn.Linear(64, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Near-uniform initial policy and a neutral value, so the first
        # searches follow visit counts instead of a random logit spike.
        nn.init.zeros_(self.policy_fc2.weight)
        nn.init.zeros_(self.policy_fc2.bias)
        nn.init.zeros_(self.value_fc2.weight)
        nn.init.zeros_(self.value_fc2.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.tower(self.stem(x))
        p = F.relu(self.policy_bn(self.policy_conv(h))).flatten(1)
        policy = self.policy_fc2(F.relu(self.policy_fc1(p)))
        v = F.relu(self.value_bn(self.value_conv(h))).flatten(1)
        value = torch.tanh(self.value_fc2(F.relu(self.value_fc1(v)))).squeeze(-1)
        return policy, value


def architecture(cfg: TrainConfig) -> dict:
    if cfg.preset not in PRESETS:
        raise ValueError(f"preset must be one of {sorted(PRESETS)}, got {cfg.preset!r}")
    arch = dict(PRESETS[cfg.preset])
    if cfg.channels is not None:
        arch["channels"] = cfg.channels
    if cfg.blocks is not None:
        arch["blocks"] = cfg.blocks
    if cfg.policy_hidden is not None:
        arch["policy_hidden"] = cfg.policy_hidden
    return arch


def load_scene(path: str | Path) -> np.ndarray:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        scene = np.asarray(json.load(handle), dtype=np.float32)
    if scene.shape != (16, 8, 8):
        raise ValueError(f"{path} has shape {scene.shape}, expected (16, 8, 8)")
    return scene


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class Replay:
    def __init__(self, capacity: int, rng: np.random.Generator) -> None:
        self.capacity = capacity
        self.rng = rng
        self.items: list = []
        self.pos = 0

    def __len__(self) -> int:
        return len(self.items)

    def add(self, samples) -> None:
        for sample in samples:
            if len(self.items) < self.capacity:
                self.items.append(sample)
            else:
                self.items[self.pos] = sample
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        replace = len(self.items) < batch_size
        idxs = self.rng.choice(len(self.items), size=batch_size, replace=replace)
        out = []
        for idx in idxs:
            k = int(self.rng.integers(0, 4))
            mirror = bool(self.rng.integers(0, 2))
            out.append(apply_symmetry(self.items[int(idx)], k, mirror))
        return out


class TorchEvaluator:
    """In-process evaluator. One game, batches of leaves, on ``device``."""

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device

    def __call__(self, planes: np.ndarray):
        x = torch.from_numpy(np.ascontiguousarray(planes, dtype=np.float32))
        x = x.to(self.device, non_blocking=self.device.type == "cuda")
        if self.device.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.inference_mode():
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.device.type == "cuda",
            ):
                policy, value = self.model(x)
        return policy.float().cpu().numpy(), value.float().cpu().numpy()


def _forward_numpy(model, planes: np.ndarray, device: torch.device):
    x = torch.from_numpy(planes).to(device, non_blocking=device.type == "cuda")
    if device.type == "cuda":
        x = x.contiguous(memory_format=torch.channels_last)
    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            policy, value = model(x)
    return policy.float().cpu().numpy(), value.float().cpu().numpy()


def generate_games(model, scene, n_games, mcts_kwargs, cfg: TrainConfig, device, iteration: int):
    """Self-play ``n_games``. One worker stays in-process; more workers share one GPU."""
    workers = cfg.workers
    if workers <= 1 or n_games <= 1:
        return _generate_inplace(model, scene, n_games, mcts_kwargs, device, cfg.seed, iteration)

    try:
        return _generate_parallel(
            model, scene, n_games, mcts_kwargs, cfg, device, iteration, workers
        )
    except Exception as exc:
        print(f"parallel self-play failed ({exc}); continuing in one process", flush=True)
        return _generate_inplace(
            model, scene, n_games, mcts_kwargs, device, cfg.seed, iteration
        )


def _generate_parallel(model, scene, n_games, mcts_kwargs, cfg, device, iteration, workers):
    import multiprocessing as mp

    from az_selfplay import worker_entry

    _drain_games.games = 0
    ctx = mp.get_context("spawn")
    counts = [n_games // workers] * workers
    for i in range(n_games % workers):
        counts[i] += 1
    counts = [c for c in counts if c > 0]
    req_q = ctx.Queue()
    out_q = ctx.Queue()
    resp_qs = [ctx.Queue() for _ in counts]
    procs = []
    for i, count in enumerate(counts):
        proc = ctx.Process(
            target=worker_entry,
            args=(
                i,
                count,
                scene,
                mcts_kwargs,
                cfg.env_root,
                req_q,
                resp_qs[i],
                out_q,
                cfg.seed + iteration * 1_000_003 + i * 97_001,
            ),
            daemon=True,
        )
        proc.start()
        procs.append(proc)

    model.eval()
    pending = len(procs)
    samples = []
    games = 0
    last_report = time.perf_counter()
    try:
        while pending > 0:
            try:
                item = req_q.get(timeout=5)
            except queue.Empty:
                dead = [p for p in procs if not p.is_alive() and p.exitcode not in (0, None)]
                if dead:
                    _drain_errors(out_q)
                    raise RuntimeError(
                        f"self-play worker exited with code {dead[0].exitcode}"
                    )
                if time.perf_counter() - last_report > 30:
                    print(f"    still searching ({games}/{n_games} games finished)", flush=True)
                    last_report = time.perf_counter()
                continue

            batch = [item]
            while len(batch) < cfg.max_infer_batch:
                try:
                    batch.append(req_q.get_nowait())
                except queue.Empty:
                    break

            evals = []
            for entry in batch:
                kind = entry[0]
                if kind == "bye":
                    pending -= 1
                elif kind == "eval":
                    evals.append(entry)
                else:
                    raise RuntimeError(f"unknown worker message {kind!r}")
            if evals:
                _serve_evals(model, evals, resp_qs, device)
            samples.extend(_drain_games(out_q, n_games))
            games = _count_marked(samples, games)
            last_report = time.perf_counter()
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
        for proc in procs:
            proc.join(timeout=10)

    samples.extend(_drain_games(out_q, n_games))
    return samples


def _count_marked(samples, games):
    # ``_drain_games`` returns a flat sample list. The caller tracks games via
    # a side channel stored on the function. See ``_drain_games``.
    return getattr(_drain_games, "games", games)


def _drain_errors(out_q) -> None:
    while True:
        try:
            kind, *rest = out_q.get_nowait()
        except queue.Empty:
            return
        if kind == "error":
            print(rest[-1], flush=True)


def _drain_games(out_q, n_games: int):
    found = []
    while True:
        try:
            msg = out_q.get_nowait()
        except queue.Empty:
            break
        kind = msg[0]
        if kind == "error":
            raise RuntimeError(f"self-play worker failed:\n{msg[2]}")
        if kind == "game":
            _drain_games.games = getattr(_drain_games, "games", 0) + 1
            found.extend(msg[3])
            print(
                f"    game {_drain_games.games}/{n_games}  {len(msg[3])} positions",
                flush=True,
            )
        elif kind == "done":
            continue
    return found


def _serve_evals(model, evals, resp_qs, device) -> None:
    planes = [entry[3] for entry in evals]
    sizes = [p.shape[0] for p in planes]
    cat = np.concatenate(planes, axis=0)
    logits, values = _forward_numpy(model, cat, device)
    offset = 0
    for (_, worker_id, rid, _), size in zip(evals, sizes, strict=True):
        resp_qs[worker_id].put(
            (
                rid,
                np.ascontiguousarray(logits[offset : offset + size]),
                np.ascontiguousarray(values[offset : offset + size]),
            )
        )
        offset += size


def _generate_inplace(model, scene, n_games, mcts_kwargs, device, seed, iteration):
    from colony_one import mcts

    model.eval()
    evaluator = TorchEvaluator(model, device)
    samples = []
    rng = np.random.default_rng(seed + iteration * 1_000_003)
    cfg = mcts.MCTSConfig(**mcts_kwargs)
    for game_i in range(n_games):
        played = play_one_game(evaluator, scene, cfg, rng)
        samples.extend(played)
        print(f"    game {game_i + 1}/{n_games}  {len(played)} positions", flush=True)
    return samples


def collate(samples, device: torch.device):
    batch = len(samples)
    obs = np.empty((batch, 16, 8, 8), dtype=np.float32)
    pi = np.zeros((batch, NUM_ACTIONS), dtype=np.float32)
    legal = np.zeros((batch, NUM_ACTIONS), dtype=bool)
    z = np.empty((batch,), dtype=np.float32)
    for i, (obs_i, legal_i, pi_i, z_i) in enumerate(samples):
        obs[i] = obs_i
        idx = legal_i.astype(np.int64, copy=False)
        if len(idx) == 0:
            raise RuntimeError("sample has no legal actions")
        legal[i, idx] = True
        probs = pi_i.astype(np.float32, copy=False)
        total = float(probs.sum())
        if total <= 0:
            pi[i, idx] = 1.0 / len(idx)
        else:
            pi[i, idx] = probs / total
        z[i] = z_i
    obs_t = torch.from_numpy(obs).to(device, non_blocking=device.type == "cuda")
    if device.type == "cuda":
        obs_t = obs_t.contiguous(memory_format=torch.channels_last)
    return (
        obs_t,
        torch.from_numpy(pi).to(device),
        torch.from_numpy(legal).to(device),
        torch.from_numpy(z).to(device),
    )


def policy_value_loss(policy, value, pi, legal, z, value_weight: float):
    # Mask in fp32. A large negative in fp16 overflows and poisons the softmax.
    logits = policy.float().masked_fill(~legal, -1e9)
    log_p = F.log_softmax(logits, dim=-1)
    policy_loss = -(pi * log_p).sum(dim=-1).mean()
    value_loss = F.mse_loss(value.float(), z)
    loss = policy_loss + value_weight * value_loss
    with torch.no_grad():
        acc = ((value.float() > 0) == (z > 0)).float().mean()
    return loss, policy_loss.detach(), value_loss.detach(), acc.detach()


def make_scaler(device: torch.device, enabled: bool):
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_on_replay(model, opt, scaler, replay: Replay, cfg: TrainConfig, device, use_amp: bool):
    steps = cfg.train_steps
    if steps <= 0:
        steps = int(8 * len(replay) / max(1, cfg.batch_size))
        steps = max(40, min(400, steps))
    model.train()
    totals = np.zeros(4, dtype=np.float64)
    for _ in range(steps):
        obs, pi, legal, z = collate(replay.sample(cfg.batch_size), device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            policy, value = model(obs)
        loss, p_loss, v_loss, acc = policy_value_loss(
            policy, value, pi, legal, z, cfg.value_weight
        )
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        totals += (
            float(loss.detach()),
            float(p_loss),
            float(v_loss),
            float(acc),
        )
    totals /= steps
    return {
        "steps": steps,
        "loss": totals[0],
        "policy": totals[1],
        "value": totals[2],
        "value_acc": totals[3],
    }


def learning_rate(iteration: int, total: int, base: float) -> float:
    warm = 2
    if iteration <= warm:
        return base * iteration / warm
    progress = (iteration - warm) / max(1, total - warm)
    return base * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def simulations_for(iteration: int, cfg: TrainConfig) -> int:
    if cfg.sim_ramp_iterations <= 1:
        return cfg.simulations
    t = min(1.0, (iteration - 1) / (cfg.sim_ramp_iterations - 1))
    return int(round(cfg.sim_warmup + t * (cfg.simulations - cfg.sim_warmup)))


def mcts_kwargs_for(sims: int, cfg: TrainConfig) -> dict:
    return {
        "simulations": int(sims),
        "c_puct": 1.5,
        "add_root_noise": True,
        "dirichlet_alpha": 0.3,
        "dirichlet_eps": 0.15,
        "warmup_temperatures": tuple(cfg.explore_temperatures),
        "warmup_step_actions": int(cfg.explore_step_actions),
        "batch_size": int(cfg.mcts_batch_size),
        "virtual_loss": 1.0,
        "adaptive_simulations": False,
    }


def export_onnx(model: nn.Module, path: Path) -> Path:
    """Write a self-contained opset-17 file with a dynamic batch axis."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for sidecar in (Path(str(path) + ".data"), path.with_name(path.name + ".data")):
        if sidecar.exists():
            sidecar.unlink()

    export_model = PolicyValueNet(
        model.channels, model.blocks, model.policy_hidden
    ).cpu().eval()
    export_model.load_state_dict(model.state_dict())
    dummy = torch.zeros(2, 16, 8, 8)
    kwargs = dict(
        input_names=["observation"],
        output_names=["policy", "value"],
        dynamic_axes={
            "observation": {0: "batch"},
            "policy": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL,
        dynamo=False,
        external_data=False,
    )
    try:
        torch.onnx.export(export_model, (dummy,), str(path), **kwargs)
    except TypeError:
        kwargs.pop("dynamo", None)
        kwargs.pop("external_data", None)
        torch.onnx.export(export_model, dummy, str(path), **kwargs)

    if Path(str(path) + ".data").exists() or path.with_name(path.name + ".data").exists():
        raise RuntimeError(f"{path} wrote an external-data sidecar; the referee rejects those")
    _promote_constants(path)
    return path


def _promote_constants(path: Path) -> None:
    """Turn ``Constant`` nodes into initializers.

    The exporter emits ``Squeeze`` axes as a ``Constant`` node. The submission
    allow-list accepts an initializer in that slot and rejects the operator.
    """
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(path))
    kept = []
    promoted = 0
    for node in model.graph.node:
        if node.op_type != "Constant":
            kept.append(node)
            continue
        value = _constant_value(node)
        model.graph.initializer.append(numpy_helper.from_array(value, name=node.output[0]))
        promoted += 1
    if promoted:
        del model.graph.node[:]
        model.graph.node.extend(kept)
        onnx.save(model, str(path))


def _constant_value(node) -> np.ndarray:
    for attr in node.attribute:
        if attr.name == "value":
            from onnx import numpy_helper
            return np.asarray(numpy_helper.to_array(attr.t))
        if attr.name == "value_int":
            return np.asarray(attr.i, dtype=np.int64)
        if attr.name == "value_ints":
            return np.asarray(list(attr.ints), dtype=np.int64)
        if attr.name == "value_float":
            return np.asarray(attr.f, dtype=np.float32)
        if attr.name == "value_floats":
            return np.asarray(list(attr.floats), dtype=np.float32)
    raise RuntimeError(f"Constant node {node.name or node.output} has no value attribute")


def contract_check(path: Path) -> str:
    from colony_one.submission_policy import ModelRejected, inspect_and_dry_run_model

    try:
        inspect_and_dry_run_model(path)
    except ModelRejected as exc:
        return f"REJECTED: {exc}"
    size = path.stat().st_size
    return f"ACCEPTED {size / (1024 * 1024):.2f} MiB"


def evaluate_vs_greedy(model, scene, games: int, sims: int, device, seed: int) -> float:
    from colony_one import constants, mcts
    from colony_one.baselines import GreedyAgent
    from colony_one.env import ColonyOneEnv

    model.eval()
    evaluator = TorchEvaluator(model, device)
    wins = 0
    for game_i in range(games):
        model_is_cyan = game_i % 2 == 0
        env = ColonyOneEnv(seed=seed + game_i, scene=scene)
        cfg = mcts.MCTSConfig(simulations=sims)
        greedy = GreedyAgent(seed=seed + game_i)
        rng = np.random.default_rng(seed + game_i)
        while not env.terminated:
            model_to_move = (env.current_team == constants.CYAN) == model_is_cyan
            if model_to_move:
                action = mcts.search_action(
                    env.board,
                    env.current_team,
                    env.actions_left,
                    env.round_number,
                    evaluator,
                    cfg,
                    rng=rng,
                )
            else:
                action = greedy.select(
                    env.board, env.current_team, env.actions_left, env.round_number
                )
            if action is None:
                break
            env.step(action)
        model_team = constants.CYAN if model_is_cyan else constants.MAGENTA
        wins += int(env.winner == model_team)
        print(f"    eval game {game_i + 1}/{games}  winner {env.winner}", flush=True)
    return wins / max(1, games)


def _set_lr(opt, lr: float) -> None:
    for group in opt.param_groups:
        group["lr"] = lr


def _checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _sync(src: Path, drive_dir: str | None) -> None:
    if not drive_dir or not src.exists():
        return
    dest_root = Path(drive_dir)
    dest_root.mkdir(parents=True, exist_ok=True)
    dest = dest_root / src.name
    shutil.copy2(src, dest)


def resolve_workers(requested: int, games: int) -> int:
    if requested > 0:
        return max(1, min(requested, games, 8))
    cpu = os.cpu_count() or 2
    if cpu <= 2:
        return 1
    return max(1, min(8, cpu - 1, games))


def run_training(cfg: TrainConfig) -> Path:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    run_dir = Path(cfg.run_dir)
    onnx_dir = run_dir / "onnx"
    ckpt_dir = run_dir / "checkpoints"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    scene = load_scene(cfg.scene_path)
    arch = architecture(cfg)
    model = PolicyValueNet(**arch)
    params = count_parameters(model)
    print(
        f"device {device}"
        + (f"  {torch.cuda.get_device_name(0)}" if device.type == "cuda" else "  (no GPU)"),
        flush=True,
    )
    print(
        f"net {cfg.preset}  channels {arch['channels']}  blocks {arch['blocks']}  "
        f"policy_hidden {arch['policy_hidden']}  params {params:,}  "
        f"~{params * 4 / (1024 * 1024):.2f} MiB of weights",
        flush=True,
    )
    if params * 4 > 4_800_000:
        raise RuntimeError(
            "This architecture is too close to the 5 MiB upload cap. "
            "Use the medium preset or a smaller policy_hidden."
        )

    model.to(device)
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = make_scaler(device, use_amp)
    replay = Replay(cfg.replay_size, np.random.default_rng(cfg.seed + 17))
    start = 1
    best_wr = -1.0
    latest_pt = ckpt_dir / "latest.pt"

    weights_pt = ckpt_dir / "weights.pt"
    if cfg.resume and (latest_pt.exists() or weights_pt.exists()):
        source = latest_pt if latest_pt.exists() else weights_pt
        try:
            saved = torch.load(source, map_location=device, weights_only=False)
        except TypeError:
            saved = torch.load(source, map_location=device)
        if saved.get("arch") != arch:
            print("checkpoint architecture differs from this run; starting fresh", flush=True)
        else:
            model.load_state_dict(saved["model"])
            if saved.get("optimizer"):
                opt.load_state_dict(saved["optimizer"])
            if saved.get("scaler") and use_amp:
                scaler.load_state_dict(saved["scaler"])
            replay.items = saved.get("replay", [])
            replay.pos = int(saved.get("replay_pos", 0))
            best_wr = float(saved.get("best_wr", -1.0))
            start = int(saved["iteration"]) + 1
            print(
                f"resumed from {source.name} at iteration {start}  "
                f"buffer {len(replay):,}  best vs greedy {best_wr:.0%}",
                flush=True,
            )

    cfg.workers = resolve_workers(cfg.workers, cfg.games_per_iteration)
    print(
        f"workers {cfg.workers}  games/iter {cfg.games_per_iteration}  "
        f"sims {cfg.sim_warmup}->{cfg.simulations}  batch {cfg.batch_size}",
        flush=True,
    )

    preflight = onnx_dir / "preflight.onnx"
    export_onnx(model, preflight)
    verdict = contract_check(preflight)
    print(f"export preflight {verdict}", flush=True)
    if not verdict.startswith("ACCEPTED"):
        raise RuntimeError("ONNX export failed the submission contract before training started")

    log_path = run_dir / "train_log.jsonl"
    if start > cfg.iterations:
        print(f"checkpoint already reached iteration {cfg.iterations}", flush=True)
        return onnx_dir / "best.onnx" if (onnx_dir / "best.onnx").exists() else onnx_dir / "latest.onnx"

    try:
        for iteration in range(start, cfg.iterations + 1):
            lr = learning_rate(iteration, cfg.iterations, cfg.lr)
            _set_lr(opt, lr)
            sims = simulations_for(iteration, cfg)
            print(
                f"\niter {iteration}/{cfg.iterations}  sims {sims}  lr {lr:.2e}  "
                f"buffer {len(replay):,}",
                flush=True,
            )
            _drain_games.games = 0
            t0 = time.perf_counter()
            samples = generate_games(
                model, scene, cfg.games_per_iteration,
                mcts_kwargs_for(sims, cfg), cfg, device, iteration,
            )
            self_play_s = time.perf_counter() - t0
            if not samples:
                print("    no samples produced; skipping the update", flush=True)
                continue
            mean_z = float(np.mean([float(z) for *_, z in samples]))
            replay.add(samples)
            t1 = time.perf_counter()
            stats = train_on_replay(model, opt, scaler, replay, cfg, device, use_amp)
            train_s = time.perf_counter() - t1
            print(
                f"    positions {len(samples)}  mean z {mean_z:+.2f}  "
                f"self-play {self_play_s:.0f}s  train {train_s:.0f}s",
                flush=True,
            )
            print(
                f"    loss {stats['loss']:.3f}  policy {stats['policy']:.3f}  "
                f"value {stats['value']:.3f}  value-acc {stats['value_acc']:.2f}  "
                f"steps {stats['steps']}",
                flush=True,
            )

            record = {
                "iteration": iteration,
                "sims": sims,
                "lr": lr,
                "positions": len(samples),
                "mean_z": mean_z,
                "buffer": len(replay),
                "self_play_s": self_play_s,
                "train_s": train_s,
                **stats,
            }

            do_export = iteration % cfg.export_every == 0 or iteration == cfg.iterations
            do_eval = cfg.eval_games > 0 and (
                iteration % cfg.eval_every == 0 or iteration == cfg.iterations
            )
            if do_export:
                latest = onnx_dir / "latest.onnx"
                export_onnx(model, latest)
                verdict = contract_check(latest)
                record["onnx"] = verdict
                print(f"    onnx {verdict}", flush=True)
                if verdict.startswith("ACCEPTED"):
                    _sync(latest, cfg.drive_dir)
                    snap = onnx_dir / f"iter_{iteration:04d}.onnx"
                    shutil.copy2(latest, snap)

            if do_eval:
                print("    evaluating vs GreedyAgent", flush=True)
                wr = evaluate_vs_greedy(
                    model, scene, cfg.eval_games, cfg.eval_simulations,
                    device, cfg.seed + 50_000 + iteration,
                )
                record["greedy_winrate"] = wr
                print(f"    vs greedy {wr:.0%}  (best {best_wr:.0%})", flush=True)
                if wr >= best_wr:
                    best_wr = wr
                    best_pt = ckpt_dir / "best.pt"
                    torch.save({"model": model.state_dict(), "arch": arch, "winrate": wr}, best_pt)
                    best_onnx = onnx_dir / "best.onnx"
                    export_onnx(model, best_onnx)
                    verdict = contract_check(best_onnx)
                    print(f"    new best  {verdict}", flush=True)
                    if verdict.startswith("ACCEPTED"):
                        _sync(best_onnx, cfg.drive_dir)

            payload = {
                "iteration": iteration,
                "arch": arch,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict() if use_amp else None,
                "replay": replay.items,
                "replay_pos": replay.pos,
                "best_wr": best_wr,
            }
            _checkpoint(latest_pt, payload)
            weights_payload = {
                "iteration": iteration,
                "arch": arch,
                "model": payload["model"],
                "optimizer": payload["optimizer"],
                "best_wr": best_wr,
            }
            _checkpoint(weights_pt, weights_payload)
            _sync(weights_pt, cfg.drive_dir)
            if iteration % cfg.sync_every == 0:
                _sync(latest_pt, cfg.drive_dir)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
    except KeyboardInterrupt:
        print("\ninterrupted; latest checkpoint is already saved after the last finished iteration", flush=True)

    best = onnx_dir / "best.onnx"
    latest = onnx_dir / "latest.onnx"
    chosen = best if best.exists() else latest
    print(f"\ndone. submission candidate: {chosen}", flush=True)
    return chosen
