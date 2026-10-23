"""Fine-tune candidate v2 on exact tactical and endgame proofs.

The exported network is unchanged: same residual trunk, same ONNX contract.
Only the targets move. Proved wins and losses replace visit counts and game
outcomes. Positions the proof does not settle keep the targets candidate v2
was trained on, with a KL penalty toward v2 so the opening does not drift.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from arena import ROOT, match
from oracle import build_cache
from workers import augment


def _batch_loss(net, ref, batch, anchor, device, value_only=False):
    import torch
    from torch.nn import functional as F

    obs = torch.from_numpy(np.stack([row[0] for row in batch]).astype(np.float32)).to(device)
    width = max(len(row[1]) for row in batch)
    idx = np.zeros((len(batch), width), np.int64)
    pi = np.zeros((len(batch), width), np.float32)
    mask = np.zeros((len(batch), width), bool)
    values = np.zeros(len(batch), np.float32)
    anchors = np.zeros(len(batch), np.float32)
    for i, row in enumerate(batch):
        legal, prob, z = row[1], row[2], row[3]
        idx[i, :len(legal)] = legal
        pi[i, :len(prob)] = prob
        mask[i, :len(legal)] = True
        values[i] = z
        anchors[i] = anchor[i]
    pi_t = torch.from_numpy(pi).to(device)
    pi_t = pi_t / pi_t.sum(1, keepdim=True).clamp_min(1e-8)
    index = torch.from_numpy(idx).to(device)
    valid = torch.from_numpy(mask).to(device)
    logits, pred = net(obs)
    legal_logits = logits.gather(1, index).masked_fill(~valid, -1e9)
    with torch.no_grad():
        ref_logits, ref_value = ref(obs)
    proved = torch.from_numpy(values).to(device)
    if value_only:
        # Replay rows keep v2's value. Proved rows (anchor 0) use the exact result.
        weight = torch.from_numpy(anchors).to(device)
        target = torch.where(weight > 0, ref_value, proved)
        value = F.mse_loss(pred, target)
        return pred.new_zeros(()), value, pred.new_zeros(())
    policy = -(pi_t * F.log_softmax(legal_logits, 1)).sum(1).mean()
    value = F.mse_loss(pred, proved)
    teacher = torch.softmax(ref_logits.gather(1, index).masked_fill(~valid, -1e9), 1)
    student = F.log_softmax(legal_logits, 1)
    kl = (teacher * (teacher.clamp_min(1e-8).log() - student)).sum(1)
    weight = torch.from_numpy(anchors).to(device)
    anchored = (kl * weight).mean() + 0.05 * ((pred - ref_value).square() * weight).mean()
    return policy, value, anchored


def _holdout(net, rows, device):
    import torch
    net.eval()
    policy_hits = policy_n = value_hits = 0
    with torch.no_grad():
        for start in range(0, len(rows), 64):
            batch = rows[start:start + 64]
            obs = torch.from_numpy(np.stack([row[0] for row in batch]).astype(np.float32)).to(device)
            logits, values = net(obs)
            logits = logits.cpu().numpy()
            values = values.cpu().numpy()
            for row, logit, value in zip(batch, logits, values):
                legal = np.asarray(row[1], np.int64)
                pi = np.asarray(row[2], np.float32)
                z = float(row[3])
                choice = legal[int(np.argmax(logit[legal]))]
                spot = int(np.where(legal == choice)[0][0])
                if z > 0 and pi.sum() > 0:
                    policy_n += 1
                    policy_hits += int(pi[spot] > 0)
                value_hits += int((value > 0) == (z > 0))
    return dict(
        policy_hit_rate=(policy_hits / policy_n) if policy_n else None,
        policy_n=policy_n,
        value_sign_rate=value_hits / max(1, len(rows)),
        n=len(rows),
    )


def main():
    import torch
    from model import Model, export

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--oracle-fraction", type=float, default=0.30)
    parser.add_argument("--hot-rate", type=float, default=0.55)
    parser.add_argument("--kl", type=float, default=0.25)
    parser.add_argument("--screen-every", type=int, default=100)
    parser.add_argument("--value-only", action="store_true",
                        help="Freeze the policy and train the value head on proved outcomes")
    parser.add_argument("--screen-games", type=int, default=40)
    parser.add_argument("--final-games", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/oracle_v3")
    args = parser.parse_args()
    args.run.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    data = build_cache(args.run / "labels.pkl.gz", seed=args.seed)
    state = torch.load(ROOT / "runs/recent_auxiliary/adapted_weights.pt", map_location="cpu", weights_only=False)
    net = Model(**state["architecture"])
    net.load_state_dict(state["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    ref = Model(**state["architecture"]).to(device)
    ref.load_state_dict(state["model"])
    ref.eval()
    for parameter in ref.parameters():
        parameter.requires_grad_(False)
    if args.value_only:
        # MCTS reads the value at unevaluated leaves. Keep v2's policy and move
        # only the value head toward proved wins and losses.
        for name, parameter in net.named_parameters():
            parameter.requires_grad_(name.startswith("value"))
    opt = torch.optim.AdamW((p for p in net.parameters() if p.requires_grad), lr=args.lr, weight_decay=1e-4)
    incumbent = ROOT / "candidate_v2.onnx"
    baseline = ROOT / "baselines/current.onnx"
    config = dict(
        method="exact short-horizon proofs and bare-kernel retrograde labels",
        sources=[
            "Winands, Bjornsson, Saito 2008 MCTS-Solver",
            "Danihelka, Guez, Schrittwieser, Silver 2022 Gumbel AlphaZero",
            "Ruoss et al. 2024 ChessBench action-value distillation",
            "Thompson 1986 retrograde endgame analysis",
        ],
        initial_weights="runs/recent_auxiliary/adapted_weights.pt",
        opponent=str(incumbent),
        steps=args.steps, batch=args.batch, lr=args.lr,
        oracle_fraction=args.oracle_fraction, seed=args.seed,
        labels=data["stats"], architecture=state["architecture"],
    )
    (args.run / "run_config.json").write_text(json.dumps(config, indent=2))
    print("Holdout before training", json.dumps(_holdout(net, data["holdout"], device)), flush=True)

    train, train_hot = data["train"], set(data["train_hot"])
    replay, replay_hot = data["replay"], set(data["replay_hot"])
    synthetic = data["synthetic"]
    best_score = -1
    history = []
    started = time.monotonic()
    best = args.run / "best.onnx"

    def screen(path, step):
        torch.set_num_threads(1)
        versus = match(path, incumbent, args.screen_games, seed=330_000_000 + step, verbose=False)
        heuristic = match(path, baseline, max(2, args.screen_games // 2), seed=340_000_000 + step, verbose=False)
        torch.set_num_threads(4)
        return versus, heuristic

    for step in range(1, args.steps + 1):
        batch, anchor = [], []
        for _ in range(args.batch):
            if synthetic and rng.random() < args.oracle_fraction:
                batch.append(augment(synthetic[int(rng.integers(len(synthetic)))], rng))
                anchor.append(0.0)
                continue
            use_replay = rng.random() < 0.5
            pool, hot = (replay, replay_hot) if use_replay else (train, train_hot)
            if hot and rng.random() < args.hot_rate:
                index = int(rng.choice(np.fromiter(hot, dtype=np.int64)))
            else:
                index = int(rng.integers(len(pool)))
            batch.append(augment(pool[index], rng))
            # Value-only rows from real games are anchored to v2's value.
            # Policy runs anchor everything the proof did not touch.
            if args.value_only:
                anchor.append(1.0)
            else:
                anchor.append(0.0 if index in hot else args.kl)
        net.train()
        opt.zero_grad(set_to_none=True)
        policy, value, anchored = _batch_loss(net, ref, batch, anchor, device, value_only=args.value_only)
        total = policy + value + anchored
        total.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        checkpoint_due = step == args.steps or step % 100 == 0
        if checkpoint_due:
            net.eval()
            hold = _holdout(net, data["holdout"], device)
            path = export(net, args.run / f"step_{step}.onnx")
            net.to(device)
            row = dict(
                step=step, holdout=hold,
                policy_loss=float(policy.detach()), value_loss=float(value.detach()),
                anchor_loss=float(anchored.detach()),
                seconds=time.monotonic() - started,
            )
            if args.screen_every:
                versus, heuristic = screen(path, step)
                row.update(versus_v2=versus["wins"], screen_games=versus["games"],
                           v2_wilson=versus["wilson_95"], heuristic_wins=heuristic["wins"],
                           heuristic_games=heuristic["games"])
                better = versus["wins"] > best_score and heuristic["wins"] * 2 >= heuristic["games"]
            else:
                # Value-only runs are chosen on proved-position value accuracy.
                score = hold["value_sign_rate"]
                better = score > best_score
                if better:
                    best_score = score
            if args.screen_every and better:
                best_score = versus["wins"]
            if better or not best.exists():
                shutil.copyfile(path, best)
                torch.save(dict(architecture=state["architecture"], model={k: v.detach().cpu() for k, v in net.state_dict().items()}), args.run / "adapted_weights.pt")
                row["selected"] = True
            history.append(row)
            (args.run / "history.json").write_text(json.dumps(history, indent=2))
            print(json.dumps(row), flush=True)

    if not best.exists():
        print("No checkpoint beat the screen gate. Keeping candidate v2.", flush=True)
        return
    final_games = args.final_games
    torch.set_num_threads(1)
    final = match(best, incumbent, final_games, seed=380_000_011, output=args.run / "final_match", verbose=False)
    heuristic = match(best, baseline, 80, seed=390_000_017, verbose=False)
    final["heuristic"] = heuristic
    final["promotion_passed"] = final["wilson_95"][0] > 0.5
    (args.run / "final_result.json").write_text(json.dumps(final, indent=2))
    print(json.dumps(final, indent=2), flush=True)
    if final["promotion_passed"]:
        shutil.copyfile(best, args.run / "submission.onnx")
        shutil.copyfile(best, ROOT / "candidate_v3.onnx")
        print(f"Promoted {args.run / 'submission.onnx'} and candidate_v3.onnx", flush=True)
    else:
        print("No established win over candidate v2. candidate_v2.onnx was left unchanged.", flush=True)


if __name__ == "__main__":
    main()
