# Claude Code independent review

Requested by the user. A bounded code consultation with tools disabled, using pasted model/training/worker source and experiment results. These are independent recommendations, not confirmed improvements.

# Review of V2 pipeline: findings and next experiment

## Possible bugs (unproven; each needs a one-line check)

1. **Legal mask may equal the visited set** (`workers.py`, `legal=sorted(root.N)`).
   - If `mcts.run` only adds children to `root.N` once they are visited, zero-visit legal moves are missing from the mask.
   - The CE then never pushes their logits down, so priors on bad moves stay inflated. A 100-sim referee search suffers directly from that.
   - Check: `assert len(root.N)==len(env.legal_actions())` on a few positions. Also check the reanalysis and tournament relabel scripts, which likely share this pattern.
   - If confirmed, rebuild masks from `env.legal_actions()` with `pi=0` on unvisited moves. This is the highest-value correctness fix.

2. **Draw or timeout labels.** `1 if t==env.winner else -1` gives −1 to both sides if `winner` is ever `None`, for example at the 104-action cap.
   - The tie→Magenta rule probably prevents this, but count `winner is None` in the replays.

3. **Warmup-temperature value noise.** z labels come from games whose first 4 actions were sampled at T=1/1/.5/.25. Opening values are trained on outcomes of deliberately weak moves. This is a minor issue, not a bug.

4. **D4 augmentation validity.** All six attack kernels are D4-symmetric, so augmentation is fine *unless* the engine has directional movement or asymmetric spawn or ring semantics.
   - The failed D4 ensemble (105/200) is weak evidence that something isn't fully symmetric.
   - Check: for random positions, compare engine legal moves on the rotated board against mapped legal moves.

5. **Aux labels.** The flipped view keeps urgency plane 14 and forces actions-left plane 13 to 1. That is fine for "enemy's next turn" threats. The labels are D4-invariant, so pairing them with augmented `x` is correct. No bug here.

## Hypotheses (not bugs)

- Prior sharpness and value scale are probably mis-calibrated for **100-sim PUCT with 8-leaf batching**. Training CE matches 600-sim visit distributions, which are sharper than what 100 sims can exploit. Batched virtual loss also flattens selection.
- Your BCE, reanalysis and soft-policy runs all changed calibration indirectly. Their noisy ±10% swings fit "calibration matters, the targets don't".

## ONE best next experiment (test FIRST): zero-training calibration sweep of frozen V2

Leave V2 untouched. Write new files that append scalar ops to a **copy** of V2's graph:

```
policy' = policy * Tp          # Tp ∈ {0.7, 0.85, 1.2, 1.5}
value'  = tanh(Tv * atanh(clip(value, -0.999, 0.999)))   # Tv ∈ {0.7, 1.4}
```

Simpler option without atanh: rebuild from the V2 checkpoint in PyTorch. In `Model.forward`, return `Tp*(p+pair)` and `torch.tanh(Tv*(v+value_head))`, then run the existing `export()`. That path already passes `validate_submission_contract`, and Mul is surely allowed.

Protocol (about 15 min of CPU):

- **Stage A:** policy-only variants Tp ∈ {0.7, 0.85, 1.2, 1.5} with Tv=1, 150 games each vs V2 on a fresh seed range (about 5 min).
- **Stage B:** take the best Tp and cross it with Tv ∈ {0.7, 1.4}, 150 games each.
- **Confirmation:** the best variant plays 400 fresh games vs V2 plus 200 vs V1. Promote only if the Wilson lower bound exceeds 0.5 vs V2 and the V1 score is ≥ V2's 61%.

Why first:
- It needs no training and stays within the 70-minute budget.
- It directly targets the search distribution shift.
- It fails safe, since Tp=Tv=1 is V2.
- If the best variant lands at about 1.0, you've learned that calibration isn't the bottleneck and can drop that whole family of ideas.

## If time remains (second)

If bug #1 is confirmed, run a 300-step fine-tune from V2:
- Full legal masks.
- Optimizer and batch: AdamW lr 1e-4, wd 1e-4, batch 128, tournament fraction 0.65, aux 0.1 (direct).
- Plain MSE value loss, so the mask fix is the only change.
- Screen at steps 150 and 300 with 96 games vs V2, then confirm with 400 fresh games.

Stack it with the calibration winner only after each change passes on its own.

## Checks performed after the review

- The legal-mask concern was ruled out. `mcts._set_priors` initializes `N` and `W` for every legal action. Twelve reachable positions searched for one simulation retained all legal indices; the opening retained all30, of which29 had zero visits. Evidence: `runs/round3/claude_mask_check.json`.
- Recorded games have named winners and the environment resolves the firewall by round26; the 104-action guard raises instead of producing a draw sample. We did not find the proposed draw-label bug.
- D4 legal sets and transitions passed160 engine checks, including actual team identity and both action phases. A failed symmetry ensemble is not evidence that these spatial symmetries are invalid.
- The policy/value calibration proposal is being tested on frozen v2, rather than repeating the earlier v1-only scale screen. Four policy scales are screened at150games each, then two pre-tanh value scales, followed by independent selection and confirmation. All files are separate from the frozen model.
- The trained classification-value finalist independently reached58/96 against v2 and42/64 against the earlier model on a second screen. Its separate600game confirmation is required before promotion.
- Submission validation found a separate numerical issue: CPU ONNX `Tanh` returned1.000000119 on the single all-ones input. Explicit output clipping fixed this; the bounded finalist passed the exact supplied validator. This was our observed finding, not a bug claimed by Claude.
