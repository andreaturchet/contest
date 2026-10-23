# Level 4: second literature review and experiments

Researched 23 October 2026. This note distinguishes published evidence from proposals for Colony: One. It does not claim that any untested variant is stronger. Work and artifacts remain inside `level-4-lab`.

## What the present evidence says

The incumbent in [RESULTS.md](RESULTS.md) beats the original submission 140/200 and the sampled hybrid 72/100, but scored 46/100 against the earlier self-play student. The latter is inconclusive and makes that student an essential opponent for selection. We should preserve the incumbent by checksum and require a direct incumbent comparison before promotion.

The existing [model](model.py) has a small learned residual trunk, hand-built tactical features, raw pair-action logits, and a bounded value. Its spatial convolutions share weights, but neither the dense value head nor learned filters guarantee rotation/reflection consistency. Training already applies spatial augmentation through [workers.py](workers.py). Augmentation does not enforce exact consistency at inference.

The actual [referee](../level-3/env/colony_one/mcts.py) normalizes logits over legal actions only and uses `Q + c_puct * P * sqrt(total + 1) / (1 + N)`. It mixes opening noise **after** normalization. Output scales therefore affect search, but cannot remove the mandated noise fraction. The server fixes 100 simulations, while the exported ONNX still has CPU and memory limits in the [contract](../level-3/env/docs/submission.md).

## Primary-source findings

### Symmetry ensembles: useful precedent, with an important distinction

Silver et al., [AlphaGo, Nature 2016, Methods “Symmetries” and Extended Data Table 3](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf), explicitly evaluate all eight rotations/reflections for **raw-network evaluation**, inverse-transform policy probabilities, and average predictions; values are averaged directly. Their actual APV-MCTS instead samples **one random orientation per evaluation**, letting search average the biases. The table reports higher raw move-prediction accuracy with more explicit symmetries, while search strength deteriorates as inference gets more expensive. Thus, the paper supports a symmetry experiment, not an unconditional claim that eight evaluations improve a constrained search.

Silver et al., [AlphaZero, 2017, comparison with AlphaGo Zero](https://arxiv.org/html/1712.01815v1), state that AlphaGo/AlphaGo Zero used symmetry augmentation and random transforms during MCTS, whereas AlphaZero omitted both because chess/shogi rules are asymmetric. Chess orientation is **not** a reason to omit valid spatial symmetries in Colony. Conversely, chess results do not establish that a symmetry ensemble improves this game's 100-simulation referee.

### Averaging weights can avoid ensemble inference cost

Izmailov et al., [Stochastic Weight Averaging, UAI 2018](https://arxiv.org/html/1803.05407v3), average checkpoints along an SGD trajectory with constant/cyclical learning rates, obtaining image-classification generalization improvements at the inference cost of one network. They recompute BatchNorm statistics after averaging when applicable. Our model has no BatchNorm. Averaging existing Adam-trained checkpoints or two endpoint models is a related experiment, **not** a reproduction of their SWA training procedure.

Wortsman et al., [Model Soups, ICML 2022, §§2 and 5](https://arxiv.org/html/2203.05482v3), average models fine-tuned from a **shared pretrained initialization**. Their greedy method adds ingredients when held-out performance improves; output ensembles cost more at inference, while weight averages do not. They warn about applicability beyond their pretraining setting and find that soups do not inherit ensembles' calibration improvement. For us, shared architecture and checkpoint ancestry matter. Independently initialized networks may represent features with different channel permutations; equal tensor shapes do not justify averaging them. Published classification gains do not imply a game win-rate gain.

### Cheap output calibration is a testable lever

Guo et al., [On Calibration of Modern Neural Networks, ICML 2017, §4.2](https://arxiv.org/html/1706.04599v2), show that fitting a positive temperature on held-out classification logits can improve probability calibration. Temperature preserves the highest-logit class. This is evidence for a cheap postprocessing method in classification, **not** evidence that its validation optimum maximizes MCTS wins. A search uses the entire prior and value distribution, so a scale change can change the selected action even when the raw policy argmax is unchanged.

### More useful targets and more diverse opponents

Wu, [KataGo, 2020, §§3.3–4.1](https://arxiv.org/html/1902.10565v5), reports gains from global pooling, a training-only auxiliary target for the opponent's next reply, and Go-specific ownership/score targets. The reply loss is downweighted and its head is unused in play. These results motivate obtaining additional supervision from existing games. Kernel-threat, bodyguard, and firewall targets are **our proposed substitutions**, not targets tested in that paper. This small board already has a wide trunk receptive field, so a new pooling architecture is lower priority than data and inference experiments.

Lanctot et al., [Policy-Space Response Oracles, NeurIPS 2017](https://arxiv.org/html/1711.00832v2), identify overfitting to training partners and train responses to mixtures selected using an empirical game. Their demonstrations include gridworld coordination and poker. Keeping several archived opponents for training and evaluation is a lightweight engineering adaptation; it is not full PSRO, a computed equilibrium, or a guarantee against unknown tournament teams.

## Experiments to run first

These are engineering proposals based on the inspected contract, existing artifacts, and evidence above. Their ordering minimizes new data generation and preserves time for direct matches.

| Priority | Variant | Concrete experiment | Cost and acceptance condition |
|---|---|---|---|
| 1 | Spatial averaging | Wrap the incumbent and earlier self-play model separately with 2, 4, or 8 symmetry evaluations. Start with a closed subgroup such as identity plus reflection, or four rotations. Inverse-map both action squares, average aligned logits, and average bounded values. | No new training; approximately proportional inference work. Measure actual CPU inference and run the exact validator before long arenas. Only full D4 averaging guarantees equivariance to the entire D4 group. |
| 2 | Shared-ancestry interpolation | If confirmed descendants, interpolate the self-play weights and replay-adapted weights with a small grid such as adaptation weights 0.25, 0.50, 0.75. Average only matching trainable tensors; preserve identical buffers. | Same single-model inference cost. First reject poor supervised/tactical validation variants, then compare surviving exports against both endpoints. Also consider a two-model output ensemble as a diagnostic if latency permits. |
| 3 | Prior/value scales | For promising models, screen a very small grid of policy scales and conservative value shrinkage, e.g. policy multipliers 0.75/1/1.25 and post-tanh value multipliers 0.75/1. | Almost no added compute. All returned values remain bounded. Call this search-scale tuning unless held-out reliability metrics actually establish calibration. Stop expanding the grid once an independent finalist exists. |
| 4 | Short supervised continuation | Continue from a verified good checkpoint using existing self-play plus tournament training data, ordinary augmentation, and conservative optimization. Save a handful of checkpoints and their running average. | Reuses cached data. Keep whole games out of training. Evaluate the exact exports; lower loss alone is insufficient. |
| 5 | Training-only tactical supervision | Add low-weight heads for exact Kernel-threat/guard/firewall labels or future reply targets; discard heads for export. Generate labels from the official transitions. | Needs code and fitting but no additional inference head. Pursue only if simpler variants leave time and measurable mistakes remain. |

### Symmetry implementation details that affect correctness

For an observation transform `g`, obtain `(l_g, v_g) = model(g(x))`. For canonical square indices `a` and `b`, the aligned logit is `l_g[g(a), g(b)]`. Returning `mean_g aligned(l_g)` and `mean_g v_g` yields an ordinary allowed policy/value model. Test the mappings against the official action encoding, for both team colors and both actions-left states.

The spatial transform applies to every spatial plane. Scalar planes stay spatially constant, and the actual Cyan/Magenta identity plane must retain its value. A spatial transform is not a team swap: Magenta's tiebreak advantage prevents treating color transposition as an interchangeable symmetry. The [existing checks](checks.py) already exercise transformed legal sets and transitions; wrapper output equivariance still needs its own numerical check.

**Averaging logits is not averaging probabilities.** The referee's legal softmax of mean logits is a geometric/product-style ensemble of legal priors. The arithmetic probability average in the original symmetry paper requires each orientation's legal-action normalizer. A conventional wrapper does not receive the legal mask. Applying softmax over all 4096 logits and averaging those probabilities is a different method because the referee later discards illegal actions and renormalizes. Any such variant must be named accurately and tested; do not describe mean logits as an exact AlphaGo implementation.

Only full group averaging enforces full group consistency. A transform subset can reduce cost but does not inherit that guarantee outside its subgroup. Valid transformed states can also differ from the narrow fixed-opening training distribution, so averaging can remove useful learned specialization as well as spurious bias. Matches decide the tradeoff.

For export, avoid runtime-generated shape or index inputs. Constant-index `Gather`/`Slice`, `Transpose`, and constant-shape `Reshape` are permitted. Random operations are prohibited, so AlphaGo's random-per-leaf orientation cannot simply be put inside this uploaded ONNX. Validate dynamic batches 1, 8, and 16 and compare ONNX to PyTorch on real game observations. An eight-orientation wrapper may fit the 5 MiB file cap but still fail the CPU limit.

### Tactical targets must respect two actions per team

The next recorded action is not necessarily an opponent action. For future reply supervision, align the target to the **next actual team change**, accounting for `actions_left`, and transform that target with its training observation. Exact immediate-loss labels must allow the opponent's whole two-action turn.

The existing [loss analysis](LOSS_ANALYSIS.md) found four final-action errors for which alternatives prevent a direct near-term loss. Such alternatives establish short-term survival, not a forced win. Use a separate tactical classification/ranking target; do not replace the actual terminal value with `+1` for a hypothetical safer action. Firewall survival also needs the shrink schedule and ring geometry, rather than only static center preference.

## Selection and reporting

Use a small balanced screening set for the grid, then fresh seeds for finalists. Evaluate at 100 referee simulations with the published opening settings, alternating colors. The minimum opponent pool is incumbent, earlier self-play student, and original; include the preserved hybrid snapshot if available. Report direct incumbent wins, each matchup, color splits, and uncertainty. A pool average can conceal a serious regression against one style.

Do not select and report superiority on the same repeatedly reused games. Paired seed/color outcomes are correlated; if treating pairs as experimental units, quantify uncertainty at that level. Report raw results regardless. The final independent set should be large enough to distinguish a useful gain; a result near 50% remains inconclusive. Preserve both candidate hashes and match/replay metadata.

The tournament replay export does not contain opponent model weights or submission hashes. It supports exact tactical diagnosis and labels, but cannot supply a reproducible weight-based opponent league. A Colab A100 can accelerate minibatch fitting and larger batched inference; it does not automatically accelerate Python board transitions, CPU MCTS, or the server's ONNX CPU inference. Use measured throughput to decide whether setup time is worthwhile.

No recommendation here promises a win against unknown teams. A stronger candidate is established only by the final exported artifact's independent arena evidence and acceptance checks.
