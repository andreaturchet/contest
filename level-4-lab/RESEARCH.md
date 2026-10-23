# Level 4: research and a practical training plan

Researched 23 October 2026. Work stays in `level-4-lab`; the existing `level-4` experiment can continue independently. Paper results below are evidence from other games. Recommendations for Colony: One are explicitly proposals, and need local matches before any claim of improvement.

## What the actual challenge permits

The local [submission specification](../level-3/env/docs/submission.md) says the referee runs **100 MCTS simulations per action in Level 4**, with a batch size of 8. This is not policy-only play. A submission supplies raw policy logits `(B, 4096)` and a bounded value `(B,)`, given float32 observations `(B, 16, 8, 8)`. The value is from the team-to-move perspective. Batch axes must be dynamic, weights embedded, ONNX opset 13–18, and the file at most 5 MiB. External Python search or a separately shipped heuristic cannot run on the server.

The [rules](../level-3/env/docs/rules.md) and [observation specification](../level-3/env/docs/observation.md) have several important implications:

- A team takes **two actions before its opponent moves**. Search/teacher backup signs change when the team changes, not automatically after each action. Preserve `actions_left` and urgency planes.
- Magenta wins tiebreaks. Preserve actual side identity in plane 15; do not assume swapping player identities is a symmetry.
- Tactical priorities include Kernel protection/bodyguards, move versus shot semantics, approaching firewall locks, and the special two-Kernel terminal rule. Material alone does not capture these.
- Roughly 12–30 actions are usually legal despite the 4096 outputs. Train policy loss over legal actions, as the referee does.

The existing [trainer](../level-4/az_level4.py) already has a residual policy/value network, masked policy cross-entropy, terminal win/loss value targets, replay, batched referee MCTS, and spatial augmentation implemented in [self-play](../level-4/az_selfplay.py). Simply naming those techniques does not improve that baseline. Its default evaluation is only four games against GreedyAgent; that is insufficient to establish that a new checkpoint beats the current submission.

## What the primary literature supports

| Source | Evidence in the paper | Implication proposed for this challenge |
|---|---|---|
| Silver et al., [AlphaZero, 2017](https://arxiv.org/html/1712.01815v1) | A joint policy/value network learns search visit distributions and terminal outcomes. Its loss combines policy cross-entropy, value squared error, and regularization. Spatial augmentation is valid only when supported by game symmetry; chess/shogi were not augmented. | Keep both heads and evaluate through the supplied MCTS. Improving terminal value accuracy and tactical policy priors should benefit the actual tournament interface. This is a suitable framework, not evidence that a short local run will surpass other teams. |
| Anthony et al., [Expert Iteration, 2017](https://arxiv.org/html/1705.08439v4) | Search supplies plans for a network to imitate; the learned network then improves search. Root visit distributions retain information about alternatives that one chosen-action label discards. The paper also collects new states with the apprentice to reduce mismatch between its training and execution distributions. | Use a verified tactical/search teacher to bootstrap a small model, keep soft legal-action targets where possible, then relabel states reached by the student. This is a more focused first experiment than starting another large random-initialized run. |
| Rusu et al., [Policy Distillation, 2016](https://arxiv.org/html/1511.06295v2) | In Atari, expert policies can be transferred into smaller networks by supervised training on teacher output distributions; multiple policies can also be consolidated. | An expensive offline teacher need not meet upload limits. Distill its policy into a compact ONNX student, while training the required value head with outcomes or separately validated teacher values. Transfer to this game remains an experiment. |
| Wu, [KataGo, 2020](https://arxiv.org/html/1902.10565v5) | Playout-cap randomization mixes cheap self-play moves with more expensive searches; policy training is restricted to the stronger searches. Global pooling and auxiliary targets are additional investigated improvements. Reported overall compute savings are for Go and a combination of changes. | If generation is the bottleneck, collect many completed games cheaply and reserve stronger searches for trustworthy policy targets. Never silently treat low-budget visits as equally good policy supervision. Benchmark a small global-context feature path only after the basic student works. Do not promise the paper's speedup here. |
| Lanctot et al., [Policy-Space Response Oracles, 2017](https://arxiv.org/html/1711.00832v2) | Policies can overfit training partners. The proposed method trains approximate responses against mixtures of policies and uses empirical games to select strategies, demonstrated in gridworld coordination and poker. | Maintain a small opponent pool containing previous checkpoints and distinct tactical styles. Judge average and weakest-matchup performance. This lightweight pool is inspired by the paper; it is not a complete PSRO implementation or a guarantee of robustness. |

## Recommended first experiment on limited local compute

These are engineering proposals inferred from the contract and literature, not established Colony: One results.

1. **Freeze the comparison before training.** Preserve the current ONNX by checksum and run a seed-fixed, color-balanced match suite from the exact Level 4 scene. Use 100 simulations and the published root-noise/temperature settings. Record winners, actions, elapsed time, terminal reason, and replayable games. A human-viewable replay helps locate errors; the authoritative environment should remain the game implementation.

2. **Establish a useful teacher before making a large dataset.** Build a legal-action teacher using the supplied environment transitions. Start with terminal outcomes, Kernel safety/bodyguards, imminently locked rings, and material; add shallow search over both actions of a turn and opponent responses where affordable. Cache repeated states and order captures/threats first. Confirm this teacher beats GreedyAgent and inspect losses. If it does not, training a student to imitate it has no demonstrated benefit.

3. **Distill a compact policy/value student.** Prefer a small convolutional model first. Gather teacher-controlled and exploratory games; save observation, legal indices, teacher/search distribution, actual current team, and final winner. Supervise policy only where teacher search was sufficiently reliable. Use true game outcomes for the value head; teacher scores may be an explicitly weighted pretraining aid, but an arbitrary heuristic score is not a calibrated win probability. Save and reuse the dataset so model fitting does not rerun search.

4. **Augment correctly and separate evaluation data.** The documented movement geometries and square-ring firewall appear invariant under all eight spatial rotations/reflections. Apply the same transformation to every spatial plane and both `from` and `to` policy squares, leaving scalar planes and team identity intact. Verify transformed legal sets against the environment. Split by complete game before augmentation; related positions from one game must not leak into validation. The prescribed initial layout need not itself be symmetric for transformed states to be useful training examples.

5. **Improve on the student's own positions.** Run short self-play or mixed-opponent batches with the student, ask the teacher/search to relabel selected positions, and continue training. Keep actual terminal outcomes. For a cheaper search schedule, log per-position search budget and separate policy versus value loss eligibility. Training may use other budgets; promotion always uses the tournament budget.

6. **Promote on matches against the current model.** A quick screen can use a small color-balanced set; promising candidates then need a larger independent seed set. Report raw wins/losses, color results, and uncertainty. For example, 8/8 wins can be promising but is still a small sample. Do not repeatedly tune against the final holdout. Also test a few archived opponents to detect regressions hidden by average performance. Winning against GreedyAgent alone is not evidence of improvement over the existing submission.

7. **Validate the exported artifact itself.** Check PyTorch/ONNX numerical agreement on real legal positions and batch sizes 1, 8, and 16; run the supplied acceptance validator; play candidate-versus-current matches through ONNX inference. The documented validator's full-match check is Level 3, so also measure the actual Level 4 scene at 100 simulations. Training loss and successful export do not establish tournament strength or runtime suitability.

## Priorities and limits

**First:** reliable arena/replays, frozen baseline, useful teacher, compact student, and direct candidate-versus-current comparison. **Next:** selective expensive labels, opponent diversity, and refreshed student-state data. **Later:** auxiliary tactical targets/global context or a larger teacher, justified by measured failures. Changing the referee's MCTS algorithm is outside the uploaded-model interface; training-time algorithm changes only help through the learned outputs.

No trained candidate or local win-rate result is claimed in this note. No paper establishes that this recipe will win the Level 4 tournament; that requires fresh evidence from the exact exported model under the actual referee.
