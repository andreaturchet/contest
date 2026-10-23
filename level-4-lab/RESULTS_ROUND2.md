# Candidate v2: confirmed improvement

Use `candidate_v2.onnx`. The original `candidate.onnx` remains unchanged. `candidate_v2_recent.onnx` is an intermediate filename; the recommended stable filename is `candidate_v2.onnx`.

File size: 256,404 bytes. SHA-256: `d385f252713ed3b0d37320d40c4a4f3d4609789b9e1f1908dbdf7d136f8e7ef1`.

Every match below uses the full Level 4 scene, ONNX CPU inference, the official 100-simulation MCTS, the published opening randomization, alternating colors, and separate seeds. The 600-game confirmation was run after checkpoint selection and uses fresh seeds.

| Opponent | Wins | 95% Wilson interval |
|---|---:|---:|
| Previous candidate (fresh confirmation) | 368/600 (61.3%) | 57.4%–65.1% |
| Original heuristic | 157/200 (78.5%) | 72.3%–83.6% |
| Earlier self-play model | 112/200 (56.0%) | 49.1%–62.7% |

Against the previous candidate, v2 won 186/300 as Cyan and 182/300 as Magenta. The direct comparison against the earlier self-play model remains inconclusive; v2 is not claimed to dominate every possible opponent. Tournament opponent weights are unavailable.

## What changed

The successful recipe combines newer tournament search targets, the most recent 12,000 cached self-play positions, and a training-only auxiliary loss. The auxiliary head predicts whether either team has a direct Kernel-winning attack within the modeled action allowance and whether our Kernel has a bodyguard. It trains the shared trunk and is absent from the submitted ONNX, so the exported architecture and file size stay the same. This experiment does not isolate the auxiliary loss from the fresher replay selection.

The auxiliary-target idea is inspired by KataGo's use of additional training targets; our tactical labels are specific to Colony and are not targets tested in that paper. Primary sources and distinctions are in `RESEARCH_ROUND2.md`. The training configuration is in `runs/recent_auxiliary/run_config.json`; selected weights are in `runs/recent_auxiliary/adapted_weights.pt`.

## Other experiments

- Full D4 logit/value averaging: 105/200 against the previous candidate, inconclusive.
- Shared-ancestry weight averaging: 97/200, no established gain.
- More mixed-opponent self-play: 109/200, inconclusive.
- Simple rule-aware inference wrappers did not improve screening results.
- A further refinement using only the candidate-specific folder reached 112/200 against v2. Its lower confidence bound is below 50%, so it was not promoted despite scoring 159/200 against the original heuristic.

## New candidate-only replays

All 19 files in the user-designated `New Matches` folder were copied and replayed through the official engine, reconstructing 774 positions: 9 wins, 10 losses. Eleven side labels come from the JSON metadata; eight were inferred from exact agreement with every post-opening candidate move on the identified side, with lower agreement on the opponent's side. This is documented in `data/candidate_only/move_agreement.json`.

Four losses ended under the bare-Kernel rule and six by Kernel capture. Three final-action tactical mistakes had alternatives that survived the opponent's next turn. Other losses need earlier planning; surviving one turn is not a proof of winning. Detailed diagnosis and replays are in `runs/candidate_only_tournament/LOSS_ANALYSIS.md` and `runs/candidate_only_tournament/replays/tournament.html`.

These protocols were used for a separate follow-up refinement and validation. The selected v2 was trained on the preceding recent JSON export, which overlaps part of that folder. The candidate-only folder is therefore not a wholly independent test of v2's tactical accuracy. Final strength is measured with the fresh seeded matches above.

## Validation and continuation

The exact v2 file passed the supplied submission validator, dynamic batches 1/8/16, numerical PyTorch/ONNX parity on real candidate-only observations, and 160 checks of symmetry-preserved legal actions and transitions. CPU runtime is unchanged in architecture from v1. Raw final games and hashes are in `replays/v2_confirmation.json` and the HTML replay alongside it.

`colony_a100_v2.ipynb` and `level4_lab_bundle_v2.zip` continue from these verified weights, cached replay, and updated examples. The bundle includes v2 as the archived champion and requires a future candidate to beat it in final matches. GPU execution has not been launched here. The default training budget is 45 minutes plus final evaluation; adjust it to leave time before the challenge deadline.
