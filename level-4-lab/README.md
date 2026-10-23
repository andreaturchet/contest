# Independent Level 4 experiment

The recommended upload is `candidate_v2.onnx`. It won 368/600 fresh matches against `candidate.onnx` and 157/200 against the original evaluator. See `RESULTS_ROUND2.md` for validation, research, and limitations. For further A100 training, use `colony_a100_v2.ipynb` with `level4_lab_bundle_v2.zip`; the older notebook below describes the first iteration.

This directory owns its engine snapshot, scene, baselines, runs, and replays. It does not edit the original `level-3` or `level-4` experiment. The existing virtual environment may be used read-only to run the scripts locally.

## Colab A100

Open `colony_a100.ipynb` in Colab, select A100, and run the cells. Upload `level4_lab_bundle.zip` when prompted. The default budget is 90 minutes plus final matches and validation, with checkpoints in a dedicated Drive folder. The notebook has not been executed on a connected A100 here. Local CPU smoke tests cover model fitting, actor inference batching, export, checkpointing, and matches. CUDA/mixed-precision execution remains to be verified in Colab.

The model is a small residual policy/value network with a compact joint source/destination policy. Its neural heads start at zero so the initial outputs reproduce the current evaluator. Training learns referee-MCTS visit counts and actual terminal outcomes, with legal-action masking, eight tested spatial symmetries, and archived opponents. Values are labeled by the actual team, which keeps the two-action turn structure correct.

## Local commands

From this directory, using the existing read-only environment:

```bash
../level-3/env/.venv/bin/python checks.py --model baselines/current.onnx
../level-3/env/.venv/bin/python train.py --minutes 10 --games 16 --workers 2 --channels 32 --blocks 2
../level-3/env/.venv/bin/python arena.py --a path/to/candidate.onnx --b baselines/current.onnx --games 100 --seed 4000000 --output replays/candidate
../level-3/env/.venv/bin/python engine/tools/validate_submission.py path/to/candidate.onnx
```

Open the generated HTML replay to inspect each action. Matches alternate colors, use separate seeded randomness per player, and retain the tournament's opening noise and temperature schedule. The Level 4 scene is explicitly supplied and all official MCTS settings are preserved, including 100 simulations. The validator's separate full-match check uses the original six-piece Level 3 scene in the frozen engine.

`RESEARCH.md` contains primary-source findings and proposed adaptations. `runs/*/history.jsonl` records training screens. Final holdout reports identify both model files by SHA-256 and include raw wins, results by color, and a 95% Wilson interval. Training screens do not establish superiority. A model is copied to `submission.onnx` only if its final holdout interval is entirely above 50%; the contest upload is left to the user.

Frozen snapshots of the original evaluator and the other local experiment's candidates are available under `baselines/`. Other teams' weights are unavailable. Beating local candidates does not establish superiority over unknown tournament opponents.

## Tournament replay adaptation

The supplied export is preserved under `data/tournament-matches.json`. `analyze_losses.py` verified 28 complete protocols against the engine, reconstructed 904 real positions, and found four avoidable last-action tactical errors. `LOSS_ANALYSIS.md` explains each loss and `replays/tournament.html` plays the actual recorded games. Two entries lack protocols. `score` is Cyan's outcome, so our result must also use the `you` color.

The source bundle includes 754 training and 166 validation examples relabeled by 600-simulation search, including extra weight for exact tactical repairs. Whole matches are separated before augmentation. The notebook feeds these examples into a quarter of training samples. `distill.py` supports a focused adaptation of an existing checkpoint.

The initial local self-play model improved from an inconclusive 54/100 to 129/200 against the frozen original on new seeds after more training. The latter model is saved as `candidates/selfplay_129_of_200.onnx`. Its results apply only against that baseline; they are not evidence of winning the tournament.

After adaptation to the tournament-derived examples, `candidate.onnx` won 140/200 (70%) against the original on another unused seed set and passed the supplied validator. The exported candidate is 256,404 bytes. The updated Colab bundle includes this local champion as an archived opponent and requires a future Colab candidate to beat both it and the original on separate final match sets before producing a submission.
