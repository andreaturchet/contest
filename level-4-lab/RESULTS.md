# Candidate results

Candidate: `candidate.onnx`, 256,404 bytes.
SHA-256: `5a370f5cea2a41834eadd22e8b312b4104c73c7041afeb394d5780b6886d9a6f`.

The candidate uses the two-block, 32-channel residual model, followed by adaptation to tournament replay positions. Training and model selection are separate from these final seeded matches. All comparisons use the frozen Level 4 scene, ONNX CPU inference, the published opening noise/temperature, alternating colors, and 100 referee MCTS simulations per action.

| Opponent | Wins | 95% Wilson interval |
|---|---:|---:|
| Original submitted heuristic | 140/200 (70%) | 63.3%–75.9% |
| Earlier self-play student | 46/100 (46%) | 36.6%–55.7% |
| Other experiment hybrid B snapshot | 72/100 (72%) | 62.5%–79.9% |

The replay-adapted candidate clearly beats the original and the sampled hybrid. The direct comparison against the earlier student is inconclusive; a higher baseline score does not establish superiority over that student. These local results do not establish superiority over unknown tournament teams. The tournament export does not identify uploaded model hashes.

The final file passed the supplied submission validator, dynamic-batch inference at 1/8/16, PyTorch/ONNX parity on real tournament observations, and 160 checks of symmetry-preserved legal actions and transitions. The validator uses the proper frozen six-piece Level 3 scene for its resource match; the independent arena uses the full Level 4 scene.

Research: `RESEARCH.md`. Exact loss diagnosis: `LOSS_ANALYSIS.md`. Actual exported tournament replays: `replays/tournament.html`. Candidate replay set: `runs/tournament_adapted/final_match.html`. Portable GPU experiment: `colony_a100.ipynb` and `level4_lab_bundle.zip`.

The Colab notebook and CUDA/mixed-precision branch have not run in an attached A100 runtime here. CPU smoke tests cover actor batching, training with tournament examples, export, checkpointing, resume, archived-opponent matches, and final promotion logic. Select A100 and run the notebook to perform the GPU experiment.
