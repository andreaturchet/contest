# Level 4: third literature review

Researched 23 October 2026. This round focuses on training targets that were not tested in round two. All proposals keep the final 32-channel, two-block network and official 100-simulation referee. Published results in other games are motivation; only independent matches of the exported artifact establish a Colony improvement.

## Recommended order under the remaining deadline

| Order | Experiment | Data and implementation cost | What changes |
|---|---|---|---|
| 1 | Binary cross entropy for the value head | Reuse cached data; short continuation from verified v2 | Train the existing pre-tanh value as a win/loss logit; export remains identical in architecture |
| 2 | Fresh v2 reanalysis with mixed historical/search value | Re-search selected replay positions at 400–600 simulations; cache root edge statistics once | Fresh policy targets and a lower-variance value target, preserving some actual outcome supervision |
| 3 | Training-only softened-policy head | Reuse cached policy distributions; add and later discard a compact auxiliary head | Predict lower-probability useful alternatives without changing the deployed policy target |
| 4 | Completed-Q policy improvement targets | Reuse the same fresh search statistics but needs careful value completion and scaling | Turn root Q estimates into regularized policy-improvement targets instead of copying raw counts |

Run a small, fixed screen against v2, then independent final seeds for a selected export. Preserve v2 throughout. Avoid combining every change into one run: a gain would then be hard to attribute, and multiple new objectives can destabilize a short continuation.

## 1. Use a classification loss for the existing bounded value

KataGo's official training code uses cross entropy for its main game-outcome head and explicitly uses `2 * pretanh` with binary cross entropy for win/loss Q targets. It converts a value target to a probability with `(1 + target) / 2`. This directly confirms the tanh/logit correspondence needed here. [Official `metrics_pytorch.py`, `loss_qvalues_samplewise` and `loss_value_samplewise`](https://github.com/lightvector/KataGo/blob/master/python/katago/train/metrics_pytorch.py#L81-L114).

**Our adaptation:** if the present model returns `v = tanh(h)`, train `binary_cross_entropy_with_logits(2*h, (1+z)/2)`. This works for both binary outcomes and bounded mixed targets. Keep the final exported value as `tanh(h)`. Obtain `h` directly from the feature value plus neural value head; computing `atanh` of a clipped output needlessly introduces numerical error and a clipping boundary.

The mathematical gradient with respect to `h` is `v-z`; current squared error gives `2*(v-z)*(1-v*v)`. Thus classification loss continues correcting confidently wrong predictions, whereas the tanh derivative can suppress the squared-error gradient. This is an analytic observation, not a measured cause of the current losses.

A read-only check of v2 on the 871 recent tournament training rows found only 0.69% with `abs(v) > .95`, and none with `v*z < -.95`. Its MSE against the historical outcome was 0.8365. Saturation is therefore **not established as the main problem** on that sample. BCE remains a cheap controlled ablation; do not promise a benefit from the gradient argument.

## 2. Refresh search targets and retain the search value

Willemsen, Baier and Kaisers compare final-outcome AlphaZero targets, root-average MCTS values (soft-Z), one greedy child, and a greedy trajectory to a search leaf (A0GB). The paper reports improved learning on Connect-Four and Breakthrough, including 100-simulation training experiments. It discusses the bias/variance tradeoff: root averaging has greater exploration bias but lower variance; following a greedy trajectory reduces that bias while relying on fewer and progressively noisier samples. Search need not change to compare these training targets. [*Value targets in off-policy AlphaZero: a new greedy backup*, §§3.2–3.4 and 4.2](https://pure.tue.nl/ws/portalfiles/portal/217685150/s00521_021_05928_5.pdf).

**Our adaptation:** current `analyze_losses.py` obtains fresh policy counts but then discards every root Q and trains value solely against the recorded game's outcome. That outcome describes the historical behavior, including older models and later mistakes. It is legitimate outcome supervision, but is not the counterfactual result of a teacher's improved policy. Re-search with the frozen v2 evaluator and no root noise, retaining:

```text
n[a] = root.N[a]
w[a] = root.W[a]
q[a] = w[a] / n[a]                         for n[a] > 0
q_root = sum_a w[a] / sum_a n[a]
pi[a] = n[a] / sum_a n[a]
y = 0.5 * historical_z + 0.5 * q_root
```

The 50/50 blend is our conservative engineering proposal, not a parameter established by that paper. It preserves actual result supervision while testing whether refreshed search values help. Terminal solved values should remain exact when known. A safer tactical repair establishes survival, and must not be assigned a fabricated `+1` outcome.

**Perspective is critical:** local `mcts.py::_backup` stores each edge's `W` in its parent node team's perspective. At the root, every `w[a]` already uses the root team. Therefore `q_root` needs no additional sign. A second same-team action does not flip the value. Never average child `leaf_value` tensors indiscriminately: those can use different teams' perspectives.

`sum(W)/sum(N)` is a visit-weighted average, **not** the unweighted mean of action Qs. A second defensible target is `q[a_star]` for `a_star = argmax N`; this reduces exploratory-action averaging but has higher variance. Choose one initial variant rather than spending the deadline on a large target grid. Whole-game validation split and fresh final arena seeds remain necessary.

## 3. An auxiliary policy distribution that reveals alternatives

KataGo's official methods note, added after the original paper, describes a second training-only policy head predicting the normalized policy target raised to `1/T`. The current documented setting is `T=4` with a nominal weight of eight; the note explains that this compensates for smaller gradients. The intent is to teach distinctions among lower-mass moves that ordinary cross entropy barely emphasizes. [Official *Other Methods Implemented in KataGo*, “Auxiliary Soft Policy Target”](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md#auxiliary-soft-policy-target).

**Our adaptation:** attach an independent compact source/destination pair head to the current trunk and train it against `normalize(pi**0.25)`. Keep the main head trained against the original search distribution. Discard the new head for ONNX export. Reusing the main head directly with softened targets is a different experiment and can flatten the deployed prior.

Preserve exact zero targets for unvisited actions; apply the same legal mask and D4 action-index transform to both heads. Do not add probability mass to illegal actions. Because the tiny warm-start model differs greatly from KataGo, screen a conservative auxiliary weight rather than assuming its nominal eight transfers. This weight choice is our proposal. A noised low-budget target can contain poor exploratory moves, so prefer the no-noise reanalysis targets.

## 4. Completed-Q targets and search-target pruning

Danihelka et al. show that conventional visit-count policy iteration can fail to improve when many root actions remain unvisited. Their alternative fills missing Q values and forms `softmax(prior_logits + sigma(completedQ))`; with accurate values the monotone improvement construction has a theoretical justification. Their complete method also changes root exploration through Gumbel sampling and sequential halving. [*Policy improvement by planning with Gumbel*, §§3–4 and Appendices C–D](https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/gumbel-alphazero.pdf).

DeepMind's official Mctx implementation fills unvisited Qs using a mixture of raw value and prior-weighted visited Qs, then optionally rescales Qs to `[0,1]` and scales by `(50 + max_visits) * 0.1` under its defaults. [Official `qtransforms.py`, `qtransform_completed_by_mix_value` and `_compute_mixed_value`](https://github.com/google-deepmind/mctx/blob/main/mctx/_src/qtransforms.py#L80-L177).

**Our adaptation:** cache fresh legal priors, `N`, `W`, and raw root value, then test completed-Q targets for supervised continuation. The current referee remains ordinary PUCT. Consequently this is **not a full Gumbel AlphaZero implementation**, and estimated Qs do not confer a guaranteed real-game improvement. With a small root Q range, min/max normalization can amplify noise; conservative fixed scaling or a blend with count targets is a safer first experiment. This is lower priority than the direct value-target ablation because it adds another scale to select.

KataGo separates exploration from policy training through forced root playouts followed by policy-target pruning: it subtracts exploratory visits when the final PUCT estimates do not justify them. Its playout-cap randomization runs a minority of expensive searches and many cheap moves, retaining full-search positions for training to balance policy quality against number of game outcomes. [Wu, *Accelerating Self-Play Learning in Go*, §§3.1–3.2](https://arxiv.org/html/1902.10565v5).

These are valuable longer-run options, but do not label arbitrary count sharpening or dropping low-count actions as a faithful reproduction. Present teacher reanalysis already disables noise; implementing the forced-playout mechanism plus its corresponding pruning before the deadline adds more code risk than the target-only changes above. Randomized training budgets cannot alter the server's fixed inference search budget.

## Two-action tactical supervision

All new heads and targets must retain `actions_left` and actual team identity. For an opponent-reply target, the next recorded action can still belong to us; align to the next actual team change. An exact end-of-turn danger label must include the opponent's two actions and official bare-Kernel/firewall resolution. Do not treat pseudo piece attacks or an isolated spatial two-action oracle as equivalent to full legal engine transitions.

Candidate strength remains the exported file's independent arena result, with both colors and more than one archived opponent. Loss reductions, better historical outcome calibration, or a successful auxiliary classifier are diagnostics, not a promotion criterion.
