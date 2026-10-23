# Claude Code follow-up after measured results

**Yes, I agree.** Heads-only distillation with the value frozen bitwise is the right controlled experiment. It isolates the one variable that every failed run shared: value drift.

**Corrections:**

1. **Fix the baseline before you compare.** Your temperature grid already shows policy-side headroom: T=0.85 scored 83/150 and T=1.2 scored 84/150, against 73/150 at T=0.7. That spread is roughly 1.3σ, so it is suggestive, not proven. Even so, the finetune must beat V2 at its best temperature, not at T=1. Otherwise a head retrain may only be reproducing a temperature change. Report the finetuned model's entropy shift compared with V2.
2. **Distill the full visit distribution, not argmax.** Use cross-entropy against the 600-sim visit counts (raw counts, τ=1). Mask the loss to legal moves only, which matches your mask fix.
3. **Use a KL anchor instead of a self-play mix.** With 1241 states, 600 steps is many epochs, so overfitting is the main risk. A KL anchor to the frozen V2 priors directly bounds that drift. Hold out 15% of the states and early-stop on held-out CE.
4. **Verify the freeze.** Hash the value-head outputs on a fixed batch before and after training. Separately, check that the factorized source/target product stays consistent with the joint visit targets.

**One setting:** AdamW at 3e-4 with weight decay 0, loss = CE + 0.5·KL(π_θ‖π_V2), batch 128, and keep the best held-out checkpoint (no more than 600 steps).

**Time budget (45 min):** Training should take about 5 minutes. Spend the rest on at least 200 games at 100 sims against V2 at its best temperature, with colors swapped. At 96 games, a 58% score is still within noise, as your BCE run showed.
