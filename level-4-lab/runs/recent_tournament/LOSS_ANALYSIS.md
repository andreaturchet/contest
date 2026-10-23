# Tournament loss analysis

Verified 27 full protocols against the frozen official engine: 15 wins, 12 losses. 3 exports lack a protocol and cannot be diagnosed.

The export's score is Cyan's result. The actual winner and the `you` color determine our win/loss; a score of zero is a win when we played Magenta.

Replayed 999 positions. Found 5 avoidable final actions that allowed a direct Kernel capture within the opponent's next turn or an immediate terminal-rule loss. The alternative actions only establish short-term survival, not a forced win.

## Match 13867 — CYAN, Kernel captured

No avoidable final-action error found by the one-turn capture test. Earlier planning or a longer tactical search needs investigation.

## Match 13822 — MAGENTA, bare-Kernel endgame

No avoidable final-action error found by the one-turn capture test. Earlier planning or a longer tactical search needs investigation.

## Match 13447 — CYAN, bare-Kernel endgame

No avoidable final-action error found by the one-turn capture test. Earlier planning or a longer tactical search needs investigation.

## Match 13362 — MAGENTA, Kernel captured

- Action 52: played E6 -> D6; opponent threat: D6 -> E6, then D6 -> E6. Safer alternatives include E6 -> F7, E6 -> F5.

## Match 12746 — CYAN, bare-Kernel endgame

No avoidable final-action error found by the one-turn capture test. Earlier planning or a longer tactical search needs investigation.

## Match 12585 — MAGENTA, Kernel captured

No avoidable final-action error found by the one-turn capture test. Earlier planning or a longer tactical search needs investigation.

## Match 12374 — MAGENTA, Kernel captured

- Action 32: played F6 -> E5; opponent threat: D4 -> E5, then D4 -> E5. Safer alternatives include E6 -> E7, F6 -> F5, F6 -> E7, F6 -> G7.

## Match 11988 — CYAN, Kernel captured

No avoidable final-action error found by the one-turn capture test. Earlier planning or a longer tactical search needs investigation.

## Match 11807 — MAGENTA, Kernel captured

- Action 28: played F7 -> G8; opponent threat: E5 -> E6, then E6 -> G8. Safer alternatives include F6 -> G6.

## Match 11199 — CYAN, Kernel captured

No avoidable final-action error found by the one-turn capture test. Earlier planning or a longer tactical search needs investigation.

## Match 11182 — MAGENTA, Kernel captured

- Action 12: played E5 -> C3; opponent threat: F5 -> F6, then F6 -> H8. Safer alternatives include H8 -> G7, H6 -> H7, E5 -> F5, E5 -> F6, E5 -> G7.

## Match 10913 — MAGENTA, Kernel captured

- Action 32: played E5 -> E4; opponent threat: C5 -> C6, then C6 -> E4. Safer alternatives include F6 -> F7, F6 -> F5, F6 -> E6, F6 -> G6, E5 -> E6.

## Training use

Training: 871 examples; validation: 148 examples. Entire match IDs divisible by five are reserved for validation. Search targets use 600 simulations and no root noise. Tactical repairs receive extra weight. Outcomes come from the recorded game, so they are not claims about the safer alternative's final result.

This export does not identify model hashes or opponent model files. Results cannot be attributed to a particular saved checkpoint, and the strongest opponents cannot be recreated from their weights. The action traces are sufficient for tactical training and exact replay.
