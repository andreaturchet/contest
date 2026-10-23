# Tournament loss analysis

Verified 28 full protocols against the frozen official engine: 17 wins, 11 losses. 2 exports lack a protocol and cannot be diagnosed.

The export's score is Cyan's result. The actual winner and the `you` color determine our win/loss; a score of zero is a win when we played Magenta.

Replayed 904 positions. Found 4 avoidable final actions that allowed a direct Kernel capture within the opponent's next turn or an immediate terminal-rule loss. The alternative actions only establish short-term survival, not a forced win.

## Match 10005 — CYAN, Kernel captured

- Action 46: played D3 -> E3; opponent threat: F5 -> G5, then G5 -> E3. Safer alternatives include D3 -> E2.

## Match 9752 — CYAN, Kernel captured

- Action 18: played E4 -> G5; opponent threat: B4 -> C2, then C2 -> A1. Safer alternatives include B3 -> B2, A1 -> B1, A1 -> B2.

## Match 9453 — CYAN, Kernel destroyed by firewall

No avoidable final-action error found by the one-turn capture test. Earlier planning or a longer tactical search needs investigation.

## Match 9196 — CYAN, Kernel captured

No avoidable final-action error found by the one-turn capture test. Earlier planning or a longer tactical search needs investigation.

## Match 8997 — CYAN, Kernel destroyed by firewall

No avoidable final-action error found by the one-turn capture test. Earlier planning or a longer tactical search needs investigation.

## Match 8943 — MAGENTA, Kernel captured

No avoidable final-action error found by the one-turn capture test. Earlier planning or a longer tactical search needs investigation.

## Match 8559 — MAGENTA, Kernel captured

- Action 24: played F5 -> F4; opponent threat: C6 -> C4, then C4 -> F4. Safer alternatives include F5 -> G4.

## Match 7986 — MAGENTA, Kernel captured

No avoidable final-action error found by the one-turn capture test. Earlier planning or a longer tactical search needs investigation.

## Match 7941 — CYAN, Kernel captured

- Action 42: played D4 -> E4; opponent threat: F6 -> G6, then G6 -> E4. Safer alternatives include D4 -> C5, D4 -> E3.

## Match 7055 — CYAN, bare-Kernel endgame

No avoidable final-action error found by the one-turn capture test. Earlier planning or a longer tactical search needs investigation.

## Match 6282 — MAGENTA, Kernel captured

No avoidable final-action error found by the one-turn capture test. Earlier planning or a longer tactical search needs investigation.

## Training use

Training: 754 examples; validation: 166 examples. Entire match IDs divisible by five are reserved for validation. Search targets use 600 simulations and no root noise. Tactical repairs receive extra weight. Outcomes come from the recorded game, so they are not claims about the safer alternative's final result.

This export does not identify model hashes or opponent model files. Results cannot be attributed to a particular saved checkpoint, and the strongest opponents cannot be recreated from their weights. The action traces are sufficient for tactical training and exact replay.
