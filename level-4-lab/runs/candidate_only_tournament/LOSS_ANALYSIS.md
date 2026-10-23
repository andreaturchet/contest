# Tournament loss analysis

Verified 19 full protocols against the frozen official engine: 9 wins, 10 losses. 0 exports lack a protocol and cannot be diagnosed.

The export's score is Cyan's result. The actual winner and the `you` color determine our win/loss; a score of zero is a win when we played Magenta.

Replayed 774 positions. Found 3 avoidable final actions that allowed a direct Kernel capture within the opponent's next turn or an immediate terminal-rule loss. The alternative actions only establish short-term survival, not a forced win.

## Match 12746 — CYAN, bare-Kernel endgame

No avoidable final-action error found by the one-turn capture test. Earlier planning or a longer tactical search needs investigation.

## Match 13362 — MAGENTA, Kernel captured

- Action 52: played E6 -> D6; opponent threat: D6 -> E6, then D6 -> E6. Safer alternatives include E6 -> F7, E6 -> F5.

## Match 13447 — CYAN, bare-Kernel endgame

No avoidable final-action error found by the one-turn capture test. Earlier planning or a longer tactical search needs investigation.

## Match 13822 — MAGENTA, bare-Kernel endgame

No avoidable final-action error found by the one-turn capture test. Earlier planning or a longer tactical search needs investigation.

## Match 13867 — CYAN, Kernel captured

No avoidable final-action error found by the one-turn capture test. Earlier planning or a longer tactical search needs investigation.

## Match 14058 — MAGENTA, bare-Kernel endgame

No avoidable final-action error found by the one-turn capture test. Earlier planning or a longer tactical search needs investigation.

## Match 14235 — CYAN, Kernel captured

No avoidable final-action error found by the one-turn capture test. Earlier planning or a longer tactical search needs investigation.

## Match 14372 — MAGENTA, Kernel captured

- Action 44: played E3 -> D2; opponent threat: B3 -> C2, then C2 -> D2. Safer alternatives include E3 -> F3, E3 -> F4, E3 -> F2.

## Match 14663 — MAGENTA, Kernel captured

- Action 24: played F6 -> E5; opponent threat: B3 -> B2, then B2 -> E5. Safer alternatives include F6 -> F7.

## Match 14885 — MAGENTA, Kernel captured

No avoidable final-action error found by the one-turn capture test. Earlier planning or a longer tactical search needs investigation.

## Training use

Training: 695 examples; validation: 91 examples. Entire match IDs divisible by five are reserved for validation. Search targets use 600 simulations and no root noise. Tactical repairs receive extra weight. Outcomes come from the recorded game, so they are not claims about the safer alternative's final result.

These 19 protocols come from the folder the user explicitly identified as containing only the supplied candidate. Eleven side labels come from the JSON exports; eight were inferred from exact agreement with every post-opening move on that side, with lower agreement on the opposing side. The protocol itself contains no model hash. Opponent weights remain unavailable. Attribution evidence is saved in `data/candidate_only/move_agreement.json`.
