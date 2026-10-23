# Colony: One Tournament Environment

Train a policy-value model for Colony: One and submit one ONNX file. The
tournament runs every submission through the same MCTS; your model supplies
policy logits and a position value.

## Build your first model

Requires **Python 3.11, 3.12, 3.13 or 3.14** (64-bit). Python 3.10 and Intel Macs are not supported:
the required `onnxruntime` cannot be installed there. Install inside the unzipped folder, preferably
in a fresh virtual environment:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS / Linux: source .venv/bin/activate
pip install -e .
```

```python
import numpy as np

from colony_one import ColonyOneEnv

env = ColonyOneEnv(seed=0)
observation = env.reset()  # float32, shape (16, 8, 8)

while not env.terminated:
    legal = env.legal_mask()                 # bool, shape (4096,)
    logits = np.zeros(4096, dtype=np.float32)  # replace with your model
    logits[~legal] = -np.inf
    action = int(np.argmax(logits))
    observation, reward, terminated, info = env.step(action)
```

`step()` applies one **action**, not one turn. A team takes two consecutive
actions; always read `env.current_team` after each step.

Your ONNX model must implement:

| Tensor | Shape | Type | Meaning |
|---|---|---|---|
| input `observation` | `(batch, 16, 8, 8)` | `float32` | canonical state |
| output `policy` | `(batch, 4096)` | `float32` | raw action logits |
| output `value` | `(batch,)` | `float32` | side-to-move value in `[-1, 1]` |

The batch axis must be dynamic. Validate every export:

```bash
python tools/validate_submission.py model.onnx
```

`ACCEPTED` means the file meets the tournament submission contract.

## Train with MCTS

The tournament chooses every action using the published MCTS implementation.
For AlphaZero-style self-play, use the same search with an evaluator callable:

```python
from colony_one import mcts

action = mcts.search_action(
    env.board,
    env.current_team,
    env.actions_left,
    env.round_number,
    evaluator,  # (B, 16, 8, 8) -> ((B, 4096), (B,))
    mcts.MCTSConfig(),
)
```

Call it once per action, including a team's second action. `MCTSConfig()` holds
the referee's search settings (root noise and opening visit sampling; see
[submission.md](docs/submission.md#the-referees-settings) for the exact schedule).
The server sets the number of simulations per action: **60 in Level 3** and
**100 in the Level 4 tournament**. Use `mcts.MCTSConfig(simulations=60)` to
reproduce Level 3 locally; the default of 300 is only a local default.

## Required reading

- [Observation and actions](docs/observation.md): canonical tensor, action
  indices, legal masking, and value target.
- [Submission](docs/submission.md): ONNX restrictions and referee MCTS.
- [Rules](docs/rules.md): complete game rules.

## Optional local matches

Run a model against another model under referee MCTS:

```bash
python play/model_vs_model.py --model-a model.onnx --model-b play/random.onnx --games 50
python play/model_vs_model.py --model-a model.onnx --model-b other.onnx --games 50
```

`play/random.onnx` is the shipped random model. Sides are alternated across games.

## Ascension chart in level 4 dashboard
The Ascension chart plots the rating after each match over the last 30 minutes for
you and up to four rivals:

- **Most played #1–#3** – the three opponents you have played the most games
  against (all-time, not just the last 30 minutes).
- **Best beaten opponent** – the highest-rated opponent you have beaten at least
  once, other than the three above.

You may see fewer rivals early on (if you have few opponents or no wins yet).

## Reference Material

- `colony_one/mcts.py`: exact referee search implementation.
- `play/random.onnx`: minimal valid random ONNX submission.

## License

See [LICENSE](LICENSE).
