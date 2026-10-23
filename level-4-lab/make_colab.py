"""Build a portable Colab notebook and a minimal independent source bundle."""
from pathlib import Path
import ast
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parent


def main():
    cells=[]
    def md(s):cells.append(dict(cell_type="markdown",metadata={},source=s.splitlines(True)))
    def code(s):
        ast.parse(s)
        cells.append(dict(cell_type="code",metadata={},execution_count=None,outputs=[],source=s.splitlines(True)))
    md("""# Colony: One — independent A100 experiment

Train a compact policy/value network from the working heuristic, then test the exact ONNX file against the frozen current model using the Level 4 scene and 100 MCTS simulations. This notebook does not upload a model to the contest.

## Goal

Learn search visit distributions and actual final outcomes with Expert Iteration. A zero-initialized neural correction starts with the baseline's policy and value, avoiding a random initial policy. Archived opponents and validated spatial symmetries diversify training.

Research and primary sources are included in `RESEARCH.md` in the source bundle. Locally tested scripts support CPU and CUDA. The full GPU run must be executed in your Colab session; no A100 speedup or final strength is assumed before that run.

Local results: self-play won 129/200 against the original evaluator; tournament-replay adaptation won 140/200 (70%) on another unused seed set. These results belong to the saved local models, not a future Colab run. The bundle includes the local champion as an archived opponent, replay examples, and whole-match validation splits.

## Setup

Select **Runtime → Change runtime type → A100 GPU**. Upload `level4_lab_bundle.zip` when prompted below. The default training budget is 90 minutes; allow additional time for final matches and validation. Colab GPU availability and runtime limits vary.
""")
    code("""import sys, subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'onnx>=1.22.0', 'onnxruntime>=1.24.3', 'psutil>=7.0'])
import torch
print('Python:', sys.version.split()[0], 'PyTorch:', torch.__version__)
assert torch.cuda.is_available(), 'Select a GPU runtime first.'
print('GPU:', torch.cuda.get_device_name(0))
""")
    md("### Upload the isolated experiment")
    code("""from google.colab import files
from pathlib import Path
import zipfile, os
uploaded = files.upload()
archives = [name for name in uploaded if name.endswith('.zip')]
assert len(archives) == 1, 'Upload level4_lab_bundle.zip.'
workspace = Path('/content/colony_level4_lab')
workspace.mkdir(exist_ok=True)
with zipfile.ZipFile(archives[0]) as archive:
    for member in archive.infolist():
        target = (workspace / member.filename).resolve()
        assert target.is_relative_to(workspace.resolve()), 'Invalid archive path'
    archive.extractall(workspace)
os.chdir(workspace)
print('Workspace:', workspace)
""")
    md("### Save checkpoints in Drive\n\nOnly this experiment's dedicated output folder is written. Enable resume to continue the same run after a disconnect.")
    code("""from google.colab import drive
drive.mount('/content/drive')
RUN = Path('/content/drive/MyDrive/colony-level4-lab/run-a100')
RUN.mkdir(parents=True, exist_ok=True)
TRAIN_MINUTES = 90
GAMES_PER_ITERATION = 32
WORKERS = min(4, os.cpu_count() or 2)
FINAL_GAMES = 100
RESUME = True
SEED = 42
""")
    md("## Steps\n\n### Verify rules and export compatibility before spending GPU time")
    code("""subprocess.check_call([sys.executable, 'checks.py', '--model', 'baselines/current.onnx'])
# Print exact dependency versions for reproducibility.
subprocess.check_call([sys.executable, '-m', 'pip', 'freeze'], stdout=(RUN / 'environment.txt').open('w'))
""")
    md("### Train and select a checkpoint\n\nCPU actors run the published game and search. The parent batches their leaf evaluations on the GPU; gradients use mixed precision. Evaluation always runs the exported ONNX model on CPU, as the referee does. Iteration results are screening measurements, not final evidence. Checkpoints and replay are saved each iteration.")
    code("""command = [sys.executable, '-u', 'train.py', '--minutes', str(TRAIN_MINUTES),
           '--games', str(GAMES_PER_ITERATION), '--workers', str(WORKERS),
           '--simulations', '100', '--channels', '64', '--blocks', '4',
           '--batch-size', '256', '--eval-every', '4', '--eval-games', '16',
           '--final-games', str(FINAL_GAMES), '--seed', str(SEED), '--run', str(RUN)]
examples = workspace / 'data/tournament_training.pkl.gz'
if examples.exists():
    command += ['--examples', str(examples), '--examples-fraction', '0.25']
if RESUME:
    command.append('--resume')
subprocess.check_call(command)
""")
    md("## Checks\n\n### Inspect final results and the game replays\n\nThe final holdout uses seeds excluded from checkpoint selection. More than 50% observed wins is not enough: the default promotion gate requires the lower endpoint of the 95% Wilson interval to exceed 50%. The gate must pass against both the original baseline and the archived local champion, on separate seed sets. This does not establish superiority over unknown teams. Do not repeatedly tune on these final seed sets.")
    code("""import json
from IPython.display import display, HTML
result = json.loads((RUN / 'final_result.json').read_text())
print(f\"Wins: {result['wins']}/{result['games']} ({result['win_rate']:.1%})\")
print('95% interval:', [f'{p:.1%}' for p in result['wilson_95']])
print('Promotion gate:', result['promotion_passed'])
print('Cyan wins:', result['cyan_wins'], 'Magenta wins:', result['magenta_wins'])
if 'champion_comparison' in result:
    champion = result['champion_comparison']
    print('Against local champion:', champion['wins'], '/', champion['games'])
display(HTML((RUN / 'final_match.html').read_text()))
""")
    md("### Validate the candidate\n\nThis runs the supplied validator on the exported file. Its full-match resource check uses the frozen Level 3 scene; the separate holdout above uses the full Level 4 scene and 100 simulations. Local acceptance cannot reproduce every server resource limit.")
    code("""if result['promotion_passed']:
    submission = RUN / 'submission.onnx'
    subprocess.check_call([sys.executable, 'checks.py', '--model', str(submission)])
    subprocess.check_call([sys.executable, 'engine/tools/validate_submission.py', str(submission)])
else:
    print('Improvement was not established. Retain the baseline and inspect losses.')
""")
    md("## Next Steps\n\nDownload only after inspecting the final results and validation. Keep the replay and checkpoint; compare against the other experiment's candidate before choosing the contest upload.")
    code("""if result['promotion_passed']:
    files.download(str(RUN / 'submission.onnx'))
files.download(str(RUN / 'final_match.html'))
""")
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(
        kernelspec=dict(name="python3",display_name="Python 3",language="python"),
        language_info=dict(name="python"),colab=dict(name="colony_a100.ipynb"),accelerator="GPU"),cells=cells)
    for i,cell in enumerate(cells):cell['id']=f'colony-{i:02}'
    (ROOT/'colony_a100.ipynb').write_text(json.dumps(nb,indent=1))
    files=[ROOT/f for f in ['arena.py','model.py','workers.py','train.py','checks.py',
           'tune.py','analyze_losses.py','distill.py','replay.html','scene.json','RESEARCH.md',
           'SUBMISSION.md','README.md','LOSS_ANALYSIS.md'] if (ROOT/f).exists()]
    files+=list((ROOT/'engine').rglob('*.py'))
    files+=[ROOT/'engine/colony_one/scene.json',ROOT/'engine/LICENSE',ROOT/'baselines/current.onnx']
    if (ROOT/'baselines/local_champion.onnx').exists():files.append(ROOT/'baselines/local_champion.onnx')
    files += [ROOT/f for f in ['data/tournament-matches.json','data/tournament_training.pkl.gz',
              'data/tournament_validation.pkl.gz','data/loss_analysis.json'] if (ROOT/f).exists()]
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (ROOT/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2))
    with zipfile.ZipFile(ROOT/'level4_lab_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.relative_to(ROOT))
        z.write(ROOT/'bundle_manifest.json','bundle_manifest.json')
    print('Built notebook and source bundle; notebook code cells parse successfully.')


if __name__=='__main__':main()
