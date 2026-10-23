"""Package the verified second candidate as a resumable Colab starting point."""
from pathlib import Path
from collections import Counter
import ast
import gzip
import hashlib
import json
import pickle
import zipfile

ROOT=Path(__file__).resolve().parent


def key(row):return tuple(v.tobytes() if hasattr(v,'tobytes') else str(v) for v in row)


def merge_examples(first,second):
    # Keep the intentional repetition of tactical repairs, while not doubling
    # identical examples from matches present in both exports.
    counts=Counter(key(s) for s in first);added=Counter();out=list(first)
    for sample in second:
        k=key(sample);added[k]+=1
        if added[k]>counts[k]:out.append(sample)
    return out


def main():
    result=json.loads((ROOT/'replays/v2_confirmation.json').read_text())['report']
    if result['wilson_95'][0]<=.5:raise ValueError('V2 did not establish an improvement')
    staging=ROOT/'runs/colab_v2';staging.mkdir(exist_ok=True)
    generated={}
    for part in ['training','validation']:
        datasets=[]
        for folder in ['recent_tournament','candidate_only_tournament']:
            with gzip.open(ROOT/f'runs/{folder}/data/tournament_{part}.pkl.gz','rb') as f:datasets.append(pickle.load(f))
        merged=merge_examples(*datasets)
        p=staging/f'{part}.pkl.gz'
        with gzip.open(p,'wb',compresslevel=1) as f:pickle.dump(merged,f,protocol=5)
        generated[f'data/tournament_{part}.pkl.gz']=p
    with gzip.open(ROOT/'runs/league_v2/replay.pkl.gz','rb') as f:replay=pickle.load(f)
    with gzip.open(staging/'initial_replay.pkl.gz','wb',compresslevel=1) as f:pickle.dump(replay[-12000:],f,protocol=5)
    nb=json.loads((ROOT/'colony_a100.ipynb').read_text())
    for cell in nb['cells']:
        s=''.join(cell['source'])
        s=s.replace('level4_lab_bundle.zip','level4_lab_bundle_v2.zip')
        s=s.replace('/content/colony_level4_lab','/content/colony_level4_lab_v2')
        s=s.replace('colony-level4-lab/run-a100','colony-level4-lab/run-a100-v2')
        s=s.replace('TRAIN_MINUTES = 90','TRAIN_MINUTES = 45')
        s=s.replace("'--final-games', str(FINAL_GAMES),", "'--final-games', str(FINAL_GAMES), '--final-seed', '200000000',")
        s=s.replace("'--simulations', '100', '--channels', '64', '--blocks', '4',", "'--simulations', '160', '--channels', '32', '--blocks', '2',\n           '--warm-start', str(workspace / 'warm_start.pt'),\n           '--initial-replay', str(workspace / 'initial_replay.pkl.gz'),\n           '--opponent', str(workspace / 'candidates/selfplay_129_of_200.onnx'),")
        if cell['cell_type']=='code':ast.parse(s)
        cell['source']=s.splitlines(True)
    lead=nb['cells'][0]
    lead['source']=("# Colony: One — continue from candidate v2\n\n"
        f"The starting model won {result['wins']}/{result['games']} fresh Level 4 matches against the previous candidate. "
        "This notebook starts from those verified weights and cached replay, with newer tournament examples. "
        "The saved local champion is an opponent and a final promotion gate. Training uses 160 simulations; final evaluation always uses the referee's 100.\n\n"
        "Select A100, run the cells, and upload `level4_lab_bundle_v2.zip`. The default training budget is 45 minutes, plus final evaluation. "
        "The notebook and CUDA branch have not been executed in an attached A100 runtime here; the corresponding CPU pipeline has been tested.\n").splitlines(True)
    nb['metadata']['colab']['name']='colony_a100_v2.ipynb'
    (ROOT/'colony_a100_v2.ipynb').write_text(json.dumps(nb,indent=1))
    names=['arena.py','model.py','workers.py','train.py','checks.py','replay.html','scene.json',
           'tactical_model.py','distill.py','RESEARCH.md','RESEARCH_ROUND2.md','RESULTS_ROUND2.md']
    entries={name:ROOT/name for name in names if (ROOT/name).exists()}
    entries.update({str(p.relative_to(ROOT)):p for p in (ROOT/'engine').rglob('*.py')})
    entries.update({'engine/colony_one/scene.json':ROOT/'engine/colony_one/scene.json',
        'engine/LICENSE':ROOT/'engine/LICENSE','baselines/current.onnx':ROOT/'baselines/current.onnx',
        'baselines/local_champion.onnx':ROOT/'runs/recent_auxiliary/best.onnx',
        'candidates/selfplay_129_of_200.onnx':ROOT/'candidates/selfplay_129_of_200.onnx',
        'warm_start.pt':ROOT/'runs/recent_auxiliary/adapted_weights.pt',
        'initial_replay.pkl.gz':staging/'initial_replay.pkl.gz',**generated})
    manifest={name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in entries.items()}
    with zipfile.ZipFile(ROOT/'level4_lab_bundle_v2.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name,p in entries.items():z.write(p,name)
        z.writestr('bundle_manifest.json',json.dumps(manifest,indent=2))
    print('Built Colab v2 notebook, warm weights, replay, examples, and checksummed source bundle')


if __name__=='__main__':main()
