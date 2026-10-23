"""Confirm the selected trained model after explicit numerical output bounds."""
import json
import hashlib
import shutil
from pathlib import Path
from arena import ROOT,match


def main():
    run=ROOT/'runs/round3'
    selected=json.loads((run/'selected.json').read_text())
    model=run/'selected_bounded.onnx'
    result=dict(selection=selected['selection'],sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
        numerical_fix='Explicit Clip(value,-1,1) after exported tanh; validator enforces exact floating-point bounds')
    for name,opponent,games,seed in [('v2',ROOT/'candidate_v2.onnx',600,270_000_000),
          ('original',ROOT/'baselines/current.onnx',200,280_000_000),
          ('earlier',ROOT/'candidates/selfplay_129_of_200.onnx',200,290_000_000)]:
        result[name]=match(model,opponent,games,seed=seed,output=run/f'bounded_final_{name}',verbose=True)
        (run/'bounded_final_result.json').write_text(json.dumps(result,indent=2))
    result['promotion_passed']=(result['v2']['wilson_95'][0]>.5 and result['original']['wilson_95'][0]>.5 and result['earlier']['win_rate']>=.5)
    if result['promotion_passed']:
        shutil.copyfile(model,ROOT/'candidate_v3.onnx')
        shutil.copyfile(Path(result['selection']['path']).with_suffix('.pt'),run/'selected_weights.pt')
    (run/'bounded_final_result.json').write_text(json.dumps(result,indent=2))
    print('BOUNDED_ROUND3_RESULT',result['promotion_passed'],flush=True)


if __name__=='__main__':main()
