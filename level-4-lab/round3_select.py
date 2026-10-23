"""Select on new screens, then confirm one frozen export on unused games."""
from __future__ import annotations
import hashlib
import json
import shutil
from pathlib import Path
from arena import ROOT,match


def main():
    run=ROOT/'runs/round3';candidates=[]
    for history in sorted(run.glob('*/history.json')):
        for row in json.loads(history.read_text()):
            candidates.append(dict(name=history.parent.name,step=row['step'],
                path=str((history.parent/f"step_{row['step']}.onnx").resolve()),screen=row['screen']))
    candidates.sort(key=lambda r:(r['screen']['win_rate'],-r['step']),reverse=True)
    # Bound selection: choose at most one checkpoint per training recipe.
    top=[];seen=set()
    for row in candidates:
        if row['name'] not in seen:top.append(row);seen.add(row['name'])
        if len(top)==3:break
    (run/'selection_protocol.json').write_text(json.dumps(dict(
        first_screen=candidates,finalist_names=[r['name'] for r in top],
        second_screen_seed=230_000_000,older_screen_seed=231_000_000,
        final_v2_seed=240_000_000,final_original_seed=250_000_000,final_earlier_seed=260_000_000,
        selection_score='v2 wins/96 + 0.25 * earlier wins/64',
        promotion='Fresh 600 vs v2 and 200 vs original Wilson lower > 0.5; earlier 200 point estimate >= 0.5'),indent=2))
    for row in top:
        row['v2_screen']=match(row['path'],ROOT/'candidate_v2.onnx',96,seed=230_000_000,verbose=False)
        row['earlier_screen']=match(row['path'],ROOT/'candidates/selfplay_129_of_200.onnx',64,seed=231_000_000,verbose=False)
        row['score']=row['v2_screen']['win_rate']+.25*row['earlier_screen']['win_rate']
        print('FINALIST',row['name'],row['step'],row['v2_screen']['wins'],'/96 v2',row['earlier_screen']['wins'],'/64 earlier',flush=True)
        (run/'finalists.json').write_text(json.dumps(top,indent=2))
    selected=max(top,key=lambda r:r['score'])
    frozen=run/'selected.onnx';shutil.copyfile(selected['path'],frozen)
    result=dict(selection=selected,sha256=hashlib.sha256(frozen.read_bytes()).hexdigest())
    (run/'selected.json').write_text(json.dumps(result,indent=2))
    result['v2']=match(frozen,ROOT/'candidate_v2.onnx',600,seed=240_000_000,output=run/'final_v2',verbose=False)
    print('FINAL_V2',result['v2']['wins'],'/600',result['v2']['wilson_95'],flush=True)
    (run/'final_result.json').write_text(json.dumps(result,indent=2))
    result['original']=match(frozen,ROOT/'baselines/current.onnx',200,seed=250_000_000,output=run/'final_original',verbose=False)
    result['earlier']=match(frozen,ROOT/'candidates/selfplay_129_of_200.onnx',200,seed=260_000_000,output=run/'final_earlier',verbose=False)
    result['promotion_passed']=(result['v2']['wilson_95'][0]>.5 and result['original']['wilson_95'][0]>.5 and result['earlier']['win_rate']>=.5)
    (run/'final_result.json').write_text(json.dumps(result,indent=2))
    if result['promotion_passed']:
        shutil.copyfile(frozen,ROOT/'candidate_v3.onnx')
        shutil.copyfile(Path(selected['path']).with_suffix('.pt'),run/'selected_weights.pt')
    print('ROUND3_RESULT',json.dumps(result),flush=True)


if __name__=='__main__':main()
