"""Independent screens for shared-ancestry weight averages and exact tactics."""
from __future__ import annotations
import json
import shutil
import time
import numpy as np
from arena import ROOT,match


def main():
    import torch
    from model import Model,export
    from tactical_model import TacticalWrapper
    torch.set_num_threads(2)
    run=ROOT/'runs/round2_followup';run.mkdir(parents=True,exist_ok=True)
    champion=run/'incumbent.onnx'
    if not champion.exists():shutil.copyfile(ROOT/'candidate.onnx',champion)
    adapted=torch.load(ROOT/'runs/tournament_adapted/adapted_weights.pt',map_location='cpu',weights_only=False)
    endpoint=torch.load(ROOT/'runs/local_neural/checkpoint.pt',map_location='cpu',weights_only=False)
    assert adapted['architecture']==endpoint['architecture']
    reference=Model(**adapted['architecture'])
    parameters={name for name,_ in reference.named_parameters()}
    specs=[]
    for fraction in [.25,.5,.75]:
        state={}
        for name,value in adapted['model'].items():
            if name in parameters:state[name]=fraction*value+(1-fraction)*endpoint['model'][name]
            else:
                assert torch.equal(value,endpoint['model'][name]),name
                state[name]=value
        model=Model(**adapted['architecture']);model.load_state_dict(state)
        path=export(model,run/f'weight_average_{fraction}.onnx')
        specs.append(dict(name=f'weight_average_{fraction}',path=str(path)))
    for boost in [0.,2.,6.]:
        model=Model(**adapted['architecture']);model.load_state_dict(adapted['model'])
        path=export(TacticalWrapper(model,boost=boost),run/f'tactical_{boost}.onnx')
        specs.append(dict(name=f'tactical_{boost}',path=str(path)))
    rows=[];start=time.monotonic()
    for spec in specs:
        r=match(spec['path'],champion,16,seed=30_000_000,verbose=False)
        row=dict(**spec,screen=r);rows.append(row)
        print(f"{spec['name']}: {r['wins']}/16 incumbent ({time.monotonic()-start:.0f}s)",flush=True)
        (run/'screening.json').write_text(json.dumps(rows,indent=2))
    top=sorted(rows,key=lambda r:r['screen']['wins'],reverse=True)[:2]
    for row in top:
        a=match(row['path'],champion,48,seed=31_000_000,verbose=False)
        b=match(row['path'],ROOT/'candidates/selfplay_129_of_200.onnx',48,seed=32_000_000,verbose=False)
        row['confirmation']=dict(incumbent=a,earlier_student=b)
        row['score']=a['wins']+b['wins']
        print(f"Confirm {row['name']}: {a['wins']}/48 incumbent, {b['wins']}/48 earlier",flush=True)
        (run/'finalists.json').write_text(json.dumps(top,indent=2))
    best=max(top,key=lambda r:r['score'])
    selected=run/'selected.onnx';shutil.copyfile(best['path'],selected)
    report=match(selected,champion,200,seed=40_000_000,output=run/'final_incumbent')
    report['selection']=best
    report['promotion_passed']=report['wilson_95'][0]>.5
    (run/'final_result.json').write_text(json.dumps(report,indent=2))
    if report['promotion_passed']:shutil.copyfile(selected,ROOT/'candidate_v2_followup.onnx')
    print('FOLLOWUP_RESULT',report['wins'],report['games'],report['promotion_passed'],flush=True)


if __name__=='__main__':main()
