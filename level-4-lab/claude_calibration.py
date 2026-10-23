"""Test Claude Code's bounded calibration proposal on frozen v2."""
import json
import hashlib
from pathlib import Path
import numpy as np
from arena import ROOT,match


def main():
    import onnx
    from onnx import helper as H,numpy_helper as NH
    from inference_variants import calibrated,write
    run=ROOT/'runs/round3/claude_calibration';run.mkdir(parents=True,exist_ok=True)
    rows=[]
    # These files are separate from the trained finalist and never replace v2.
    for scale in [.7,.85,1.2,1.5]:
        path=calibrated(ROOT/'candidate_v2.onnx',run/f'policy_{scale}.onnx',policy_scale=scale)
        report=match(path,ROOT/'candidate_v2.onnx',150,seed=300_000_000,verbose=False)
        rows.append(dict(policy_scale=scale,value_pretanh_scale=1.,path=str(path),screen=report))
        (run/'policy_screens.json').write_text(json.dumps(rows,indent=2))
        print('CLAUDE_POLICY',scale,report['wins'],'/150',flush=True)
    best=max(rows,key=lambda x:x['screen']['win_rate'])
    final=[best]
    for scale in [.7,1.4]:
        model=onnx.load(best['path'])
        # Scale the existing raw value immediately before Tanh. No inverse
        # tanh or approximated raw logit is needed at inference.
        tanh=[n for n in model.graph.node if n.op_type=='Tanh']
        if len(tanh)!=1:raise ValueError('Expected one value Tanh')
        target=tanh[0];original=target.input[0];target.input[0]='claude_scaled_raw_value'
        model.graph.initializer.append(NH.from_array(np.asarray(scale,np.float32),'claude_raw_scale'))
        index=next(i for i,n in enumerate(model.graph.node) if n.op_type=='Tanh')
        model.graph.node.insert(index,H.make_node('Mul',[original,'claude_raw_scale'],['claude_scaled_raw_value']))
        path=write(model,run/f'policy_{best["policy_scale"]}_raw_{scale}.onnx')
        report=match(path,ROOT/'candidate_v2.onnx',150,seed=301_000_000,verbose=False)
        final.append(dict(policy_scale=best['policy_scale'],value_pretanh_scale=scale,path=str(path),screen=report))
        (run/'value_screens.json').write_text(json.dumps(final,indent=2))
        print('CLAUDE_VALUE',scale,report['wins'],'/150',flush=True)
    # A fresh common seed set compares the final scale candidates fairly.
    for row in final:
        row['selection_match']=match(row['path'],ROOT/'candidate_v2.onnx',100,seed=302_000_000,verbose=False)
    selected=max(final,key=lambda x:x['selection_match']['win_rate'])
    (run/'selected.json').write_text(json.dumps(selected,indent=2))
    # Bound the screen family; no extra grid is added after seeing this set.
    if selected['selection_match']['win_rate']>.5:
        report=match(selected['path'],ROOT/'candidate_v2.onnx',400,seed=310_000_000,output=run/'final_v2',verbose=True)
        (run/'final_result.json').write_text(json.dumps(dict(selection=selected,v2=report,promotion_passed=report['wilson_95'][0]>.5),indent=2))
        print('CLAUDE_FINAL',report['wins'],'/400',report['wilson_95'],flush=True)
    else:
        (run/'final_result.json').write_text(json.dumps(dict(selection=selected,promotion_passed=False,reason='No positive selection-match result'),indent=2))


if __name__=='__main__':main()
