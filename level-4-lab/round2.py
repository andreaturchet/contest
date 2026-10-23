"""Screen literature-inspired variants, then test one on untouched seeds."""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from arena import ROOT,match
from inference_variants import calibrated,mixture,symmetry


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--screen-games',type=int,default=12)
    p.add_argument('--confirm-games',type=int,default=40)
    p.add_argument('--final-games',type=int,default=200)
    args=p.parse_args()
    run=ROOT/'runs/round2';run.mkdir(parents=True,exist_ok=True)
    champion=run/'incumbent.onnx'
    if not champion.exists():shutil.copyfile(ROOT/'candidate.onnx',champion)
    old=ROOT/'candidates/selfplay_129_of_200.onnx'
    original=ROOT/'baselines/current.onnx'
    specs=[]
    for group in ['halfturn','rotations','reflections','d4']:
        path=run/f'symmetry_{group}.onnx';symmetry(champion,path,group)
        specs.append(dict(name=f'symmetry_{group}',path=str(path)))
    for w in [.25,.5,.75]:
        path=run/f'mix_{w}.onnx';mixture([champion,old],[w,1-w],path)
        specs.append(dict(name=f'mix_{w}',path=str(path)))
    for ps,vs in [(.5,1.),(.75,1.),(1.5,1.),(2.,1.),(1.,.5),(1.,.75),(1.,1.5),(1.,2.)]:
        path=run/f'calibrate_p{ps}_v{vs}.onnx';calibrated(champion,path,ps,vs)
        specs.append(dict(name=f'calibrate_p{ps}_v{vs}',path=str(path)))
    rows=[];start=time.monotonic()
    for spec in specs:
        result=match(spec['path'],champion,args.screen_games,seed=10_000_000,verbose=False)
        row=dict(**spec,screen=result,seconds=time.monotonic()-start)
        rows.append(row)
        print(f"{spec['name']}: {result['wins']}/{args.screen_games} vs incumbent ({time.monotonic()-start:.0f}s)",flush=True)
        (run/'screening.json').write_text(json.dumps(rows,indent=2))
    finalists=sorted(rows,key=lambda r:r['screen']['wins'],reverse=True)[:3]
    for row in finalists:
        vs_champion=match(row['path'],champion,args.confirm_games,seed=11_000_000,verbose=False)
        vs_old=match(row['path'],old,args.confirm_games,seed=12_000_000,verbose=False)
        row['confirmation']=dict(incumbent=vs_champion,earlier_student=vs_old)
        row['score']=vs_champion['wins']+vs_old['wins']
        print(f"Confirm {row['name']}: {vs_champion['wins']}/{args.confirm_games} vs incumbent, {vs_old['wins']}/{args.confirm_games} vs earlier",flush=True)
        (run/'finalists.json').write_text(json.dumps(finalists,indent=2))
    best=max(finalists,key=lambda r:r['score'])
    selected=run/'selected.onnx';shutil.copyfile(best['path'],selected)
    (run/'selection.json').write_text(json.dumps(best,indent=2))
    # No final seed is read during selection.
    final=match(selected,champion,args.final_games,seed=20_000_000,output=run/'final_incumbent')
    final['selection']=best['name']
    final['promotion_passed']=final['wilson_95'][0]>.5
    if final['promotion_passed']:
        against_old=match(selected,old,100,seed=21_000_000,output=run/'final_earlier')
        against_original=match(selected,original,100,seed=22_000_000,output=run/'final_original')
        final['earlier_student']=against_old;final['original']=against_original
        # An ensemble must also maintain a positive result against the original.
        final['promotion_passed']=against_original['wilson_95'][0]>.5
        if final['promotion_passed']:shutil.copyfile(selected,ROOT/'candidate_v2.onnx')
    (run/'final_result.json').write_text(json.dumps(final,indent=2))
    print('ROUND2_RESULT',json.dumps(final),flush=True)


if __name__=='__main__':main()
