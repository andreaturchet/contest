"""Train a small evaluator by win/loss based parameter search, then hold out seeds."""
from __future__ import annotations
import argparse
import json
import shutil
import time
from pathlib import Path
import numpy as np
from arena import ROOT, match


def main():
    import torch
    from model import FeatureOnly,export
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates",type=int,default=24)
    p.add_argument("--screen-games",type=int,default=8)
    p.add_argument("--confirm-games",type=int,default=32)
    p.add_argument("--holdout-games",type=int,default=100)
    p.add_argument("--seed",type=int,default=771)
    p.add_argument("--local",action="store_true",help="Search near the baseline rather than large changes")
    args=p.parse_args()
    torch.set_num_threads(2)
    rng=np.random.default_rng(args.seed)
    run=ROOT/("runs/tuning_local" if args.local else "runs/tuning");run.mkdir(parents=True,exist_ok=True)
    baseline=ROOT/"baselines/current.onnx"
    rows=[];start=time.monotonic()
    for i in range(args.candidates):
        model=FeatureOnly()
        with torch.no_grad():
            model.features.material[:5]*=torch.tensor(np.exp(rng.normal(0,.35,5)),dtype=torch.float32)
            # Value scale and all policy scores are tuned, rather than fixed
            # to an arbitrary material-to-probability conversion.
            model.features.material*=float(rng.uniform(.7,1.4))
            model.features.capture.copy_(torch.tensor(rng.uniform(.8,3.5,6),dtype=torch.float32))
            model.features.extra.copy_(torch.tensor([
                rng.uniform(.01,.10),rng.uniform(.0,.35),rng.uniform(0,.25),
                rng.uniform(.4,2.5),rng.uniform(0,.25),rng.uniform(0,.4),
                rng.uniform(0,.8),rng.uniform(0,.8),rng.uniform(0,2.5)],dtype=torch.float32))
            if args.local:
                model.features.material.copy_(torch.tensor([.30,.90,.50,.60,1.,2.])*.55)
                model.features.capture.fill_(2.)
                model.features.extra.zero_();model.features.extra[0]=.044
                # Ablate one family at a time before combining changes. Broad
                # perturbations are too destructive near a working incumbent.
                family=i%5
                if family==0:
                    model.features.capture.fill_(float(rng.uniform(.4,4.)))
                elif family==1:
                    model.features.material[:5]*=torch.tensor(np.exp(rng.normal(0,.2,5)),dtype=torch.float32)
                elif family==2:
                    model.features.extra[0]=float(rng.uniform(0,.15))
                elif family==3:
                    feature=int(rng.integers(1,9))
                    model.features.extra[feature]=float(rng.uniform(.01,.2))
                else:
                    model.features.capture.copy_(torch.tensor(2.+rng.normal(0,.5,6),dtype=torch.float32))
        path=run/f"trial_{i:03d}.onnx"
        export(model,path)
        params={k:v.detach().tolist() for k,v in model.features.named_parameters()}
        report=match(path,baseline,args.screen_games,seed=20_000,verbose=False)
        row=dict(trial=i,screen=report["wins"],path=str(path),params=params)
        rows.append(row)
        print(f"Trial {i+1}/{args.candidates}: {report['wins']}/{args.screen_games} "
              f"({time.monotonic()-start:.0f}s)",flush=True)
        (run/"search.json").write_text(json.dumps(rows,indent=2))
    finalists=sorted(rows,key=lambda x:x["screen"],reverse=True)[:min(5,len(rows))]
    for row in finalists:
        r=match(row["path"],baseline,args.confirm_games,seed=60_000,verbose=False)
        row["confirm"]=r["wins"]
        print(f"Confirm trial {row['trial']}: {r['wins']}/{args.confirm_games}",flush=True)
    best=max(finalists,key=lambda x:x["confirm"])
    candidate=run/"candidate.onnx";shutil.copyfile(best["path"],candidate)
    (run/"selected_parameters.json").write_text(json.dumps(best,indent=2))
    # These seeds are never used in parameter selection above.
    report=match(candidate,baseline,args.holdout_games,seed=1_000_000,
                 output=ROOT/("replays/local_holdout" if args.local else "replays/tuned_holdout"))
    report["selection"]=best
    report["promotion_passed"]=report["wilson_95"][0]>.5
    (run/"holdout.json").write_text(json.dumps(report,indent=2))
    if report["promotion_passed"]:
        shutil.copyfile(candidate,ROOT/"candidate.onnx")
        print("Candidate passes the baseline holdout gate.",flush=True)
    else:
        print("Improvement not established; retain baseline.",flush=True)


if __name__=="__main__":main()
