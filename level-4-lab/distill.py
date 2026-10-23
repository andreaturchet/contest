"""Adapt the student to replay-derived deep-search targets and tactical repairs."""
from __future__ import annotations
import argparse
import gzip
import json
import pickle
import shutil
import time
from pathlib import Path
import numpy as np
from arena import ROOT,match
from workers import augment


def main():
    import torch
    from torch.nn import functional as F
    from model import Model,export
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,default=ROOT/"runs/local_neural/checkpoint.pt")
    p.add_argument("--replay",type=Path,default=ROOT/"runs/local_neural/replay.pkl.gz")
    p.add_argument("--steps",type=int,default=600)
    p.add_argument("--screen-games",type=int,default=24)
    p.add_argument("--final-games",type=int,default=200)
    p.add_argument("--run",type=Path,default=ROOT/"runs/tournament_adapted")
    args=p.parse_args();args.run.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(12091)
    rng=np.random.default_rng(12091)
    state=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    net=Model(**state['architecture']);net.load_state_dict(state['model'])
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");net.to(device)
    with gzip.open(ROOT/"data/tournament_training.pkl.gz","rb") as f:training=pickle.load(f)
    with gzip.open(ROOT/"data/tournament_validation.pkl.gz","rb") as f:validation=pickle.load(f)
    with gzip.open(args.replay,"rb") as f:replay=pickle.load(f)
    opt=torch.optim.AdamW(net.parameters(),lr=2e-4,weight_decay=1e-4)
    best=args.run/"best.onnx";export(net,best);net.to(device)
    baseline=ROOT/"baselines/current.onnx"
    anchor=ROOT/"runs/local_neural/best.onnx"
    best_score=-1;history=[];t=time.monotonic()

    def loss(batch,training_mode=True):
        x=torch.from_numpy(np.stack([s[0] for s in batch]).astype(np.float32)).to(device)
        width=max(len(s[1]) for s in batch)
        idx=np.zeros((len(batch),width),np.int64);pi=np.zeros((len(batch),width),np.float32);mask=np.zeros((len(batch),width),bool)
        for i,(_,legal,prob,_) in enumerate(batch):idx[i,:len(legal)]=legal;pi[i,:len(prob)]=prob;mask[i,:len(legal)]=True
        logits,v=net(x)
        logits=logits.gather(1,torch.from_numpy(idx).to(device)).masked_fill(~torch.from_numpy(mask).to(device),-1e9)
        pl=-(torch.from_numpy(pi).to(device)*F.log_softmax(logits,1)).sum(1).mean()
        vl=F.mse_loss(v,torch.tensor([s[3] for s in batch],device=device))
        return pl,vl

    for step in range(1,args.steps+1):
        net.train();batch=[]
        for _ in range(128):
            source=training if rng.random()<.65 else replay
            batch.append(augment(source[int(rng.integers(len(source)))],rng))
        opt.zero_grad(set_to_none=True)
        pl,vl=loss(batch);(pl+vl).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
        if step%100==0 or step==args.steps:
            net.eval()
            with torch.inference_mode():vp,vv=loss(validation,False)
            candidate=export(net,args.run/f"step_{step}.onnx");net.to(device)
            # Fixed screening seeds are deliberately separate from final tests.
            base=match(candidate,baseline,args.screen_games,seed=610_000,verbose=False)
            old=match(candidate,anchor,args.screen_games,seed=710_000,verbose=False)
            row=dict(step=step,baseline_wins=base['wins'],anchor_wins=old['wins'],
                     validation_policy_ce=vp.item(),validation_value_mse=vv.item(),
                     seconds=time.monotonic()-t)
            score=base['wins']+old['wins']
            if score>best_score:
                best_score=score;shutil.copyfile(candidate,best);row['selected']=True
                torch.save(dict(architecture=state['architecture'],model={k:v.detach().cpu() for k,v in net.state_dict().items()}),args.run/"adapted_weights.pt")
            history.append(row);print(json.dumps(row),flush=True)
            (args.run/"history.json").write_text(json.dumps(history,indent=2))
    result=match(best,baseline,args.final_games,seed=4_000_000,output=args.run/"final_match")
    result['promotion_passed']=result['wilson_95'][0]>.5
    incumbent=match(best,anchor,100,seed=5_000_000,output=args.run/"incumbent_match")
    result['incumbent_comparison']=incumbent
    result['promotion_passed']=result['promotion_passed'] and incumbent['wilson_95'][0]>.5
    (args.run/"final_result.json").write_text(json.dumps(result,indent=2))
    if result['promotion_passed']:
        shutil.copyfile(best,args.run/"submission.onnx")
        shutil.copyfile(best,ROOT/"candidate.onnx")
        print("Tournament-adapted candidate passes the baseline holdout gate.",flush=True)
    else:print("No established improvement on the new holdout; retain the baseline.",flush=True)


if __name__=="__main__":main()
