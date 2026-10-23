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
    p.add_argument("--training",type=Path,default=ROOT/"data/tournament_training.pkl.gz")
    p.add_argument("--validation",type=Path,default=ROOT/"data/tournament_validation.pkl.gz")
    p.add_argument("--anchor",type=Path,default=ROOT/"runs/local_neural/best.onnx")
    p.add_argument("--seed",type=int,default=12091)
    p.add_argument("--final-seed",type=int,default=4_000_000)
    p.add_argument("--tournament-fraction",type=float,default=.65)
    p.add_argument("--auxiliary",type=float,default=0.,help="Training-only tactical classification loss weight")
    p.add_argument("--aux-oracle",choices=["direct","two-action"],default="direct")
    args=p.parse_args();args.run.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed)
    state=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    net=Model(**state['architecture']);net.load_state_dict(state['model'])
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");net.to(device)
    with gzip.open(args.training,"rb") as f:training=pickle.load(f)
    with gzip.open(args.validation,"rb") as f:validation=pickle.load(f)
    with gzip.open(args.replay,"rb") as f:replay=pickle.load(f)
    auxiliary=None;train_labels=None;replay_labels=None
    if args.auxiliary>0:
        from tactical_model import DirectKernelAttack,TwoActionKernelAttack
        # Extra heads train the shared trunk and disappear from the export.
        auxiliary=torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten(),
            torch.nn.Linear(state['architecture']['channels'],32),torch.nn.ReLU(),torch.nn.Linear(32,3)).to(device)
        oracle=(TwoActionKernelAttack() if args.aux_oracle=="two-action" else DirectKernelAttack()).to(device)
        replay=replay[-12000:]
        def labels(items):
            targets=[]
            with torch.inference_mode():
                for begin in range(0,len(items),128):
                    x=torch.from_numpy(np.stack([row[0] for row in items[begin:begin+128]]).astype(np.float32)).to(device)
                    own,_=oracle(x)
                    flipped=torch.cat([x[:,6:12],x[:,:6],x[:,12:]],dim=1).clone()
                    flipped[:,13]=1.;flipped[:,15]=1.-flipped[:,15]
                    enemy,_=oracle(flipped)
                    guards=(F.conv2d(x[:,:5].sum(1,keepdim=True),torch.ones(1,1,3,3,device=device),padding=1)*x[:,5:6]).sum((1,2,3))>0
                    targets.append(torch.stack([own,enemy,guards],1).float().cpu().numpy())
            return np.concatenate(targets)
        train_labels=labels(training);replay_labels=labels(replay)
        print('Auxiliary label means:',train_labels.mean(0).tolist(),replay_labels.mean(0).tolist(),flush=True)
    trainable=list(net.parameters())+(list(auxiliary.parameters()) if auxiliary is not None else [])
    opt=torch.optim.AdamW(trainable,lr=2e-4,weight_decay=1e-4)
    best=args.run/"best.onnx";export(net,best);net.to(device)
    baseline=ROOT/"baselines/current.onnx"
    anchor=args.anchor
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
        net.train();batch=[];aux_targets=[]
        for _ in range(128):
            use_tournament=rng.random()<args.tournament_fraction
            source=training if use_tournament else replay
            index=int(rng.integers(len(source)))
            batch.append(augment(source[index],rng))
            if auxiliary is not None:aux_targets.append((train_labels if use_tournament else replay_labels)[index])
        opt.zero_grad(set_to_none=True)
        pl,vl=loss(batch);total=pl+vl
        if auxiliary is not None:
            x=torch.from_numpy(np.stack([s[0] for s in batch]).astype(np.float32)).to(device)
            predictions=auxiliary(net.trunk(x))
            al=F.binary_cross_entropy_with_logits(predictions,torch.from_numpy(np.stack(aux_targets)).to(device),
                pos_weight=torch.tensor([5.,5.,1.],device=device))
            total=total+args.auxiliary*al
        total.backward()
        torch.nn.utils.clip_grad_norm_(trainable,1.);opt.step()
        if step%100==0 or step==args.steps:
            net.eval()
            with torch.inference_mode():vp,vv=loss(validation,False)
            candidate=export(net,args.run/f"step_{step}.onnx");net.to(device)
            # Fixed screening seeds are deliberately separate from final tests.
            base=match(candidate,baseline,args.screen_games,seed=args.final_seed-2_000_000,verbose=False)
            old=match(candidate,anchor,args.screen_games,seed=args.final_seed-1_000_000,verbose=False)
            row=dict(step=step,baseline_wins=base['wins'],anchor_wins=old['wins'],
                     validation_policy_ce=vp.item(),validation_value_mse=vv.item(),
                     seconds=time.monotonic()-t)
            if auxiliary is not None:row['auxiliary_loss']=al.item()
            score=base['wins']+old['wins']
            if score>best_score:
                best_score=score;shutil.copyfile(candidate,best);row['selected']=True
                torch.save(dict(architecture=state['architecture'],model={k:v.detach().cpu() for k,v in net.state_dict().items()}),args.run/"adapted_weights.pt")
            history.append(row);print(json.dumps(row),flush=True)
            (args.run/"history.json").write_text(json.dumps(history,indent=2))
    result=match(best,baseline,args.final_games,seed=args.final_seed,output=args.run/"final_match")
    result['promotion_passed']=result['wilson_95'][0]>.5
    incumbent=match(best,anchor,args.final_games,seed=args.final_seed+1_000_000,output=args.run/"incumbent_match")
    result['incumbent_comparison']=incumbent
    result['promotion_passed']=result['promotion_passed'] and incumbent['wilson_95'][0]>.5
    (args.run/"final_result.json").write_text(json.dumps(result,indent=2))
    if result['promotion_passed']:
        shutil.copyfile(best,args.run/"submission.onnx")
        # Preserve the previously offered upload file; publish a separate version.
        shutil.copyfile(best,ROOT/"candidate_v2_recent.onnx")
        print("Tournament-adapted candidate passes the baseline holdout gate.",flush=True)
    else:print("No established improvement on the new holdout; retain the baseline.",flush=True)


if __name__=="__main__":main()
