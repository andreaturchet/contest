"""Improve move priors while preserving the frozen v2 value function."""
from __future__ import annotations
import argparse
import copy
import gzip
import json
import pickle
import time
from pathlib import Path
import numpy as np
from arena import ROOT,match
from workers import augment


def main():
    import torch
    from torch.nn import functional as F
    from model import Model,export
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--kl',type=float,default=.1)
    parser.add_argument('--kl-direction',choices=['anchor','current'],default='anchor')
    parser.add_argument('--weight-decay',type=float,default=1e-4)
    parser.add_argument('--steps',type=int,default=900)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();args.run.mkdir(parents=True,exist_ok=True)
    (args.run/'config.json').write_text(json.dumps(vars(args),default=str,indent=2))
    torch.set_num_threads(2);torch.manual_seed(3471);rng=np.random.default_rng(3471)
    state=torch.load(ROOT/'runs/recent_auxiliary/adapted_weights.pt',map_location='cpu',weights_only=False)
    net=Model(**state['architecture']);net.load_state_dict(state['model']);anchor=copy.deepcopy(net).eval()
    for p in net.parameters():p.requires_grad_(False)
    for module in [net.source,net.target,net.from_bias,net.to_bias]:
        for p in module.parameters():p.requires_grad_(True)
    net.features.capture.requires_grad_(True)
    trainable=[p for p in net.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(trainable,lr=args.lr,weight_decay=args.weight_decay)
    for p in anchor.parameters():p.requires_grad_(False)
    with gzip.open(ROOT/'runs/round3/data/training.pkl.gz','rb') as f:source=pickle.load(f)
    # Value targets are intentionally unused; only deep-search legal policies
    # supervise the trainable heads. KL retains the frozen network's prior.
    source=[(x,l,p,z) for x,l,p,z,q in source]
    with gzip.open(ROOT/'runs/round3/data/validation.pkl.gz','rb') as f:validation=pickle.load(f)
    test=torch.from_numpy(np.stack([r[0] for r in validation]).astype(np.float32))
    with torch.inference_mode():_,original_values=anchor(test)
    history=[];started=time.monotonic()
    for step in range(1,args.steps+1):
        batch=[augment(source[int(i)],rng) for i in rng.integers(len(source),size=128)]
        x=torch.from_numpy(np.stack([r[0] for r in batch]))
        width=max(len(s[1]) for s in batch)
        indices=np.zeros((len(batch),width),np.int64);pi=np.zeros((len(batch),width),np.float32);mask=np.zeros((len(batch),width),bool)
        for i,(_,l,p,_) in enumerate(batch):indices[i,:len(l)]=l;pi[i,:len(l)]=p;mask[i,:len(l)]=True
        idx=torch.from_numpy(indices);valid=torch.from_numpy(mask);targets=torch.from_numpy(pi)
        opt.zero_grad(set_to_none=True);policy,_=net(x)
        legal=policy.gather(1,idx).masked_fill(~valid,-1e9);logp=F.log_softmax(legal,1)
        with torch.no_grad():
            old,_=anchor(x);oldlogp=F.log_softmax(old.gather(1,idx).masked_fill(~valid,-1e9),1);oldp=oldlogp.exp()
        pl=-(targets*logp).sum(1).mean()
        kl=(oldp*(oldlogp-logp)).sum(1).mean() if args.kl_direction=='anchor' else (logp.exp()*(logp-oldlogp)).sum(1).mean()
        loss=pl+args.kl*kl;loss.backward();torch.nn.utils.clip_grad_norm_(trainable,1.);opt.step()
        if step%300==0 or step==args.steps:
            net.eval()
            with torch.inference_mode():_,values=net(test)
            assert torch.equal(original_values,values),'Value function changed'
            with torch.inference_mode():
                allpol,_=net(test);oldpol,_=anchor(test);ce=[];ent=[];oldent=[]
                for i,(_,legal,prob,z,q) in enumerate(validation):
                    ids=torch.from_numpy(legal.astype(np.int64));target=torch.from_numpy(prob.astype(np.float32))
                    lp=F.log_softmax(allpol[i,ids],0);op=F.log_softmax(oldpol[i,ids],0)
                    ce.append(float(-(target*lp).sum()));ent.append(float(-(lp.exp()*lp).sum()));oldent.append(float(-(op.exp()*op).sum()))
            frozen_keys=[name for name,_ in net.named_parameters() if not dict(net.named_parameters())[name].requires_grad]
            assert all(torch.equal(net.state_dict()[name],state['model'][name]) for name in frozen_keys)
            path=export(net,args.run/f'step_{step}.onnx')
            torch.save(dict(architecture=state['architecture'],model=net.state_dict()),args.run/f'step_{step}.pt')
            result=match(path,ROOT/'candidate_v2.onnx',64,seed=320_000_000,verbose=False)
            row=dict(step=step,trainable_parameters=sum(p.numel() for p in trainable),policy_loss=pl.item(),
                kl=kl.item(),validation_policy_ce=float(np.mean(ce)),validation_entropy=float(np.mean(ent)),
                original_validation_entropy=float(np.mean(oldent)),max_value_change=0.,screen=result,seconds=time.monotonic()-started)
            history.append(row);(args.run/'history.json').write_text(json.dumps(history,indent=2))
            print(json.dumps(row),flush=True)


if __name__=='__main__':main()
