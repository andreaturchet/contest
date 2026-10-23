"""Controlled warm-v2 objective experiments; inference remains the same model."""
from __future__ import annotations
import argparse
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
    from tactical_model import DirectKernelAttack
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['mse','bce'],default='bce')
    p.add_argument('--reanalysed',action='store_true')
    p.add_argument('--q-mix',type=float,default=.5)
    p.add_argument('--steps',type=int,default=600)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--soft-policy',type=float,default=0.,help='Weight of a discarded temperature-four auxiliary policy head')
    p.add_argument('--seed',type=int,default=931)
    p.add_argument('--screen-seed',type=int,default=220_000_000)
    p.add_argument('--screen-games',type=int,default=48)
    p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();args.run.mkdir(parents=True,exist_ok=True)
    (args.run/'config.json').write_text(json.dumps(vars(args),default=str,indent=2))
    torch.set_num_threads(2);torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    state=torch.load(ROOT/'runs/recent_auxiliary/adapted_weights.pt',map_location='cpu',weights_only=False)
    net=Model(**state['architecture']);net.load_state_dict(state['model'])
    data=ROOT/'runs/round3/data' if args.reanalysed else ROOT/'runs/colab_v2'
    def load(part):
        with gzip.open(data/f'{part}.pkl.gz','rb') as f:rows=pickle.load(f)
        return [(x,l,pi,np.float32((1-args.q_mix)*z+args.q_mix*q)) for x,l,pi,z,q in rows] if args.reanalysed else rows
    training=load('training');validation=load('validation')
    with gzip.open(ROOT/'runs/league_v2/replay.pkl.gz','rb') as f:replay=pickle.load(f)[-12000:]
    auxiliary=torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten(),
        torch.nn.Linear(state['architecture']['channels'],32),torch.nn.ReLU(),torch.nn.Linear(32,3))
    softhead=torch.nn.ModuleDict({name:torch.nn.Conv2d(state['architecture']['channels'],width,1)
        for name,width in [('source',16),('target',16),('from_bias',1),('to_bias',1)]}) if args.soft_policy else None
    oracle=DirectKernelAttack()
    def labels(rows):
        result=[]
        with torch.inference_mode():
            for i in range(0,len(rows),128):
                x=torch.from_numpy(np.stack([r[0] for r in rows[i:i+128]]).astype(np.float32))
                own,_=oracle(x)
                flipped=torch.cat([x[:,6:12],x[:,:6],x[:,12:]],1).clone()
                flipped[:,13]=1.;flipped[:,15]=1.-flipped[:,15]
                enemy,_=oracle(flipped)
                guards=(F.conv2d(x[:,:5].sum(1,keepdim=True),torch.ones(1,1,3,3),padding=1)*x[:,5:6]).sum((1,2,3))>0
                result.append(torch.stack([own,enemy,guards],1).float())
        return torch.cat(result)
    trainlabels=labels(training);replaylabels=labels(replay)
    params=list(net.parameters())+list(auxiliary.parameters())+(list(softhead.parameters()) if softhead is not None else [])
    opt=torch.optim.AdamW(params,lr=args.lr,weight_decay=1e-4)

    def tensors(batch):
        width=max(len(s[1]) for s in batch)
        idx=np.zeros((len(batch),width),np.int64);pi=np.zeros((len(batch),width),np.float32);mask=np.zeros((len(batch),width),bool)
        for i,(_,l,p,_) in enumerate(batch):idx[i,:len(l)]=l;pi[i,:len(p)]=p;mask[i,:len(l)]=True
        return (torch.from_numpy(np.stack([s[0] for s in batch]).astype(np.float32)),torch.from_numpy(idx),
            torch.from_numpy(pi),torch.from_numpy(mask),torch.tensor([s[3] for s in batch],dtype=torch.float32))

    def forward_raw(x):
        # Same arithmetic and exported architecture as Model.forward, exposing
        # the pre-tanh value for a numerically stable binary classification loss.
        p,v=net.features(x);h=net.trunk(x)
        src=net.source(h).reshape(-1,16,64).transpose(1,2);dst=net.target(h).reshape(-1,16,64)
        pair=torch.bmm(src,dst)*.25+net.from_bias(h).reshape(-1,64,1)+net.to_bias(h).reshape(-1,1,64)
        return p+pair.reshape(-1,4096),v+net.value(h).squeeze(-1),h

    test=tensors(validation[:64])[0]
    net.eval()
    with torch.inference_mode():
        refp,refv=net(test);p0,h0,_=forward_raw(test)
        assert torch.equal(refp,p0) and torch.equal(refv,torch.tanh(h0))
    history=[];started=time.monotonic()
    for step in range(1,args.steps+1):
        net.train();batch=[];targets=[]
        for _ in range(128):
            fresh=rng.random()<.5;source=training if fresh else replay;i=int(rng.integers(len(source)))
            batch.append(augment(source[i],rng));targets.append((trainlabels if fresh else replaylabels)[i])
        x,idx,pi,mask,z=tensors(batch)
        opt.zero_grad(set_to_none=True);pol,raw,h=forward_raw(x)
        legal=pol.gather(1,idx).masked_fill(~mask,-1e9)
        pl=-(pi*F.log_softmax(legal,1)).sum(1).mean()
        vl=F.binary_cross_entropy_with_logits(2*raw,(z+1)*.5) if args.mode=='bce' else F.mse_loss(torch.tanh(raw),z)
        al=F.binary_cross_entropy_with_logits(auxiliary(h),torch.stack(targets),pos_weight=torch.tensor([5.,5.,1.]))
        total=pl+vl+.15*al
        if softhead is not None:
            s=softhead['source'](h).reshape(-1,16,64).transpose(1,2)
            d=softhead['target'](h).reshape(-1,16,64)
            pair=torch.bmm(s,d)*.25+softhead['from_bias'](h).reshape(-1,64,1)+softhead['to_bias'](h).reshape(-1,1,64)
            softlogits=pair.reshape(-1,4096).gather(1,idx).masked_fill(~mask,-1e9)
            softtarget=pi.pow(.25);softtarget=softtarget/softtarget.sum(1,keepdim=True)
            sl=-(softtarget*F.log_softmax(softlogits,1)).sum(1).mean()
            total=total+args.soft_policy*sl
        total.backward();torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
        if step%300==0 or step==args.steps:
            net.eval()
            with torch.inference_mode():
                vx,vi,vp,vm,vz=tensors(validation);pp,rr,_=forward_raw(vx)
                valce=-(vp*F.log_softmax(pp.gather(1,vi).masked_fill(~vm,-1e9),1)).sum(1).mean().item()
                valmse=F.mse_loss(torch.tanh(rr),vz).item()
            path=export(net,args.run/f'step_{step}.onnx')
            torch.save(dict(architecture=state['architecture'],model={k:v.detach().cpu() for k,v in net.state_dict().items()}),args.run/f'step_{step}.pt')
            result=match(path,ROOT/'candidate_v2.onnx',args.screen_games,seed=args.screen_seed,verbose=False)
            row=dict(step=step,policy_loss=pl.item(),value_loss=vl.item(),auxiliary_loss=al.item(),
                validation_policy_ce=valce,validation_value_mse=valmse,screen=result,seconds=time.monotonic()-started)
            history.append(row);(args.run/'history.json').write_text(json.dumps(history,indent=2))
            print(json.dumps(row),flush=True)


if __name__=='__main__':main()
