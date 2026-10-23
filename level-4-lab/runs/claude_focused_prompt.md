We are collaborating on a Colony: One Level 4 contest model. The user explicitly asked Codex to ask Claude Code to help both improve the model. You are the independent reviewer; please inspect the files and respond with concrete findings and experiment proposals. Read only: do not modify files or start training, do not contact other sessions, and do not access credentials or unrelated directories.

Working directory: /Users/ciwrl/Downloads/contest/level-4-lab. Read RESULTS_ROUND2.md, model.py, distill.py, workers.py, tactical_model.py, engine/colony_one/mcts.py and encoding.py, and RESEARCH_ROUND2.md as needed.

Our verified candidate_v2.onnx is a 32-channel, 2-residual-block policy/value model, 256 KB, with trainable heuristic skip features, spatial source/destination dot-product policy and small dense value head. V2 warm started from V1, then used newer tournament protocols relabeled with 600-simulation MCTS and the latest 12k self-play positions, plus a training-only auxiliary classifier for own/enemy direct Kernel capture threats and own bodyguard. It beat V1 368/600 fresh official 100-simulation games (61.3%, Wilson 57.4-65.1) and original heuristic157/200. File frozen; do not change it.

Failed tests: full D4 logit/value ensemble105/200 V1; weight soup97/200; more old selfplay109/200; plain newer-data imitation108/200; candidate-only follow-up from V2 with two-action aux112/200 V2 (inconclusive). Naive exact direct-kill priors and bare-Kernel wrappers failed screens. Do not recommend rerunning those blindly.

Game: two consecutive actions by each team, values switch perspective only when the actual team changes. Adjacent non-Kernel bodyguard redirects attacks on the Kernel, attacker stays on redirected attacks. End-turn bare-Kernel rule: if both teams only have Kernels and distance<=2 then just-acting loses, otherwise higher inner ring wins, tie Magenta. Firewall shrinks after Magenta on rounds8/16/24/26. The network observes canonical own/enemy planes, locked squares, actions-left, urgency and true Magenta team. Referee wraps ONNX in exactly100-simulation PUCT with batched8leaf inference. We can change network/training only. Dynamic B, outputs4096logits and B bounded values; 5MiB max. Colab A100 available to human but not attached; CPU runs ~600games in5min. About70min contest time left.

Please independently identify the best 2-4 feasible new experiments and any concrete correctness or target-label bugs. Prioritize a small modification implementable and trainable in5-15min, with clear evaluation against frozen V2. Consider value-target quality, policy target correction, action-phase dependence, geometric inductive bias, search distribution shift, exact tactics supervision, local/global pooling. Distinguish hypotheses from proven bugs. Include suggested code changes or pseudocode and precise hyperparameters. Tell us what you would test FIRST. Do not give generic RL advice.

This is a focused bounded consultation: do not call tools. The relevant source is pasted below. Reply in at most 700 words. Identify any real bugs, distinguish them from hypotheses, and propose ONE best next experiment. We are already testing BCE pre-tanh value loss, fresh600sim V2 reanalysis with .5historicaloutcome+.5rootQ, and training-only softened-policy auxiliary head pi**.25. On initial48game screens these scored52-65%, but softened variant fell to48/96V2 and27/64earlier; original MSEwarmfinetune fell39/96V2 but41/64earlier. So no confirmed gain yet. We need a different small change or a concrete target correction.

SOURCE model.py
```python
"""Small residual learner with a heuristic starting point and compact pair policy."""
from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Features(nn.Module):
    """Conventional ONNX operations; extra tactical features start at zero."""
    def __init__(self):
        super().__init__()
        self.material = nn.Parameter(torch.tensor([.30,.90,.50,.60,1.,2.])*0.55)
        self.capture = nn.Parameter(torch.full((6,),2.))
        # value: centre, kernel-centre, guards, fire, pressure
        # policy: centre delta, kernel-centre delta, kernel danger, fire escape
        self.extra = nn.Parameter(torch.tensor([.044,0.,0.,0.,0.,0.,0.,0.,0.]))
        rings=np.array([[min(r,c,7-r,7-c) for c in range(8)] for r in range(8)],np.float32)
        self.register_buffer("rings",torch.tensor(rings)[None,None])
        self.register_buffer("center",torch.tensor(rings)[None,None])
        self.register_buffer("near",torch.ones((1,1,3,3)))
        attack=np.zeros((1,6,7,7),np.float32)
        for dr in range(-3,4):
            for dc in range(-3,4):
                a,b=abs(dr),abs(dc)
                attack[0,0,dr+3,dc+3]=a==b and 1<=a<=2
                attack[0,1,dr+3,dc+3]=(a==0) != (b==0)
                attack[0,2,dr+3,dc+3]=sorted([a,b])==[1,2]
                attack[0,3,dr+3,dc+3]=a==b and 2<=a<=3
                attack[0,4,dr+3,dc+3]=0<max(a,b)<=2 and (a==b or a==0 or b==0)
                attack[0,5,dr+3,dc+3]=max(a,b)==1
        self.register_buffer("attack",torch.tensor(attack))

    def forward(self,x):
        own,enemy=x[:,:6],x[:,6:12]
        diff=own-enemy
        ownk,enemyk=own[:,5:6],enemy[:,5:6]
        ownocc,enemyocc=own.sum(1,keepdim=True),enemy.sum(1,keepdim=True)
        ownattack=F.conv2d(own,self.attack,padding=3)
        enemyattack=F.conv2d(enemy,self.attack,padding=3)
        own_guards=F.conv2d(ownocc-ownk,self.near,padding=1)
        enemy_guards=F.conv2d(enemyocc-enemyk,self.near,padding=1)
        locked=x[:,12:13].sum((2,3),keepdim=True)
        stage=(locked>0).to(x.dtype)+(locked>28).to(x.dtype)+(locked>48).to(x.dtype)
        burns=(self.rings==stage).to(x.dtype)
        urgency=x[:,14:15]
        value=(diff*self.material[None,:,None,None]).sum((1,2,3))
        value=value+self.extra[0]*(diff*self.center).sum((1,2,3))
        value=value+self.extra[1]*((ownk-enemyk)*self.center).sum((1,2,3))
        value=value+self.extra[2]*(ownk*own_guards-enemyk*enemy_guards).sum((1,2,3))
        value=value-self.extra[3]*((ownk-enemyk)*burns*urgency).sum((1,2,3))
        value=value+self.extra[4]*(enemyk*ownattack-ownk*enemyattack).sum((1,2,3))
        target=(enemy*self.capture[None,:,None,None]).sum(1).reshape(-1,1,64)
        delta=self.center.reshape(1,1,64)-self.center.reshape(1,64,1)
        policy=target+self.extra[5]*delta
        policy=policy+self.extra[6]*ownk.reshape(-1,64,1)*delta
        policy=policy-self.extra[7]*ownk.reshape(-1,64,1)*enemyattack.reshape(-1,1,64)
        escape=burns.reshape(-1,64,1)-burns.reshape(-1,1,64)
        policy=policy+self.extra[8]*ownk.reshape(-1,64,1)*escape*urgency.mean((2,3)).reshape(-1,1,1)
        return policy.reshape(-1,4096),value


class Block(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.seq=nn.Sequential(nn.Conv2d(c,c,3,padding=1),nn.ReLU(),nn.Conv2d(c,c,3,padding=1))
    def forward(self,x):
        return F.relu(x+self.seq(x))


class Model(nn.Module):
    def __init__(self,channels=32,blocks=2):
        super().__init__()
        self.features=Features()
        self.channels,self.blocks=channels,blocks
        self.trunk=nn.Sequential(nn.Conv2d(16,channels,3,padding=1),nn.ReLU(),
                                 *[Block(channels) for _ in range(blocks)])
        # Joint source/destination features avoid a 4096-wide dense hidden layer.
        self.source=nn.Conv2d(channels,16,1)
        self.target=nn.Conv2d(channels,16,1)
        self.from_bias=nn.Conv2d(channels,1,1)
        self.to_bias=nn.Conv2d(channels,1,1)
        self.value=nn.Sequential(nn.Conv2d(channels,4,1),nn.ReLU(),nn.Flatten(),
                                 nn.Linear(256,32),nn.ReLU(),nn.Linear(32,1))
        nn.init.zeros_(self.target.weight); nn.init.zeros_(self.target.bias)
        for layer in (self.from_bias,self.to_bias,self.value[-1]):
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)

    def forward(self,x):
        p,v=self.features(x)
        h=self.trunk(x)
        src=self.source(h).reshape(-1,16,64).transpose(1,2)
        dst=self.target(h).reshape(-1,16,64)
        pair=torch.bmm(src,dst)*.25
        pair=pair+self.from_bias(h).reshape(-1,64,1)+self.to_bias(h).reshape(-1,1,64)
        return p+pair.reshape(-1,4096),torch.tanh(v+self.value(h).squeeze(-1))


class FeatureOnly(nn.Module):
    def __init__(self):
        super().__init__();self.features=Features()
    def forward(self,x):
        p,v=self.features(x)
        return p,torch.tanh(v)


def export(model,path):
    import onnx
    from colony_one.submission_policy import validate_submission_contract
    model=model.cpu().eval()
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    torch.onnx.export(model,torch.zeros(1,16,8,8),str(path),
        input_names=["observation"],output_names=["policy","value"],opset_version=17,
        dynamic_axes={"observation":{0:"batch"},"policy":{0:"batch"},"value":{0:"batch"}},
        dynamo=False,external_data=False)
    m=onnx.load(path)
    # The supplied allow-list omits Constant nodes, so bake their tensors into
    # initializers (also makes every reshape/index argument visibly constant).
    nodes=[]
    for node in m.graph.node:
        if node.op_type=="Constant":
            tensor=onnx.TensorProto()
            tensor.CopyFrom(next(a.t for a in node.attribute if a.name=="value"))
            tensor.name=node.output[0]
            m.graph.initializer.append(tensor)
        else:nodes.append(node)
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    m.graph.output[0].type.tensor_type.shape.dim[1].dim_value=4096
    # Constant-only reshape/index arguments are checked by the actual contract.
    onnx.checker.check_model(m)
    validate_submission_contract(m)
    onnx.save(m,path)
    if path.stat().st_size>5*1024**2:raise ValueError("Export exceeds 5 MiB")
    return path

```

SOURCE distill.py
```python
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

```

SOURCE workers.py
```python
"""NumPy-only actors; spawned processes never initialize PyTorch or CUDA."""
from __future__ import annotations
import sys
import traceback
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"engine"))


def episode(evaluator,scene,seed,simulations,opponent_path=None):
    from colony_one import mcts
    from colony_one.env import ColonyOneEnv
    from colony_one.onnx_runner import OnnxEvaluator
    rng=np.random.default_rng(seed)
    env=ColonyOneEnv(scene=scene,seed=seed)
    opponent=OnnxEvaluator(opponent_path) if opponent_path else None
    opponent_team="CYAN" if seed%2 else "MAGENTA"
    cfg=mcts.MCTSConfig(simulations=simulations,
        warmup_temperatures=(1.,1.,.5,.25),warmup_step_actions=4)
    samples=[]
    while not env.terminated:
        ev=opponent if opponent and env.current_team==opponent_team else evaluator
        root=mcts.run(env.board,env.current_team,env.actions_left,env.round_number,
                      ev,cfg,rng=rng)
        if not root.N:raise RuntimeError("Unfinished episode has no legal actions")
        legal=np.array(sorted(root.N),dtype=np.int16)
        visits=np.array([root.N[int(a)] for a in legal],dtype=np.float32)
        if visits.sum()==0:raise RuntimeError("Search produced no visits")
        pi=(visits/visits.sum()).astype(np.float16)
        # Mixed-opponent episodes learn the search distributions of both players.
        # Values always use the real team's final outcome, never action parity.
        samples.append((env.observation().astype(np.float16),legal,pi,env.current_team))
        temperature=mcts.warmup_temperature(env.action,cfg.warmup_temperatures,cfg.warmup_step_actions)
        action=mcts.select_action_index(root,temperature,rng=rng)
        env.step(action)
        if len(samples)>104:raise RuntimeError("Episode exceeded firewall termination")
    return [(o,l,p,np.float32(1 if t==env.winner else -1)) for o,l,p,t in samples]


def actor(worker_id,jobs,scene,simulations,requests,responses,results,opponents):
    class Proxy:
        def __call__(self,planes):
            requests.put((worker_id,planes))
            result=responses.get(timeout=180)
            if result is None:raise RuntimeError("Inference server stopped")
            return result
    try:
        ev=Proxy()
        for game_i,seed in jobs:
            # A third of episodes meet archived opponents to reduce self-play
            # specialization; the remaining episodes use the current model.
            opp=opponents[(seed//3)%len(opponents)] if opponents and seed%3==0 else None
            samples=episode(ev,scene,seed,simulations,opp)
            results.put(("game",worker_id,game_i,samples))
        results.put(("done",worker_id))
    except BaseException:
        results.put(("error",worker_id,traceback.format_exc()))


def augment(sample,rng):
    obs,legal,pi,z=sample
    k=int(rng.integers(4));mirror=bool(rng.integers(2))
    squares=np.arange(64).reshape(8,8)
    if mirror:
        obs=obs[:,:,::-1];squares=squares[:,::-1]
    if k:
        obs=np.rot90(obs,k,axes=(1,2));squares=np.rot90(squares,k)
    mapping=np.argsort(squares.reshape(-1))
    legal=mapping[legal.astype(np.int64)//64]*64+mapping[legal.astype(np.int64)%64]
    return np.ascontiguousarray(obs,dtype=np.float32),legal,pi.astype(np.float32),z

```
