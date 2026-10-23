"""Time-bounded, resumable Expert Iteration with centralized batched GPU inference."""
from __future__ import annotations
import argparse
import gzip
import json
import multiprocessing as mp
import os
import pickle
import queue
import shutil
import sys
import time
from collections import deque
from pathlib import Path
import numpy as np
from arena import ROOT,scene,match
from workers import actor,augment,episode


def collect(evaluate,n_games,n_workers,simulations,seed,opponents):
    if n_workers==0:
        return [s for g in range(n_games) for s in episode(evaluate,scene(),seed+g,
                  simulations,opponents[(g//3)%len(opponents)] if opponents and g%3==0 else None)]
    ctx=mp.get_context("spawn")
    n_workers=min(n_workers,n_games)
    requests,results=ctx.Queue(),ctx.Queue()
    responses=[ctx.Queue() for _ in range(n_workers)]
    processes=[]
    for w in range(n_workers):
        jobs=[(g,seed+g) for g in range(w,n_games,n_workers)]
        p=ctx.Process(target=actor,args=(w,jobs,scene(),simulations,requests,responses[w],results,opponents))
        p.start();processes.append(p)
    samples=[];done=0;last_progress=time.monotonic()
    try:
        while done<n_workers:
            # Each actor contributes up to 8 leaves, batched across games on GPU.
            pending=[]
            try:pending.append(requests.get(timeout=.01))
            except queue.Empty:pass
            until=time.monotonic()+.001
            while len(pending)<n_workers and time.monotonic()<until:
                try:pending.append(requests.get(timeout=max(.0001,until-time.monotonic())))
                except queue.Empty:break
            if pending:
                batches=[x[1] for x in pending]
                policy,value=evaluate(np.concatenate(batches))
                start=0
                for (wid,x) in pending:
                    end=start+len(x)
                    responses[wid].put((policy[start:end].copy(),value[start:end].copy()))
                    start=end
                last_progress=time.monotonic()
            while True:
                try:message=results.get_nowait()
                except queue.Empty:break
                if message[0]=="game":samples.extend(message[3]);last_progress=time.monotonic()
                elif message[0]=="done":done+=1
                else:raise RuntimeError(message[2])
            for process in processes:
                if process.exitcode not in (None,0):raise RuntimeError(f"Actor died: {process.exitcode}")
            if time.monotonic()-last_progress>180:raise TimeoutError("Actors stopped making progress")
    finally:
        for p in processes:
            p.join(timeout=1)
            if p.is_alive():p.terminate();p.join(timeout=3)
        for q in [requests,results,*responses]:q.close();q.cancel_join_thread()
    return samples


def main():
    # Lazy imports keep spawned search actors free of CUDA/PyTorch runtime.
    import torch
    import torch.nn.functional as F
    from model import Model,export
    from colony_one.onnx_runner import OnnxEvaluator
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--minutes",type=float,default=90)
    p.add_argument("--iterations",type=int,default=1000)
    p.add_argument("--games",type=int,default=32)
    p.add_argument("--workers",type=int,default=min(4,os.cpu_count() or 2))
    p.add_argument("--simulations",type=int,default=100)
    p.add_argument("--channels",type=int,default=64)
    p.add_argument("--blocks",type=int,default=4)
    p.add_argument("--batch-size",type=int,default=256)
    p.add_argument("--eval-every",type=int,default=4)
    p.add_argument("--eval-games",type=int,default=16)
    p.add_argument("--final-games",type=int,default=100)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--final-seed",type=int,default=2_000_000)
    p.add_argument("--run",type=Path,default=ROOT/"runs/neural")
    p.add_argument("--initial-features",type=Path,default=None)
    p.add_argument("--examples",type=Path,default=None,help="Replay-derived tactical training examples")
    p.add_argument("--examples-fraction",type=float,default=.25)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--warm-start",type=Path,default=None)
    p.add_argument("--initial-replay",type=Path,default=None)
    p.add_argument("--opponent",type=Path,action="append",default=[])
    p.add_argument("--skip-final",action="store_true")
    args=p.parse_args()
    if args.games<1 or args.minutes<=0 or args.workers<0:raise ValueError("Invalid training budget")
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type=="cuda":
        torch.backends.cudnn.benchmark=True
        torch.set_float32_matmul_precision("high")
    args.run.mkdir(parents=True,exist_ok=True)
    state_path=args.run/"checkpoint.pt"
    state=torch.load(state_path,map_location="cpu",weights_only=False) if args.resume and state_path.exists() else None
    warm=torch.load(args.warm_start,map_location="cpu",weights_only=False) if args.warm_start and not state else None
    arch=state["architecture"] if state else warm["architecture"] if warm else dict(channels=args.channels,blocks=args.blocks)
    net=Model(**arch)
    if state:net.load_state_dict(state["model"])
    elif warm:net.load_state_dict(warm["model"])
    elif args.initial_features is not None:
        params=json.loads(args.initial_features.read_text())["params"]
        with torch.no_grad():
            for name,param in net.features.named_parameters():param.copy_(torch.tensor(params[name]))
    net.to(device)
    if device.type=="cuda":net.to(memory_format=torch.channels_last)
    opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-4)
    if state:opt.load_state_dict(state["optimizer"])
    replay=deque(maxlen=50_000)
    examples=[]
    if args.examples:
        with gzip.open(args.examples,"rb") as f:examples=pickle.load(f)
        print(f"Loaded {len(examples)} tournament-derived training examples",flush=True)
    replay_path=args.run/"replay.pkl.gz"
    if state:
        rng.bit_generator.state=state["rng"]
        torch.set_rng_state(state["torch_rng"])
        if replay_path.exists():
            with gzip.open(replay_path,"rb") as f:replay.extend(pickle.load(f))
    elif args.initial_replay:
        with gzip.open(args.initial_replay,"rb") as f:replay.extend(pickle.load(f))
    baseline=ROOT/"baselines/current.onnx"
    champion=ROOT/"baselines/local_champion.onnx"
    best=args.run/"best.onnx"
    if not best.exists():
        export(net,args.run/"initial.onnx");net.to(device)
        shutil.copyfile(args.run/"initial.onnx",best)
        torch.save(dict(architecture=arch,model={k:v.detach().cpu() for k,v in net.state_dict().items()}),args.run/"best_weights.pt")
    print(f"Device={device}; architecture={arch}; parameters={sum(p.numel() for p in net.parameters()):,}",flush=True)
    print("Baseline and original challenge files are frozen. Only this run directory changes.",flush=True)
    amp_dtype=torch.bfloat16 if device.type=="cuda" and torch.cuda.is_bf16_supported() else torch.float16
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda" and amp_dtype==torch.float16)
    if state and "scaler" in state:scaler.load_state_dict(state["scaler"])

    def evaluate(planes):
        x=torch.from_numpy(np.ascontiguousarray(planes,dtype=np.float32)).to(device)
        if device.type=="cuda":x=x.contiguous(memory_format=torch.channels_last)
        with torch.inference_mode(),torch.autocast(device_type=device.type,dtype=amp_dtype,enabled=device.type=="cuda"):
            pol,val=net(x)
        return pol.float().cpu().numpy(),val.float().cpu().numpy()

    started=time.monotonic();first=state["iteration"]+1 if state else 0
    last_iteration=first-1
    for iteration in range(first,first+args.iterations):
        if time.monotonic()-started>args.minutes*60:break
        net.eval();t=time.monotonic()
        opponents=[str(baseline),str(best)]
        if champion.exists():opponents.append(str(champion))
        opponents.extend(str(path.resolve()) for path in args.opponent)
        samples=collect(evaluate,args.games,args.workers,args.simulations,
                        args.seed+iteration*10_000,opponents)
        replay.extend(samples)
        generation=time.monotonic()-t
        if not samples:raise RuntimeError("No completed self-play data")
        net.train();losses=[]
        steps=max(8,min(128,4*len(samples)//args.batch_size))
        for _ in range(steps):
            batch=[]
            for i in rng.integers(len(replay),size=args.batch_size):
                item=examples[int(rng.integers(len(examples)))] if examples and rng.random()<args.examples_fraction else replay[int(i)]
                batch.append(augment(item,rng))
            width=max(len(s[1]) for s in batch)
            indices=np.zeros((len(batch),width),np.int64)
            targets=np.zeros((len(batch),width),np.float32)
            valid=np.zeros((len(batch),width),bool)
            for i,(_,legal,pi,_) in enumerate(batch):
                indices[i,:len(legal)]=legal;targets[i,:len(pi)]=pi;valid[i,:len(legal)]=True
            x=torch.from_numpy(np.stack([s[0] for s in batch])).to(device)
            idx=torch.from_numpy(indices).to(device);y=torch.from_numpy(targets).to(device)
            mask=torch.from_numpy(valid).to(device)
            z=torch.tensor([s[3] for s in batch],device=device,dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=amp_dtype,enabled=device.type=="cuda"):
                logits,value=net(x)
                # Softmax only over legal moves; duplicates in padded indices
                # are masked before normalization.
                legal_logits=logits.float().gather(1,idx).masked_fill(~mask,-1e9)
                policy_loss=-(y*F.log_softmax(legal_logits,dim=1)).sum(1).mean()
                value_loss=F.mse_loss(value.float(),z)
                loss=policy_loss+value_loss
            scaler.scale(loss).backward();scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            scaler.step(opt);scaler.update()
            losses.append([policy_loss.item(),value_loss.item()])
        net.eval()
        last_iteration=iteration
        stats=dict(iteration=iteration+1,new_samples=len(samples),replay=len(replay),
                   generation_seconds=generation,loss=np.mean(losses,axis=0).tolist(),
                   elapsed_seconds=time.monotonic()-started)
        if (iteration+1)%args.eval_every==0 or args.iterations==1:
            candidate=export(net,args.run/"latest.onnx");net.to(device)
            # Screening changes seeds across iterations. Final holdout is kept
            # separate and is run once after checkpoint selection has ended.
            result=match(candidate,best,args.eval_games,seed=200_000+iteration*1009,verbose=False)
            stats["incumbent_wins"]=result["wins"]
            baseline_result=match(candidate,baseline,args.eval_games,seed=300_000+iteration*1009,verbose=False)
            stats["baseline_wins"]=baseline_result["wins"]
            if result["win_rate"]>=.625 and baseline_result["win_rate"]>=.5:
                shutil.copyfile(candidate,best)
                torch.save(dict(architecture=arch,model={k:v.detach().cpu() for k,v in net.state_dict().items()}),args.run/"best_weights.pt")
                stats["provisional_promotion"]=True
        print(json.dumps(stats),flush=True)
        with (args.run/"history.jsonl").open("a") as f:f.write(json.dumps(stats)+"\n")
        checkpoint=dict(model={k:v.detach().cpu() for k,v in net.state_dict().items()},
            architecture=arch,optimizer=opt.state_dict(),iteration=iteration,
            rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),scaler=scaler.state_dict())
        tmp=state_path.with_suffix(".tmp");torch.save(checkpoint,tmp);tmp.replace(state_path)
        replay_tmp=replay_path.with_suffix(".tmp")
        with gzip.open(replay_tmp,"wb",compresslevel=1) as f:pickle.dump(list(replay),f,protocol=5)
        replay_tmp.replace(replay_path)
    export(net,args.run/"latest.onnx")
    if not args.skip_final:
        report=match(best,baseline,args.final_games,seed=args.final_seed,
                     output=args.run/"final_match")
        report["promotion_passed"]=report["wilson_95"][0]>.5
        if champion.exists():
            against_champion=match(best,champion,args.final_games,
                seed=args.final_seed+500_000,output=args.run/"champion_match")
            report["champion_comparison"]=against_champion
            report["promotion_passed"]=report["promotion_passed"] and against_champion["wilson_95"][0]>.5
        (args.run/"final_result.json").write_text(json.dumps(report,indent=2))
        if report["promotion_passed"]:shutil.copyfile(best,args.run/"submission.onnx")
        else:print("Holdout did not establish improvement. Keep the baseline.",flush=True)
    print(f"Finished through iteration {last_iteration+1}. Artifacts: {args.run}",flush=True)


if __name__=="__main__":main()
