"""Reanalyse fixed match splits with v2; retain historical and search values."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import pickle
import time
from pathlib import Path
import numpy as np
from arena import ROOT
from colony_one import constants, encoding, mcts
from colony_one.board import Board
from colony_one.onnx_runner import OnnxEvaluator


def restore(observation):
    x=np.asarray(observation,dtype=np.float32)
    team=constants.MAGENTA if x[15,0,0]>.5 else constants.CYAN
    board=Board.from_observation(x,team)
    left=round(float(x[13,0,0])*2)
    locked=int(x[12].sum())
    possible=[r for r in range(1,27)
        if sum(1 for s in constants.SHRINK_ROUNDS if s<r)=={0:0,28:1,48:2,60:3}[locked]
        and abs(encoding.firewall_urgency(r)-float(x[14,0,0]))<1e-5]
    if len(possible)!=1:raise ValueError((locked,float(x[14,0,0]),possible))
    r=possible[0]
    assert np.array_equal(encoding.encode_board(board,team,left,r),x)
    return board,team,left,r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--simulations',type=int,default=600)
    p.add_argument('--out',type=Path,default=ROOT/'runs/round3/data')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    teacher=ROOT/'candidate_v2.onnx';ev=OnnxEvaluator(teacher)
    cfg=mcts.MCTSConfig(simulations=args.simulations,add_root_noise=False,warmup_temperatures=())
    rng=np.random.default_rng(31031);cache={};stats={};t=time.monotonic()
    for part in ['training','validation']:
        with gzip.open(ROOT/f'runs/colab_v2/{part}.pkl.gz','rb') as f:source=pickle.load(f)
        out=[];differences=[]
        for i,(x,legal,pi,z) in enumerate(source):
            key=x.tobytes()
            if key not in cache:
                board,team,left,r=restore(x)
                root=mcts.run(board,team,left,r,ev,cfg,rng=rng)
                indices=np.array(sorted(root.N),np.int16)
                assert set(indices.tolist())==set(legal.tolist())
                n=np.array([root.N[int(a)] for a in indices],np.float32)
                q=float(sum(root.W.values())/sum(root.N.values()))
                assert -1.00001<=q<=1.00001
                cache[key]=(indices,(n/n.sum()).astype(np.float16),np.float32(np.clip(q,-1,1)))
            indices,prob,q=cache[key]
            out.append((x,indices,prob,np.float32(z),q))
            differences.append(abs(float(z)-float(q)))
            if (i+1)%100==0:print(part,i+1,'/',len(source),'unique',len(cache),'seconds',round(time.monotonic()-t,1),flush=True)
        with gzip.open(args.out/f'{part}.pkl.gz','wb',compresslevel=1) as f:pickle.dump(out,f,protocol=5)
        stats[part]={'samples':len(out),'mean_abs_outcome_search_difference':float(np.mean(differences))}
    stats.update(teacher_sha256=hashlib.sha256(teacher.read_bytes()).hexdigest(),simulations=args.simulations,
        unique_positions=len(cache),seconds=time.monotonic()-t,
        value_definition='Root-team q = sum(root.W.values()) / sum(root.N.values()); no extra sign flip',
        split='Preserved whole-match training/validation files from round 2; shared positions reuse search')
    (args.out/'manifest.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats),flush=True)


if __name__=='__main__':main()
