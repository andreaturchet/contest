"""Check game symmetries and the actual exported model on real reachable boards."""
from __future__ import annotations
import argparse
import sys
import numpy as np
from arena import ROOT,scene
from workers import augment
from colony_one.board import Board
from colony_one.env import ColonyOneEnv
from colony_one.encoding import action_index,encode_board
from colony_one.rules import legal_actions_for_team


class Fixed:
    def __init__(self,k,m):self.values=iter([k,int(m)])
    def integers(self,n):return next(self.values)


def transformed(obs,k,m):
    return augment((obs,np.array([0]),np.array([1.]),0.),Fixed(k,m))[0]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",default=None)
    args=p.parse_args()
    rng=np.random.default_rng(9281)
    env=ColonyOneEnv(scene=scene(),seed=9281)
    positions=[];checks=0
    for step in range(80):
        if env.terminated:env.reset()
        if step%4==0:
            positions.append(env.observation())
            legal=env.legal_action_indices()
            for k in range(4):
                for mirror in [False,True]:
                    obs,indices,_,_=augment((env.observation(),legal,np.ones(len(legal)),0.),Fixed(k,mirror))
                    board=Board.from_observation(obs,env.current_team)
                    actual={action_index(a,env.current_team) for a in legal_actions_for_team(board,env.current_team)}
                    assert set(indices)==actual,(step,k,mirror,"legal mismatch")
                    original=env.clone();clone=env.clone();clone.board=board
                    choice=int(rng.integers(len(legal)))
                    original.step(int(legal[choice]));clone.step(int(indices[choice]))
                    assert clone.winner==original.winner
                    assert clone.current_team==original.current_team
                    np.testing.assert_array_equal(clone.observation(),transformed(original.observation(),k,mirror))
                    checks+=1
        env.step(int(env.sample_legal_action_index()))
    print(f"PASS: {checks} D4 legal-action and transition checks on reachable boards")
    if args.model:
        import onnxruntime as ort
        from colony_one.submission_policy import validate_submission_contract
        import onnx
        validate_submission_contract(onnx.load(args.model))
        opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
        ev=ort.InferenceSession(args.model,sess_options=opts,providers=["CPUExecutionProvider"])
        for batch in [1,8,16]:
            x=np.stack(positions[:batch]).astype(np.float32)
            policy,value=ev.run(None,{ev.get_inputs()[0].name:x})
            assert policy.shape==(batch,4096) and value.shape==(batch,)
            assert policy.dtype==np.float32 and value.dtype==np.float32
            assert np.isfinite(policy).all() and np.isfinite(value).all()
            assert np.abs(value).max()<=1
        print("PASS: submission structure and real-board inference at batches 1, 8, 16")


if __name__=="__main__":main()
