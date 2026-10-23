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
