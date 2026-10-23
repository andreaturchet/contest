"""Preserve candidate-only replays; recover missing side labels by move agreement."""
from __future__ import annotations
import json
import re
import shutil
import numpy as np
from arena import ROOT,scene
from analyze_losses import LINE,square
from colony_one import mcts
from colony_one.env import ColonyOneEnv
from colony_one.onnx_runner import OnnxEvaluator


def main():
    source=ROOT.parent/'New Matches'
    out=ROOT/'data/candidate_only';out.mkdir(exist_ok=True)
    (out/'protocols').mkdir(exist_ok=True)
    metadata={}
    for p in [ROOT/'data/tournament-matches.json',ROOT/'data/recent/tournament-matches.json']:
        if p.exists():metadata.update({r['match_id']:r for r in json.loads(p.read_text())})
    evaluator=OnnxEvaluator(ROOT/'candidate.onnx')
    cfg=mcts.MCTSConfig(simulations=100)
    records=[];agreements=[];unknown=[]
    for path in sorted(source.glob('match-*.txt')):
        shutil.copyfile(path,out/'protocols'/path.name)
        ident=int(re.search(r'(\d+)',path.name).group(1));protocol=path.read_text()
        winner=re.findall(r'\*\*\* (CYAN|MAGENTA) WINS',protocol)[-1]
        env=ColonyOneEnv(scene=scene());scores={t:[0,0] for t in ['CYAN','MAGENTA']}
        for line in protocol.splitlines():
            parsed=LINE.match(line)
            if not parsed:continue
            move,team,abbr,fr,to,kind=parsed.groups()
            action=next(a for a in env.legal_actions() if a.from_pos==square(fr) and a.to_pos==square(to))
            assert env.current_team==team and env.action==int(move)
            if env.action>8:
                predicted=mcts.search_action(env.board,env.current_team,env.actions_left,env.round_number,evaluator,cfg,rng=np.random.default_rng(1))
                scores[team][0]+=predicted==action;scores[team][1]+=1
            env.step(action)
        assert env.terminated and env.winner==winner
        known=metadata.get(ident,{}).get('you');inferred=False
        if known is None:
            rates={t:a/max(1,n) for t,(a,n) in scores.items()}
            best=max(rates,key=rates.get);other='MAGENTA' if best=='CYAN' else 'CYAN'
            if scores[best][1]>=4 and rates[best]>=.9 and rates[best]-rates[other]>=.2:
                known=best;inferred=True
        row=dict(match_id=ident,agreements=scores,you=known,side_inferred=inferred,winner=winner)
        agreements.append(row)
        record=dict(match_id=ident,you=known,opponent=None,score=int(winner=='CYAN'),
            played_at=metadata.get(ident,{}).get('played_at'),starting_observation=scene().tolist(),protocol=protocol,
            source='User-designated candidate-only folder',side_inferred=inferred)
        if known:records.append(record)
        else:unknown.append(record)
        print(ident,known,'inferred' if inferred else 'metadata' if known else 'unknown',scores,flush=True)
    (out/'tournament-matches.json').write_text(json.dumps(records))
    (out/'unattributed-matches.json').write_text(json.dumps(unknown))
    (out/'move_agreement.json').write_text(json.dumps(agreements,indent=2))
    print('Preserved',len(agreements),'replays; side known/inferred:',len(records),'ambiguous:',len(unknown),flush=True)


if __name__=='__main__':main()
