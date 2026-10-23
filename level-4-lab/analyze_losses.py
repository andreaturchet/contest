"""Replay the tournament export exactly, identify blunders, and relabel examples."""
from __future__ import annotations
import argparse
import collections
import gzip
import hashlib
import json
import pickle
import re
from pathlib import Path
import numpy as np
from arena import ROOT,snapshot,wilson
from colony_one import constants,mcts
from colony_one.encoding import action_index
from colony_one.env import ColonyOneEnv
from colony_one.onnx_runner import OnnxEvaluator
from colony_one.pieces import PieceType
from colony_one.rules import legal_actions_for_team

LINE=re.compile(r"^\s*(\d+)\.\s+(CYAN|MAGENTA)\s+([A-Z]{2})\s+([A-H][1-8])\s+->\s+([A-H][1-8])\s+(MOVE|SHOOT)")


def square(s):return 8-int(s[1]),ord(s[0])-65


def immediate_capture(board,team,actions_left):
    """Exact Kernel capture within the remaining actions, including bodyguards.

    This tests captures only; firewall/bare-Kernel terminal effects are handled
    by the real environment when evaluating an alternative final action.
    """
    for action in legal_actions_for_team(board,team):
        record=board.apply_action(action)
        try:
            if record.captured_piece and record.captured_piece.piece_type==PieceType.KERNEL:
                return [action]
            if actions_left>1:
                for second in legal_actions_for_team(board,team):
                    rec=board.apply_action(second)
                    try:
                        if rec.captured_piece and rec.captured_piece.piece_type==PieceType.KERNEL:
                            return [action,second]
                    finally:board.undo_action(rec)
        finally:board.undo_action(record)
    return None


def safe_final_actions(env):
    safe=[];winning=[]
    for action in env.legal_actions():
        child=env.clone();child.step(action)
        idx=action_index(action,env.current_team)
        if child.terminated:
            if child.winner==env.current_team:winning.append(idx);safe.append(idx)
        elif not immediate_capture(child.board,child.current_team,child.actions_left):safe.append(idx)
    return winning or safe


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,default=ROOT/"data/tournament-matches.json")
    p.add_argument("--teacher",type=Path,default=ROOT/"baselines/current.onnx")
    p.add_argument("--simulations",type=int,default=600)
    p.add_argument("--output-dir",type=Path,default=ROOT)
    args=p.parse_args()
    output=args.output_dir
    (output/"data").mkdir(parents=True,exist_ok=True)
    (output/"replays").mkdir(parents=True,exist_ok=True)
    exported=json.loads(args.input.read_text())
    assert len({x['match_id'] for x in exported})==len(exported),"Duplicate match IDs"
    ev=OnnxEvaluator(args.teacher)
    rng=np.random.default_rng(900_001)
    cfg=mcts.MCTSConfig(simulations=args.simulations,add_root_noise=False,warmup_temperatures=())
    games=[];issues=[];stats=collections.Counter();dataset=[];validation=[]
    missing=[];loss_summaries=[];position_count=0
    for number,game in enumerate(exported):
        if not game.get("protocol"):
            missing.append(game['match_id']);continue
        env=ColonyOneEnv(scene=np.asarray(game['starting_observation'],np.float32))
        reported=re.findall(r"\*\*\* (CYAN|MAGENTA) WINS",game['protocol'])[-1]
        assert game['score']==int(reported=="CYAN"),(game['match_id'],"score meaning changed")
        won=reported==game['you']
        frames=[snapshot(env)];positions=[];our_last_actions=[];captures=[];burns=[]
        for line in game['protocol'].splitlines():
            parsed=LINE.match(line)
            if not parsed:continue
            move,team,abbr,origin,target,kind=parsed.groups()
            assert int(move)==env.action and team==env.current_team,(game['match_id'],line)
            fr,to=square(origin),square(target)
            actions=[a for a in env.legal_actions() if a.from_pos==fr and a.to_pos==to]
            assert len(actions)==1,(game['match_id'],line,"not uniquely legal")
            assert env.board.piece_at(*fr).piece_type.abbr==abbr
            positions.append((env.clone(),action_index(actions[0],team)))
            if team==game['you'] and env.actions_left==1:
                our_last_actions.append((int(move),env.clone(),action_index(actions[0],team)))
            before={(r,c):(q.team,q.piece_type.value) for r in range(8) for c in range(8)
                    if (q:=env.board.piece_at(r,c)) is not None}
            env.step(actions[0])
            if 'x' in line:captures.append(dict(action=int(move),team=team,text=line.strip()))
            after={(r,c) for r in range(8) for c in range(8) if env.board.piece_at(r,c)}
            removed=[(pos,before[pos]) for pos in before if pos not in after and env.board.is_locked(*pos)]
            if removed:burns.append(dict(action=int(move),pieces=removed))
            frames.append(snapshot(env,line.strip(),team,[list(fr),list(to)]))
        assert env.terminated and env.winner==reported,(game['match_id'],"terminal disagreement")
        stats['verified_games']+=1;stats['wins' if won else 'losses']+=1
        stats[f"{game['you']}_{'wins' if won else 'losses'}"]+=1
        game_report=dict(id=game['match_id'],you=game['you'],winner=reported,won=won,
                         actions=len(positions),firewall_losses=burns)
        if not won:
            terminal_line=game['protocol'].splitlines()[-2]
            if "BARE KERNEL" in game['protocol']:reason="bare-Kernel endgame"
            elif "FIREWALL" in terminal_line:reason="Kernel destroyed by firewall"
            else:reason="Kernel captured"
            stats[reason]+=1
            blunders=[]
            for move,before,played in our_last_actions:
                child=before.clone();child.step(played)
                threat=None
                if child.terminated:
                    if child.winner==game['you']:continue
                    vulnerable=True
                else:
                    threat=immediate_capture(child.board,child.current_team,child.actions_left)
                    vulnerable=bool(threat)
                if not vulnerable:continue
                safe=safe_final_actions(before)
                if safe and played not in safe:
                    legal=before.legal_action_indices()
                    pi=np.isin(legal,safe).astype(np.float32);pi/=pi.sum()
                    record=dict(match_id=game['match_id'],action=move,played=played,
                        played_label=before.format_action(next(a for a in before.legal_actions() if action_index(a,before.current_team)==played)),
                        safe_actions=safe,
                        safe_labels=[before.format_action(a) for a in before.legal_actions() if action_index(a,before.current_team) in safe],
                        threat=[child.format_action(a) for a in threat] if threat else ["terminal rule loss"])
                    blunders.append(record);issues.append(record)
                    sample=(before.observation().astype(np.float16),legal.astype(np.int16),pi.astype(np.float16),np.float32(-1))
                    (validation if game['match_id']%5==0 else dataset).extend([sample]*4)
            game_report.update(reason=reason,avoidable_threats=blunders)
            loss_summaries.append(game_report)
        # Relabel every real position with a deeper search; terminal outcome
        # labels use the actual side to move. Split by whole match, before D4.
        for before,played in positions:
            root=mcts.run(before.board,before.current_team,before.actions_left,before.round_number,ev,cfg,rng=rng)
            legal=np.array(sorted(root.N),np.int16)
            visits=np.array([root.N[int(a)] for a in legal],np.float32)
            visits/=visits.sum()
            z=np.float32(1 if before.current_team==reported else -1)
            sample=(before.observation().astype(np.float16),legal,visits.astype(np.float16),z)
            (validation if game['match_id']%5==0 else dataset).append(sample)
            position_count+=1
        games.append(dict(won=won,winner=reported,a_team=game['you'],seed=game['match_id'],moves=len(positions),frames=frames))
        print(f"Verified match {game['match_id']}: {'win' if won else 'loss'} as {game['you']}, {len(positions)} actions",flush=True)
    summary=dict(stats=stats,missing_protocols=missing,positions=position_count,
                 training_samples=len(dataset),validation_samples=len(validation),
                 avoidable_last_actions=len(issues),losses=loss_summaries,
                 input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                 teacher_sha256=hashlib.sha256(args.teacher.read_bytes()).hexdigest(),
                 teacher_simulations=args.simulations)
    (output/"data/loss_analysis.json").write_text(json.dumps(summary,indent=2))
    for name,items in [('tournament_training',dataset),('tournament_validation',validation)]:
        with gzip.open(output/f"data/{name}.pkl.gz","wb") as f:pickle.dump(items,f,protocol=5)
    report=dict(model_a="Our exported tournament games",model_b="Tournament opponents",
        wins=stats['wins'],games=len(games),win_rate=stats['wins']/len(games),simulations=100)
    data=dict(report=report,games=games)
    (output/"replays/tournament.html").write_text((ROOT/"replay.html").read_text().replace(
        "/* MATCH_DATA */null",json.dumps(data).replace("</","<\\/")))
    lines=["# Tournament loss analysis", "",f"Verified {len(games)} full protocols against the frozen official engine: {stats['wins']} wins, {stats['losses']} losses. {len(missing)} exports lack a protocol and cannot be diagnosed.",
        "", "The export's score is Cyan's result. The actual winner and the `you` color determine our win/loss; a score of zero is a win when we played Magenta.",
        "",f"Replayed {position_count} positions. Found {len(issues)} avoidable final actions that allowed a direct Kernel capture within the opponent's next turn or an immediate terminal-rule loss. The alternative actions only establish short-term survival, not a forced win.", ""]
    for loss in loss_summaries:
        lines.extend([f"## Match {loss['id']} — {loss['you']}, {loss['reason']}",""])
        for b in loss['avoidable_threats']:
            lines.append(f"- Action {b['action']}: played {b['played_label']}; opponent threat: {', then '.join(b['threat'])}. Safer alternatives include {', '.join(b['safe_labels'][:5])}.")
        if not loss['avoidable_threats']:lines.append("No avoidable final-action error found by the one-turn capture test. Earlier planning or a longer tactical search needs investigation.")
        lines.append("")
    lines.extend(["## Training use", "",f"Training: {len(dataset)} examples; validation: {len(validation)} examples. Entire match IDs divisible by five are reserved for validation. Search targets use {args.simulations} simulations and no root noise. Tactical repairs receive extra weight. Outcomes come from the recorded game, so they are not claims about the safer alternative's final result.",
        "", "This export does not identify model hashes or opponent model files. Results cannot be attributed to a particular saved checkpoint, and the strongest opponents cannot be recreated from their weights. The action traces are sufficient for tactical training and exact replay."])
    (output/"LOSS_ANALYSIS.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k!='losses'},indent=2),flush=True)


if __name__=='__main__':main()
