"""A conservative, exact direct-Kernel-attack feature for the trained evaluator."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class DirectKernelAttack(nn.Module):
    """Recognize a legal direct attack, repeatable after one guard sacrifice.

    This is a sufficient test for winning this turn, not a complete two-action
    solver: move-then-attack combinations are intentionally left to MCTS.
    """
    def __init__(self):
        super().__init__()
        geometry=np.zeros((1,6,64,64),np.float32)
        indices=np.zeros((2,64,64),np.int64)
        needed=np.zeros((2,64,64),np.float32)
        for fr in range(64):
            r,c=divmod(fr,8)
            for to in range(64):
                rr,cc=divmod(to,8);dr,dc=rr-r,cc-c
                a,b=abs(dr),abs(dc);dist=max(a,b)
                diag=a==b and a>0;orth=(a==0)!=(b==0)
                geometry[0,:,fr,to]=[
                    diag and dist<=2,orth and dist<=3,sorted([a,b])==[1,2],
                    diag and 2<=dist<=3,(diag or orth) and dist<=2,dist==1]
                if (diag or orth) and 2<=dist<=3:
                    for step in range(1,dist):
                        indices[step-1,fr,to]=(r+step*int(np.sign(dr)))*8+c+step*int(np.sign(dc))
                        needed[step-1,fr,to]=1
        self.register_buffer('geometry',torch.from_numpy(geometry))
        self.register_buffer('path1',torch.from_numpy(indices[0].reshape(-1)))
        self.register_buffer('path2',torch.from_numpy(indices[1].reshape(-1)))
        self.register_buffer('need1',torch.from_numpy(needed[0])[None])
        self.register_buffer('need2',torch.from_numpy(needed[1])[None])
        self.register_buffer('ray_type',torch.tensor([1,1,0,1,1,1],dtype=torch.float32).reshape(1,6,1,1))
        self.register_buffer('near',torch.ones(1,1,3,3))

    def forward(self,x):
        own=x[:,:6]*(1-x[:,12:13])
        enemy_kernel=x[:,11:12]*(1-x[:,12:13])
        blockers=(x[:,:12].sum(1)+x[:,12]).reshape(-1,64)
        block1=blockers.index_select(1,self.path1).reshape(-1,64,64)*self.need1
        block2=blockers.index_select(1,self.path2).reshape(-1,64,64)*self.need2
        clear=((block1+block2)<.5).to(x.dtype).unsqueeze(1)
        valid=self.geometry*(1-self.ray_type+self.ray_type*clear)
        attacks=(valid*own.reshape(-1,6,64,1)).sum(1)
        targets=attacks*enemy_kernel.reshape(-1,1,64)
        guards=(F.conv2d(x[:,6:11].sum(1,keepdim=True),self.near,padding=1)*enemy_kernel).sum((1,2,3))
        actions=x[:,13].mean((1,2))*2
        eligible=(guards<actions).to(x.dtype)
        winning_moves=(targets>0).to(x.dtype)*eligible.reshape(-1,1,1)
        has_win=winning_moves.sum((1,2))>0
        return has_win,winning_moves.reshape(-1,4096)


class TwoActionKernelAttack(DirectKernelAttack):
    """Also prove empty-square reposition-then-capture wins in the same turn.

    Old source squares remain blockers while checking the second action. This
    is conservative: it can miss wins, but cannot invent a clear attack ray.
    """
    def __init__(self):
        super().__init__()
        moves=np.zeros((1,6,64,64),np.float32)
        for fr in range(64):
            r,c=divmod(fr,8)
            for to in range(64):
                rr,cc=divmod(to,8);a,b=abs(rr-r),abs(cc-c);d=max(a,b)
                orth=(a==0)!=(b==0);diag=a==b and a>0
                moves[0,:,fr,to]=[orth and d<=2,orth and d<=3,
                    sorted([a,b])==[1,2],orth and d==1,(orth or diag) and d<=2,d==1]
        self.register_buffer('move_geometry',torch.from_numpy(moves))

    def forward(self,x):
        direct,winning=super().forward(x)
        own=x[:,:6]*(1-x[:,12:13]);kernel=x[:,11:12]
        blockers=(x[:,:12].sum(1)+x[:,12]).reshape(-1,64)
        b1=blockers.index_select(1,self.path1).reshape(-1,64,64)*self.need1
        b2=blockers.index_select(1,self.path2).reshape(-1,64,64)*self.need2
        clear=((b1+b2)<.5).to(x.dtype).unsqueeze(1)
        clear_by_type=1-self.ray_type+self.ray_type*clear
        can_attack=(self.geometry*clear_by_type*kernel.reshape(-1,1,1,64)).sum(3)>0
        empty=(blockers<.5).to(x.dtype).reshape(-1,1,1,64)
        moves=self.move_geometry*clear_by_type*own.reshape(-1,6,64,1)*empty
        guards=(F.conv2d(x[:,6:11].sum(1,keepdim=True),self.near,padding=1)*kernel).sum((1,2,3))
        eligible=((guards==0)&(x[:,13].mean((1,2))==1.)).to(x.dtype).reshape(-1,1,1)
        two_step=((moves*can_attack.to(x.dtype).unsqueeze(2)).sum(1)>0).to(x.dtype)*eligible
        winning=winning+two_step.reshape(-1,4096)
        return direct|(two_step.sum((1,2))>0),winning


class TwoActionWrapper(nn.Module):
    def __init__(self,model,boost=3.):
        super().__init__();self.model=model;self.tactic=TwoActionKernelAttack();self.boost=boost
    def forward(self,x):
        p,v=self.model(x);won,moves=self.tactic(x)
        return p+self.boost*moves,torch.where(won,v*0+1,v)


class TacticalWrapper(nn.Module):
    def __init__(self,model,boost=4.,override=True):
        super().__init__();self.model=model;self.tactic=DirectKernelAttack()
        self.boost=boost;self.override=override
    def forward(self,x):
        policy,value=self.model(x)
        winning,moves=self.tactic(x)
        if self.override:value=torch.where(winning,value*0+1.,value)
        return policy+self.boost*moves,value


class BareKernelEndgame(nn.Module):
    """Exact terminal move labels when the last non-Kernel can disappear."""
    def __init__(self):
        super().__init__()
        positions=np.array([divmod(i,8) for i in range(64)])
        distance=np.abs(positions[:,None]-positions[None,:]).max(-1)
        rings=np.array([min(r,c,7-r,7-c) for r,c in positions],np.float32)
        self.register_buffer('king_moves',torch.tensor(distance==1,dtype=torch.float32)[None])
        self.register_buffer('close',torch.tensor(distance<=2,dtype=torch.float32))
        self.register_buffer('rings',torch.from_numpy(rings).reshape(1,64))
        self.register_buffer('near',torch.ones(1,1,3,3))

    def forward(self,x):
        own_kernel=x[:,5].reshape(-1,64)
        enemy_kernel=x[:,11].reshape(-1,64)
        own_non=x[:,:5].sum((1,2,3))
        enemy_non_map=x[:,6:11].sum(1)
        enemy_non=enemy_non_map.sum((1,2))
        last_action=x[:,13].mean((1,2))==.5
        eligible=((own_non==0)&last_action).to(x.dtype).reshape(-1,1,1)
        bare=(enemy_non==0).to(x.dtype).reshape(-1,1,1)
        one=(enemy_non==1).to(x.dtype).reshape(-1,1,1)
        king_moves=self.king_moves*own_kernel.reshape(-1,64,1)*(1-x[:,12].reshape(-1,1,64))
        enemy_at=enemy_kernel.reshape(-1,1,64)
        direct_targets=bare*(1-enemy_at)+one*enemy_non_map.reshape(-1,1,64)
        terminal=eligible*king_moves*direct_targets
        enemy_ring=(enemy_kernel*self.rings).sum(1,keepdim=True)
        close=(enemy_kernel@self.close)>0
        magenta=x[:,15].mean((1,2))>.5
        wins=(~close)&((self.rings>enemy_ring)|((self.rings==enemy_ring)&magenta.unsqueeze(1)))
        good=terminal*wins.to(x.dtype).unsqueeze(1)
        bad=terminal*(~wins).to(x.dtype).unsqueeze(1)
        guarded=(F.conv2d(enemy_non_map.unsqueeze(1),self.near,padding=1)*x[:,11:12]).sum((1,2,3))>0
        # Redirecting a last-action Kernel attack sacrifices the final guard,
        # leaves the two Kernels adjacent, and loses by the bare-Kernel rule.
        bad=bad+eligible*king_moves*enemy_at*one*guarded.to(x.dtype).reshape(-1,1,1)
        has_capture=(king_moves*enemy_at).sum((1,2))>0
        proven_win=(good.sum((1,2))>0)|(eligible.reshape(-1)>0)&(bare.reshape(-1)>0)&has_capture
        proven_loss=(eligible.reshape(-1)>0)&(bare.reshape(-1)>0)&(~proven_win)&(king_moves.sum((1,2))>0)
        return good.reshape(-1,4096),bad.reshape(-1,4096),proven_win,proven_loss


class EndgameWrapper(nn.Module):
    def __init__(self,model,boost=6.,direct=False):
        super().__init__();self.model=model;self.endgame=BareKernelEndgame()
        self.tactic=DirectKernelAttack() if direct else None
        self.boost=boost
    def forward(self,x):
        p,v=self.model(x)
        good,bad,won,lost=self.endgame(x)
        p=p+self.boost*(good-bad)
        if self.tactic is not None:
            direct,moves=self.tactic(x);p=p+2*moves;won=won|direct
        v=torch.where(won,v*0+1,torch.where(lost,v*0-1,v))
        return p,v
