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
    # CPU Tanh approximations can exceed one by a few float32 ulps on the
    # validator's extreme inputs. The contract uses strict numeric bounds.
    value_name=m.graph.output[1].name
    raw_name='export_unclipped_value'
    for node in m.graph.node:
        for i,name in enumerate(node.output):
            if name==value_name:node.output[i]=raw_name
        for i,name in enumerate(node.input):
            if name==value_name:node.input[i]=raw_name
    m.graph.initializer.extend([
        onnx.numpy_helper.from_array(np.asarray(-1.,np.float32),'export_value_min'),
        onnx.numpy_helper.from_array(np.asarray(1.,np.float32),'export_value_max')])
    m.graph.node.append(onnx.helper.make_node('Clip',
        [raw_name,'export_value_min','export_value_max'],[value_name]))
    m.graph.output[0].type.tensor_type.shape.dim[1].dim_value=4096
    # Constant-only reshape/index arguments are checked by the actual contract.
    onnx.checker.check_model(m)
    validate_submission_contract(m)
    onnx.save(m,path)
    if path.stat().st_size>5*1024**2:raise ValueError("Export exceeds 5 MiB")
    return path
