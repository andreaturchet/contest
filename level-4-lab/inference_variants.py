"""Build bounded ONNX evaluator variants without changing the referee search."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as H,numpy_helper as NH,TensorProto,compose


def init(name,value):return NH.from_array(np.asarray(value),name)


def make(nodes,inits,opset=17):
    x=H.make_tensor_value_info('observation',TensorProto.FLOAT,['batch',16,8,8])
    p=H.make_tensor_value_info('policy',TensorProto.FLOAT,['batch',4096])
    v=H.make_tensor_value_info('value',TensorProto.FLOAT,['batch'])
    graph=H.make_graph(nodes,'colony_inference_variant',[x],[p,v],inits)
    m=H.make_model(graph,opset_imports=[H.make_opsetid('',opset)],ir_version=9)
    return m


def write(m,path):
    from colony_one.submission_policy import validate_submission_contract
    onnx.checker.check_model(m);validate_submission_contract(m)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    onnx.save(m,path)
    if path.stat().st_size>5*1024**2:raise ValueError('Exceeds 5 MiB')
    return path


def embed(m,prefix,input_name):
    m=compose.add_prefix(m,prefix)
    old=m.graph.input[0].name
    for node in m.graph.node:
        for i,name in enumerate(node.input):
            if name==old:node.input[i]=input_name
    return list(m.graph.node),list(m.graph.initializer),[o.name for o in m.graph.output]


def calibrated(path,output,policy_scale=1.,value_scale=1.):
    m=onnx.load(path);nodes,inits,(p,v)=embed(m,'base_','observation')
    inits += [init('policy_scale',np.float32(policy_scale)),init('value_scale',np.float32(value_scale)),
              init('minus_one',np.float32(-1)),init('plus_one',np.float32(1))]
    nodes += [H.make_node('Mul',[p,'policy_scale'],['policy']),
              H.make_node('Mul',[v,'value_scale'],['scaled_value']),
              H.make_node('Clip',['scaled_value','minus_one','plus_one'],['value'])]
    return write(make(nodes,inits,m.opset_import[0].version),output)


def mixture(paths,weights,output):
    weights=np.asarray(weights,np.float32);weights/=weights.sum()
    if len(paths)!=len(weights) or np.min(weights)<0:raise ValueError('Mixture weights must be nonnegative')
    nodes=[];inits=[];ps=[];vs=[]
    for i,(path,weight) in enumerate(zip(paths,weights)):
        ns,ins,(p,v)=embed(onnx.load(path),f'm{i}_','observation')
        nodes+=ns;inits+=ins;inits.append(init(f'weight_{i}',weight))
        ps.append(f'policy_{i}');vs.append(f'value_{i}')
        nodes += [H.make_node('Mul',[p,f'weight_{i}'],[ps[-1]]),
                  H.make_node('Mul',[v,f'weight_{i}'],[vs[-1]])]
    nodes += [H.make_node('Sum',ps,['policy']),H.make_node('Sum',vs,['value'])]
    return write(make(nodes,inits),output)


GROUPS={
    'identity':[(0,False)],
    'halfturn':[(0,False),(2,False)],
    'rotations':[(i,False) for i in range(4)],
    'reflections':[(0,False),(2,False),(0,True),(2,True)],
    'd4':[(i,m) for m in [False,True] for i in range(4)],
}


def symmetry(path,output,group='d4'):
    transforms=GROUPS[group];count=len(transforms)
    nodes=[H.make_node('Reshape',['observation','flatten_shape'],['flat_observation'])]
    inits=[init('flatten_shape',np.array([-1,16,64],np.int64)),
           init('expand_axis',np.array([1],np.int64)),
           init('model_shape',np.array([-1,16,8,8],np.int64)),
           init('policy_cube_shape',np.array([-1,count,4096],np.int64)),
           init('value_cube_shape',np.array([-1,count],np.int64)),
           init('average_factor',np.float32(1/count))]
    expanded=[];aligned=[]
    for i,(turns,mirror) in enumerate(transforms):
        grid=np.arange(64).reshape(8,8)
        if mirror:grid=grid[:,::-1]
        grid=np.rot90(grid,turns).reshape(-1)
        mapping=np.argsort(grid)
        action_map=(mapping[:,None]*64+mapping[None,:]).reshape(-1)
        inits += [init(f'board_perm_{i}',grid.astype(np.int64)),
                  init(f'action_perm_{i}',action_map.astype(np.int64)),
                  init(f'orientation_{i}',np.int64(i))]
        nodes += [H.make_node('Gather',['flat_observation',f'board_perm_{i}'],[f'flat_{i}'],axis=2),
                  H.make_node('Unsqueeze',[f'flat_{i}','expand_axis'],[f'expanded_{i}'])]
        expanded.append(f'expanded_{i}')
    nodes += [H.make_node('Concat',expanded,['observation_stack'],axis=1),
              H.make_node('Reshape',['observation_stack','model_shape'],['model_observation'])]
    ns,ins,(p,v)=embed(onnx.load(path),'base_','model_observation')
    nodes+=ns;inits+=ins
    nodes += [H.make_node('Reshape',[p,'policy_cube_shape'],['policy_cube']),
              H.make_node('Reshape',[v,'value_cube_shape'],['value_cube'])]
    for i in range(count):
        nodes += [H.make_node('Gather',['policy_cube',f'orientation_{i}'],[f'pol_{i}'],axis=1),
                  H.make_node('Gather',[f'pol_{i}',f'action_perm_{i}'],[f'aligned_{i}'],axis=1)]
        aligned.append(f'aligned_{i}')
    nodes += [H.make_node('Sum',aligned,['policy_sum']),
              H.make_node('Mul',['policy_sum','average_factor'],['policy']),
              H.make_node('ReduceMean',['value_cube'],['value'],axes=[1],keepdims=0)]
    return write(make(nodes,inits),output)
