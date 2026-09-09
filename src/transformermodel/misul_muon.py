"""NorMuon/Polar Express matrix updates with 16-bit state and compensated adds.

Uses the qualified local PE coefficients and RMS convention. Auxiliary vectors
and embeddings use AdamW. The algorithm and its learning-rate schedule are
separate. No floating model or optimizer array is constructed in FP32.
"""
import math
import mlx.core as mx
from mlx.utils import tree_flatten,tree_unflatten
from .budget_muon import _start,_iterate


def matrix_parameter(name,value):
    return value.ndim==2 and (name.startswith('blocks.') or name.startswith('flow_') and name.endswith('.weight'))


def polar16(matrix):
    x=_iterate(_start(matrix,'bf16'),0,5)
    return x.T if matrix.shape[0]>matrix.shape[1] else x


class Muon16:
    def __init__(self,parameters,learning_rate=.006,aux_learning_rate=.001,momentum=.7,weight_decay=.01,kind='nor'):
        if kind not in ('nor','adamw') or not all(math.isfinite(v) and v>0 for v in (learning_rate,aux_learning_rate)):
            raise ValueError('invalid optimizer recipe')
        if not 0<=momentum<1 or not 0<=weight_decay<1:raise ValueError('invalid momentum or decay')
        self.learning_rate=learning_rate;self.aux_learning_rate=aux_learning_rate
        self.momentum=momentum;self.weight_decay=weight_decay;self.kind=kind
        self.state={'step':mx.array(0,dtype=mx.int32),'leaves':{}}
        for name,value in tree_flatten(parameters):
            if value.dtype!=mx.bfloat16:raise ValueError('BF16 working parameters are required; no FP32 masters')
            hidden=kind=='nor' and matrix_parameter(name,value)
            self.state['leaves'][name]={'momentum':mx.zeros_like(value),
                'second':mx.zeros((value.shape[0],1),dtype=mx.bfloat16) if hidden else mx.zeros_like(value),
                'compensation':mx.zeros_like(value)}

    def update(self,model,gradients,scale):
        gradients=dict(tree_flatten(gradients))
        self.state['step']=self.state['step']+1
        beta1=mx.array(.9,dtype=mx.bfloat16);beta2=mx.array(.95,dtype=mx.bfloat16)
        step=self.state['step'].astype(mx.bfloat16)
        updated=[]
        for name,parameter in tree_flatten(model.parameters()):
            gradient=gradients[name];state=self.state['leaves'][name]
            if self.kind=='nor' and matrix_parameter(name,parameter):
                moment=self.momentum*state['momentum']+gradient
                state['momentum']=moment
                direction=polar16(gradient+self.momentum*moment)
                second=beta2*state['second']+(1-beta2)*mx.mean(direction*direction,axis=1,keepdims=True)
                direction=direction/(mx.sqrt(second)+1e-7)
                direction=.2*direction/mx.maximum(mx.sqrt(mx.mean(direction*direction)),1e-7)
                lr=self.learning_rate*scale
            else:
                moment=beta1*state['momentum']+(1-beta1)*gradient
                second=beta2*state['second']+(1-beta2)*gradient*gradient
                state['momentum']=moment
                direction=(moment/(1-beta1**step))/(mx.sqrt(second/(1-beta2**step))+1e-7)
                lr=self.aux_learning_rate*scale
            state['second']=second
            # Compensation retains small BF16 updates that would round away near1.
            delta=-lr*(direction+self.weight_decay*parameter)-state['compensation']
            value=parameter+delta
            state['compensation']=(value-parameter)-delta
            updated.append((name,value))
        model.update(tree_unflatten(updated))


def clip16(gradients,maximum=1.):
    flat=tree_flatten(gradients)
    norm=mx.sqrt(sum(mx.sum(value*value) for _,value in flat))
    factor=mx.minimum(1.,maximum/mx.maximum(norm,1e-7))
    return tree_unflatten([(name,value*factor) for name,value in flat]),norm
