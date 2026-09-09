"""CoFrGeNet's staged coefficient activation, using ordinary MLX AdamW groups.

Separate optimizer counters start when each depth is first activated. Frozen
coefficient rows receive neither moments nor weight decay; upstream gradients
through their forward function remain part of backpropagation.
"""
import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten,tree_unflatten


class DyadicAdamW:
    def __init__(self,learning_rate,depth=2,coefficient_scale=.1):
        self.learning_rate=learning_rate
        self.coefficient_scale=coefficient_scale
        self.depth=depth
        self.base=optim.AdamW(learning_rate,weight_decay=.01)
        self.ladders=[optim.AdamW(learning_rate,weight_decay=.01) for _ in range(depth)]

    def group(self,flat,level=None):
        result={}
        for name,value in flat.items():
            coefficient='.ffn.denominators.' in name
            if level is None and not coefficient:result[name]=value
            elif level is not None and coefficient:result[name]=value[level::self.depth]
        return result

    def init(self,parameters):
        flat=dict(tree_flatten(parameters))
        self.base.init(self.group(flat))
        for i,opt in enumerate(self.ladders):opt.init(self.group(flat,i))
        self.state=dict(base=self.base.state,ladders=[opt.state for opt in self.ladders])

    def mask_gradients(self,gradients,phase):
        result=[]
        for name,g in tree_flatten(gradients):
            if '.ffn.denominators.' in name:
                mask=(mx.arange(g.shape[0])%self.depth)<phase
                g=g*mask.reshape((g.shape[0],)+(1,)*(g.ndim-1))
            result.append((name,g))
        return tree_unflatten(result)

    def update(self,model,gradients,phase,scale):
        parameters=dict(tree_flatten(model.parameters()))
        gradients=dict(tree_flatten(gradients))
        self.base.learning_rate=self.learning_rate*scale
        updated=self.base.apply_gradients(self.group(gradients),self.group(parameters))
        active=[]
        for i,opt in enumerate(self.ladders[:phase]):
            opt.learning_rate=self.learning_rate*self.coefficient_scale*scale
            active.append(opt.apply_gradients(self.group(gradients,i),self.group(parameters,i)))
        for name,value in parameters.items():
            if '.ffn.denominators.' in name:
                pieces=[active[i][name] if i<phase else value[i::self.depth] for i in range(self.depth)]
                updated[name]=mx.stack(pieces,axis=1).reshape(value.shape)
        model.update(tree_unflatten(list(updated.items())))
