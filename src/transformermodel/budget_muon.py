"""Experimental Muon combinations and measured polynomial-work allocation.

PE coefficients: arxiv:2505.16932 appendix A, including 1.01 safety scaling.
Post-normalization uses Muon+/NorMuon mechanisms with explicitly common RMS .2.
The work controller is an unverified hypothesis, not a guaranteed descent rule.
"""
import math
import mlx.core as mx
import mlx.optimizers as optim

PE_COEFFICIENTS = (
    (8.28721201814563,-23.595886519098837,17.300387312530933),
    (4.107059111542203,-2.9478499167379106,.5448431082926601),
    (3.9486908534822946,-2.908902115962949,.5518191394370137),
    (3.3184196573706015,-2.488488024314874,.51004894012372),
    (2.300652019954817,-1.6689039845747493,.4188073119525673),
)


def _iterate(x, start, stop):
    for a,b,c in PE_COEFFICIENTS[start:stop]:
        gram=x@x.T
        x=(a/1.01)*x+((b/1.01**3)*gram+(c/1.01**5)*(gram@gram))@x
    return x


def _start(matrix,precision):
    if matrix.ndim!=2 or precision not in ('fp32','bf16'):
        raise ValueError('expected matrix and fp32/bf16 precision')
    x=matrix.astype(mx.bfloat16 if precision=='bf16' else mx.float32)
    if matrix.shape[0]>matrix.shape[1]:x=x.T
    return x/(mx.linalg.norm(x)*1.01+1e-7)


def polar(matrix,steps=5,precision='bf16'):
    if steps not in (3,4,5):raise ValueError('supported work budgets are 3, 4 and 5')
    x=_iterate(_start(matrix,precision),0,steps)
    return (x.T if matrix.shape[0]>matrix.shape[1] else x).astype(mx.float32)


def work_steps(records,cheap_steps=3):
    """All matrices in a shape group must pass; no negative-gain shortcut."""
    return cheap_steps if records and all(math.isfinite(c+low+full) and c>=.95
        and full>0 and low>=.99*full for c,low,full in records) else 5


class BudgetMuon(optim.Optimizer):
    def __init__(self,learning_rate=.001,kind='budget',precision='bf16',
                 momentum=.95,beta2=.95,weight_decay=.01,steps=5,cheap_steps=3):
        super().__init__()
        if kind not in ('pe','plus','nor','combined','budget'):
            raise ValueError(kind)
        if precision not in ('fp32','bf16') or not 0<=momentum<1 or not 0<=beta2<1:
            raise ValueError('invalid precision or moments')
        if not math.isfinite(learning_rate) or learning_rate<=0 or not math.isfinite(weight_decay) or weight_decay<0:
            raise ValueError('invalid learning rate or decay')
        self._maybe_schedule('learning_rate',learning_rate)
        if steps not in (3,4,5) or cheap_steps not in (3,4):raise ValueError("invalid work budget")
        self.steps,self.cheap_steps=steps,cheap_steps
        self.kind,self.precision=kind,precision
        self.momentum,self.beta2,self.weight_decay=momentum,beta2,weight_decay
        # Host decisions are static per compiled function, refreshed every block.
        self.calibrate=False
        self.steps_by_shape={}
        self.last_diagnostics=[]

    def init_single(self,parameter,state):
        if parameter.ndim!=2:raise ValueError('Muon hidden parameters must be matrices')
        state['momentum']=mx.zeros_like(parameter)
        if self.kind in ('nor','combined','budget'):
            state['row_second']=mx.zeros((parameter.shape[0],1),mx.float32)
        if self.kind=='budget':state['diagnostic']=mx.zeros((3,),mx.float32)

    def postprocess(self,direction,second):
        d=direction.astype(mx.float32)
        if self.kind in ('plus','combined','budget'):
            d=d/mx.sqrt(mx.sum(d*d,axis=0,keepdims=True)+1e-8)
        if self.kind=='plus':d=d/mx.sqrt(mx.sum(d*d,axis=1,keepdims=True)+1e-8)
        if self.kind in ('nor','combined','budget'):
            second=self.beta2*second+(1-self.beta2)*mx.mean(d*d,axis=1,keepdims=True)
            d=d/(mx.sqrt(second)+1e-8)
        return .2*d/mx.maximum(mx.sqrt(mx.mean(d*d)),1e-7),second

    def apply_single(self,gradient,parameter,state):
        moment=self.momentum*state['momentum']+gradient
        state['momentum']=moment
        momentum_direction=gradient+self.momentum*moment
        second=state.get('row_second')
        if self.kind=='budget' and self.calibrate:
            x=_start(momentum_direction,self.precision);cheap=_iterate(x,0,self.cheap_steps)
            full=_iterate(cheap,self.cheap_steps,5)
            if parameter.shape[0]>parameter.shape[1]:cheap,full=cheap.T,full.T
            low,_=self.postprocess(cheap,second)
            direction,second=self.postprocess(full,second)
            cosine=mx.sum(low*direction)/mx.maximum(mx.linalg.norm(low)*mx.linalg.norm(direction),1e-12)
            gnorm=mx.maximum(mx.linalg.norm(gradient),1e-12)
            state['diagnostic']=mx.stack([cosine,mx.sum(gradient*low)/gnorm,
                                         mx.sum(gradient*direction)/gnorm])
        else:
            steps=self.steps_by_shape.get(tuple(parameter.shape),5) if self.kind=='budget' else self.steps
            direction,second=self.postprocess(polar(momentum_direction,steps,self.precision),second)
        if second is not None:state['row_second']=second
        return parameter*(1-self.learning_rate*self.weight_decay)-self.learning_rate*direction

    def choose_policy(self):
        """Called only after a synchronized calibration step, outside compilation."""
        groups={};details=[]
        def visit(node):
            if not isinstance(node,dict):return
            if 'diagnostic' in node and 'momentum' in node:
                shape=tuple(node['momentum'].shape);record=node['diagnostic'].tolist()
                groups.setdefault(shape,[]).append(record);details.append({'shape':shape,'scores':record})
            else:
                for child in node.values():
                    if isinstance(child,list):
                        for item in child:visit(item)
                    else:visit(child)
        visit(self.state)
        self.steps_by_shape={shape:work_steps(records,self.cheap_steps) for shape,records in groups.items()}
        self.last_diagnostics=details
        return self.steps_by_shape


def compile_policy_step(function,state,optimizer):
    """Give every static work policy its own Python function identity for MLX."""
    compiled={}
    def run(x,y,calibrate=False):
        if optimizer is not None and optimizer.kind=='budget':
            optimizer.calibrate=calibrate
            policy=tuple(sorted(optimizer.steps_by_shape.items()))
            signature=(calibrate,policy)
        else:signature=None
        if signature not in compiled:
            # Recompiling `function` itself reuses its original cached trace.
            def specialized(x,y):
                return function(x,y)
            compiled[signature]=mx.compile(specialized,inputs=state,outputs=state)
        return compiled[signature](x,y)
    run.compiled=compiled
    return run
