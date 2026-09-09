"""Isolate low-precision KDA gradient cancellation against a NumPy oracle.

Hypothesis: equivalent chunk algebra has more cancellation than the sequential
update for almost collinear keys. Null: chunk length/type has no useful effect.
This is a numerical diagnostic, not a model-training or performance result.
"""
import hashlib
import json
import argparse
from pathlib import Path
import numpy as np
import mlx.core as mx
from transformermodel.combined import delta_scan,delta_chunk
from transformermodel.safe_run import require_guard


def oracle(q,k,v,alpha,beta):
    state=np.zeros((*q.shape[:2],q.shape[-1],v.shape[-1]))
    states=[state.copy()];decayed=[];errors=[];outputs=[]
    for tick in range(q.shape[-2]):
        d=alpha[...,tick,:,None]*state
        error=v[...,tick,:]-np.einsum('bhk,bhkv->bhv',k[...,tick,:],d)
        state=d+beta[...,tick,None,None]*k[...,tick,:,None]*error[...,None,:]
        states.append(state.copy());decayed.append(d);errors.append(error)
        outputs.append(np.einsum('bhk,bhkv->bhv',q[...,tick,:],state))
    outputs=np.stack(outputs,axis=-2)
    do=2*outputs/outputs.size;ds=2*state/state.size
    dq,dk,dv,da,db=[np.zeros_like(x) for x in (q,k,v,alpha,beta)]
    for tick in reversed(range(q.shape[-2])):
        qq,kk,bb=q[...,tick,:],k[...,tick,:],beta[...,tick]
        dq[...,tick,:]=np.einsum('bhkv,bhv->bhk',states[tick+1],do[...,tick,:])
        ds=ds+qq[...,None]*do[...,tick,None,:]
        error=errors[tick]
        db[...,tick]=np.sum(ds*kk[...,None]*error[...,None,:],axis=(-2,-1))
        dk[...,tick,:]=bb[...,None]*np.einsum('bhkv,bhv->bhk',ds,error)
        de=bb[...,None]*np.einsum('bhk,bhkv->bhv',kk,ds)
        dv[...,tick,:]=de
        dk[...,tick,:]-=np.einsum('bhkv,bhv->bhk',decayed[tick],de)
        dd=ds-kk[...,None]*de[...,None,:]
        da[...,tick,:]=np.sum(dd*states[tick],axis=-1)
        ds=dd*alpha[...,tick,:,None]
    return outputs,state,[dq,dk,dv,da,db]


def main():
    require_guard();mx.set_memory_limit(256*2**20);mx.set_cache_limit(16*2**20)
    parser=argparse.ArgumentParser();parser.add_argument('--compiled',action='store_true');args=parser.parse_args()
    rng=np.random.default_rng(984);records=[]
    for correlated in (False,True):
        k=rng.normal(size=(1,2,32,16))
        if correlated:k=k*.03+rng.normal(size=(1,2,1,16))
        k=k/np.linalg.norm(k,axis=-1,keepdims=True)
        arrays=(k,k,rng.normal(size=(1,2,32,32)),rng.uniform(.94,.99,k.shape),np.ones(k.shape[:-1])*.9)
        # Same BF16-representable inputs isolate subsequent arithmetic precision.
        values=[mx.array(x,dtype=mx.bfloat16) for x in arrays]
        ref,rs,rg=oracle(*[np.array(x.tolist(),dtype=np.float64) for x in values])
        for dtype in (mx.bfloat16,mx.float16):
            for chunk in (0,1,2,4,8):
                function=delta_scan if not chunk else lambda *x:delta_chunk(*x,chunk=chunk,compute_dtype=dtype)
                def objective(*x):
                    out,state=function(*x)
                    return mx.mean(out*out)+mx.mean(state*state)
                inputs=[x.astype(dtype) for x in values]
                out,state=function(*inputs)
                grad=mx.grad(objective,argnums=list(range(5)))
                gradients=(mx.compile(grad) if args.compiled else grad)(*inputs)
                def error(x,y):
                    delta=np.array(x.tolist(),dtype=np.float64)-y
                    return dict(relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(y),1e-30)),maximum_absolute=float(np.max(np.abs(delta))))
                records.append(dict(correlated=correlated,dtype=str(dtype),chunk=chunk,compiled=args.compiled,
                    output=error(out,ref),state=error(state,rs),
                    gradients={name:error(x,y) for name,x,y in zip(('q','k','v','alpha','beta'),gradients,rg)}))
    report=dict(hypothesis=__doc__,records=records,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        precision='BF16 and FP16 numerical controls; no FP32 model/gradient arrays',peak_mlx_bytes=mx.get_peak_memory())
    name='misul-memory-diagnosis-compiled' if args.compiled else 'misul-memory-diagnosis'
    Path(f'evidence/{name}.json').write_text(json.dumps(report,indent=2))
    for r in records:print(json.dumps(r),flush=True)


if __name__=='__main__':main()
