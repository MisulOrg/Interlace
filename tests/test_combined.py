import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map
from transformermodel.combined import fraction, delta_chunk, delta_scan, selected_attention, CombinedModel


def test_fraction_oracle_and_gradients():
    def oracle(a):
        # Matrix products form continuants independently of the backward loop.
        result = np.broadcast_to(np.eye(2), (*a.shape[:-1],2,2)).copy()
        for i in range(a.shape[-1]):
            mat = np.zeros_like(result)
            mat[...,0,0] = a[...,i]
            mat[...,0,1] = mat[...,1,0] = 1
            result = result@mat
        den = result[...,0,0]
        return result[...,1,0]/(np.where(den<0,-1,1)*np.maximum(np.abs(den),.01))
    rng = np.random.default_rng(901)
    for depth in (1,2,3,5):
        a = rng.uniform(.5,1.5,(3,4,depth))
        np.testing.assert_allclose(np.array(fraction(mx.array(a))),oracle(a),rtol=3e-6)
        gradient = np.array(mx.grad(lambda z:mx.sum(fraction(z)))(mx.array(a)))
        expected = np.zeros_like(a)
        for idx in np.ndindex(a.shape):
            plus=a.copy();minus=a.copy();plus[idx]+=1e-5;minus[idx]-=1e-5
            expected[idx]=(oracle(plus).sum()-oracle(minus).sum())/2e-5
        np.testing.assert_allclose(gradient,expected,rtol=2e-4,atol=2e-5)
    near = mx.array([[.001],[-.001],[0.],[.009],[-.011]])
    np.testing.assert_allclose(np.array(fraction(near)),[100,-100,100,100,-1/.011],rtol=2e-6)
    np.testing.assert_array_equal(np.array(mx.grad(lambda z:mx.sum(fraction(z)))(near))[:4],np.zeros((4,1)))


def test_kda_recurrence_chunk_state_and_gradients():
    rng=np.random.default_rng(902)
    q,k,v=[rng.normal(size=(2,2,19,d))*.2 for d in (3,3,4)]
    alpha=rng.uniform(.6,.99,q.shape);beta=rng.uniform(.1,.9,q.shape[:-1])
    initial=rng.normal(size=(2,2,3,4))*.1
    state=initial.copy();reference=[]
    for t in range(19):
        for b in range(2):
            for h in range(2):
                decay=np.diag(alpha[b,h,t])
                kk=k[b,h,t]
                state[b,h]=(np.eye(3)-beta[b,h,t]*np.outer(kk,kk))@decay@state[b,h]+beta[b,h,t]*np.outer(kk,v[b,h,t])
        reference.append(np.einsum('bhk,bhkv->bhv',q[:,:,t],state))
    values=[mx.array(z) for z in (q,k,v,alpha,beta,initial)]
    for chunk in (1,4,16):
        actual,final=delta_chunk(*values,chunk=chunk)
        np.testing.assert_allclose(np.array(actual),np.stack(reference,axis=-2),rtol=1e-4,atol=2e-6)
        np.testing.assert_allclose(np.array(final),state,rtol=1e-4,atol=2e-6)
    def loss(fn,*inputs):
        out,state=fn(*inputs)
        return mx.sum(out*out)+mx.sum(state*state)
    ref=mx.grad(lambda *a:loss(delta_scan,*a),argnums=list(range(6)))(*values)
    got=mx.grad(lambda *a:loss(delta_chunk,*a),argnums=list(range(6)))(*values)
    for a,b in zip(ref,got):np.testing.assert_allclose(np.array(a),np.array(b),rtol=2e-4,atol=3e-6)
    # State handoff across a boundary must preserve the recurrence exactly.
    left=[a[...,:7,:] for a in values[:4]]+[values[4][...,:7]]
    right=[a[...,7:,:] for a in values[:4]]+[values[4][...,7:]]
    o1,s1=delta_chunk(*left,state=values[5]);o2,s2=delta_chunk(*right,state=s1)
    np.testing.assert_allclose(np.array(mx.concatenate([o1,o2],axis=-2)),np.stack(reference,axis=-2),rtol=1e-4,atol=2e-6)


def test_routed_attention_dense_limit_and_future_isolation():
    mx.random.seed(903)
    for streams in (1,3):
        n=streams*19
        q,k,v=[mx.random.normal((2,2,n,8)) for _ in range(3)]
        row=mx.arange(n)//streams
        mask=mx.where(row[:,None]>=row[None,:],0.,-float('inf'))
        ref=mx.fast.scaled_dot_product_attention(q,k,v,scale=8**-.5,mask=mask)
        got=selected_attention(q,k,v,streams=streams,selected=10)
        np.testing.assert_allclose(np.array(got),np.array(ref),rtol=3e-5,atol=3e-6)
        sparse=selected_attention(q,k,v,streams=streams,selected=1)
        native=selected_attention(q,k,v,streams=streams,selected=1,backend='native')
        np.testing.assert_allclose(np.array(native),np.array(sparse),rtol=3e-5,atol=3e-6)
        kc=mx.array(k);vc=mx.array(v)
        kc[:,:,11*streams:]=100.;vc[:,:,11*streams:]=-100.
        changed=selected_attention(q,kc,vc,streams=streams,selected=1)
        np.testing.assert_array_equal(np.array(sparse[:,:,:11*streams]),np.array(changed[:,:,:11*streams]))


def test_correlated_memory_at_accelerated_matrix_shapes():
    # The earlier 3-channel oracle missed the accelerator path and cancellation.
    # Near-collinear normalized keys and beta=1 are legal, stable delta updates.
    rng=np.random.default_rng(915)
    k=rng.normal(size=(1,2,128,16))*.05+rng.normal(size=(1,2,1,16))
    k=k/np.sqrt(np.sum(k*k,axis=-1,keepdims=True)+1e-6)
    values=[mx.array(z) for z in (k,k,rng.normal(size=(1,2,128,32)),
        rng.uniform(.94,.99,k.shape),np.ones(k.shape[:-1]))]
    ref,state=delta_scan(*values);got,final=delta_chunk(*values)
    np.testing.assert_allclose(np.array(got),np.array(ref),rtol=2e-4,atol=2e-5)
    np.testing.assert_allclose(np.array(final),np.array(state),rtol=2e-4,atol=2e-5)


def test_combined_roles_gradients_causality_and_export(tmp_path):
    mx.set_memory_limit(768*2**20);mx.set_cache_limit(32*2**20)
    mx.random.seed(904)
    config=dict(vocab_size=32,width=32,layers=2,heads=2,ladders=8,context=24,loops=2,precision='bf16')
    m=CombinedModel(**config)
    rows=mx.array(np.random.default_rng(904).integers(0,32,(1,19,3)).astype(np.int32))
    initial=m(rows)
    changed=mx.array(rows);changed[:,11:]=(changed[:,11:]+7)%32
    np.testing.assert_array_equal(np.array(initial[:,:11]),np.array(m(changed)[:,:11]))
    altered=mx.array(rows);altered[:,3,2]=(altered[:,3,2]+7)%32
    assert float(mx.max(mx.abs(m(altered)[:,3,0]-initial[:,3,0])))>1e-6
    x,y=rows[:,:-1],rows[:,1:]
    def loss(model,x,y):return nn.losses.cross_entropy(model(x)[:,:,1:],y[:,:,1:],reduction='mean')
    value,g=nn.value_and_grad(m,loss)(m,x,y)
    assert np.isfinite(float(value))
    for name,gradient in tree_flatten(g):
        assert bool(mx.all(mx.isfinite(gradient))),name
        assert float(mx.sum(mx.abs(gradient)))>0,name
    path=tmp_path/'weights.safetensors'
    m.save_weights(str(path));reload=CombinedModel(**config);reload.load_weights(str(path))
    np.testing.assert_array_equal(np.array(initial),np.array(reload(rows)))
    reload.update(tree_map(lambda w:w.astype(mx.bfloat16),reload.parameters()))
    np.testing.assert_array_equal(np.array(initial),np.array(reload(rows)))
    assert m(rows[:,:,0]).shape==(1,19,32)


def test_native_routed_gradients_and_bf16():
    mx.random.seed(916)
    for dtype,tolerance in ((mx.float32,3e-5),(mx.bfloat16,.02)):
        args=[mx.random.normal((1,4,32,96)).astype(dtype) for _ in range(3)]
        def objective(backend,*args):
            out=selected_attention(*args,selected=1,backend=backend)
            return mx.mean(out.astype(mx.float32)**2)
        ref=mx.grad(lambda *a:objective('gather',*a),argnums=[0,1,2])(*args)
        got=mx.grad(lambda *a:objective('native',*a),argnums=[0,1,2])(*args)
        for a,b in zip(ref,got):
            error=float(mx.linalg.norm((a-b).astype(mx.float32))/mx.maximum(mx.linalg.norm(a.astype(mx.float32)),1e-12))
            assert error<tolerance,error
