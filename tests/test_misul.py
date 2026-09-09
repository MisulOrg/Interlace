import math
import numpy as np
import mlx.core as mx
from transformermodel.misul import rational_gate,memory16,MisulModel
from mlx.utils import tree_flatten
from experiments.check_misul_memory import oracle


def host(value):
    return np.array(value.tolist(),dtype=np.float64)


def test_rational_gate_independent_values_derivatives_and_extremes():
    rng=np.random.default_rng(983)
    a,b=rng.normal(size=(2,1024))*3
    # Include the zero, extrema, sign changes and the rescaling boundaries.
    a=np.r_[a,0.,math.sqrt(2),-math.sqrt(2),1.,-1.,0.,1e30,-1e30]
    b=np.r_[b,0.,1.,1.,1.,-1.,-1.,1e30,0.]
    am,bm=[mx.array(v,dtype=mx.bfloat16) for v in (a,b)]
    a,b=host(am),host(bm)
    denominator=1+a*a+b*b
    expected=a*(1+b)/denominator
    expected_da=(1+b)*(1+b*b-a*a)/denominator**2
    expected_db=a*(1+a*a-b*b-2*b)/denominator**2
    actual=rational_gate(am,bm)
    gradients=mx.grad(lambda aa,bb:mx.sum(rational_gate(aa,bb)),argnums=[0,1])(am,bm)
    assert actual.dtype==mx.bfloat16
    assert np.max(np.abs(host(actual)))<=1/math.sqrt(2)+.004
    assert np.all(np.isfinite(host(actual)))
    assert np.linalg.norm(host(actual)-expected)/np.linalg.norm(expected)<=.008
    for got,ref in zip(gradients,(expected_da,expected_db)):
        assert got.dtype==mx.bfloat16
        assert np.all(np.isfinite(host(got)))
        assert np.linalg.norm(host(got)-ref)/np.linalg.norm(ref)<=.025
    # A high-precision finite difference independently checks the written formulas.
    for aa,bb in ((0.,0.),(1.,1.),(-1.,.5),(2.,-1.),(.25,2.)):
        def f(x,y):return x*(1+y)/(1+x*x+y*y)
        d=1+aa*aa+bb*bb
        oracle=np.array([(1+bb)*(1+bb*bb-aa*aa)/d**2,aa*(1+aa*aa-bb*bb-2*bb)/d**2])
        finite=np.array([(f(aa+1e-6,bb)-f(aa-1e-6,bb))/2e-6,(f(aa,bb+1e-6)-f(aa,bb-1e-6))/2e-6])
        np.testing.assert_allclose(finite,oracle,rtol=1e-7,atol=1e-10)


def test_memory16_against_independent_values_and_reverse_derivatives():
    rng=np.random.default_rng(984)
    for correlated in (False,True):
        for length in (32,128):
            k=rng.normal(size=(1,2,length,16))
            if correlated:k=k*.03+rng.normal(size=(1,2,1,16))
            k=k/np.linalg.norm(k,axis=-1,keepdims=True)
            values=[mx.array(z,dtype=mx.bfloat16) for z in (k,k,
                rng.normal(size=(1,2,length,32)),rng.uniform(.94,.99,k.shape),
                np.ones(k.shape[:-1])*.9)]
            ref,state,gradients=oracle(*map(host,values))
            actual,final=memory16(*values)
            for got,want in ((actual,ref),(final,state)):
                assert got.dtype==mx.float16
                error=np.linalg.norm(host(got)-want)/np.linalg.norm(want)
                assert error<=.02,(correlated,length,error)
            def objective(*values):
                output,state=memory16(*values)
                return (mx.mean(output*output)+mx.mean(state*state))*128
            got=mx.grad(objective,argnums=list(range(5)))(*values)
            for name,actual,expected in zip(('q','k','v','alpha','beta'),got,gradients):
                error=np.linalg.norm(host(actual)/128-expected)/np.linalg.norm(expected)
                assert error<=.02,(correlated,length,name,error)


def test_fused_memory_against_independent_values_and_reverse_derivatives():
    from transformermodel.misul_memory_metal import memory_metal
    rng=np.random.default_rng(1015)
    for correlated in (False,True):
        for length,width in ((1,32),(6,80),(32,32),(128,80)):
            k=rng.normal(size=(1,2,length,16))
            if correlated:k=k*.03+rng.normal(size=(1,2,1,16))
            k=k/np.linalg.norm(k,axis=-1,keepdims=True)
            values=[mx.array(z,dtype=mx.bfloat16) for z in (k,k,
                rng.normal(size=(1,2,length,width)),rng.uniform(.94,.99,k.shape),
                np.ones(k.shape[:-1])*.9)]
            ref,state,gradients=oracle(*map(host,values))
            actual,final=memory_metal(*values)
            for got,want in ((actual,ref),(final,state)):
                assert got.dtype==mx.float16
                error=np.linalg.norm(host(got)-want)/np.linalg.norm(want)
                assert error<=.02,(correlated,length,'output/state',error)
            def objective(*values):
                output,state=memory_metal(*values)
                return (mx.mean(output*output)+mx.mean(state*state))*128
            got=mx.grad(objective,argnums=list(range(5)))(*values)
            for name,actual,expected in zip(('q','k','v','alpha','beta'),got,gradients):
                denominator=np.linalg.norm(expected)
                error=np.linalg.norm(host(actual)/128-expected)/max(denominator,1e-12)
                assert error<=.02,(correlated,length,name,error)


def test_fp8_backbone_causality_gradient_dtypes_and_packed_reload(tmp_path):
    import mlx.nn as nn
    mx.set_memory_limit(512*2**20);mx.set_cache_limit(16*2**20)
    mx.random.seed(985)
    m=MisulModel(vocab_size=64,width=64,hidden=96,layers=2,heads=2,context=20,memory_backend='metal')
    x=mx.array(np.random.default_rng(985).integers(0,64,(1,12,3)),dtype=mx.int32)
    baseline=m(x)
    changed=mx.array(x);changed[:,7:]=(changed[:,7:]+3)%64
    assert mx.array_equal(baseline[:,:7],m(changed)[:,:7]).item()
    value,gradient=nn.value_and_grad(m,lambda model:nn.losses.cross_entropy(model(x),x,reduction='mean')*128)(m)
    assert math.isfinite(value.item())
    for name,g in tree_flatten(gradient):
        assert g.dtype==mx.bfloat16,name
        assert mx.all(mx.isfinite(g)).item(),name
    assert all(v.dtype==mx.bfloat16 for _,v in tree_flatten(m.parameters()))
    storage=m.export(tmp_path/'release')
    arrays=mx.load(str(tmp_path/'release/weights.safetensors'))
    assert all(v.dtype in (mx.uint32,mx.uint8) for v in arrays.values())
    assert storage['tensor_bytes']<1.05*storage['parameters']
    reload=MisulModel.load(tmp_path/'release')
    assert mx.array_equal(baseline,reload(x)).item()
    clues=x[:,:,0];noisy=mx.random.normal((1,12,10),dtype=mx.bfloat16)
    feedback=mx.zeros_like(noisy);time=mx.array([.25],dtype=mx.bfloat16)
    flow=m.flow(clues,noisy,time,feedback)
    assert flow.shape==(1,12,10) and flow.dtype==mx.bfloat16
    assert mx.array_equal(flow,reload.flow(clues,noisy,time,feedback)).item()
    from transformermodel.misul_flow import flow_loss
    targets=mx.zeros(clues.shape,dtype=mx.int32);mask=mx.ones(clues.shape,dtype=mx.bool_)
    loss,g=nn.value_and_grad(m,lambda model:flow_loss(model,clues,targets,mask,time,noisy,True)*128)(m)
    assert loss.dtype in (mx.float16,mx.bfloat16)
    for name,value in tree_flatten(g):
        assert value.dtype==mx.bfloat16 and mx.all(mx.isfinite(value)).item(),name


def test_fp8_small_row_forward_and_backward_keep_other_storage_intact():
    from transformermodel.misul import fp8_matmul,packed_value,unpacked_value
    mx.random.seed(1013)
    for rows,outputs in ((1,4),(6,10),(33,32)):
        x=mx.random.normal((rows,64),dtype=mx.bfloat16)
        w=mx.random.normal((outputs,64),dtype=mx.bfloat16)*.125
        sentinel=mx.ones((128,),dtype=mx.bfloat16);mx.eval(x,w,sentinel)
        def quantized(z):return unpacked_value(*packed_value(z),z.shape)
        expected=host(quantized(x))@host(quantized(w)).T
        actual=host(fp8_matmul(x,w))
        assert np.linalg.norm(actual-expected)/np.linalg.norm(expected)<.015
        gradient=mx.grad(lambda a,b:mx.mean(fp8_matmul(a,b)**2),argnums=[0,1])
        for _ in range(16):
            dx,dw=gradient(x,w);w=w-.001*dw;mx.eval(dx,dw,w)
            assert mx.all(mx.isfinite(dx)).item() and mx.all(mx.isfinite(dw)).item()
            assert mx.array_equal(sentinel,mx.ones_like(sentinel)).item()


def test_muon16_zero_rectangular_direction_and_small_update_retention():
    import mlx.nn as nn
    from transformermodel.misul_muon import Muon16,polar16
    for shape in ((4,8),(8,4),(4,4)):
        zero=polar16(mx.zeros(shape,dtype=mx.bfloat16))
        assert mx.all(zero==0).item() and zero.dtype==mx.bfloat16
        rank_one=polar16(mx.ones(shape,dtype=mx.bfloat16))
        assert mx.all(mx.isfinite(rank_one)).item()
        assert np.max(host(rank_one))-np.min(host(rank_one))<.001
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks=[{'weight':mx.zeros((4,8),dtype=mx.bfloat16)}]
            self.embedding=mx.ones((4,),dtype=mx.bfloat16)
    model=Tiny();optimizer=Muon16(model.parameters(),aux_learning_rate=.0001,weight_decay=0)
    gradient={'blocks':[{'weight':mx.ones((4,8),dtype=mx.bfloat16)}],
              'embedding':mx.ones((4,),dtype=mx.bfloat16)}
    for _ in range(64):optimizer.update(model,gradient,mx.array(1.,dtype=mx.bfloat16))
    np.testing.assert_allclose(host(model.blocks[0]['weight']),-64*.006*.2,rtol=.01,atol=.001)
    np.testing.assert_allclose(host(model.embedding),1-64*.0001,rtol=0,atol=.002)
    assert float(mx.min(1-model.embedding))>.005
    for name,value in tree_flatten(optimizer.state):
        assert value.dtype in (mx.int32,mx.bfloat16),name


def test_flow_revision_clues_feedback_and_wrong_stable_counterexample():
    from transformermodel.misul_flow import sample_flow,stability_score
    class Revises:
        def __init__(self):self.calls=[]
        def flow(self,clues,noisy,time,feedback):
            self.calls.append((clues.tolist(),feedback.tolist()))
            digit=mx.where(time[:,None,None]<.5,0,1)
            return mx.where(mx.arange(10)[None,None,:]==digit,10.,-10.).astype(mx.bfloat16)+mx.zeros_like(noisy)
    model=Revises();clues=mx.array([[3,7]],dtype=mx.int32)
    _,_,trace=sample_flow(model,clues,steps=4)
    assert trace[0]['candidate']==[[0,0]] and trace[-1]['candidate']==[[1,1]]
    assert all(call[0]==[[3,7]] for call in model.calls)
    assert model.calls[0][1]!=model.calls[1][1]
    class WrongFixedPoint:
        def flow(self,clues,noisy,time,feedback):
            return mx.where(mx.arange(10)[None,None,:]==0,10.,-10.).astype(mx.bfloat16)+mx.zeros_like(noisy)
    wrong=WrongFixedPoint()
    candidate,_,_=sample_flow(wrong,clues,steps=8)
    score=stability_score(wrong,clues,candidate,mx.ones(clues.shape,dtype=mx.bool_))
    # The true prefix of[3,7] starts at3. A stable zero prediction is still wrong.
    assert candidate.tolist()==[[0,0]] and float(score.item())<.001
