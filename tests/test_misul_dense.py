import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from transformermodel.misul_dense import DenseModel, SwiGLU, DenseAttention


def host(x):
    return np.array(x.tolist(), dtype=np.float64)


def test_swiglu_independent_value_and_input_gradient():
    mx.random.seed(1101)
    f = SwiGLU(32,32,'bf16')
    x = mx.random.normal((2,32), dtype=mx.bfloat16) * .1
    X,U,D = map(host,(x,f.up.weight,f.down.weight))
    a,b = np.split(X @ U.T,2,axis=-1)
    sigmoid = 1/(1+np.exp(-a))
    ref = (a*sigmoid*b) @ D.T
    downstream = np.ones_like(ref) @ D
    dz = np.concatenate([downstream*b*sigmoid*(1+a*(1-sigmoid)),
                         downstream*a*sigmoid],axis=-1)
    grad = mx.grad(lambda z:mx.sum(f(z)))(x)
    assert np.linalg.norm(host(f(x))-ref)/np.linalg.norm(ref)<.025
    expected = dz @ U
    assert np.linalg.norm(host(grad)-expected)/np.linalg.norm(expected)<.025


def test_dense_modes_causality_and_attention_oracle():
    mx.random.seed(1102)
    m=DenseModel(vocab_size=64,width=64,hidden=96,layers=2,heads=2,context=16,precision='bf16',loops=1)
    x=mx.array(np.random.default_rng(1102).integers(0,64,(1,8,3)))
    changed=mx.array(x);changed[:,5:]=(changed[:,5:]+1)%64
    assert mx.array_equal(m(x)[:,:5],m(changed)[:,:5]).item()
    assert m(x[:,:,0]).shape==(1,8,64)
    noise=mx.zeros((1,8,10),dtype=mx.bfloat16)
    assert m.flow(x[:,:,0],noise,mx.array([0.],dtype=mx.bfloat16),noise).shape==(1,8,10)
    att=DenseAttention(32,2,'bf16')
    z=mx.random.normal((1,4,3,32),dtype=mx.bfloat16)*.1
    projected=host(att.qkv(z)).reshape(1,12,3,2,16).transpose(2,0,3,1,4)
    q,k,v=projected
    scores=q @ k.swapaxes(-1,-2)/4
    row=np.arange(12)//3
    scores=np.where(row[:,None]>=row[None,:],scores,-np.inf)
    weights=np.exp(scores-scores.max(axis=-1,keepdims=True));weights/=weights.sum(axis=-1,keepdims=True)
    expected=(weights@v).transpose(0,2,1,3).reshape(1,4,3,32) @ host(att.output.weight).T
    assert np.linalg.norm(host(att(z))-expected)/np.linalg.norm(expected)<.025
    assert len(m.blocks)==2 and m.loops==1
    assert all('decay' not in k and 'beta' not in k for k,v in tree_flatten(m.parameters()))
