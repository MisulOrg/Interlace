"""Combined causal backbone. Primary contracts: docs/.../first-combined-model.md.

CoFrGeNet collapsed FFN; Monodratic-style completed-block selection; chunked
KDA recurrence. This is a local adaptation, not any paper's full training recipe.
"""
import os
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map
from .model import DenseFFN


def fraction(a, epsilon=.01):
    """Fractional part with one signed continuant floor, sign(0) defined as +1.

    The floor has zero derivative inside (-epsilon,epsilon). Its discontinuity
    at zero is intentional. No STE or unstated per-level denominator change.
    """
    a = a.astype(mx.float32)
    numerator = mx.ones_like(a[..., 0])
    denominator = a[..., -1]
    for i in range(a.shape[-1]-2, -1, -1):
        numerator, denominator = denominator, a[..., i]*denominator+numerator
    safe = mx.where(denominator < 0, -1., 1.)*mx.maximum(mx.abs(denominator), epsilon)
    return numerator/safe


class ContinuedFFN(nn.Module):
    def __init__(self, width, ladders, depth=2):
        super().__init__()
        self.depth = depth
        self.ladders = ladders
        self.gate = nn.Linear(width, width)
        self.linear = nn.Linear(width, width)
        self.denominators = nn.Linear(width, ladders*depth)
        self.denominators.weight = self.denominators.weight*.1
        self.denominators.bias = mx.ones((ladders*depth,))
        self.mix = nn.Linear(ladders, width, bias=False)
        self.mix.weight = self.mix.weight*.1

    def __call__(self, x):
        z = x*nn.silu(self.gate(x))
        a = self.denominators(z).reshape(*z.shape[:-1], self.ladders, self.depth)
        return self.linear(z)+self.mix(fraction(a).astype(x.dtype))


def delta_scan(q, k, v, alpha, beta, state=None):
    """Ordinary recurrence, independent of the chunk implementation below."""
    if state is None:
        state = mx.zeros((*q.shape[:2], q.shape[-1], v.shape[-1]),dtype=q.dtype)
    outputs = []
    for t in range(q.shape[-2]):
        state = alpha[..., t, :, None]*state
        error = v[..., t, :]-mx.sum(k[..., t, :, None]*state, axis=-2)
        state = state+beta[..., t, None, None]*k[..., t, :, None]*error[..., None, :]
        outputs.append(mx.sum(q[..., t, :, None]*state, axis=-2))
    return mx.stack(outputs, axis=-2), state


def unit_lower_inverse(lower):
    """Inverse of I+strict_lower by block substitution, without a power series."""
    n=lower.shape[-1]
    if n==1:return mx.ones_like(lower)
    half=n//2
    a,d=lower[...,:half,:half],lower[...,half:,half:]
    if n%2==0:
        pair=unit_lower_inverse(mx.stack([a,d]))
        ai,di=pair[0],pair[1]
    else:
        ai,di=unit_lower_inverse(a),unit_lower_inverse(d)
    cross=-di@lower[...,half:,:half]@ai
    top=mx.concatenate([ai,mx.zeros((*lower.shape[:-2],half,n-half),dtype=lower.dtype)],axis=-1)
    return mx.concatenate([top,mx.concatenate([cross,di],axis=-1)],axis=-2)


def delta_chunk(q, k, v, alpha, beta, state=None, chunk=16, compute_dtype=mx.float32):
    """Exact KDA chunk algebra; default FP32, explicitly selected dtype otherwise.

    Invert I+strict_lower by recursive block substitution. A nilpotent power
    series lost accuracy for correlated keys; preserve that witness in evidence.
    No generic inverse, token loop or detached chunk state.
    """
    q, k, v, alpha, beta = [z.astype(compute_dtype) for z in (q,k,v,alpha,beta)]
    if state is None:
        state = mx.zeros((*q.shape[:2], q.shape[-1], v.shape[-1]),dtype=compute_dtype)
    outputs = []
    for start in range(0, q.shape[-2], chunk):
        qc,kc,vc,ac,bc = [z[..., start:start+chunk, :] for z in (q,k,v,alpha,beta[...,None])]
        n = qc.shape[-2]
        g = mx.exp(mx.cumsum(mx.log(ac), axis=-2))
        kg = kc*g
        ki = kc/g
        lower = mx.tril(bc*(kg@mx.swapaxes(ki,-1,-2)), k=-1)
        inverse = unit_lower_inverse(lower)
        transform = inverse*mx.swapaxes(bc,-1,-2)
        u = transform@vc
        w = transform@kg
        residual = u-w@state
        qg = qc*g
        outputs.append(qg@state+mx.tril(qg@mx.swapaxes(ki,-1,-2))@residual)
        state = g[..., -1, :, None]*state+mx.swapaxes(ki*g[..., -1:, :],-1,-2)@residual
    return mx.concatenate(outputs,axis=-2), state


class DeltaMemory(nn.Module):
    def __init__(self, width, heads, key_dim=16):
        super().__init__()
        if os.environ.get('MLX_ENABLE_TF32')!='0':
            raise ValueError('KDA recurrence requires MLX_ENABLE_TF32=0 on this qualified MLX build')
        self.heads = heads
        self.key_dim = key_dim
        self.value_dim = width//heads
        self.q = nn.Linear(width, heads*key_dim, bias=False)
        self.k = nn.Linear(width, heads*key_dim, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.decay = nn.Linear(width, heads*key_dim)
        self.beta = nn.Linear(width, heads)
        self.gate = nn.Linear(width, width)
        self.output = nn.Linear(width, width, bias=False)
        self.norm = nn.RMSNorm(self.value_dim)

    def __call__(self, x):
        # Synchronous rows: all three tokens are completed history. Pool writes;
        # role-specific residual/gate reads remain separate and causal.
        source = mx.mean(x, axis=2)
        b,t,_ = source.shape
        def heads(z, dim):
            return z.reshape(b,t,self.heads,dim).transpose(0,2,1,3).astype(mx.float32)
        q = heads(self.q(source), self.key_dim)
        k = heads(self.k(source), self.key_dim)
        q = q*mx.rsqrt(mx.sum(q*q,axis=-1,keepdims=True)+1e-6)
        k = k*mx.rsqrt(mx.sum(k*k,axis=-1,keepdims=True)+1e-6)
        v = heads(self.v(source), self.value_dim)
        # Bounded log decay avoids underflow within a chunk. This is an explicit
        # local gate parameterization, not the released Kimi gate parameterization.
        a = mx.exp(-.05*nn.softplus(mx.clip(heads(self.decay(source),self.key_dim),-10,10)))
        beta = mx.sigmoid(self.beta(source).transpose(0,2,1).astype(mx.float32))
        o,_ = delta_chunk(q,k,v,a,beta)
        o = self.norm(o.astype(x.dtype)).transpose(0,2,1,3).reshape(b,t,1,-1)
        return self.output(o*nn.silu(self.gate(x)))


def selected_attention(q, k, v, streams=1, block_rows=8, selected=2, backend='gather', compute_dtype=mx.float32):
    """Content-route completed prior blocks, then exact attention on gathered K/V.

    Selection has no surrogate gradient. Current partial block is always present.
    Routing summaries from the current block cannot affect any selection.
    """
    n = q.shape[-2]
    block = block_rows*streams
    count = (n+block-1)//block
    pad = count*block-n
    kp = mx.pad(k, [(0,0),(0,0),(0,pad),(0,0)])
    vp = mx.pad(v, [(0,0),(0,0),(0,pad),(0,0)])
    means = mx.mean(kp.reshape(*k.shape[:2],count,block,k.shape[-1]),axis=-2)
    position = mx.arange(n)
    current = position//block
    score = q.astype(compute_dtype)@mx.swapaxes(means.astype(compute_dtype),-1,-2)
    score = mx.where(mx.arange(count)[None,:]<current[:,None], score, -float('inf'))
    take = min(selected,count)
    chosen = mx.argsort(mx.stop_gradient(score),axis=-1)[...,-take:] if take else mx.zeros((*q.shape[:3],0),mx.int32)
    if backend=='native':
        key_blocks=position//block
        previous=mx.any(chosen[...,None]==key_blocks,axis=-2)&(key_blocks<current[:,None])
        validity=(previous|(key_blocks==current[:,None]))&(position[None,:]//streams<=position[:,None]//streams)
        mask=mx.where(validity,0.,-float('inf')).astype(q.dtype)
        # Native masked SDPA is a bounded quadratic reference for short contexts.
        # It preserves selected-token semantics without materializing 5D K/V.
        return mx.fast.scaled_dot_product_attention(q,k,v,scale=q.shape[-1]**-.5,mask=mask)
    if backend!='gather':raise ValueError('unknown attention backend')
    local = mx.broadcast_to(current,(*q.shape[:2],n))[...,None]
    blocks = mx.concatenate([chosen,local],axis=-1)
    ids = (blocks[...,None]*block+mx.arange(block)).reshape(*blocks.shape[:-1],-1)
    keys = mx.take_along_axis(kp[:,:,None,:,:],ids[...,None],axis=-2)
    values = mx.take_along_axis(vp[:,:,None,:,:],ids[...,None],axis=-2)
    prior_valid = mx.repeat(chosen<current[:,None],block,axis=-1)
    validity = mx.concatenate([prior_valid,mx.ones((*local.shape[:-1],block),mx.bool_)],axis=-1)
    validity = validity & (ids<n) & (ids//streams <= position[:,None]//streams)
    logits = mx.sum(q[...,None,:]*keys,axis=-1)*q.shape[-1]**-.5
    probabilities = mx.softmax(mx.where(validity,logits.astype(compute_dtype),-float('inf')),axis=-1).astype(v.dtype)
    return mx.sum(probabilities[...,None]*values,axis=-2)


class RoutedAttention(nn.Module):
    def __init__(self,width,heads,selected=2,backend='gather'):
        super().__init__()
        self.heads = heads
        self.selected = selected
        self.backend = backend
        self.qkv = nn.Linear(width,width*3,bias=False)
        self.output = nn.Linear(width,width,bias=False)

    def __call__(self,x):
        b,t,r,w = x.shape
        q,k,v = mx.split(self.qkv(x).reshape(b,t*r,3,self.heads,w//self.heads).transpose(2,0,3,1,4),3,axis=0)
        o = selected_attention(q[0],k[0],v[0],streams=r,selected=self.selected,backend=self.backend)
        return self.output(o.transpose(0,2,1,3).reshape(b,t,r,w))


class CombinedBlock(nn.Module):
    def __init__(self,width,heads,ladders,mixer,ffn,attention_backend='gather'):
        super().__init__()
        self.norm1 = nn.RMSNorm(width)
        self.norm2 = nn.RMSNorm(width)
        self.kind = mixer
        self.mixer = DeltaMemory(width,heads) if mixer=='delta' else RoutedAttention(width,heads,backend=attention_backend) if mixer=='routed' else nn.MultiHeadAttention(width,heads)
        self.ffn = ContinuedFFN(width,ladders) if ffn=='continued' else DenseFFN(width)

    def __call__(self,x):
        z = self.norm1(x)
        if self.kind == 'dense':
            b,t,r,w = x.shape
            z = z.reshape(b,t*r,w)
            row = mx.arange(t*r)//r
            mask = mx.where(row[:,None]>=row[None,:],0.,-float('inf')).astype(z.dtype)
            z = self.mixer(z,z,z,mask=mask).reshape(b,t,r,w)
        else:
            z = self.mixer(z)
        x = x+z
        return x+self.ffn(self.norm2(x))


class CombinedModel(nn.Module):
    def __init__(self,vocab_size=2048,width=256,layers=4,heads=4,ladders=64,
                 context=128,loops=2,architecture='combined',precision='bf16',attention_backend='gather'):
        super().__init__()
        if architecture not in ('combined','dense','continued_dense','routed','delta') or precision not in ('fp32','bf16'):
            raise ValueError('unsupported architecture or precision')
        if attention_backend not in ('gather','native'):raise ValueError('unknown attention backend')
        if not 1<=loops<=4 or width%heads or not 1<=context<=512 or not 1<=layers<=8 or not 16<=width<=512 or not 1<=ladders<=width or not 16<=vocab_size<=8192:
            raise ValueError('invalid or unbounded model configuration')
        self.precision = precision
        self.loops = loops
        self.embedding = nn.Embedding(vocab_size,width)
        self.position = mx.random.normal((context,width))*.01
        self.stream_embedding = mx.random.normal((3,width))*.01
        self.loop_embedding = mx.random.normal((loops,width))*.01
        self.layers = [CombinedBlock(width,heads,ladders,
            ('routed' if i%2==0 else 'delta') if architecture=='combined' else architecture if architecture in ('routed','delta') else 'dense',
            'dense' if architecture=='dense' else 'continued',attention_backend) for i in range(layers)]
        self.norm = nn.RMSNorm(width)

    def features(self,tokens):
        text = tokens.ndim==2
        if text:tokens=tokens[:,:,None]
        if tokens.ndim!=3 or tokens.shape[2] not in (1,3):
            raise ValueError('expected text [B,T] or streams [B,T,3]')
        b,t,r = tokens.shape
        if not 1<=t<=self.position.shape[0] or b*t*r>4096:
            raise ValueError('context or activation allocation bound exceeded')
        role = self.stream_embedding[2:3] if r==1 else self.stream_embedding
        h = self.embedding(tokens)+self.position[None,:t,None,:]+role[None,None,:,:]
        for i in range(self.loops):
            h = h+self.loop_embedding[i]
            for layer in self.layers:h=layer(h)
        h = self.norm(h)
        return h[:,:,0] if text else h

    def __call__(self,tokens):
        master = self.parameters()
        if self.precision=='bf16':self.update(tree_map(lambda w:w.astype(mx.bfloat16),master))
        try:
            logits = self.features(tokens)@self.embedding.weight.T
        finally:
            self.update(master)
        return logits.astype(mx.float32)

    def answer(self,tokens):
        return self(tokens)[:,-1]
