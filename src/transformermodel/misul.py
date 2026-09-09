"""Misul model mechanisms. Contracts: 2026-09-09-misul-fp8-flow.md.

The paired rational gate takes a positive-quadratic route rather than a
continued-fraction ladder. Rational activations and pairwise rational features
have prior art; this module defines a local parameterization, not novelty proof.
"""
import hashlib
import json
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten,tree_unflatten
from .combined import delta_scan,selected_attention


def rational_gate(a,b):
    """a(1+b)/(1+a^2+b^2), evaluated without squaring unbounded inputs.

    For any positive s the rescaled expression is algebraically identical.
    Detaching s therefore preserves the derivative of the original function,
    including at the max operation's branch boundaries. The chosen scale makes
    its denominator at least one and the output at most 1/sqrt(2) in magnitude.
    """
    s=mx.stop_gradient(mx.maximum(1.,mx.maximum(mx.abs(a),mx.abs(b))))
    p,q,t=a/s,b/s,1./s
    return p*(t+q)/(t*t+p*p+q*q)


def packed_value(value):
    """Quantize every logical value, counting zero padding and group scales."""
    shape=value.shape
    rows=value.reshape(-1,shape[-1])
    padding=(-shape[-1])%32
    if padding:rows=mx.pad(rows,[(0,0),(0,padding)])
    packed,scales=mx.quantize(rows,mode='mxfp8')
    return packed,scales


def unpacked_value(packed,scales,shape):
    rows=mx.dequantize(packed,scales,mode='mxfp8',dtype=mx.bfloat16)
    return rows[:,:shape[-1]].reshape(shape)


def fp8_value(value):
    """FP8 forward with an explicit identity surrogate for non-matrix weights."""
    packed,scales=packed_value(value)
    quantized=unpacked_value(packed,scales,value.shape)
    return mx.stop_gradient(quantized)+(value-mx.stop_gradient(value))


def weight(shape,scale):
    return mx.random.normal(shape,dtype=mx.bfloat16)*scale


def fp8_matmul(x,w):
    """Keep all native FP8 VJP contraction dimensions block-aligned.

    The unaligned joint-backward witness produced an unrelated norm tensor
    containing packed FP8 bit patterns. Zero-padding M and N also keeps the
    transposed backward contractions aligned; slicing preserves logical shape.
    Padding is temporary and does not add learned or retained model state.
    """
    shape=x.shape;rows=x.reshape(-1,shape[-1]);count=rows.shape[0];outputs=w.shape[0]
    if count%32:rows=mx.pad(rows,[(0,(-count)%32),(0,0)])
    if outputs%32:w=mx.pad(w,[(0,(-outputs)%32),(0,0)])
    return mx.qqmm(rows,w,mode='mxfp8')[:count,:outputs].reshape(*shape[:-1],outputs)


class FP8Linear(nn.Module):
    def __init__(self,inputs,outputs,precision='mxfp8',zero=False):
        super().__init__()
        if inputs%32:raise ValueError('FP8 matrix input dimension must be a multiple of32')
        self.weight=mx.zeros((outputs,inputs),dtype=mx.bfloat16) if zero else weight((outputs,inputs),inputs**-.5)
        self.precision=precision

    def __call__(self,x):
        return fp8_matmul(x,self.weight) if self.precision=='mxfp8' else x@self.weight.T


class FP8Norm(nn.Module):
    def __init__(self,width,precision):
        super().__init__();self.weight=mx.ones((width,),dtype=mx.bfloat16);self.precision=precision

    def __call__(self,x):
        w=fp8_value(self.weight) if self.precision=='mxfp8' else self.weight
        return mx.fast.rms_norm(x,w,1e-5)


class RationalFFN(nn.Module):
    def __init__(self,width,hidden,precision):
        super().__init__()
        self.up=FP8Linear(width,2*hidden,precision)
        self.down=FP8Linear(hidden,width,precision)

    def __call__(self,x):
        a,b=mx.split(self.up(x),2,axis=-1)
        return self.down(rational_gate(a,b))


def memory16(q,k,v,alpha,beta):
    """Sequential KDA with FP16 working state; BF16 interfaces, no FP32 arrays.

    The almost-collinear-key witness rejects BF16 and low-precision chunk
    gradients. This ordinary recurrence keeps its qualified numerical contract.
    """
    return delta_scan(*[x.astype(mx.float16) for x in (q,k,v,alpha,beta)])


class MisulMemory(nn.Module):
    def __init__(self,width,heads,precision,memory_backend):
        super().__init__();self.heads=heads;self.key_dim=16;self.value_dim=width//heads
        self.memory_backend=memory_backend
        self.q=FP8Linear(width,heads*16,precision)
        self.k=FP8Linear(width,heads*16,precision)
        self.v=FP8Linear(width,width,precision)
        self.decay=FP8Linear(width,heads*16,precision)
        self.beta=FP8Linear(width,heads,precision)
        self.gate=FP8Linear(width,width,precision)
        self.output=FP8Linear(width,width,precision)
        self.norm=FP8Norm(self.value_dim,precision)

    def __call__(self,x):
        source=mx.mean(x,axis=2);b,t,_=source.shape
        def heads(z,d):return z.reshape(b,t,self.heads,d).transpose(0,2,1,3)
        q,k=[heads(layer(source),16) for layer in (self.q,self.k)]
        q=q*mx.rsqrt(mx.sum(q*q,axis=-1,keepdims=True)+1e-5)
        k=k*mx.rsqrt(mx.sum(k*k,axis=-1,keepdims=True)+1e-5)
        v=heads(self.v(source),self.value_dim)
        alpha=mx.exp(-.05*nn.softplus(mx.clip(heads(self.decay(source),16),-10,10)))
        beta=mx.sigmoid(self.beta(source).transpose(0,2,1))
        if self.memory_backend=='metal':
            from .misul_memory_metal import memory_metal
            out,_=memory_metal(q,k,v,alpha,beta)
        else:out,_=memory16(q,k,v,alpha,beta)
        out=self.norm(out.astype(mx.bfloat16)).transpose(0,2,1,3).reshape(b,t,1,-1)
        return self.output(out*nn.silu(self.gate(x)))


class MisulAttention(nn.Module):
    def __init__(self,width,heads,precision):
        super().__init__();self.heads=heads
        self.qkv=FP8Linear(width,width*3,precision)
        self.output=FP8Linear(width,width,precision)

    def __call__(self,x,causal=True):
        b,t,r,w=x.shape
        q,k,v=mx.split(self.qkv(x).reshape(b,t*r,3,self.heads,w//self.heads).transpose(2,0,3,1,4),3,axis=0)
        if causal:
            out=selected_attention(q[0],k[0],v[0],streams=r,backend='native',compute_dtype=mx.bfloat16)
        else:
            # Flow's current candidate is known in full; bidirectional reads are legal.
            out=mx.fast.scaled_dot_product_attention(q[0],k[0],v[0],scale=(w//self.heads)**-.5)
        return self.output(out.transpose(0,2,1,3).reshape(b,t,r,w))


class MisulBlock(nn.Module):
    def __init__(self,width,hidden,heads,index,precision,memory_backend):
        super().__init__();self.is_attention=index%2==0
        self.norm1=FP8Norm(width,precision);self.norm2=FP8Norm(width,precision)
        self.mixer=MisulAttention(width,heads,precision) if self.is_attention else MisulMemory(width,heads,precision,memory_backend)
        self.ffn=RationalFFN(width,hidden,precision)

    def __call__(self,x,causal=True):
        z=self.norm1(x)
        x=x+(self.mixer(z,causal) if self.is_attention else self.mixer(z))
        return x+self.ffn(self.norm2(x))


class MisulModel(nn.Module):
    def __init__(self,vocab_size=4096,width=640,hidden=1536,layers=6,heads=8,
                 context=128,loops=2,precision='mxfp8',memory_backend='scan'):
        super().__init__()
        if precision not in ('mxfp8','bf16'):raise ValueError('explicit BF16 control or native MXFP8 required')
        if memory_backend not in ('scan','metal'):raise ValueError('unknown memory backend')
        if not 32<=width<=640 or width%32 or width%heads or hidden%32 or not 32<=hidden<=1536:
            raise ValueError('invalid bounded channel configuration')
        if not 32<=vocab_size<=8192 or not 1<=layers<=6 or not 1<=loops<=2 or not 1<=context<=256:
            raise ValueError('invalid bounded model configuration')
        self.config=dict(vocab_size=vocab_size,width=width,hidden=hidden,layers=layers,
            heads=heads,context=context,loops=loops,precision=precision,memory_backend=memory_backend)
        self.precision=precision;self.loops=loops
        self.embedding=weight((vocab_size,width),width**-.5)
        self.position=weight((context,width),.01)
        self.roles=weight((3,width),.05)
        self.mode_embedding=weight((3,width),.05)
        self.loop_embedding=weight((loops,width),.01)
        self.blocks=[MisulBlock(width,hidden,heads,i,precision,memory_backend) for i in range(layers)]
        self.norm=FP8Norm(width,precision)
        # Flow operates on ten digit categories, padded to32 for native FP8 ops.
        self.flow_state=FP8Linear(32,width,precision)
        self.flow_feedback=FP8Linear(32,width,precision,zero=True)
        self.flow_time=weight((2,width),.01)
        self.flow_output=FP8Linear(width,10,precision)

    def effective(self,value):
        return fp8_value(value) if self.precision=='mxfp8' else value

    def backbone(self,h,causal=True):
        _,t,_,_=h.shape
        h=h+self.effective(self.position)[None,:t,None,:]
        for i in range(self.loops):
            h=h+self.effective(self.loop_embedding)[i]
            for block in self.blocks:h=block(h,causal)
        return self.norm(h)

    def __call__(self,tokens):
        text=tokens.ndim==2
        if text:tokens=tokens[:,:,None]
        if tokens.ndim!=3 or tokens.shape[2] not in (1,3):raise ValueError('expected text or three synchronous streams')
        b,t,r=tokens.shape
        if not 1<=t<=self.position.shape[0] or b*t*r>2048:raise ValueError('activation allocation bound exceeded')
        embedding=self.effective(self.embedding)
        role=self.effective(self.roles)[2:3] if r==1 else self.effective(self.roles)
        mode=self.effective(self.mode_embedding)[0 if text else 1]
        h=self.backbone(embedding[tokens]+role[None,None,:,:]+mode)
        out=fp8_matmul(h,self.embedding) if self.precision=='mxfp8' else h@self.embedding.T
        return out[:,:,0] if text else out

    def flow(self,clues,noisy,time,feedback):
        """Conditional program-state denoiser; labels never enter this interface."""
        b,t=clues.shape
        if not 1<=t<=self.position.shape[0] or b*t*3>2048 or noisy.shape!=(b,t,10) or feedback.shape!=noisy.shape:
            raise ValueError('invalid bounded Flow input')
        def pad(x):return mx.pad(x,[(0,0),(0,0),(0,22)])
        temporal=time.reshape(b,1,1)*self.effective(self.flow_time)[0]+self.effective(self.flow_time)[1]
        inputs=self.effective(self.embedding)[clues]
        candidate=self.flow_state(pad(noisy))+self.flow_feedback(pad(feedback))+temporal
        roles=self.effective(self.roles)
        h=mx.stack([inputs+roles[0],candidate+roles[1],candidate+roles[2]],axis=2)+self.effective(self.mode_embedding)[2]
        return self.flow_output(self.backbone(h,causal=False)[:,:,2])

    def export(self,directory):
        """The FP8 release contains no floating model arrays or latent masters."""
        if self.precision!='mxfp8':raise ValueError('the FP8 release requires FP8 operand training')
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
        shapes={};arrays={};logical=0
        for name,value in tree_flatten(self.parameters()):
            packed,scales=packed_value(value)
            arrays[name+'.packed']=packed;arrays[name+'.scales']=scales
            shapes[name]=list(value.shape);logical+=value.size
        mx.eval(arrays)
        mx.save_safetensors(str(directory/'weights.safetensors'),arrays)
        (directory/'config.json').write_text(json.dumps(self.config,indent=2))
        info=dict(format='MXFP8 E4M3 values, E8M0 scales per32 padded values',shapes=shapes,
            parameters=logical,tensor_bytes=sum(v.nbytes for v in arrays.values()),
            weight_file_bytes=(directory/'weights.safetensors').stat().st_size,
            runtime='BF16 decoded working weights; native MXFP8 projection operands; FP16 KDA state',
            weight_sha256=hashlib.sha256((directory/'weights.safetensors').read_bytes()).hexdigest())
        (directory/'storage.json').write_text(json.dumps(info,indent=2))
        return info

    @classmethod
    def load(cls,directory):
        directory=Path(directory);config=json.loads((directory/'config.json').read_text())
        model=cls(**config)
        info=json.loads((directory/'storage.json').read_text());arrays=mx.load(str(directory/'weights.safetensors'))
        decoded=[(name,unpacked_value(arrays[name+'.packed'],arrays[name+'.scales'],shape)) for name,shape in info['shapes'].items()]
        model.update(tree_unflatten(decoded));mx.eval(model.parameters())
        return model
