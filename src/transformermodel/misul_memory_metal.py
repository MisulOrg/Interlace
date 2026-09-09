"""Local FP16 delta-memory fusion with an explicit reverse pass.

The public contract is the ordinary recurrence in combined.delta_scan with
zero initial state, K=16, and bounded local shapes. Each Metal thread owns one
value channel and its sixteen state coordinates. Forward histories are retained
only for backpropagation; no history is part of a released model's knowledge.
"""
import mlx.core as mx


_forward=mx.fast.metal_kernel(
    name='misul_memory16_forward',input_names=['q','k','v','a','b'],
    output_names=['out','history'],source=r'''
    const uint lane=thread_position_in_grid.x;
    const uint bh=lane/V, j=lane%V;
    if(bh>=q_shape[0]*q_shape[1]) return;
    half state[16], product[16];
    for(int c=0;c<16;++c) {
        state[c]=half(0);
        history[(bh*(L+1)*16+c)*V+j]=half(0);
    }
    for(int t=0;t<L;++t) {
        const uint base=(bh*L+t)*16;
        for(int c=0;c<16;++c) {
            state[c]=half(a[base+c]*state[c]);
            product[c]=half(k[base+c]*state[c]);
        }
        for(int stride=8;stride>0;stride/=2)
            for(int c=0;c<stride;++c) product[c]=half(product[c]+product[c+stride]);
        const half error=half(v[(bh*L+t)*V+j]-product[0]);
        const half beta=b[bh*L+t];
        for(int c=0;c<16;++c) {
            const half gain=half(beta*k[base+c]);
            state[c]=half(state[c]+half(gain*error));
            history[((bh*(L+1)+t+1)*16+c)*V+j]=state[c];
            product[c]=half(q[base+c]*state[c]);
        }
        for(int stride=8;stride>0;stride/=2)
            for(int c=0;c<stride;++c) product[c]=half(product[c]+product[c+stride]);
        out[(bh*L+t)*V+j]=product[0];
    }
    ''')


_reverse=mx.fast.metal_kernel(
    name='misul_memory16_reverse',input_names=['q','k','v','a','b','history','grad_out','dh'],
    output_names=['dq','dk','dv','da','db'],source=r'''
    const uint bh=threadgroup_position_in_grid.x;
    const uint j=thread_position_in_threadgroup.x;
    const uint warp=j/32, warp_lane=j%32;
    const bool active=j<V;
    threadgroup half scratch[196];
    half adjoint[16], product[16], decayed[16];
    for(int c=0;c<16;++c) adjoint[c]=half(0);
    for(int t=L-1;t>=0;--t) {
        const uint base=(bh*L+t)*16;
        const uint current=(bh*(L+1)+t+1)*16;
        const uint previous=current-16;
        const half dout=active?grad_out[(bh*L+t)*V+j]:half(0);
        const half beta=b[bh*L+t];
        for(int c=0;c<16;++c) {
            adjoint[c]=half(adjoint[c]+(active?dh[(current+c)*V+j]:half(0)));
            adjoint[c]=half(adjoint[c]+half(q[base+c]*dout));
            const half dquery=active?half(history[(current+c)*V+j]*dout):half(0);
            const half sum_query=metal::simd_sum(dquery);
            if(warp_lane==0) scratch[c*4+warp]=sum_query;
            decayed[c]=active?half(a[base+c]*history[(previous+c)*V+j]):half(0);
            product[c]=half(k[base+c]*decayed[c]);
        }
        for(int stride=8;stride>0;stride/=2)
            for(int c=0;c<stride;++c) product[c]=half(product[c]+product[c+stride]);
        const half error=active?half(v[(bh*L+t)*V+j]-product[0]):half(0);
        for(int c=0;c<16;++c) product[c]=half(adjoint[c]*half(beta*k[base+c]));
        for(int stride=8;stride>0;stride/=2)
            for(int c=0;c<stride;++c) product[c]=half(product[c]+product[c+stride]);
        const half derror=product[0];
        if(active) dv[(bh*L+t)*V+j]=derror;
        for(int c=0;c<16;++c) {
            const half dkey=half(half(half(adjoint[c]*error)*beta)-half(decayed[c]*derror));
            const half sum_key=metal::simd_sum(dkey);
            if(warp_lane==0) scratch[64+c*4+warp]=sum_key;
            product[c]=half(half(adjoint[c]*error)*k[base+c]);
            const half ddecay=half(adjoint[c]-half(k[base+c]*derror));
            const half dalpha=active?half(ddecay*history[(previous+c)*V+j]):half(0);
            const half sum_alpha=metal::simd_sum(dalpha);
            if(warp_lane==0) scratch[128+c*4+warp]=sum_alpha;
            adjoint[c]=half(ddecay*a[base+c]);
        }
        for(int stride=8;stride>0;stride/=2)
            for(int c=0;c<stride;++c) product[c]=half(product[c]+product[c+stride]);
        const half sum_beta=metal::simd_sum(product[0]);
        if(warp_lane==0) scratch[192+warp]=sum_beta;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(j<16) {
            const uint offset=j*4;
            dq[base+j]=half(half(scratch[offset]+scratch[offset+1])+half(scratch[offset+2]+scratch[offset+3]));
            dk[base+j]=half(half(scratch[64+offset]+scratch[65+offset])+half(scratch[66+offset]+scratch[67+offset]));
            da[base+j]=half(half(scratch[128+offset]+scratch[129+offset])+half(scratch[130+offset]+scratch[131+offset]));
        }
        if(j==0) db[bh*L+t]=half(half(scratch[192]+scratch[193])+half(scratch[194]+scratch[195]));
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    ''')


@mx.custom_function
def _scan(q,k,v,a,b):
    batch,heads,length,_=q.shape;width=v.shape[-1]
    return tuple(_forward(inputs=[q,k,v,a,b],template=[('L',length),('V',width)],
        grid=(batch*heads*width,1,1),threadgroup=(32,1,1),
        output_shapes=[v.shape,(batch,heads,length+1,16,width)],
        output_dtypes=[mx.float16,mx.float16]))


@_scan.vjp
def _scan_reverse(primals,cotangents,outputs):
    q,k,v,a,b=primals;do,dh=cotangents;_,history=outputs
    batch,heads,length,_=q.shape;width=v.shape[-1]
    dq,dk,dv,da,db=_reverse(inputs=[q,k,v,a,b,history,do,dh],
        template=[('L',length),('V',width)],grid=(batch*heads*128,1,1),threadgroup=(128,1,1),
        output_shapes=[q.shape,k.shape,v.shape,a.shape,b.shape],output_dtypes=[mx.float16]*5)
    return dq,dk,dv,da,db


def memory_metal(q,k,v,alpha,beta):
    if q.shape!=k.shape or q.shape!=alpha.shape or q.shape[-1]!=16 or beta.shape!=q.shape[:-1] or v.shape[:-1]!=beta.shape:
        raise ValueError('fused memory requires compatible K=16 inputs')
    if not 1<=q.shape[-2]<=256 or not 1<=v.shape[-1]<=80 or q.shape[0]*q.shape[1]>256:
        raise ValueError('fused memory allocation bound exceeded')
    output,history=_scan(*[x.astype(mx.float16) for x in (q,k,v,alpha,beta)])
    return output,history[:,:,-1]
