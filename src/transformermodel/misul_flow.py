"""Conditional Flow refinement over learned program states.

Adaptation of Helbling et al., arxiv:2606.29150: linear noise interpolation,
categorical denoising, raw-logit self-conditioning, and re-noise stability.
Dataset arithmetic is separate from inference. Stability is not correctness.
"""
import numpy as np
import mlx.core as mx
import mlx.nn as nn


def program_batch(programs,digit_ids,idle,length):
    clues=np.full((len(programs),length),idle,dtype=np.int32)
    answers=np.zeros_like(clues);mask=np.zeros_like(clues,dtype=np.bool_)
    for row,digits in enumerate(programs):
        if not 1<=len(digits)<=length or any(not 0<=d<10 for d in digits):raise ValueError('invalid digit sequence')
        clues[row,:len(digits)]=[digit_ids[d] for d in digits]
        answers[row,:len(digits)]=np.cumsum(digits)%10
        mask[row,:len(digits)]=True
    return mx.array(clues),mx.array(answers),mx.array(mask)


def flow_loss(model,clues,targets,mask,time,noise,feedback_on):
    endpoint=mx.eye(10,dtype=mx.bfloat16)[targets]
    t=time[:,None,None]
    noisy=(1-t)*noise+t*endpoint
    feedback=mx.zeros_like(noisy)
    if feedback_on:
        feedback=mx.stop_gradient(model.flow(clues,noisy,time,feedback))
    logits=model.flow(clues,noisy,time,feedback)
    loss=nn.losses.cross_entropy(logits.astype(mx.float16),targets,reduction='none')
    return mx.sum(loss*mask)/mx.sum(mask)


def sample_flow(model,clues,steps=8,seed=987,feedback_on=True,start=None,start_time=0.,trace=True):
    """Read only clues and model outputs; every candidate position can change."""
    if not 1<=steps<=32 or not 0<=start_time<1:raise ValueError('invalid bounded integration schedule')
    b,t=clues.shape
    state=mx.random.normal((b,t,10),dtype=mx.bfloat16,key=mx.random.key(seed)) if start is None else start
    feedback=mx.zeros_like(state);history=[]
    for i in range(steps):
        time=start_time+(1-start_time)*i/steps
        clocks=mx.full((b,),time,dtype=mx.bfloat16)
        logits=model.flow(clues,state,clocks,feedback)
        posterior=mx.softmax(logits.astype(mx.float16),axis=-1).astype(mx.bfloat16)
        # dt/(1-t) is exactly1/(steps-i) for this uniform sub-interval.
        state=state+(posterior-state)/(steps-i)
        prediction=mx.argmax(logits,axis=-1)
        if trace:history.append(dict(step=i+1,time=time,candidate=prediction.tolist()))
        feedback=logits if feedback_on else mx.zeros_like(logits)
    mx.eval(prediction,logits)
    return prediction,logits,history


def stability_score(model,clues,candidate,mask,seed=988,steps=4):
    """Return cross-entropy after perturbing the candidate, without a gold answer."""
    endpoint=mx.eye(10,dtype=mx.bfloat16)[candidate]
    noise=mx.random.normal(endpoint.shape,dtype=mx.bfloat16,key=mx.random.key(seed))
    _,logits,_=sample_flow(model,clues,steps,seed,start=.4*endpoint+.6*noise,start_time=.4,trace=False)
    loss=nn.losses.cross_entropy(logits.astype(mx.float16),candidate,reduction='none')
    return mx.sum(loss*mask,axis=-1)/mx.sum(mask,axis=-1)


def evaluate_flow(model,programs,digit_ids,idle,steps=(1,2,4,8),seed=990,feedback_on=True):
    length=max(map(len,programs));clues,targets,mask=program_batch(programs,digit_ids,idle,length)
    truth=np.array(targets);valid=np.array(mask);last=np.array([len(d)-1 for d in programs])
    reports={}
    for count in steps:
        predicted=[];traces=[]
        for start in range(0,len(programs),8):
            part,_,trace=sample_flow(model,clues[start:start+8],count,seed+start,feedback_on,trace=start==0)
            predicted.append(np.array(part))
            if start==0:traces=trace
        got=np.concatenate(predicted);exact=np.all((got==truth)|~valid,axis=-1)
        reports[str(count)]=dict(examples=len(programs),exact_correct=int(exact.sum()),
            exact_accuracy=float(exact.mean()),state_accuracy=float(np.mean(got[valid]==truth[valid])),
            answer_accuracy=float(np.mean(got[np.arange(len(got)),last]==truth[np.arange(len(got)),last])),
            forward_passes_per_program=count,first_batch_trace=traces,
            failures=[dict(digits=list(programs[i]),candidate=got[i,:len(programs[i])].tolist(),target=truth[i,:len(programs[i])].tolist()) for i in np.flatnonzero(~exact)])
    return reports
