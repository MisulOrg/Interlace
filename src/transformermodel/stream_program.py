"""Train or run the combined model's three streams on a verifiable program.

Input supplies digits then END. Thought emits the running sum modulo ten one
tick later. Output emits the final answer after END. No external model supplies
labels or inference answers; arithmetic below constructs training/evaluation
targets only. Free-running decoding predicts both learned streams itself.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import sys
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten,tree_map
import numpy as np
from tokenizers import Tokenizer
from .combined import CombinedModel
from .text_data import token_windows
from .safe_run import require_guard


def program_rows(digits, digit_ids, idle, end, length):
    if not digits or any(not isinstance(n,(int,np.integer)) or not 0<=n<10 for n in digits):
        raise ValueError('provide digits from 0 through 9')
    if len(digits)+3>length:raise ValueError('program exceeds row context')
    rows=np.full((length,3),idle,dtype=np.int32)
    total=0
    for i,digit in enumerate(digits):
        rows[i+1,0]=digit_ids[digit]
        total=(total+int(digit))%10
        rows[i+2,1]=digit_ids[total]
    rows[len(digits)+1,0]=end
    rows[len(digits)+2,2]=digit_ids[total]
    return rows


def make_programs(seed,count,min_digits,max_digits,buckets,exclude=()):
    rng=np.random.default_rng(seed);seen=set(exclude);programs=[]
    while len(programs)<count:
        digits=tuple(int(n) for n in rng.integers(0,10,rng.integers(min_digits,max_digits+1)))
        bucket=int.from_bytes(hashlib.sha256(bytes(digits)).digest()[:4],'little')%10
        if digits not in seen and bucket in buckets:
            seen.add(digits);programs.append(digits)
    return programs


def test_programs(seed):
    """Reconstruct prior tests with their exclusions, then exclude their union."""
    seen_held,seen_long=set(),set()
    for previous in [s for s in (934,956) if s<seed]+[seed]:
        held=make_programs(previous,256,2,6,{9},exclude=seen_held)
        longer=make_programs(previous+1,128,8,10,{9},exclude=seen_long)
        seen_held.update(held);seen_long.update(longer)
    return held,longer


def rollout(model,external,idle,thought_override=None):
    """Only the external input stream is supplied. Full-vocabulary greedy decode."""
    batch,time_steps=external.shape
    rows=np.full((batch,1,3),idle,dtype=np.int32)
    for tick in range(1,time_steps):
        predicted=np.array(mx.argmax(model(mx.array(rows))[:,-1],axis=-1))
        predicted[:,0]=external[:,tick]
        if thought_override is not None:predicted[:,1]=thought_override
        rows=np.concatenate([rows,predicted[:,None]],axis=1)
    return rows


def evaluate_programs(model,programs,digit_ids,idle,end,thought_override=None,include_failures=False):
    length=max(map(len,programs))+3
    targets=np.stack([program_rows(d,digit_ids,idle,end,length) for d in programs])
    predictions=[]
    for start in range(0,len(targets),8):
        predictions.append(rollout(model,targets[start:start+8,:,0],idle,thought_override))
    predicted=np.concatenate(predictions)
    answer_rows=np.array([len(d)+2 for d in programs]);indices=np.arange(len(programs))
    thought_mask=targets[:,:,1]!=idle
    result=dict(examples=len(programs),answer_correct=int(np.sum(predicted[indices,answer_rows,2]==targets[indices,answer_rows,2])),
        thought_correct=int(np.sum((predicted[:,:,1]==targets[:,:,1])&thought_mask)),thought_targets=int(thought_mask.sum()),
        answer_accuracy=float(np.mean(predicted[indices,answer_rows,2]==targets[indices,answer_rows,2])),
        thought_accuracy=float(np.mean(predicted[:,:,1][thought_mask]==targets[:,:,1][thought_mask])),
        output_idle_accuracy=float(np.mean(predicted[:,:,2][targets[:,:,2]==idle]==idle)),
        samples=[dict(digits=list(d),expected=int(sum(d)%10),rows=predicted[i].tolist()) for i,d in enumerate(programs[:6])])
    if include_failures:
        failed=predicted[indices,answer_rows,2]!=targets[indices,answer_rows,2]
        result['failures']=[dict(digits=list(programs[i]),expected=int(sum(programs[i])%10),rows=predicted[i].tolist()) for i in np.flatnonzero(failed)]
    return result


def load(directory):
    directory=Path(directory)
    config=json.loads((directory/'config.json').read_text())
    model=CombinedModel(**config);model.load_weights(str(directory/'weights.safetensors'))
    tokenizer=Tokenizer.from_file(str(directory/'tokenizer.json'))
    return model,tokenizer,config


def main():
    require_guard()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--output',help='train a separate adapted artifact when provided')
    p.add_argument('--digits',default='2,5,3,7')
    p.add_argument('--steps',type=int,default=2048)
    p.add_argument('--seconds',type=float,default=240)
    p.add_argument('--seed',type=int,default=931)
    p.add_argument('--replay-every',type=int,default=4,help='zero disables text replay for a stream-specialized artifact')
    p.add_argument('--freeze-coefficients',action='store_true')
    p.add_argument('--test-seed',type=int,default=934,help='new seed excludes the original exposed test sequences')
    a=p.parse_args()
    if not 1<=a.steps<=16384 or not 0<a.seconds<=600:p.error('invalid bounded adaptation budget')
    if a.replay_every<0 or a.replay_every==1:p.error('replay interval must be zero or at least two')
    mx.set_memory_limit(1100*2**20);mx.set_cache_limit(32*2**20)
    start=time.perf_counter();mx.random.seed(a.seed)
    m,tokenizer,config=load(a.model)
    if not a.output:
        digits=[int(n) for n in a.digits.split(',')]
        if len(digits)>min(config['context']-3,16):p.error('at most 16 digits in the live demo')
        idle=tokenizer.token_to_id('<|idle|>');end=tokenizer.token_to_id('<|end-input|>')
        if idle is None or end is None:p.error('this artifact has no trained stream program')
        digit_ids=[tokenizer.encode(str(i)).ids[0] for i in range(10)]
        if not digits or any(not 0<=d<10 for d in digits):p.error('provide digits0-9')
        external=np.full((1,len(digits)+3),idle,dtype=np.int32)
        external[0,1:len(digits)+1]=[digit_ids[d] for d in digits]
        external[0,len(digits)+1]=end
        predicted=rollout(m,external,idle)[0]
        for t,row in enumerate(predicted):
            print(json.dumps(dict(tick=t,input=tokenizer.decode([int(row[0])],skip_special_tokens=False),
                thought=tokenizer.decode([int(row[1])],skip_special_tokens=False),
                output=tokenizer.decode([int(row[2])],skip_special_tokens=False))),flush=True)
        return
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    old_size=config['vocab_size'];tokenizer.add_special_tokens(['<|idle|>','<|end-input|>'])
    new_size=tokenizer.get_vocab_size()
    if new_size-old_size!=2:raise ValueError('adapt a text-only source checkpoint once')
    m.update(tree_map(lambda w:w.astype(mx.float32),m.parameters()))
    m.embedding.weight=mx.concatenate([m.embedding.weight,mx.random.normal((2,config['width']))*.01])
    config=dict(config,vocab_size=new_size)
    idle=tokenizer.token_to_id('<|idle|>');end=tokenizer.token_to_id('<|end-input|>')
    encoded=[tokenizer.encode(str(i)).ids for i in range(10)]
    if any(len(ids)!=1 for ids in encoded):raise ValueError('digit tokenizer contract failed')
    digit_ids=[ids[0] for ids in encoded]
    training=make_programs(a.seed,4096,2,6,set(range(8)))
    monitor=make_programs(933,64,2,6,{8})
    rows=mx.array(np.stack([program_rows(d,digit_ids,idle,end,9) for d in training]))
    x,y=rows[:,:-1],rows[:,1:]
    # Replay retains a directly measured part of language behavior during adaptation.
    tx,ty,_,_=token_windows('data/wikitext2-v1/prepared/manifest.json',Path(a.model)/'tokenizer.json','train',config['context'])
    vx,vy,_,_=token_windows('data/wikitext2-v1/prepared/manifest.json',Path(a.model)/'tokenizer.json','validation',config['context'])
    tx,ty=mx.array(tx),mx.array(ty)
    retention=np.random.default_rng(909).permutation(len(vx))[:64]
    def language_loss():
        return sum(float(nn.losses.cross_entropy(m(mx.array(vx[idx])),mx.array(vy[idx]),reduction='sum'))
            for idx in np.array_split(retention,8))/(64*config['context'])
    before_language=language_loss()
    if config['architecture']!='dense':
        from .dyadic import DyadicAdamW
        opt=DyadicAdamW(.0003,coefficient_scale=.1)
    else:opt=optim.AdamW(.0003,weight_decay=.01)
    opt.init(m.trainable_parameters());mx.eval(m.parameters(),opt.state,x,y,tx,ty)
    def stream_loss(model,x,y):
        logits=model(x)[:,:,1:];targets=y[:,:,1:]
        weights=mx.where(targets!=idle,5.,1.)
        return mx.sum(nn.losses.cross_entropy(logits,targets,reduction='none')*weights)/mx.sum(weights)
    stream_vg=nn.value_and_grad(m,stream_loss)
    language_vg=nn.value_and_grad(m,lambda model,x,y:nn.losses.cross_entropy(model(x),y,reduction='mean'))
    state=[m.state,opt.state,mx.random.state]
    def apply(g):
        if config['architecture']!='dense' and a.freeze_coefficients:g=opt.mask_gradients(g,0)
        g,norm=optim.clip_grad_norm(g,1.)
        if config['architecture']!='dense':opt.update(m,g,0 if a.freeze_coefficients else 2,mx.array(1.))
        else:opt.update(m,g)
        return norm
    def stream_update(x,y):
        loss,g=stream_vg(m,x,y);return loss,apply(g)
    def language_update(x,y):
        loss,g=language_vg(m,x,y);return loss,apply(g)
    stream_step=mx.compile(stream_update,inputs=state,outputs=state)
    language_step=mx.compile(language_update,inputs=state,outputs=state)
    manifest=dict(command=[sys.executable,*sys.argv],configuration=config,initialization='adapted from locally random-initialized text model',
        source_model=str(Path(a.model).resolve()),source_weights_sha256=hashlib.sha256((Path(a.model)/'weights.safetensors').read_bytes()).hexdigest(),
        training_programs_sha256=hashlib.sha256(json.dumps(training).encode()).hexdigest(),
        split='SHA256(digit sequence) mod10: 0-7 train, 8 monitor, 9 held-out; lengths2-6 train',
        supervision='running sum modulo10 from deterministic Python arithmetic; no teacher model',
        evaluation='full-vocabulary free-running thought/output; external inputs only supplied',
        recipe=dict(optimizer='AdamW',lr=.0003,coefficient_lr_scale=.1,freeze_coefficients=a.freeze_coefficients,weight_decay=.01,batch=16,nonidle_weight=5,language_replay_interval=a.replay_every,steps_cap=a.steps,seconds_cap=a.seconds),
        before_language_monitor_loss=before_language,seed=a.seed,test_seed=a.test_seed)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'source-snapshot.json').write_text(json.dumps({str(f):f.read_text() for f in Path(__file__).parent.glob('*.py')}))
    (out/'config.json').write_text(json.dumps(config,indent=2));tokenizer.save(str(out/'tokenizer.json'))
    rng=np.random.default_rng(a.seed);best=-1.;steps=0;replay=0;last_loss=None;last_norm=None
    with (out/'metrics.jsonl').open('w') as log:
        for i in range(a.steps+1):
            if i%256==0 or i==a.steps or time.perf_counter()-start>=a.seconds:
                report=evaluate_programs(m,monitor,digit_ids,idle,end)
                score=report['answer_accuracy']+report['thought_accuracy']
                if score>best:
                    best=score;m.save_weights(str(out/'best-master.safetensors'))
                record=dict(step=i,elapsed=time.perf_counter()-start,monitor=report,last_loss=last_loss,last_gradient_norm=last_norm)
                log.write(json.dumps(record)+'\n');log.flush()
                print(json.dumps({k:v for k,v in record.items() if k!='monitor'}|{k:report[k] for k in ('answer_accuracy','thought_accuracy')}),flush=True)
            if i==a.steps or time.perf_counter()-start>=a.seconds:break
            if a.replay_every and i%a.replay_every==a.replay_every-1:
                ids=mx.array(rng.integers(len(tx),size=8));loss,norm=language_step(tx[ids],ty[ids]);replay+=1
            else:
                ids=mx.array(rng.integers(len(x),size=16));loss,norm=stream_step(x[ids],y[ids])
            mx.eval(m.parameters(),opt.state,loss,norm);steps=i+1
            last_loss=float(loss);last_norm=float(norm)
            if not math.isfinite(float(loss)) or not math.isfinite(float(norm)):
                m.save_weights(str(out/'nonfinite-master.safetensors'));raise RuntimeError('nonfinite adaptation')
    m.save_weights(str(out/'last-master.safetensors'));m.load_weights(str(out/'best-master.safetensors'))
    # Open these fixed tests only after the selected checkpoint is frozen.
    held,long=test_programs(a.test_seed)
    held_report=evaluate_programs(m,held,digit_ids,idle,end)
    long_report=evaluate_programs(m,long,digit_ids,idle,end)
    after_language=language_loss()
    probe=rows[:2,:-1];expected=np.array(m(probe))
    m.update(tree_map(lambda w:w.astype(mx.bfloat16),m.parameters()));m.save_weights(str(out/'weights.safetensors'))
    reload,_,_=load(out);exact=np.array_equal(expected,np.array(reload(probe)))
    if not exact:raise RuntimeError('stream export mismatch')
    summary=dict(steps=steps,language_replay_updates=replay,elapsed_seconds=time.perf_counter()-start,
        held_out=held_report,longer_untrained_lengths=long_report,
        before_language_monitor_loss=before_language,after_language_monitor_loss=after_language,
        parameter_count=sum(v.size for _,v in tree_flatten(m.parameters())),
        tensor_bytes=sum(v.nbytes for _,v in tree_flatten(m.parameters())),
        weight_file_bytes=(out/'weights.safetensors').stat().st_size,tokenizer_bytes=(out/'tokenizer.json').stat().st_size,
        export_reload_exact=bool(exact),mlx_peak_bytes=mx.get_peak_memory())
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
