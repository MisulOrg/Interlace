"""Guarded, resumable joint training of one Misul text/streams/Flow checkpoint."""
import time
PROCESS_START=time.perf_counter()
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten,tree_unflatten
import numpy as np
from transformermodel.misul_dense import DenseModel as MisulModel
from transformermodel.misul_muon import Muon16
from transformermodel.misul_update import CheckedUpdate
from transformermodel.misul_flow import program_batch,flow_loss,evaluate_flow
from transformermodel.stream_program import make_programs,program_rows,evaluate_programs
from transformermodel.text_data import token_windows
from transformermodel.safe_run import require_guard
from transformermodel.train import command_output


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def schedule(index,total):
    if index<64:return max(.05,(index+1)/64)
    start=int(.8*total)
    return 1. if index<start else max(.1,1.-.9*(index-start)/max(1,total-start))


def save_training(path,model,optimizer):
    path=Path(path)
    arrays={'model/'+k:v for k,v in tree_flatten(model.parameters())}
    arrays['step']=optimizer.state['step']
    for name,state in optimizer.state['leaves'].items():
        for key,value in state.items():arrays['optimizer/'+key+'/'+name]=value
    temporary=path.with_name(path.stem+'.pending'+path.suffix)
    mx.save_safetensors(str(temporary),arrays)
    temporary.replace(path)


def load_training(path,model,optimizer):
    arrays=mx.load(str(path))
    model.update(tree_unflatten([(k[6:],v) for k,v in arrays.items() if k.startswith('model/')]))
    optimizer.state['step']=arrays['step']
    for name,state in optimizer.state['leaves'].items():
        for key in state:state[key]=arrays['optimizer/'+key+'/'+name]
    mx.eval(model.parameters(),optimizer.state)
    return int(optimizer.state['step'])


def main():
    require_guard()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--precision',choices=['mxfp8','bf16'],default='mxfp8')
    p.add_argument('--memory-backend',choices=['scan','metal'],default='metal')
    p.add_argument('--width',type=int,default=640);p.add_argument('--hidden',type=int,default=1536)
    p.add_argument('--layers',type=int,default=6);p.add_argument('--heads',type=int,default=8)
    p.add_argument('--context',type=int,default=128);p.add_argument('--batch',type=int,default=4)
    p.add_argument('--program-batch',type=int,default=16)
    p.add_argument('--steps',type=int,default=16384);p.add_argument('--seconds',type=float,default=1200)
    p.add_argument('--stop-after',type=int,default=0,help='stop at this absolute update for an exact resumable segment')
    p.add_argument('--seed',type=int,default=1001);p.add_argument('--eval-every',type=int,default=512)
    p.add_argument('--validation-windows',type=int,default=0,help='0 uses all official validation windows at completion')
    p.add_argument('--optimizer',choices=['nor','adamw'],default='nor')
    p.add_argument('--mode',choices=['joint','text'],default='joint')
    p.add_argument('--resume',action='store_true');p.add_argument('--skip-program-eval',action='store_true')
    a=p.parse_args()
    if not 1<=a.steps<=32768 or not 0<a.seconds<=3600 or not 1<=a.batch<=8 or a.batch*a.context>1024 or not 1<=a.program_batch<=32:
        p.error('invalid bounded run configuration')
    if not 1<=a.eval_every or a.validation_windows<0 or a.stop_after<0:p.error('invalid evaluation/update bound')
    out=Path(a.output)
    if not a.resume:out.mkdir(parents=True,exist_ok=False)
    elif not (out/'training-state.safetensors').exists():p.error('missing resumable training state')
    mx.set_memory_limit(1000*2**20);mx.set_cache_limit(32*2**20)
    manifest_path='data/wikitext2-v1/prepared/manifest.json';tokenizer_path='data/misul-v2/tokenizer.json'
    tx,ty,tokenizer,_=token_windows(manifest_path,tokenizer_path,'train',a.context)
    vx,vy,_,byte_lengths=token_windows(manifest_path,tokenizer_path,'validation',a.context)
    if tx.nbytes+ty.nbytes+vx.nbytes+vy.nbytes>256*2**20:raise ValueError('data allocation bound exceeded')
    config=dict(vocab_size=tokenizer.get_vocab_size(),width=a.width,hidden=a.hidden,layers=a.layers,
        heads=a.heads,context=a.context,loops=1,precision=a.precision,memory_backend=a.memory_backend)
    estimate=config['vocab_size']*a.width+3*a.width*a.hidden*a.layers+4*a.layers*a.width*a.width
    if estimate>36_000_000:raise ValueError('reduce model dimensions before allocation')
    mx.random.seed(a.seed);model=MisulModel(**config)
    optimizer=Muon16(model.parameters(),kind=a.optimizer)
    mx.eval(model.parameters(),optimizer.state)
    start_step=load_training(out/'training-state.safetensors',model,optimizer) if a.resume else 0
    sources={str(Path('src/transformermodel')/f):digest(Path('src/transformermodel')/f) for f in
        ['misul_dense.py','misul.py','misul_memory_metal.py','misul_muon.py','misul_update.py','misul_flow.py','train_misul.py','combined.py','budget_muon.py','text_data.py','stream_program.py']}
    sources[__file__]=digest(__file__)
    old_summary=json.loads((out/'summary.json').read_text()) if a.resume else {}
    if a.resume:
        previous=json.loads((out/'manifest.json').read_text())
        if previous['source_sha256']!=sources or previous['configuration']!=config:
            raise ValueError('source or configuration changed across a resumed experiment')
        for key in ('seed','steps','batch','program_batch','mode','optimizer'):
            if previous['recipe'][key]!=vars(a)[key]:raise ValueError('training recipe changed across a resume')
        if previous['complete']:raise ValueError('a completed experiment is immutable')
    idle=tokenizer.token_to_id('<|idle|>');end=tokenizer.token_to_id('<|end-input|>')
    digit_ids=[tokenizer.encode(str(i)).ids[0] for i in range(10)]
    programs=make_programs(1009,8192,2,6,set(range(8)))
    rows=mx.array(np.stack([program_rows(d,digit_ids,idle,end,9) for d in programs]))
    clues,answers,flow_mask=program_batch(programs,digit_ids,idle,6)
    monitor_programs=make_programs(1011,32,2,6,{8})
    order=np.random.default_rng(1003).permutation(len(vx));monitor=order[:32]
    tx,ty=mx.array(tx),mx.array(ty);mx.eval(tx,ty,rows,clues,answers,flow_mask)
    metadata=dict(command=[sys.executable,*sys.argv],configuration=config,recipe=vars(a),complete=False,
        initialization='random, no pretrained weights or external teacher',seed=a.seed,
        parameters=sum(v.size for _,v in tree_flatten(model.parameters())),
        working_parameter_bytes=sum(v.nbytes for _,v in tree_flatten(model.parameters())),
        optimizer_bytes=sum(v.nbytes for _,v in tree_flatten(optimizer.state)),
        training_windows=len(tx),validation_windows=len(vx),training_tokens_available=ty.size,
        monitor_indices=monitor.tolist(),source_sha256=sources,data_sha256=digest(manifest_path),
        tokenizer_sha256=digest(tokenizer_path),program_sha256=hashlib.sha256(json.dumps(programs).encode()).hexdigest(),
        precision='MXFP8 weights/projection operands; BF16 working parameters and optimizer; FP16 recurrence/loss; no FP32 master arrays' if a.precision=='mxfp8' else 'explicit BF16 quality control, with FP16 recurrence/loss; no FP32 master arrays',
        recipe_notes='NorMuon PE5 momentum.7 RMS.2 hidden matrices; AdamW auxiliary, beta1.9 beta2.95; LR.006/.001; decay.01; clip1; checked loss scale128 halved on overflow through1, retry same batch before optimizer; 64-update warmup, stable to80%, then linear decay to.1',
        task_cycle=['text','text','streams','text','flow','text','streams','flow'] if a.mode=='joint' else ['text'],
        program_split='SHA256(sequence) mod10:0-7 training,8 validation,9 final test; train length2-6',
        hypothesis='parameter-matched dense architecture control; compare quality and elapsed cost to frozen Misul BF16 seed1001',
        null='Misul does not improve held-out loss or task accuracy, and does not reach fixed quality sooner',
        scope='exploratory pilot' if a.validation_windows else 'declared full-validation training run',
        stop_rule='fixed updates or segment/time bound, nonfinite value, or external guard',
        device=mx.device_info(),mlx=importlib.metadata.version('mlx'),macos=platform.mac_ver(),
        power=command_output(['pmset','-g','batt']),thermal=command_output(['pmset','-g','therm']))
    if not a.resume:
        (out/'manifest.json').write_text(json.dumps(metadata,indent=2))
        (out/'source-snapshot.json').write_text(json.dumps({name:Path(name).read_text() for name in sources}))
        (out/'tokenizer.json').write_bytes(Path(tokenizer_path).read_bytes())
    else:metadata=previous
    def language_loss(model,x,y):
        return nn.losses.cross_entropy(model(x).astype(mx.float16),y,reduction='mean')
    def stream_loss(model,x,y):
        logits=model(x)[:,:,1:];target=y[:,:,1:]
        weights=mx.where(target!=idle,mx.array(5.,dtype=mx.float16),mx.array(1.,dtype=mx.float16))
        return mx.sum(nn.losses.cross_entropy(logits.astype(mx.float16),target)*weights)/mx.sum(weights)
    compiled={}
    loss_scales=old_summary.get('loss_scales',{})
    overflow_retries=old_summary.get('overflow_retries',0)
    def get_step(kind,feedback_on=False):
        signature=(kind,feedback_on)
        if signature not in compiled:
            loss_function=language_loss if kind=='text' else stream_loss if kind=='streams' else lambda m,*x:flow_loss(m,*x,feedback_on)
            key=kind+('_feedback' if feedback_on else '')
            compiled[signature]=CheckedUpdate(model,optimizer,loss_function,loss_scales.get(key,128))
        return compiled[signature]
    def evaluate(ids):
        total=0.;count=0
        for start in range(0,len(ids),a.batch):
            part=ids[start:start+a.batch]
            losses=nn.losses.cross_entropy(model(mx.array(vx[part])).astype(mx.float16),mx.array(vy[part]),reduction='none')
            values=np.array(losses.tolist(),dtype=np.float64);total+=values.sum();count+=values.size
        return float(total/count)
    timings=old_summary.get('update_timings',{})
    steps=start_step;last_values={};all_modes=metadata['task_cycle']
    with (out/'metrics.jsonl').open('a' if a.resume else 'w') as log:
        for index in range(start_step,a.steps+1):
            elapsed=time.perf_counter()-PROCESS_START
            done=index==a.steps or elapsed>=a.seconds or a.stop_after and index>=a.stop_after
            if index==start_step or index%a.eval_every==0 or done:
                validation=evaluate(monitor)
                record=dict(step=index,session_elapsed=time.perf_counter()-PROCESS_START,
                    monitor_loss=validation,last_values=last_values,peak_mlx_bytes=mx.get_peak_memory())
                log.write(json.dumps(record)+'\n');log.flush();print(json.dumps(record),flush=True)
                if not math.isfinite(validation):
                    save_training(out/'nonfinite-validation-state.safetensors',model,optimizer)
                    raise RuntimeError('nonfinite validation')
                if index>start_step and not done:
                    save_training(out/'training-state.safetensors',model,optimizer)
                    checkpoint=dict(complete=False,steps=index,update_timings=timings,loss_scales=loss_scales,overflow_retries=overflow_retries,
                        elapsed_including_data_validation_export=old_summary.get('elapsed_including_data_validation_export',0.)+time.perf_counter()-PROCESS_START)
                    (out/'summary.json').write_text(json.dumps(checkpoint,indent=2))
            if done:break
            tick=time.perf_counter();rng=np.random.default_rng(a.seed*1_000_000+index)
            kind=all_modes[index%len(all_modes)];scale=mx.array(schedule(index,a.steps),dtype=mx.bfloat16)
            if kind=='text':
                ids=mx.array(rng.integers(len(tx),size=a.batch));inputs=[tx[ids],ty[ids]];feedback=False
            else:
                ids=mx.array(rng.integers(len(rows),size=a.program_batch))
                if kind=='streams':inputs=[rows[ids,:-1],rows[ids,1:]];feedback=False
                else:
                    times=rng.uniform(size=a.program_batch);times[rng.uniform(size=a.program_batch)<.25]=0.
                    clocks=mx.array(times,dtype=mx.bfloat16)
                    noise=mx.random.normal((a.program_batch,6,10),dtype=mx.bfloat16,key=mx.random.key(a.seed+2*index))
                    inputs=[clues[ids],answers[ids],flow_mask[ids],clocks,noise];feedback=index%16>=8
            key=kind+('_feedback' if feedback else '')
            def on_overflow(event):
                nonlocal overflow_retries
                overflow_retries+=1
                event.update(step=index+1,kind=key)
                with (out/'overflow.jsonl').open('a') as overflow_log:
                    overflow_log.write(json.dumps(event)+'\n')
                print(json.dumps(dict(overflow=event)),flush=True)
                if not (out/'first-overflow-state.safetensors').exists():
                    save_training(out/'first-overflow-state.safetensors',model,optimizer)
            update=get_step(kind,feedback)
            try:value,norm=update(*inputs,learning_rate_scale=scale,on_overflow=on_overflow)
            except FloatingPointError:
                save_training(out/'unrecoverable-state.safetensors',model,optimizer)
                raise
            loss_scales[key]=update.loss_scale
            duration=time.perf_counter()-tick;steps=index+1
            timing=timings.setdefault(key,dict(updates=0,seconds=0.,first_including_compile=duration,steady_seconds=0.))
            if timing['updates']:timing['steady_seconds']+=duration
            timing['seconds']+=duration;timing['updates']+=1
            last_values[kind]=dict(loss=float(value),gradient_norm=float(norm))
            if not math.isfinite(float(value)) or not math.isfinite(float(norm)):
                save_training(out/'nonfinite-state.safetensors',model,optimizer)
                print(json.dumps(dict(failed_step=steps,kind=kind,value=float(value),gradient_norm=float(norm))),flush=True)
                raise RuntimeError('nonfinite training')
    save_training(out/'training-state.safetensors',model,optimizer)
    complete=steps==a.steps
    ids=order[:a.validation_windows] if a.validation_windows else order
    validation=evaluate(ids) if complete else None
    programs_report={}
    if a.mode=='joint' and not a.skip_program_eval:
        programs_report['validation_streams']=evaluate_programs(model,monitor_programs,digit_ids,idle,end)
        programs_report['validation_flow']=evaluate_flow(model,monitor_programs,digit_ids,idle,steps=(1,4,8))
    export=None
    if complete and a.precision=='mxfp8':
        probe=mx.array(vx[monitor[:1]]);reference=model(probe)
        export=model.export(out/'release')
        (out/'release/tokenizer.json').write_bytes(Path(tokenizer_path).read_bytes())
        reloaded=MisulModel.load(out/'release');export['reload_exact']=bool(mx.array_equal(reference,reloaded(probe)).item())
        if not export['reload_exact']:raise RuntimeError('FP8 release reload mismatch')
    dtypes=sorted({str(value.dtype) for _,value in tree_flatten(model.parameters())+tree_flatten(optimizer.state)})
    if 'mlx.core.float32' in dtypes:raise RuntimeError('FP32 state appeared')
    summary=dict(complete=complete,steps=steps,configuration=config,parameters=metadata['parameters'],
        language_updates=sum(index%len(all_modes) in [i for i,k in enumerate(all_modes) if k=='text'] for index in range(steps)),
        update_timings=timings,loss_scales=loss_scales,overflow_retries=overflow_retries,session_elapsed=time.perf_counter()-PROCESS_START,
        elapsed_including_data_validation_export=old_summary.get('elapsed_including_data_validation_export',0.)+time.perf_counter()-PROCESS_START,
        final_validation_loss=validation,validation_windows=len(ids) if complete else 0,
        final_validation_nats_per_byte=validation*vy[ids].size/int(byte_lengths[vy[ids]].sum()) if complete else None,
        validation_target_bytes=int(byte_lengths[vy[ids]].sum()) if complete else 0,
        program_validation=programs_report,export=export,peak_mlx_bytes=mx.get_peak_memory(),state_dtypes=dtypes,
        training_state_sha256=digest(out/'training-state.safetensors'))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    metadata['complete']=complete;(out/'manifest.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
