"""Synchronized matched BF16 inference timings; no optimizer or adaptation."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import mlx.core as mx
from mlx.utils import tree_unflatten
import numpy as np
from tokenizers import Tokenizer
from transformermodel.misul import MisulModel
from transformermodel.misul_dense import DenseModel
from transformermodel.misul_flow import sample_flow
from transformermodel.safe_run import require_guard
from transformermodel.train import command_output

require_guard()
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--arm', choices=['misul','dense'],required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
mx.set_memory_limit(1000*2**20);mx.set_cache_limit(32*2**20)
root=Path('artifacts/misul-joint-checked-bf16-recovered' if a.arm=='misul' else 'artifacts/misul-dense-bf16')
summary=json.loads((root/'summary.json').read_text())
assert summary['complete']
constructor=MisulModel if a.arm=='misul' else DenseModel
model=constructor(**summary['configuration'])
weights=root/'training-state.safetensors'
state_hash=hashlib.sha256(weights.read_bytes()).hexdigest()
assert state_hash==summary['training_state_sha256']
arrays=mx.load(str(weights));model.update(tree_unflatten([(k[6:],v) for k,v in arrays.items() if k.startswith('model/')]))
mx.eval(model.parameters());del arrays;gc.collect();mx.clear_cache()
tokenizer=Tokenizer.from_file(str(root/'tokenizer.json'))
base=tokenizer.encode('The city was founded near the river and grew into a center of trade. ').ids
cases={}
for length in (32,128):
    tokens=mx.array([(base*((length+len(base)-1)//len(base)))[:length]])
    cases[f'text_forward_context{length}_batch1']=lambda tokens=tokens: model(tokens)
clues=mx.array([[tokenizer.encode(str(d)).ids[0] for d in (2,5,3,7,4,1)]]*8)
for steps in (1,4,8):
    cases[f'flow_length6_batch8_passes{steps}']=lambda steps=steps:sample_flow(model,clues,steps=steps,seed=1111,trace=False)[0]

def generate():
    tokens=tokenizer.encode('The city of').ids
    for _ in range(32):
        token=int(mx.argmax(model(mx.array([tokens[-128:]]))[0,-1]).item())
        tokens.append(token)
    return mx.array(tokens)
cases['uncached_greedy32_tokens_batch1']=generate
result={}
for name,function in cases.items():
    tick=time.perf_counter();mx.eval(function());mx.synchronize()
    first=time.perf_counter()-tick
    elapsed=[]
    for _ in range(9):
        mx.synchronize();tick=time.perf_counter();mx.eval(function());mx.synchronize()
        elapsed.append(time.perf_counter()-tick)
    result[name]=dict(first_seconds=first,repetitions=elapsed,median_seconds=float(np.median(elapsed)),
                     p10_seconds=float(np.quantile(elapsed,.1)),p90_seconds=float(np.quantile(elapsed,.9)))
assert hashlib.sha256(weights.read_bytes()).hexdigest()==state_hash
report=dict(arm=a.arm,configuration=model.config,parameters=summary['parameters'],precision='BF16, same as architecture quality comparison',
 checkpoint_sha256=state_hash,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 device=mx.device_info(),macos=platform.mac_ver(),mlx=importlib.metadata.version('mlx'),
 power=command_output(['pmset','-g','batt']),thermal=command_output(['pmset','-g','therm']),
 cases=result,peak_mlx_bytes=mx.get_peak_memory(),checkpoint_unchanged=True,
 limitations='Single host, nine repetitions, no KV cache in either arm, no distributed or other accelerator measurements.')
Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps(report))
