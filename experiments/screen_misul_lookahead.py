"""Run one preregistered local lookahead arm; official test data remain sealed."""
import time
START = time.perf_counter()
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np
from transformermodel.misul_mtp import MisulMTP, lookahead_loss
from transformermodel.misul_muon import Muon16
from transformermodel.misul_update import CheckedUpdate
from transformermodel.train_misul import schedule, digest
from transformermodel.text_data import token_windows
from transformermodel.safe_run import require_guard
from transformermodel.train import command_output

require_guard()
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--scale', type=int, choices=(0, 1, 2), required=True)
p.add_argument('--auxiliary-weight', type=float, choices=(0., .25), required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
out = Path(a.output)
out.mkdir(parents=True, exist_ok=False)
protocol_path = Path('evidence/misul-lookahead-protocol.json')
protocol = json.loads(protocol_path.read_text())
config = protocol['scales'][a.scale] | dict(vocab_size=4096, loops=2, precision='mxfp8', memory_backend='metal')
cap = 300 if a.scale == 2 else 180
mx.set_memory_limit(1000 * 2**20)
mx.set_cache_limit(32 * 2**20)
data_path = 'data/wikitext2-v1/prepared/manifest.json'
tokenizer_path = 'data/misul-v2/tokenizer.json'
tx, ty, tokenizer, _ = token_windows(data_path, tokenizer_path, 'train', config['context'])
vx, vy, _, _ = token_windows(data_path, tokenizer_path, 'validation', config['context'])
assert tokenizer.get_vocab_size() == config['vocab_size']
order = np.random.default_rng(1054).permutation(len(vx))
tx, ty = mx.array(tx), mx.array(ty)
mx.random.seed(1053)
model = MisulMTP(**config)
optimizer = Muon16(model.parameters())
mx.eval(model.parameters(), optimizer.state, tx, ty)
source_files = [Path(__file__)] + [Path('src/transformermodel') / name for name in
    ('misul.py', 'misul_mtp.py', 'misul_muon.py', 'misul_update.py', 'misul_memory_metal.py',
     'combined.py', 'budget_muon.py', 'text_data.py', 'train_misul.py')]
manifest = dict(configuration=model.config, auxiliary_weight=a.auxiliary_weight,
                protocol_sha256=digest(protocol_path), source_sha256={str(p): digest(p) for p in source_files},
                data_sha256=digest(data_path), tokenizer_sha256=digest(tokenizer_path),
                parameters=sum(v.size for _, v in tree_flatten(model.parameters())),
                working_parameter_bytes=sum(v.nbytes for _, v in tree_flatten(model.parameters())),
                optimizer_bytes=sum(v.nbytes for _, v in tree_flatten(optimizer.state)),
                device=mx.device_info(), mlx=importlib.metadata.version('mlx'), macos=platform.mac_ver(),
                command=[sys.executable, *sys.argv], power=command_output(['pmset', '-g', 'batt']),
                thermal=command_output(['pmset', '-g', 'therm']))
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
(out / 'source-snapshot.json').write_text(json.dumps({str(p): p.read_text() for p in source_files}))
(out / 'tokenizer.json').write_bytes(Path(tokenizer_path).read_bytes())
update = CheckedUpdate(model, optimizer, lambda m, x, y: lookahead_loss(m, x, y, a.auxiliary_weight))

def evaluate(ids):
    total = count = 0
    for start in range(0, len(ids), 4):
        part = ids[start:start + 4]
        losses = nn.losses.cross_entropy(model(mx.array(vx[part])).astype(mx.float16),
                                         mx.array(vy[part]), reduction='none')
        values = np.array(losses.tolist(), dtype=np.float64)
        total += values.sum()
        count += values.size
    return float(total / count)

records = []
events = []
steps = 0
crossing = None
with (out / 'metrics.jsonl').open('w') as log:
    for index in range(385):
        done = index == 384 or time.perf_counter() - START >= cap - 15
        if index % 64 == 0 or done:
            loss = evaluate(order[:64])
            row = dict(step=index, elapsed_including_data_and_validation=time.perf_counter() - START,
                       validation_loss=loss, loss_scale=update.loss_scale)
            assert math.isfinite(loss)
            if crossing is None and loss <= 6.5:
                crossing = row.copy()
            records.append(row)
            log.write(json.dumps(row) + '\n')
            log.flush()
            print(json.dumps(row), flush=True)
        if done:
            break
        rng = np.random.default_rng(1053000000 + index)
        ids = mx.array(rng.integers(len(tx), size=4))
        def overflow(event):
            event['step'] = index + 1
            events.append(event)
            print(json.dumps(dict(overflow=event)), flush=True)
        value, norm = update(tx[ids], ty[ids], learning_rate_scale=mx.array(schedule(index, 384), dtype=mx.bfloat16),
                             on_overflow=overflow)
        steps = index + 1
final_loss = evaluate(order[:256])
export = model.export(out / 'release')
(out / 'release/tokenizer.json').write_bytes(Path(tokenizer_path).read_bytes())
summary = dict(complete=steps == 384, steps=steps, parameters=manifest['parameters'],
               final_validation_loss=final_loss, first_observed_target_crossing=crossing,
               elapsed_including_data_validation_export=time.perf_counter() - START,
               records=records, overflow_events=events, loss_scale=update.loss_scale,
               peak_mlx_bytes=mx.get_peak_memory(), export=export)
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary), flush=True)
