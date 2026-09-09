"""Reproduce and recover the preserved real Flow overflow before resuming."""
import hashlib
import json
from pathlib import Path
import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np
from tokenizers import Tokenizer
from transformermodel.misul import MisulModel
from transformermodel.misul_muon import Muon16
from transformermodel.misul_fractional_update import FractionalCheckedUpdate
from transformermodel.misul_flow import program_batch, flow_loss
from transformermodel.stream_program import make_programs
from transformermodel.train_misul import load_training, schedule
from transformermodel.safe_run import require_guard

require_guard()
mx.set_memory_limit(1000 * 2**20)
mx.set_cache_limit(32 * 2**20)
root = Path('artifacts/misul-joint-checked-bf16')
source = root / 'unrecoverable-state.safetensors'
model = MisulModel(**json.loads((root / 'manifest.json').read_text())['configuration'])
optimizer = Muon16(model.parameters())
index = load_training(source, model, optimizer)
tokenizer = Tokenizer.from_file(str(root / 'tokenizer.json'))
digits = [tokenizer.encode(str(i)).ids[0] for i in range(10)]
programs = make_programs(1009, 8192, 2, 6, set(range(8)))
clues, answers, mask = program_batch(programs, digits, tokenizer.token_to_id('<|idle|>'), 6)
rng = np.random.default_rng(1001000000 + index)
ids = mx.array(rng.integers(len(programs), size=16))
times = rng.uniform(size=16)
times[rng.uniform(size=16) < .25] = 0.
inputs = (clues[ids], answers[ids], mask[ids], mx.array(times, dtype=mx.bfloat16),
          mx.random.normal((16, 6, 10), dtype=mx.bfloat16, key=mx.random.key(1001 + 2 * index)))
loss = lambda m, *x: flow_loss(m, *x, index % 16 >= 8)
update = FractionalCheckedUpdate(model, optimizer, loss, 1)
before = tree_flatten([model.parameters(), optimizer.state])
events = []
def observe(event):
    now = tree_flatten([model.parameters(), optimizer.state])
    assert all(bool(mx.array_equal(a, b)) for (_, a), (_, b) in zip(before, now))
    assert int(optimizer.state['step']) == index
    events.append(event)
scale = mx.array(schedule(index, 16384), dtype=mx.bfloat16)
value, norm = update(*inputs, learning_rate_scale=scale, on_overflow=observe)
assert [event['loss_scale'] for event in events] == [1]
assert update.loss_scale == .5 and int(optimizer.state['step']) == index + 1
del before
actual = tree_flatten([model.parameters(), optimizer.state])
assert all(bool(mx.all(mx.isfinite(v))) for _, v in actual)
assert not any(v.dtype == mx.float32 for _, v in actual)
load_training(source, model, optimizer)
direct = FractionalCheckedUpdate(model, optimizer, loss, .5)
direct_value, direct_norm = direct(*inputs, learning_rate_scale=scale)
expected = tree_flatten([model.parameters(), optimizer.state])
differences = [dict(actual_name=ka, expected_name=kb,
                    unequal=int(mx.sum(a != b)), maximum_error=float(mx.max(mx.abs(a - b))))
               for (ka, a), (kb, b) in zip(actual, expected) if not bool(mx.array_equal(a, b))]
print(json.dumps(dict(differences=differences, value=float(value), norm=float(norm),
                      direct_value=float(direct_value), direct_norm=float(direct_norm))), flush=True)

# A separate direct-versus-direct witness tests whether strict state equality
# is available from the unchanged MLX backward path on this real batch.
load_training(source, model, optimizer)
direct(*inputs, learning_rate_scale=scale)
repeated = tree_flatten([model.parameters(), optimizer.state])
repeat_differences = [dict(name=k, unequal=int(mx.sum(a != b)), maximum_error=float(mx.max(mx.abs(a - b))))
                      for (k, a), (_, b) in zip(expected, repeated) if not bool(mx.array_equal(a, b))]
model_equal = all(bool(mx.array_equal(a, b)) for (k, a), (_, b) in zip(actual, expected) if k.startswith('0.'))
nonembedding_equal = all(bool(mx.array_equal(a, b)) for (k, a), (_, b) in zip(actual, expected)
                         if not k.startswith('1.leaves.embedding.'))
assert model_equal and nonembedding_equal
assert repeat_differences and all(d['name'].startswith('1.leaves.embedding.') for d in repeat_differences)
report = dict(numerical_recovery_supported=True, strict_full_state_parity_passed=not differences,
              model_update_bit_exact=model_equal, nonembedding_optimizer_bit_exact=nonembedding_equal,
              direct_control_also_not_bit_exact=bool(repeat_differences),
              preupdate_step=index, accepted_scale=update.loss_scale, loss=float(value), gradient_norm=float(norm),
              state_unchanged_on_overflow=True, all_state_finite=True, no_fp32_state=True,
              retry_vs_direct=differences, direct_vs_direct=repeat_differences,
              diagnosis='Differences are confined to the embedding optimizer state and also arise between two direct updates. Accumulation-order variation in the unchanged embedding backward is plausible; strict full-state replay remains unqualified.',
              source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
Path('evidence/misul-fractional-repeat-qualification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
