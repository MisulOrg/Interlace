"""Compare ordinary and trajectory-feedback continuation on validation only."""
import time
START = time.perf_counter()
import argparse
import hashlib
import json
from pathlib import Path
import math
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from transformermodel.misul import MisulModel
from transformermodel.misul_fixedpoint import trajectory_feedback, loss_with_feedback
from transformermodel.misul_flow import program_batch, evaluate_flow
from transformermodel.misul_muon import Muon16
from transformermodel.misul_update import CheckedUpdate
from transformermodel.safe_run import require_guard
from transformermodel.stream_program import make_programs, program_rows, evaluate_programs
from transformermodel.text_data import token_windows


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    require_guard()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('standard', 'fixedpoint'), required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    protocol = Path('evidence/misul-fixedpoint-protocol.json')
    if not protocol.exists():
        parser.error('register the experiment first')
    base = Path('artifacts/misul-joint-checked-fp8')
    prior = json.loads((base / 'summary.json').read_text())
    manifest = json.loads((base / 'manifest.json').read_text())
    if not prior['complete'] or not manifest['complete'] or prior['steps'] != 16384:
        parser.error('finish and freeze the registered initial model first')
    for path, expected in manifest['source_sha256'].items():
        if digest(path) != expected:
            raise ValueError(f'trained source changed: {path}')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    mx.set_memory_limit(1000 * 2**20)
    mx.set_cache_limit(32 * 2**20)
    mx.random.seed(1063)
    model = MisulModel.load(base / 'release')
    optimizer = Muon16(model.parameters(), learning_rate=.001, aux_learning_rate=.0002)
    mx.eval(model.parameters(), optimizer.state)
    tokenizer_path = base / 'release/tokenizer.json'
    tx, ty, tokenizer, _ = token_windows('data/wikitext2-v1/prepared/manifest.json',
                                        tokenizer_path, 'train', 128)
    vx, vy, _, _ = token_windows('data/wikitext2-v1/prepared/manifest.json',
                                 tokenizer_path, 'validation', 128)
    monitor = np.random.default_rng(1003).permutation(len(vx))[:32]
    idle = tokenizer.token_to_id('<|idle|>')
    end = tokenizer.token_to_id('<|end-input|>')
    digit_ids = [tokenizer.encode(str(i)).ids[0] for i in range(10)]
    programs = make_programs(1009, 8192, 2, 6, set(range(8)))
    validation = make_programs(1061, 256, 2, 6, {8})
    if hashlib.sha256(json.dumps(programs).encode()).hexdigest() != manifest['program_sha256']:
        raise ValueError('training programs changed')
    rows = mx.array(np.stack([program_rows(d, digit_ids, idle, end, 9) for d in programs]))
    clues, targets, mask = program_batch(programs, digit_ids, idle, 6)
    tx, ty = mx.array(tx), mx.array(ty)
    mx.eval(tx, ty, rows, clues, targets, mask)

    def language(ids):
        values = []
        for start in range(0, len(ids), 4):
            part = ids[start:start + 4]
            loss = nn.losses.cross_entropy(model(mx.array(vx[part])).astype(mx.float16),
                                           mx.array(vy[part]), reduction='none')
            values.extend(np.asarray(loss.tolist(), dtype=np.float64).reshape(-1))
        result = float(np.mean(values))
        if not math.isfinite(result):
            raise FloatingPointError('nonfinite validation')
        return result

    def text_loss(m, x, y):
        return nn.losses.cross_entropy(m(x).astype(mx.float16), y, reduction='mean')

    def stream_loss(m, x, y):
        logits, target = m(x)[:, :, 1:], y[:, :, 1:]
        weights = mx.where(target != idle, mx.array(5., dtype=mx.float16),
                           mx.array(1., dtype=mx.float16))
        return mx.sum(nn.losses.cross_entropy(logits.astype(mx.float16), target) * weights) / mx.sum(weights)

    updates = {'text': CheckedUpdate(model, optimizer, text_loss, 128),
               'streams': CheckedUpdate(model, optimizer, stream_loss, 128),
               'flow': CheckedUpdate(model, optimizer, loss_with_feedback, 16)}
    sources = dict(manifest['source_sha256'])
    for path in (Path(__file__), Path('src/transformermodel/misul_fixedpoint.py'), protocol):
        sources[str(path)] = digest(path)
    report = dict(arm=args.arm, initial_model_sha256=digest(base / 'release/weights.safetensors'),
                  initial_validation_loss=prior['final_validation_loss'],
                  parameters=prior['parameters'], configuration=model.config,
                  source_sha256=sources, protocol_sha256=digest(protocol),
                  validation_program_sha256=hashlib.sha256(json.dumps(validation).encode()).hexdigest(),
                  data_sha256=manifest['data_sha256'], tokenizer_sha256=digest(tokenizer_path),
                  seed=1063, fixed_updates=1536, metrics=[], overflows=[], complete=False)
    def save_report():
        (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')

    def monitor_at(step):
        flow = evaluate_flow(model, validation, digit_ids, idle, steps=(8,), seed=1067)['8']
        monitor_loss = language(monitor)
        row = dict(step=step, elapsed_seconds=time.perf_counter() - START,
                   monitor_loss=monitor_loss, flow_exact=flow['exact_accuracy'],
                   flow_state=flow['state_accuracy'], flow_answer=flow['answer_accuracy'])
        report['metrics'].append(row)
        save_report()
        print(json.dumps(row), flush=True)

    monitor_at(0)
    steps = 0
    cycle = ('flow', 'text', 'flow', 'flow', 'streams', 'flow')
    for index in range(1536):
        if time.perf_counter() - START > 540:
            break
        rng = np.random.default_rng(1063000000 + index)
        kind = cycle[index % 6]
        warm = max(.05, min(1., (index + 1) / 32))
        decay = min(1., max(.1, 1 - .9 * (index - int(.8 * 1536)) / (1536 - int(.8 * 1536))))
        rate = mx.array(warm * decay, dtype=mx.bfloat16)
        if kind == 'text':
            ids = mx.array(rng.integers(len(tx), size=4))
            inputs = [tx[ids], ty[ids]]
        else:
            ids = mx.array(rng.integers(len(rows), size=16))
            if kind == 'streams':
                inputs = [rows[ids, :-1], rows[ids, 1:]]
            else:
                times = rng.uniform(size=16)
                times[rng.uniform(size=16) < .25] = 0.
                clocks = mx.array(times, dtype=mx.bfloat16)
                start_fraction = mx.array(rng.uniform(size=16), dtype=mx.bfloat16)
                feedback_on = bool(rng.uniform() < .5)
                noise = mx.random.normal((16, 6, 10), dtype=mx.bfloat16,
                                         key=mx.random.key(1065 + 2 * index))
                feedback = mx.zeros_like(noise)
                if feedback_on:
                    if args.arm == 'fixedpoint':
                        feedback = trajectory_feedback(model, clues[ids], targets[ids],
                                                        clocks, noise, start_fraction, depth=4)
                    else:
                        t = clocks[:, None, None]
                        noisy = (1 - t) * noise + t * mx.eye(10, dtype=mx.bfloat16)[targets[ids]]
                        feedback = mx.stop_gradient(model.flow(clues[ids], noisy, clocks, feedback))
                        mx.eval(feedback)
                inputs = [clues[ids], targets[ids], mask[ids], clocks, noise, feedback]
        def overflow(event):
            report['overflows'].append(event | dict(step=index + 1, kind=kind))
            save_report()
            print(json.dumps(dict(overflow=report['overflows'][-1])), flush=True)
        try:
            updates[kind](*inputs, learning_rate_scale=rate, on_overflow=overflow)
        except FloatingPointError:
            mx.save_safetensors(str(out / 'unrecoverable-model.safetensors'),
                                dict(tree_flatten(model.parameters())))
            raise
        steps = index + 1
        if steps % 256 == 0:
            monitor_at(steps)
    validation_loss = language(np.arange(len(vx)))
    streams = evaluate_programs(model, validation, digit_ids, idle, end, include_failures=True)
    flow = evaluate_flow(model, validation, digit_ids, idle, steps=(1, 2, 4, 8), seed=1067)
    probe = mx.array(vx[monitor[:1]])
    reference = model(probe)
    storage = model.export(out / 'release')
    (out / 'release/tokenizer.json').write_bytes(tokenizer_path.read_bytes())
    reloaded = MisulModel.load(out / 'release')
    if not bool(mx.array_equal(reference, reloaded(probe))):
        raise ValueError('packed continuation reload differs')
    report.update(steps=steps, complete=steps == 1536, final_validation_loss=validation_loss,
                  relative_language_loss_change=validation_loss / prior['final_validation_loss'] - 1,
                  streams=streams, flow=flow, storage=storage, reload_exact=True,
                  elapsed_seconds=time.perf_counter() - START, peak_mlx_bytes=mx.get_peak_memory(),
                  loss_scales={k: v.loss_scale for k, v in updates.items()})
    save_report()
    print(json.dumps({k: report[k] for k in ('arm', 'steps', 'complete', 'final_validation_loss',
                                            'relative_language_loss_change', 'elapsed_seconds')}), flush=True)


if __name__ == '__main__':
    main()
