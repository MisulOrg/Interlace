"""Sealed text/program evaluation of a completed shared Misul checkpoint."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np
from tokenizers import Tokenizer
from transformermodel.misul import MisulModel
from transformermodel.misul_flow import evaluate_flow
from transformermodel.safe_run import require_guard
from transformermodel.stream_program import test_programs, evaluate_programs
from transformermodel.text_data import token_windows


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_saved_state(arrays):
    dtypes = sorted({str(v.dtype) for v in arrays.values()})
    if mx.float32 in {v.dtype for v in arrays.values()}:
        raise ValueError('FP32 model or optimizer state appeared')
    if not all(bool(mx.all(mx.isfinite(v))) for v in arrays.values()):
        raise ValueError('nonfinite saved model or optimizer state')
    return dict(all_finite=True, dtypes=dtypes, tensors=len(arrays),
                model_bytes=sum(v.nbytes for k, v in arrays.items() if k.startswith('model/')),
                optimizer_bytes=sum(v.nbytes for k, v in arrays.items() if not k.startswith('model/')))


def text_sample(model, tokenizer, prompt, temperature, seed=1047, tokens=64):
    """Generate the registered illustration without a task-specific prompt."""
    sequence = tokenizer.encode(prompt).ids
    continuation = []
    controls = [tokenizer.token_to_id(name) for name in ('<|idle|>', '<|end-input|>')]
    for index in range(tokens):
        logits = model(mx.array([sequence[-model.config['context']:]], dtype=mx.int32))[0, -1].astype(mx.float16)
        for control in controls:
            if control is not None:
                logits[control] = -float('inf')
        token = int(mx.argmax(logits)) if temperature == 0 else int(mx.random.categorical(
            logits / temperature, key=mx.random.key(seed + index)))
        sequence.append(token)
        continuation.append(token)
    return dict(prompt=prompt, temperature=temperature, seed=seed,
                tokens=tokens, continuation=tokenizer.decode(continuation),
                token_ids=continuation)


def main():
    require_guard()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        parser.error('preserve the existing evaluation; choose a new output')
    started = time.perf_counter()
    mx.set_memory_limit(1000 * 2**20)
    mx.set_cache_limit(32 * 2**20)
    run = Path(args.run)
    summary = json.loads((run / 'summary.json').read_text())
    manifest = json.loads((run / 'manifest.json').read_text())
    if not summary['complete'] or not manifest['complete']:
        parser.error('finish and freeze this training run before opening tests')
    for path, expected in manifest['source_sha256'].items():
        if digest(path) != expected:
            raise ValueError(f'trained source changed: {path}')
    amendment_path = run / 'numerical-amendment.json'
    amendment = json.loads(amendment_path.read_text()) if amendment_path.exists() else None
    if amendment:
        for path, expected in amendment['amendment_source_sha256'].items():
            if digest(path) != expected:
                raise ValueError(f'numerical amendment source changed: {path}')
    config = summary['configuration']
    weights = run / 'training-state.safetensors'
    if digest(weights) != summary['training_state_sha256']:
        raise ValueError('training checkpoint digest changed')
    if config['precision'] == 'mxfp8':
        model = MisulModel.load(run / 'release')
        inference_weights = run / 'release/weights.safetensors'
    else:
        model = MisulModel(**config)
        arrays = mx.load(str(weights))
        saved_state = check_saved_state(arrays)
        model.update(tree_unflatten([(k[6:], v) for k, v in arrays.items()
                                     if k.startswith('model/')]))
        mx.eval(model.parameters())
        del arrays
        gc.collect()
        mx.clear_cache()
        inference_weights = weights
    tokenizer_path = run / 'tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    idle = tokenizer.token_to_id('<|idle|>')
    end = tokenizer.token_to_id('<|end-input|>')
    encoded = [tokenizer.encode(str(i)).ids for i in range(10)]
    if any(len(ids) != 1 for ids in encoded) or idle is None or end is None:
        raise ValueError('tokenizer does not meet the stream contract')
    digit_ids = [ids[0] for ids in encoded]
    model_hash_before = digest(inference_weights)
    reload_parity = None
    if config['precision'] == 'mxfp8':
        reference = MisulModel(**config)
        original = mx.load(str(weights))
        saved_state = check_saved_state(original)
        reference.update(tree_unflatten([(k[6:], v) for k, v in original.items()
                                         if k.startswith('model/')]))
        mx.eval(reference.parameters())
        del original
        gc.collect()
        text_probe = mx.array([tokenizer.encode('The city of').ids], dtype=mx.int32)
        stream_probe = np.full((1, 7, 3), idle, dtype=np.int32)
        stream_probe[0, 1:5, 0] = [digit_ids[d] for d in (2, 5, 3, 7)]
        stream_probe[0, 5, 0] = end
        stream_probe = mx.array(stream_probe)
        clue_probe = mx.array([[digit_ids[d] for d in (2, 5, 3, 7)]], dtype=mx.int32)
        noise_probe = mx.random.normal((1, 4, 10), dtype=mx.bfloat16, key=mx.random.key(1049))
        clock_probe = mx.array([.375], dtype=mx.bfloat16)
        feedback_probe = mx.zeros_like(noise_probe)
        first = reference.flow(clue_probe, noise_probe, clock_probe, feedback_probe)
        reload_parity = {
            'text': bool(mx.array_equal(reference(text_probe), model(text_probe))),
            'streams': bool(mx.array_equal(reference(stream_probe), model(stream_probe))),
            'flow_without_feedback': bool(mx.array_equal(first, model.flow(
                clue_probe, noise_probe, clock_probe, feedback_probe))),
            'flow_with_feedback': bool(mx.array_equal(
                reference.flow(clue_probe, noise_probe, clock_probe, first),
                model.flow(clue_probe, noise_probe, clock_probe, first))),
        }
        if not all(reload_parity.values()):
            raise ValueError(f'final trained-state reload differs: {reload_parity}')
        del reference, first
        gc.collect()
        mx.clear_cache()
    # Seed and exclusions were fixed before tests were opened. These sequences
    # exclude the prior model's exposed seeds 934/956 and SHA256 buckets 0-8.
    held, longer = test_programs(1081)
    for group in (held, longer):
        if len(set(group)) != len(group):
            raise ValueError('duplicate final test sequence')
        if any(int.from_bytes(hashlib.sha256(bytes(d)).digest()[:4], 'little') % 10 != 9
               for d in group):
            raise ValueError('final program split overlaps development')
    report = dict(run=str(run.resolve()), configuration=config,
                  parameters=summary['parameters'], training_steps=summary['steps'],
                  training_state_sha256=summary['training_state_sha256'],
                  inference_weights_sha256=model_hash_before,
                  packed_reload_parity=reload_parity,
                  saved_training_state=saved_state,
                  tokenizer_sha256=digest(tokenizer_path),
                  evaluation_source_sha256=digest(__file__),
                  test_seed=1081, numerical_amendment=amendment,
                  test_program_sha256=hashlib.sha256(json.dumps([held, longer]).encode()).hexdigest(),
                  final_validation_loss=summary['final_validation_loss'],
                  final_validation_nats_per_byte=summary['final_validation_nats_per_byte'])
    test_manifest = 'data/misul-test/manifest.json'
    report['text_test_manifest_sha256'] = digest(test_manifest)
    report['text_test_provenance'] = {k: v for k, v in json.loads(Path(test_manifest).read_text()).items()
                                      if k != 'records'}
    x, y, _, byte_lengths = token_windows(test_manifest,
                                         tokenizer_path, 'test', config['context'])
    window_sums = []
    for offset in range(0, len(x), 4):
        loss = nn.losses.cross_entropy(model(mx.array(x[offset:offset + 4])).astype(mx.float16),
                                       mx.array(y[offset:offset + 4]), reduction='none')
        values = np.array(loss.tolist(), dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError('nonfinite official test loss')
        window_sums.extend(values.sum(axis=1).tolist())
    total_nats = float(sum(window_sums))
    target_bytes = int(byte_lengths[y].sum())
    report['text_test'] = dict(windows=len(x), target_tokens=int(y.size),
                               target_bytes=target_bytes, total_nats=total_nats,
                               loss=total_nats / y.size,
                               nats_per_byte=total_nats / target_bytes,
                               bits_per_byte=total_nats / target_bytes / np.log(2),
                               window_total_nats=window_sums)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Preserve incremental results if a guard stops a later, independent check.
    destination.write_text(json.dumps(report | {'complete': False}, indent=2))
    for name, programs in [('held_out', held), ('longer_lengths', longer)]:
        report[name] = dict(programs=[list(d) for d in programs],
                            streams=evaluate_programs(model, programs, digit_ids, idle, end,
                                                      include_failures=True),
                            flow=evaluate_flow(model, programs, digit_ids, idle,
                                               steps=(1, 2, 4, 8), seed=1043),
                            flow_without_feedback=evaluate_flow(model, programs, digit_ids, idle,
                                                                steps=(8,), seed=1043,
                                                                feedback_on=False))
        destination.write_text(json.dumps(report | {'complete': False}, indent=2))
        print(json.dumps(dict(group=name,
                              stream_answer_accuracy=report[name]['streams']['answer_accuracy'],
                              flow_exact={k: v['exact_accuracy'] for k, v in report[name]['flow'].items()})),
              flush=True)
    report['held_out']['streams_with_idle_thought'] = evaluate_programs(
        model, held, digit_ids, idle, end, thought_override=idle, include_failures=True)
    report['text_samples'] = [text_sample(model, tokenizer, prompt, temperature)
                              for prompt, temperature in (('The city of', 0),
                                                           ('In the early', .7),
                                                           ('The history of science', .7))]
    report.update(complete=True, elapsed_seconds=time.perf_counter() - started,
                  peak_mlx_bytes=mx.get_peak_memory(),
                  model_working_dtypes=sorted({str(v.dtype) for _, v in tree_flatten(model.parameters())}))
    if digest(inference_weights) != model_hash_before:
        raise ValueError('inference modified the checkpoint')
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ('text_test', 'held_out', 'longer_lengths')}), flush=True)


if __name__ == '__main__':
    main()
