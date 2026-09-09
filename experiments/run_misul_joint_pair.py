"""Complete the registered precision pair in sequential guarded segments."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from transformermodel.safe_run import memory_free_percent

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--precision', choices=('mxfp8', 'bf16', 'both'), default='both')
a = p.parse_args()
protocol = Path('evidence/misul-joint-checked-protocol.json')
if not protocol.exists():
    raise SystemExit('register the corrected recipe before running the pair')
env = os.environ | {'PYTHONPATH': 'src'}
for precision in (('mxfp8', 'bf16') if a.precision == 'both' else (a.precision,)):
    label = 'fp8' if precision == 'mxfp8' else 'bf16'
    name = f'misul-joint-checked-{label}'
    out = Path('artifacts') / name
    for _ in range(12):
        summary_path = out / 'summary.json'
        previous = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        if previous.get('complete'):
            print(f'Completed {name}: {previous["steps"]} updates', flush=True)
            break
        step = previous.get('steps', 0)
        target = 512 if step == 0 else min(16384, ((step // 4096) + 1) * 4096)
        # Leave room for the measured 1.25 GB child before starting. The guard
        # still enforces the unchanged 40% headroom throughout execution.
        started_wait = time.monotonic()
        while memory_free_percent() < 50:
            if time.monotonic() - started_wait >= 600:
                raise SystemExit('headroom did not recover within ten minutes; checkpoints preserved')
            print(f'Waiting for 50% preflight memory headroom before {name}', flush=True)
            time.sleep(30)
        with Path('evidence/misul-joint-waits.jsonl').open('a') as waits:
            waits.write(json.dumps(dict(run=name, step=step,
                                        seconds=time.monotonic() - started_wait)) + '\n')
        segment = 1
        while Path(f'evidence/{name}-segment{segment}.log').exists():
            segment += 1
        stem = f'evidence/{name}-segment{segment}'
        command = [sys.executable, '-m', 'transformermodel.safe_run', '--report', stem + '-guard.json',
                   '--seconds', '1200', '--', sys.executable, '-m', 'transformermodel.train_misul',
                   '--precision', precision, '--output', str(out), '--steps', '16384', '--seconds', '1100',
                   '--stop-after', str(target), '--eval-every', '512']
        if (out / 'training-state.safetensors').exists():
            command.append('--resume')
        print(f'Starting {name}, segment {segment}: update {step} toward {target}', flush=True)
        with Path(stem + '.log').open('w') as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        if summary_path.exists():
            Path(stem + '-summary.json').write_bytes(summary_path.read_bytes())
        print(f'Finished {name}, segment {segment}: exit {result.returncode}', flush=True)
        if result.returncode:
            guard_path = Path(stem + '-guard.json')
            if guard_path.exists() and json.loads(guard_path.read_text()).get('stop_reason') == 'system memory headroom':
                print('Memory guard preserved the last checkpoint; retry after headroom recovers', flush=True)
                time.sleep(30)
                continue
            raise SystemExit(result.returncode)
    else:
        raise SystemExit('bounded segment count exhausted; preserve state for inspection')
