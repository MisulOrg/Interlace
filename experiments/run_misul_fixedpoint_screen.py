"""Run the registered validation-only feedback comparison sequentially."""
import json
import os
from pathlib import Path
import subprocess
import sys

test_guard = json.loads(Path('evidence/misul-fixedpoint-tests-guard.json').read_text())
if test_guard['returncode'] or test_guard['stop_reason'] is not None:
    raise SystemExit('qualify the detached trajectory before learning')
env = os.environ | {'PYTHONPATH': 'src'}
runs = {}
for arm in ('standard', 'fixedpoint'):
    name = f'misul-fixedpoint-{arm}'
    out = Path('artifacts') / name
    stem = Path('evidence') / name
    if out.exists() or stem.with_suffix('.log').exists():
        raise FileExistsError('preserve the existing screen before selecting an explicitly named retry')
    command = [sys.executable, '-m', 'transformermodel.safe_run',
               '--report', str(stem) + '-guard.json', '--seconds', '600', '--',
               sys.executable, 'experiments/screen_misul_fixedpoint.py',
               '--arm', arm, '--output', str(out)]
    print(f'Starting {name}', flush=True)
    with stem.with_suffix('.log').open('w') as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise SystemExit(result.returncode)
    summary = json.loads((out / 'summary.json').read_text())
    if not summary['complete']:
        raise SystemExit('incomplete feedback screen; keep the original default model')
    runs[arm] = dict(summary=summary,
                    guard=json.loads(Path(str(stem) + '-guard.json').read_text()),
                    path=str(out))
    print(f'Finished {name}: {summary["steps"]} updates', flush=True)
standard, fixedpoint = [runs[k]['summary'] for k in ('standard', 'fixedpoint')]
for key in ('initial_model_sha256', 'source_sha256', 'protocol_sha256', 'parameters',
            'configuration', 'data_sha256', 'tokenizer_sha256', 'validation_program_sha256',
            'seed', 'fixed_updates'):
    assert standard[key] == fixedpoint[key], key
assert standard['metrics'][0]['flow_exact'] == fixedpoint['metrics'][0]['flow_exact']
assert abs(standard['metrics'][0]['monitor_loss'] - fixedpoint['metrics'][0]['monitor_loss']) < 1e-7
gain = fixedpoint['flow']['8']['exact_accuracy'] - standard['flow']['8']['exact_accuracy']
passes = (fixedpoint['flow']['8']['exact_accuracy'] >= .8 and gain >= .1
          and fixedpoint['relative_language_loss_change'] <= .01)
comparison = dict(runs=runs, exact_accuracy_gain_over_standard=gain,
                  usefulness_threshold=.8, improvement_threshold=.1,
                  language_loss_change_threshold=.01, passed=passes,
                  default_model_changed=False,
                  next_gate='Matched BF16 continuation and the existing precision gate are required before promotion.' if passes
                  else 'Retain this negative screen and the original jointly trained model.',
                  scope='One local initialization, fixed data and budget; validation-only comparison.')
Path('evidence/misul-fixedpoint-comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
print(json.dumps({k: v for k, v in comparison.items() if k != 'runs'}, indent=2))
