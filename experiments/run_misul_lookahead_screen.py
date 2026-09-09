"""Sequentially launch the six registered guarded screen arms."""
import os
from pathlib import Path
import subprocess
import sys

env = os.environ | {'PYTHONPATH': 'src'}
for scale in (0, 1, 2):
    for auxiliary, label in ((0., 'control'), (.25, 'lookahead')):
        name = f'misul-lookahead-scale{scale}-{label}'
        command = [sys.executable, '-m', 'transformermodel.safe_run', '--report',
                   f'evidence/{name}-guard.json', '--seconds', str(300 if scale == 2 else 180),
                   '--', sys.executable, 'experiments/screen_misul_lookahead.py', '--scale', str(scale),
                   '--auxiliary-weight', str(auxiliary), '--output', f'artifacts/{name}']
        print(f'Starting {name}', flush=True)
        with Path(f'evidence/{name}.log').open('w') as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        print(f'Finished {name}: exit {result.returncode}', flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)
