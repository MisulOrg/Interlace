"""Run the bundled model through its local resource guard."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('mode', choices=('text', 'streams', 'flow'))
args, options = parser.parse_known_args()
command = [sys.executable, '-m', 'transformermodel.safe_run', '--report',
           str(root / 'evidence' / ('demo-' + uuid.uuid4().hex[:12] + '.json')),
           '--seconds', '60', '--', sys.executable, '-m', 'transformermodel.misul_demo',
           '--model', str(root / 'model'), '--mode', args.mode, *options]
result = subprocess.run(command, cwd=root, env=os.environ | {'PYTHONPATH': str(root / 'src')},
                        capture_output=True, text=True)
lines = result.stdout.splitlines()
if lines:
    try:
        if 'sampled_peak_phys_bytes' in json.loads(lines[-1]):
            lines.pop()
    except (json.JSONDecodeError, TypeError):
        pass
if args.mode == 'streams' and result.returncode == 0:
    print(f"{'tick':>4}  {'input':>7}  {'thought':>7}  {'output':>7}")
    for line in lines:
        row = json.loads(line)
        values = [('-' if row[k] == '<|idle|>' else 'END' if row[k] == '<|end-input|>' else row[k])
                  for k in ('input', 'thought', 'output')]
        print(f"{row['tick']:4d}  {values[0]:>7}  {values[1]:>7}  {values[2]:>7}")
else:
    print('\n'.join(lines))
if result.stderr:
    sys.stderr.write(result.stderr)
raise SystemExit(result.returncode)
