"""Run the frozen dense comparison sequentially under unchanged resource caps."""
import json
import os
from pathlib import Path
import subprocess
import sys

out=Path('artifacts/misul-dense-bf16')
for segment in range(1,5):
    args=[sys.executable,'-m','transformermodel.safe_run','--report',
          f'evidence/misul-dense-segment{segment}-guard.json','--seconds','1100','--',
          sys.executable,'experiments/train_misul_dense.py','--output',str(out),
          '--precision','bf16','--memory-backend','scan','--width','640','--hidden','1504',
          '--layers','6','--heads','8','--context','128','--batch','4','--program-batch','16',
          '--steps','16384','--seconds','1000','--seed','1001','--eval-every','512',
          '--skip-program-eval']
    if segment>1:args.append('--resume')
    print(f'Starting dense segment {segment}',flush=True)
    with Path(f'evidence/misul-dense-segment{segment}.log').open('w') as log:
        r=subprocess.run(args,env=os.environ|{'PYTHONPATH':'src'},stdout=log,stderr=subprocess.STDOUT)
    print(f'Dense segment {segment}: {r.returncode}',flush=True)
    if r.returncode:raise SystemExit(r.returncode)
    if json.loads((out/'summary.json').read_text())['complete']:break
