"""Counterbalance two fresh-process inference measurements per architecture."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import statistics

order=[('misul',1),('dense',1),('dense',2),('misul',2)]
for name,round in order:
    stem=f'misul-architecture-benchmark-{name}-round{round}'
    command=[sys.executable,'-m','transformermodel.safe_run','--report',f'evidence/{stem}-guard.json',
             '--seconds','180','--',sys.executable,'experiments/benchmark_misul_architecture.py',
             '--arm',name,'--output',f'evidence/{stem}.json']
    with Path(f'evidence/{stem}.log').open('w') as log:
        result=subprocess.run(command,env=os.environ|{'PYTHONPATH':'src'},stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise SystemExit(result.returncode)
for name in ('misul','dense'):
    rounds=[json.loads(Path(f'evidence/misul-architecture-benchmark-{name}-round{n}.json').read_text()) for n in (1,2)]
    for key in ('configuration','checkpoint_sha256','source_sha256','parameters'):
        assert rounds[0][key]==rounds[1][key]
    result=rounds[0].copy()
    result['cases']={}
    for key in rounds[0]['cases']:
        values=sum([r['cases'][key]['repetitions'] for r in rounds],[])
        quantiles=statistics.quantiles(values,n=10,method='inclusive')
        result['cases'][key]=dict(first_seconds_by_process=[r['cases'][key]['first_seconds'] for r in rounds],
           repetitions=values,median_seconds=statistics.median(values),p10_seconds=quantiles[0],p90_seconds=quantiles[8])
    result['rounds']=rounds
    result['order']=order
    result['peak_mlx_bytes']=max(r['peak_mlx_bytes'] for r in rounds)
    result['protocol_sha256']=hashlib.sha256(Path('evidence/interlace-inference-protocol.json').read_bytes()).hexdigest()
    Path(f'evidence/misul-architecture-benchmark-{name}.json').write_text(json.dumps(result,indent=2))
print('Counterbalanced inference measurements complete',flush=True)
