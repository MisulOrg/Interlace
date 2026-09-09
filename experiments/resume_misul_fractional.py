"""Resume the explicitly amended BF16 control with fractional overflow recovery."""
import hashlib
import json
from pathlib import Path
import sys
from transformermodel import train_misul
from transformermodel.misul_fractional_update import FractionalCheckedUpdate

protocol = Path('evidence/misul-fractional-amendment.json')
amendment = json.loads(protocol.read_text())
for path, expected in amendment['amendment_source_sha256'].items():
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
        raise ValueError(f'amendment source changed: {path}')
arguments = sys.argv[1:]
output = Path(arguments[arguments.index('--output') + 1])
if output != Path('artifacts/misul-joint-checked-bf16-recovered'):
    raise ValueError('use the preserved, explicitly amended control directory')
(output / 'numerical-amendment.json').write_bytes(protocol.read_bytes())
train_misul.CheckedUpdate = FractionalCheckedUpdate
train_misul.main()
