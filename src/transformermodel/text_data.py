import hashlib
import json
from pathlib import Path

import numpy as np


def byte_windows(manifest_path, split, context):
    """Non-overlapping target windows, never crossing a document boundary."""
    manifest = json.loads(Path(manifest_path).read_text())
    x, y = [], []
    for record in manifest['records']:
        if record['split'] != split:
            continue
        raw = Path(record['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != record['sha256']:
            raise ValueError('corpus digest mismatch')
        data = np.frombuffer(raw, dtype=np.uint8)
        count = (len(data) - 1) // context
        if count:
            x.append(data[:count * context].reshape(count, context))
            y.append(data[1:count * context + 1].reshape(count, context))
    if not x:
        raise ValueError('empty split')
    return np.concatenate(x), np.concatenate(y)


def token_windows(manifest_path,tokenizer_path,split,context):
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(tokenizer_path))
    manifest=json.loads(Path(manifest_path).read_text())
    xs,ys=[],[]
    for record in manifest['records']:
        if record['split']!=split:continue
        raw=Path(record['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=record['sha256']:raise ValueError('corpus digest mismatch')
        text=raw.decode('utf-8')
        ids=tokenizer.encode(text).ids
        if tokenizer.decode(ids)!=text:raise ValueError('tokenizer is not lossless on source')
        data=np.array(ids,dtype=np.int32)
        count=(len(data)-1)//context
        if count:
            xs.append(data[:count*context].reshape(count,context))
            ys.append(data[1:count*context+1].reshape(count,context))
    if not xs:raise ValueError('empty split')
    # ByteLevel maps each original byte to one Unicode character before BPE.
    # There are no special tokens or normalization in the registered tokenizer.
    lengths=np.array([len(tokenizer.id_to_token(i)) for i in range(tokenizer.get_vocab_size())],dtype=np.int32)
    return np.concatenate(xs),np.concatenate(ys),tokenizer,lengths
