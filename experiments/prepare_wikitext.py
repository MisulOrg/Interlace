"""Preserve source rows, audit cross-split overlap, fit train-only ByteLevel BPE."""
import hashlib
import json
import re
import time
from pathlib import Path
import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformermodel.safe_run import require_guard

require_guard()
start = time.perf_counter()
root = Path('data/wikitext2-v1')
out = root / 'prepared'
out.mkdir(exist_ok=False)
records = []
paragraphs = {}
for split in ['train', 'validation']:
    source = root / 'raw' / f'{split}-00000-of-00001.parquet'
    docs = []
    title, rows = None, []
    for batch in pq.ParquetFile(source).iter_batches(batch_size=256, columns=['text']):
        for row in batch.column(0).to_pylist():
            match = re.fullmatch(r'\s*= ([^=].*?) =\s*', row)
            if match:
                if title is not None:
                    docs.append((title, rows))
                title, rows = match.group(1), []
            if title is None:
                if row.strip():
                    raise ValueError('nonblank source text before first article')
            else:
                rows.append(row)
    if title is not None:
        docs.append((title, rows))
    paragraphs[split] = {}
    for i, (title, rows) in enumerate(docs):
        raw = ''.join(rows).encode('utf-8')
        path = out / f'{split}-{i:04}.txt'
        path.write_bytes(raw)
        records.append(dict(split=split, title=title, path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)))
        for row in rows:
            normalized = ' '.join(row.split())
            if len(normalized) >= 80:
                digest = hashlib.sha256(normalized.encode()).hexdigest()
                paragraphs[split].setdefault(digest, []).append(title)
train = [r for r in records if r['split'] == 'train']
validation = [r for r in records if r['split'] == 'validation']
overlap = dict(titles=sorted({r['title'] for r in train} & {r['title'] for r in validation}), documents=sorted({r['sha256'] for r in train} & {r['sha256'] for r in validation}), normalized_paragraphs=[dict(sha256=d, train=paragraphs['train'][d], validation=paragraphs['validation'][d]) for d in sorted(paragraphs['train'].keys() & paragraphs['validation'].keys())])
(out / 'overlap.json').write_text(json.dumps(overlap, indent=2))
if any(overlap.values()):
    raise ValueError('cross-split overlap requires explicit treatment before training')
tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(vocab_size=2048, min_frequency=2, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
tok.train([r['path'] for r in train], trainer)
tok.save(str(out / 'tokenizer.json'))
for r in records:
    text = Path(r['path']).read_text()
    assert tok.decode(tok.encode(text).ids) == text
manifest = dict(records=records, source_manifest=str((root/'download.json').resolve()), tokenizer_sha256=hashlib.sha256((out/'tokenizer.json').read_bytes()).hexdigest(), fit_on_train_only=True, test_opened=False, overlap=overlap, seconds=time.perf_counter()-start, source_rows_concatenated_without_added_separators=True)
(out/'manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(dict(documents={s:sum(r['split']==s for r in records) for s in ['train','validation']}, bytes={s:sum(r['bytes'] for r in records if r['split']==s) for s in ['train','validation']}, overlap=overlap, seconds=manifest['seconds'])))
