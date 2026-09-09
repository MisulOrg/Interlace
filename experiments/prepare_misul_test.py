"""Fetch the pinned official test split without changing training data."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import urllib.request
import pyarrow.parquet as pq

root = Path('data/wikitext2-v1')
prior = json.loads((root / 'download.json').read_text())
entry = next(r for r in json.loads((root / 'source-tree.json').read_text())
             if r['path'].endswith('/test-00000-of-00001.parquet'))
out = Path('data/misul-test')
out.mkdir(exist_ok=False)
if shutil.disk_usage(out).free < 20 * 2**30 + 8 * 2**20:
    raise RuntimeError('preserve the existing disk reserve')
url = f"https://huggingface.co/datasets/Salesforce/wikitext/resolve/{prior['revision']}/{entry['path']}"
source = out / 'test.parquet'
with urllib.request.urlopen(url, timeout=30) as response:
    payload = response.read(2 * 2**20 + 1)
if len(payload) != entry['size'] or hashlib.sha256(payload).hexdigest() != entry['lfs']['oid']:
    raise ValueError('test source does not match the previously pinned upstream size and hash')
source.write_bytes(payload)
rows, title, documents = [], None, []
for batch in pq.ParquetFile(source).iter_batches(batch_size=256, columns=['text']):
    for row in batch.column(0).to_pylist():
        match = re.fullmatch(r'\s*= ([^=].*?) =\s*', row)
        if match:
            if title is not None:
                documents.append((title, rows))
            title, rows = match.group(1), []
        if title is None:
            if row.strip():
                raise ValueError('nonblank text before first test article')
        else:
            rows.append(row)
if title is not None:
    documents.append((title, rows))
records = []
for index, (title, rows) in enumerate(documents):
    raw = ''.join(rows).encode('utf-8')
    path = out / f'test-{index:04}.txt'
    path.write_bytes(raw)
    records.append(dict(split='test', title=title, path=str(path.resolve()),
                        bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
training_manifest = root / 'prepared/manifest.json'
prior_records = json.loads(training_manifest.read_text())['records']
def paragraph_hashes(records):
    return {hashlib.sha256(' '.join(line.split()).encode()).hexdigest()
            for r in records for line in Path(r['path']).read_text().splitlines()
            if len(' '.join(line.split())) >= 80}
overlap = {}
for split in ('train', 'validation'):
    group = [r for r in prior_records if r['split'] == split]
    overlap[split] = dict(titles=sorted({r['title'] for r in group} & {r['title'] for r in records}),
                         documents=sorted({r['sha256'] for r in group} & {r['sha256'] for r in records}),
                         normalized_paragraph_hashes=sorted(paragraph_hashes(group) & paragraph_hashes(records)))
manifest = dict(records=records, source_url=url, source_sha256=entry['lfs']['oid'],
                revision=prior['revision'], training_manifest_sha256=hashlib.sha256(training_manifest.read_bytes()).hexdigest(),
                tokenizer='Existing data/misul-v2/tokenizer.json, fitted on training only; no fitting on test.',
                assembly='Same article-boundary regex and direct row concatenation as the frozen training preparation.',
                overlap=overlap)
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(dict(documents=len(records), bytes=sum(r['bytes'] for r in records),
                      source_sha256=entry['lfs']['oid'], overlap=overlap)))
