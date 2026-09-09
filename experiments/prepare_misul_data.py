"""Fit the larger model's lossless tokenizer on existing training documents only."""
import hashlib
import json
from pathlib import Path
import time
from tokenizers import Tokenizer,decoders,models,pre_tokenizers,trainers
from transformermodel.safe_run import require_guard

require_guard();start=time.perf_counter()
root=Path('data/misul-v2');root.mkdir(parents=True,exist_ok=False)
source=Path('data/wikitext2-v1/prepared/manifest.json')
corpus=json.loads(source.read_text());training=[r for r in corpus['records'] if r['split']=='train']
for record in training:
    if hashlib.sha256(Path(record['path']).read_bytes()).hexdigest()!=record['sha256']:raise ValueError('corpus digest mismatch')
tokenizer=Tokenizer(models.BPE())
tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False);tokenizer.decoder=decoders.ByteLevel()
tokenizer.train([r['path'] for r in training],trainers.BpeTrainer(vocab_size=4096,min_frequency=2,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),show_progress=False,
    special_tokens=['<|idle|>','<|end-input|>']))
for text in ['Hello, world!','\nShe had never seen such a thing.','Élodie, 中文, 😀']:
    assert tokenizer.decode(tokenizer.encode(text).ids)==text
assert all(len(tokenizer.encode(str(d)).ids)==1 for d in range(10))
path=root/'tokenizer.json';tokenizer.save(str(path))
manifest=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    tokenizer_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),vocab_size=tokenizer.get_vocab_size(),
    training_documents=len(training),fit_on_train_only=True,normalizer=None,decoder='ByteLevel',
    special_tokens=['<|idle|>','<|end-input|>'],seconds=time.perf_counter()-start)
(root/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)
