"""Download only pinned train/validation source files, with size/hash checks."""
import hashlib,json,shutil,urllib.request
from pathlib import Path

revision='f776294184f13b8ff2337b3841cf9269a6216d1e'
root=Path('data/wikitext2-v1');root.mkdir(exist_ok=False);raw=root/'raw';raw.mkdir()
if shutil.disk_usage(root).free<21*2**30:raise RuntimeError('disk reserve insufficient')
api=f'https://huggingface.co/api/datasets/Salesforce/wikitext/tree/{revision}/wikitext-2-raw-v1'
with urllib.request.urlopen(api,timeout=30) as response:metadata=json.loads(response.read(100000))
(root/'source-tree.json').write_text(json.dumps(metadata,indent=2));records=[]
for name,cap in [('README.md',100000),('wikitext-2-raw-v1/train-00000-of-00001.parquet',10*2**20),('wikitext-2-raw-v1/validation-00000-of-00001.parquet',2*2**20)]:
    url=f'https://huggingface.co/datasets/Salesforce/wikitext/resolve/{revision}/{name}';path=raw/Path(name).name;digest=hashlib.sha256();count=0
    with urllib.request.urlopen(url,timeout=30) as response,path.open('xb') as output:
        while chunk:=response.read(65536):
            count+=len(chunk)
            if count>cap:raise RuntimeError('source exceeds registered download limit')
            digest.update(chunk);output.write(chunk)
    expected=next((entry.get('lfs',{}).get('oid') for entry in metadata if entry['path']==name),None)
    if expected and digest.hexdigest()!=expected:raise RuntimeError('upstream LFS digest mismatch')
    records.append({'path':str(path),'url':url,'sha256':digest.hexdigest(),'upstream_lfs_sha256':expected,'bytes':count});print(json.dumps(records[-1]),flush=True)
(root/'download.json').write_text(json.dumps({'revision':revision,'dataset':'Salesforce/wikitext','subset':'wikitext-2-raw-v1','records':records,'test_downloaded':False},indent=2))
