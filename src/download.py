"""Download official masks and the timm mirror of the original images."""
from pathlib import Path
import urllib.request,tarfile,hashlib,json
ROOT=Path(__file__).resolve().parents[1]; D=ROOT/'data';D.mkdir(exist_ok=True)
SOURCES={
'annotations.tar.gz':'https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz',
'mirror_train.parquet':'https://huggingface.co/datasets/timm/oxford-iiit-pet/resolve/089695c834a7deb60505b7cc506672db1c31a6aa/data/train-00000-of-00001.parquet',
'mirror_test.parquet':'https://huggingface.co/datasets/timm/oxford-iiit-pet/resolve/089695c834a7deb60505b7cc506672db1c31a6aa/data/test-00000-of-00001.parquet'}
for name,url in SOURCES.items():
    dest=D/name
    if not dest.exists():
        part=dest.with_suffix(dest.suffix+'.part');print('Downloading',name,flush=True)
        urllib.request.urlretrieve(url,part);part.replace(dest)
    h=hashlib.sha256()
    with dest.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    expected_file=ROOT/'artifacts/archive_hashes.json'
    if expected_file.exists():assert h.hexdigest()==json.loads(expected_file.read_text())[name],f'Hash mismatch: {name}'
    print(name,h.hexdigest(),flush=True)
with tarfile.open(D/'annotations.tar.gz') as archive:
    members=[m for m in archive.getmembers() if m.name.startswith('annotations/') and '..' not in Path(m.name).parts and not m.issym() and not m.islnk()]
    archive.extractall(D,members=members)
print('Download and extraction complete')
