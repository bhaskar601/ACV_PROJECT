"""Prepare a breed-balanced, frozen subset using official annotation IDs."""
import io,json,hashlib,random
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'data'; A=ROOT/'artifacts'; A.mkdir(exist_ok=True)
SEED=23115002; SIZE=96

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def main():
    rng=random.Random(SEED)
    official={}
    for split in ['trainval','test']:
        rows=[l.split() for l in (D/'annotations'/f'{split}.txt').read_text().splitlines() if l and not l.startswith('#')]
        official[split]=rows
    groups={}
    for r in official['trainval']: groups.setdefault(int(r[1]),[]).append(r[0])
    splits={'train':[],'val':[],'test':[]}
    for breed,ids in sorted(groups.items()):
        ids=sorted(ids); rng.shuffle(ids)
        splits['train']+=ids[:40]; splits['val']+=ids[40:50]
    groups={}
    for r in official['test']: groups.setdefault(int(r[1]),[]).append(r[0])
    for breed,ids in sorted(groups.items()):
        ids=sorted(ids); rng.shuffle(ids); splits['test']+=ids[:20]
    assert all(not set(splits[a])&set(splits[b]) for a,b in [('train','val'),('train','test'),('val','test')])
    wanted=set(sum(splits.values(),[])); images={}
    for source in ['train','test']:
        table=pq.read_table(D/f'mirror_{source}.parquet',columns=['image','image_id'])
        for row in table.to_pylist():
            if row['image_id'] in wanted: images[row['image_id']]=row['image']['bytes']
    assert wanted==set(images), f'Missing images: {wanted-set(images)}'
    hashes=[]; pixel_counts={}
    for split,ids in splits.items():
        xs=[];ys=[];counts=np.zeros(3,dtype=np.int64)
        for name in ids:
            raw=images[name]; mask_path=D/'annotations'/'trimaps'/f'{name}.png'
            with Image.open(io.BytesIO(raw)) as im:
                x=np.asarray(im.convert('RGB').resize((SIZE,SIZE),Image.Resampling.BILINEAR)).copy()
            with Image.open(mask_path) as im:
                rawmask=np.asarray(im.resize((SIZE,SIZE),Image.Resampling.NEAREST))
            assert set(np.unique(rawmask)) <= {1,2,3}
            y=np.full(rawmask.shape,255,dtype=np.uint8); y[rawmask==1]=1; y[rawmask==2]=0
            counts += [(y==0).sum(),(y==1).sum(),(y==255).sum()]
            xs.append(x);ys.append(y)
            hashes.append({'id':name,'split':split,'image_sha256':hashlib.sha256(raw).hexdigest(),'mask_sha256':sha(mask_path)})
        np.savez_compressed(D/f'{split}.npz',x=np.stack(xs),y=np.stack(ys),ids=np.array(ids))
        pixel_counts[split]=counts.tolist(); print(split,len(ids),counts.tolist(),flush=True)
    (A/'splits.json').write_text(json.dumps(splits,indent=2))
    (A/'data_manifest.json').write_text(json.dumps({'seed':SEED,'resolution':SIZE,'counts':{k:len(v) for k,v in splits.items()},'pixel_counts_background_pet_ignore':pixel_counts,'files':hashes},indent=2))
    (A/'archive_hashes.json').write_text(json.dumps({p.name:sha(p) for p in [D/'annotations.tar.gz',D/'mirror_train.parquet',D/'mirror_test.parquet']},indent=2))
if __name__=='__main__':main()
