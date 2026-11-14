"""Record absent target regions without changing the frozen data set."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];audit={}
for split in ['train','val','test']:
    z=np.load(ROOT/'data'/f'{split}.npz');items=[]
    for i in np.where((z['y']==1).sum((1,2))==0)[0]:
        name=str(z['ids'][i]);raw=np.array(Image.open(ROOT/'data/annotations/trimaps'/f'{name}.png'))
        vals,counts=np.unique(raw,return_counts=True)
        items.append({'id':name,'original_mask_counts':{str(int(v)):int(c) for v,c in zip(vals,counts)},'pet_pixels_after_resize':0})
    audit[split]={'images':len(z['ids']),'no_pet_target_count':len(items),'cases':items}
(ROOT/'artifacts/annotation_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
