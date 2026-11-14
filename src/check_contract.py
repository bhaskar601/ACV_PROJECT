"""Checks for data leakage, label preservation and IoU/ignore handling."""
import json
import torch
import numpy as np
from model import Segmenter
from train import ROOT,confusion,scores
s=json.loads((ROOT/'artifacts/splits.json').read_text())
for name,n in [('train',1480),('val',370),('test',740)]:
    assert len(s[name])==len(set(s[name]))==n
    z=np.load(ROOT/'data'/f'{name}.npz');assert set(np.unique(z['y']))=={0,1,255}
    assert z['x'].shape==(n,96,96,3)
assert not(set(s['train'])&set(s['val']) or set(s['train'])&set(s['test']) or set(s['val'])&set(s['test']))
a=Segmenter(False);b=Segmenter(True)
assert {k:v.shape for k,v in a.state_dict().items()}=={k:v.shape for k,v in b.state_dict().items()}
y=torch.tensor([[0,0,1,1,255]]);p=torch.tensor([[0,1,1,1,0]])
cm=confusion(p,y);assert cm.tolist()==[[1,1],[0,2]]
r=scores(cm);assert abs(r['miou']-(.5+2/3)/2)<1e-10
assert scores(confusion(y[:,:4],y[:,:4]))['miou']==1
print('PASS: split sizes and disjointness, label IDs, image dimensions, identical parameter shapes, hand-calculated IoU, ignored pixel exclusion, perfect-mask IoU')
# Validate the same initial tensors and the ignored-pixel invariance directly.
torch.manual_seed(23115002);a=Segmenter(False)
torch.manual_seed(23115002);b=Segmenter(True)
assert all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict())
p2=p.clone();p2[0,-1]=1
assert torch.equal(confusion(p,y),confusion(p2,y))
for name,pool in [('train','trainval'),('val','trainval'),('test','test')]:
    legal={l.split()[0] for l in (ROOT/'data/annotations'/f'{pool}.txt').read_text().splitlines() if l and not l.startswith('#')}
    assert set(s[name])<=legal
print('PASS: identical initial weights, ignored-pixel prediction invariance, official source-pool membership')
import hashlib
h=hashlib.sha256()
for k,v in a.state_dict().items():h.update(k.encode());h.update(v.numpy().tobytes())
g=torch.Generator().manual_seed(23115002);schedule=hashlib.sha256()
for _ in range(20):
    order=torch.randperm(1480,generator=g);schedule.update(order.numpy().tobytes())
    for indices in order.split(16):schedule.update((torch.rand(len(indices),generator=g)<.5).numpy().tobytes())
(ROOT/'artifacts/controlled_comparison.json').write_text(json.dumps({'seed':23115002,'shared_initial_weights_sha256':h.hexdigest(),'shared_order_and_flip_schedule_sha256':schedule.hexdigest(),'parameters_per_variant':sum(p.numel() for p in a.parameters()),'updates_per_variant':1860,'only_changed_mechanism':'two additive encoder-to-decoder skips'},indent=2))
absent=scores(confusion(torch.zeros((1,3),dtype=torch.long),torch.zeros((1,3),dtype=torch.long)))
assert absent['background_iou']==1 and absent['pet_iou'] is None and absent['miou']==1
print('PASS: absent-union class IoU is undefined and omitted only from image-level mean')
