"""Fixed-budget training. Test data are never loaded here."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import argparse,json,random,time,platform,subprocess
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import Segmenter
ROOT=Path(__file__).resolve().parents[1]; A=ROOT/'artifacts'
SEED=23115002

def setup():
    random.seed(SEED); np.random.seed(SEED);torch.manual_seed(SEED)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)

def load(split):
    z=np.load(ROOT/'data'/f'{split}.npz')
    x=torch.from_numpy(z['x'].copy()).permute(0,3,1,2).float()/255
    y=torch.from_numpy(z['y'].copy()).long()
    return x,y,z['ids']

def confusion(pred,y):
    keep=y!=255
    return torch.bincount((2*y[keep]+pred[keep]).reshape(-1),minlength=4).reshape(2,2)

def scores(cm):
    cm=cm.double(); den=cm.sum(0)+cm.sum(1)-cm.diag()
    iou=cm.diag()/den
    return {'background_iou':iou[0].item() if den[0]>0 else None,'pet_iou':iou[1].item() if den[1]>0 else None,'miou':iou[den>0].mean().item(),'confusion_matrix':cm.long().tolist()}

@torch.inference_mode()
def evaluate(model,x,y):
    model.eval();cm=torch.zeros(2,2,dtype=torch.long);loss=0;valid=0
    for start in range(0,len(x),16):
        logits=model(x[start:start+16]); target=y[start:start+16]
        loss+=F.cross_entropy(logits,target,ignore_index=255,reduction='sum').item();valid+=(target!=255).sum().item()
        cm+=confusion(logits.argmax(1),target)
    return {'loss':loss/valid,**scores(cm)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['baseline','skip'],required=True);args=p.parse_args()
    setup(); x,y,_=load('train'); vx,vy,_=load('val')
    model=Segmenter(args.variant=='skip'); opt=torch.optim.Adam(model.parameters(),lr=.001)
    env={'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),'processor':platform.processor(),'device':'cpu','threads':torch.get_num_threads(),'seed':SEED,'deterministic_algorithms':True,'resolution':96,'batch_size':16,'epochs':20,'optimizer':'Adam','lr':.001,'weight_decay':0,'augmentation':'paired horizontal flip p=0.5','stopping':'20 epochs; select highest validation global mIoU; earliest tie','parameters':sum(p.numel() for p in model.parameters())}
    (A/'environment.json').write_text(json.dumps(env,indent=2))
    rng=torch.Generator().manual_seed(SEED); best=-1; history=[];total=time.perf_counter()
    for epoch in range(1,21):
        tic=time.perf_counter();model.train();order=torch.randperm(len(x),generator=rng);loss_sum=0;valid=0
        for ids in order.split(16):
            bx=x[ids].clone();by=y[ids].clone();flip=torch.rand(len(ids),generator=rng)<.5
            bx[flip]=bx[flip].flip(-1);by[flip]=by[flip].flip(-1)
            opt.zero_grad(set_to_none=True);logits=model(bx); loss=F.cross_entropy(logits,by,ignore_index=255)
            loss.backward();opt.step();n=(by!=255).sum().item();loss_sum+=loss.item()*n;valid+=n
        val=evaluate(model,vx,vy)
        row={'epoch':epoch,'train_loss':loss_sum/valid,**{f'val_{k}':v for k,v in val.items() if k!='confusion_matrix'},'seconds':time.perf_counter()-tic}
        history.append(row)
        if val['miou']>best:
            best=val['miou'];torch.save({'state_dict':model.state_dict(),'variant':args.variant,'epoch':epoch,'val_miou':best,'seed':SEED},A/f'{args.variant}.pt')
        (A/f'{args.variant}_history.json').write_text(json.dumps(history,indent=2))
        print(f"{args.variant:8s} epoch={epoch:02d}/20 train_loss={row['train_loss']:.4f} val_mIoU={val['miou']:.4f} seconds={row['seconds']:.2f}",flush=True)
    meta={'variant':args.variant,'seconds':time.perf_counter()-total,'steps':20*((len(x)+15)//16),'best_val_miou':best,'parameters':env['parameters']}
    (A/f'{args.variant}_training.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
if __name__=='__main__':main()
