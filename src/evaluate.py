"""Exact held-out evaluation and forward-only CPU latency measurement."""
import argparse,csv,json,time,hashlib,statistics
import numpy as np
import torch
from train import setup,load,evaluate,scores,confusion,ROOT
from model import Segmenter
A=ROOT/'artifacts'
@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='results.json');args=p.parse_args()
    setup();x,y,ids=load('test'); results={}; per_image=[]
    for variant in ['baseline','skip']:
        ckpt=torch.load(A/f'{variant}.pt',map_location='cpu',weights_only=True)
        model=Segmenter(variant=='skip');model.load_state_dict(ckpt['state_dict']);model.eval()
        result=evaluate(model,x,y);result.update({'best_epoch':ckpt['epoch'],'best_val_miou':ckpt['val_miou'],'parameters':sum(p.numel() for p in model.parameters()),'checkpoint_sha256':hashlib.sha256((A/f'{variant}.pt').read_bytes()).hexdigest()})
        predictions=[]
        for start in range(0,len(x),16):predictions.append(model(x[start:start+16]).argmax(1).numpy().astype('uint8'))
        preds=np.concatenate(predictions);np.save(A/f'{variant}_predictions.npy',preds)
        for idx,name in enumerate(ids):
            s=scores(confusion(torch.from_numpy(preds[idx]).long(),y[idx]))
            per_image.append({'variant':variant,'id':str(name),'miou':s['miou'],'pet_iou':s['pet_iou'],'background_iou':s['background_iou']})
        sample=x[:1].contiguous()
        for _ in range(30):model(sample)
        times=[]
        for _ in range(200):
            tic=time.perf_counter_ns();model(sample);times.append((time.perf_counter_ns()-tic)/1e6)
        result['latency_ms']={'median':statistics.median(times),'mean':statistics.mean(times),'p95':float(np.percentile(times,95)),'warmups':30,'iterations':200,'batch':1,'device':'cpu','threads':4,'includes':'model forward only; float32; inference_mode; input already resident; excludes decode, resize, copies, argmax and metrics','raw':times}
        results[variant]=result;print(variant,json.dumps({k:v for k,v in result.items() if k!='latency_ms'}),flush=True)
    results['delta_miou']=results['skip']['miou']-results['baseline']['miou']
    (A/args.output).write_text(json.dumps(results,indent=2))
    with open(A/'per_image.csv','w') as f:
        writer=csv.DictWriter(f,fieldnames=list(per_image[0]),lineterminator='\n');writer.writeheader();writer.writerows(per_image)
if __name__=='__main__':main()
