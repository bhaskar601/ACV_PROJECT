"""Generate publication figures from saved measurements; never invent results."""
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'artifacts';F=A/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
COLORS={'baseline':'#65748b','skip':'#087e8b'}

def main():
    r=json.loads((A/'results.json').read_text())
    fig,ax=plt.subplots(1,2,figsize=(11,3.8),layout='constrained')
    for v in ['baseline','skip']:
        h=json.loads((A/f'{v}_history.json').read_text()); e=[x['epoch'] for x in h]
        ax[0].plot(e,[x['train_loss'] for x in h],color=COLORS[v],label=v.title())
        ax[1].plot(e,[x['val_miou']*100 for x in h],color=COLORS[v],label=v.title())
        b=r[v]['best_epoch'];ax[1].scatter([b],[h[b-1]['val_miou']*100],color=COLORS[v],s=60)
    ax[0].set(xlabel='Epoch',ylabel='Valid-pixel cross entropy',title='Training loss')
    ax[1].set(xlabel='Epoch',ylabel='Validation mIoU (%)',title='Validation and checkpoint selection')
    for a in ax:a.legend(frameon=False);a.grid(alpha=.16);a.set_xticks([1,5,10,15,20])
    fig.savefig(F/'learning_curves.png',dpi=190);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,3.7),layout='constrained'); xx=np.arange(3)
    for j,v in enumerate(['baseline','skip']):
        vals=[100*r[v][k] for k in ['background_iou','pet_iou','miou']]
        bars=ax.bar(xx+(j-.5)*.33,vals,.33,label=v.title(),color=COLORS[v]);ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=10)
    ax.set(xticks=xx,xticklabels=['Background IoU','Pet IoU','Mean IoU'],ylabel='IoU (%)',ylim=(0,100),title='Held-out test subset of 740 images');ax.legend(frameon=False);ax.grid(axis='y',alpha=.15)
    fig.savefig(F/'test_metrics.png',dpi=190);plt.close(fig)
    z=np.load(ROOT/'data/test.npz');x=z['x'];y=z['y'];ids=z['ids'];preds={v:np.load(A/f'{v}_predictions.npy') for v in ['baseline','skip']}
    records=list(csv.DictReader((A/'per_image.csv').open()));rows={v:{a['id']:a for a in records if a['variant']==v} for v in ['baseline','skip']}
    ranked=sorted(range(len(ids)),key=lambda i:float(rows['skip'][ids[i]]['miou']))
    picks={'representative':[ranked[int(len(ranked)*q)] for q in [.25,.5,.75]],'failures':ranked[:3]}
    selected={}
    pal=np.array([[242,245,248],[12,145,150],[167,177,189]],dtype=np.uint8)
    for kind,indices in picks.items():
        fig,axes=plt.subplots(3,4,figsize=(10,8.3),layout='constrained')
        selected[kind]=[]
        for row,i in enumerate(indices):
            name=str(ids[i]);selected[kind].append({'id':name,'baseline':rows['baseline'][name],'skip':rows['skip'][name]})
            gt=y[i].copy();gt[gt==255]=2
            arrays=[x[i],pal[gt],pal[preds['baseline'][i]],pal[preds['skip'][i]]]
            for col,a in enumerate(axes[row]):
                a.imshow(arrays[col]);a.set_xticks([]);a.set_yticks([])
                for sp in a.spines.values():sp.set_visible(False)
                if row==0:a.set_title(['Source RGB at 96 × 96','Human mask','Baseline prediction','Skip prediction'][col],fontsize=11)
                if col==0:a.set_xlabel(name,fontsize=9)
                if col in [2,3]:a.set_xlabel(f"Image mIoU {float(rows[['baseline','skip'][col-2]][name]['miou'])*100:.1f}%",fontsize=10)
        fig.suptitle(('Representative examples at skip-score quartiles' if kind=='representative' else 'Three lowest skip-score test examples')+'\nTeal = pet    Light grey = background    Slate = ignored human boundary',fontsize=12)
        fig.savefig(F/f'{kind}.png',dpi=180);plt.close(fig)
    (A/'selected_examples.json').write_text(json.dumps(selected,indent=2))
    # Diagram is a code-native plot; boxes carry real tensor dimensions and operation names.
    fig,ax=plt.subplots(figsize=(10.6,5.7));ax.set(xlim=(0,10.6),ylim=(0,5.7));ax.axis('off')
    def box(x,y,w,h,text,fill='#eef3f7'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06',fc=fill,ec='#6c7d8d',lw=1));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
    def arrow(a,b,style='-',color='#526374'):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='->',mutation_scale=12,lw=1.5,linestyle=style,color=color))
    box(.2,4.5,1.6,.65,'RGB input\n3 × 96 × 96');box(2.3,4.4,2,.85,'Encoder E1\nConv 3→8→8\n8 × 96 × 96');arrow((1.85,4.8),(2.2,4.8))
    box(2.3,2.65,2,.85,'Encoder E2\nConv 8→16→16\n16 × 48 × 48');arrow((3.3,4.3),(3.3,3.6));ax.text(3.45,3.9,'MaxPool 2',fontsize=9)
    box(2.3,.8,2,.85,'Bottleneck\nConv 16→32→32\n32 × 24 × 24');arrow((3.3,2.55),(3.3,1.75));ax.text(3.45,2.05,'MaxPool 2',fontsize=9)
    box(6.1,2.65,2.2,.85,'Decoder D2\nUp + 1×1 32→16\nConv 16→16→16');arrow((4.4,1.2),(7.2,2.55));box(6.1,4.4,2.2,.85,'Decoder D1\nUp + 1×1 16→8\nConv 8→8→8');arrow((7.2,3.6),(7.2,4.3))
    arrow((4.4,3.05),(6,3.05),'--','#087e8b');arrow((4.4,4.82),(6,4.82),'--','#087e8b');ax.text(5.2,3.23,'g × E2  (+)',ha='center',fontsize=9,color='#087e8b');ax.text(5.2,5,'g × E1  (+)',ha='center',fontsize=9,color='#087e8b')
    box(8.8,4.4,1.5,.85,'Head 1×1\n8→2 logits\nArgmax');arrow((8.4,4.82),(8.7,4.82))
    ax.text(5.3,.25,'Baseline g = 0     Skip variant g = 1     Both 24,674 trainable parameters',ha='center',fontsize=11)
    ax.text(6.65,1.05,'Every Conv block: 3×3, stride 1, padding 1,\nbias enabled, ReLU after each convolution.\nUp: bilinear ×2, align_corners=False.\nSkip additions follow the 1×1 projections.',fontsize=10,ha='left')
    fig.savefig(F/'architecture.png',dpi=190,bbox_inches='tight');plt.close(fig)
    # Data contract figure uses actual first training example.
    tr=np.load(ROOT/'data/train.npz');raw=tr['y'][0].copy();raw[raw==255]=2
    fig,axes=plt.subplots(1,2,figsize=(7.5,3.6),layout='constrained')
    axes[0].imshow(tr['x'][0]);axes[0].set_title(str(tr['ids'][0]));axes[1].imshow(pal[raw]);axes[1].set_title('Target labels after nearest resize')
    for a in axes:a.axis('off')
    fig.suptitle('0 background (light grey)   1 pet (teal)   255 ignore (slate)',fontsize=11)
    fig.savefig(F/'data_contract.png',dpi=180);plt.close(fig)
    print('Generated figures and deterministic example selection',flush=True)
if __name__=='__main__':main()
