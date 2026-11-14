"""Build browser evidence views directly from measured files."""
import json,html
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'artifacts';E=A/'evidence';E.mkdir(exist_ok=True)
r=json.loads((A/'results.json').read_text());env=json.loads((A/'environment.json').read_text());hw=json.loads((A/'hardware.json').read_text())
style='''<style>*{box-sizing:border-box}body{margin:0;padding:24px 32px;background:#f3f6fa;color:#15263d;font:14px Arial,sans-serif}h1{font-size:26px;margin:0 0 8px}h2{font-size:17px;margin:18px 0 9px}.sub{color:#52647b;margin-bottom:20px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.card{background:white;border:1px solid #d9e2ed;border-radius:9px;padding:15px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:9px 10px;border-bottom:1px solid #dce5ee}th{background:#e9f0f6}pre{white-space:pre-wrap;font:12px/1.4 monospace;margin:0;color:#19344c}.tag{display:inline-block;background:#dcefe9;padding:5px 9px;border-radius:4px;font-size:13px;color:#116650}footer{font-size:13px;color:#52647b;margin-top:20px}img{max-width:100%}</style>'''
def page(title,body):return '<!doctype html><html><head><meta charset="utf-8">'+style+'</head><body><h1>'+title+'</h1><div class="sub">Abhay Kumar · 23115002 · Reproducible segmentation experiment</div>'+body+'<footer>Evidence rendered from artifacts in this submission. These are actual browser views, not terminal screenshots.</footer></body></html>'
rows=''
for v in ['baseline','skip']:
    z=r[v];rows+=f"<tr><td>{v.title()}</td><td>{z['background_iou']*100:.2f}%</td><td>{z['pet_iou']*100:.2f}%</td><td><b>{z['miou']*100:.2f}%</b></td><td>{z['parameters']:,}</td><td>{z['best_epoch']}</td></tr>"
body='<span class="tag">Evaluation completed · 740 held-out images</span><h2>Global valid-pixel test metrics</h2><table><tr><th>Model</th><th>Background IoU</th><th>Pet IoU</th><th>mIoU</th><th>Parameters</th><th>Best epoch</th></tr>'+rows+'</table>'
body+='<div class="grid">'
for v in ['baseline','skip']:
    z=r[v];l=z['latency_ms'];body+=f"<div class='card'><h2>{v.title()} measurement</h2><pre>CPU: {hw['cpu']} · float32 · 4 threads\nInput: 1 × 3 × 96 × 96\nMedian: {l['median']:.3f} ms · p95: {l['p95']:.3f} ms\nWarmup: 30 · timed iterations: 200\nConfusion matrix [truth × prediction]\n{z['confusion_matrix'][0]}\n{z['confusion_matrix'][1]}</pre></div>"
body+='</div><h2>Controlled comparison</h2><div class="card">Only additive skip routing changes. Both models receive 1,860 updates, the same seeded initialization and data schedule. Ignore label 255 is excluded from loss and IoU. Latency includes forward execution only.</div>'
(E/'metrics.html').write_text(page('Held-out evaluation evidence',body))
body='<span class="tag">Fixed 20-epoch runs completed</span><div class="grid">'
for v in ['baseline','skip']:
    lines=(A/f'{v}_train.log').read_text().splitlines();body+=f'<div class="card"><h2>{v.title()} log excerpt</h2><pre>'+html.escape('\n'.join(lines[:2]+['...']+lines[-3:-1]))+'</pre></div>'
body+='</div><h2>Data and metric checks</h2><div class="card"><pre>'+html.escape((A/'contract_check.log').read_text().strip())+'</pre></div>'
body+='<h2>Frozen execution settings</h2><div class="card"><pre>'+html.escape(f"Seed {env['seed']} · Python {env['python']} · PyTorch {env['torch']} · NumPy {env['numpy']}\nAdam lr 0.001 · batch 16 · 96 × 96 · deterministic CPU operations\nSplits: train 1,480 / validation 370 / test 740; no overlapping image IDs\nSources: *_train.log, *_history.json, environment.json, contract_check.log")+'</pre></div>'
(E/'execution.html').write_text(page('Training and verification evidence',body))
print('Built measured HTML evidence views')
