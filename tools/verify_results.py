"""Compare independent checkpoint evaluation with recorded accuracy values."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'artifacts'
a=json.loads((A/'results.json').read_text());b=json.loads((A/'verification_results.json').read_text())
for variant in ['baseline','skip']:
    for key in ['background_iou','pet_iou','miou','confusion_matrix','checkpoint_sha256','best_epoch','parameters']:
        assert a[variant][key]==b[variant][key],(variant,key)
    cm=a[variant]['confusion_matrix'];assert sum(map(sum,cm))==3897740+2069115
    history=json.loads((A/f'{variant}_history.json').read_text());assert len(history)==20
    selected=max(history,key=lambda row:row['val_miou']);assert selected['epoch']==a[variant]['best_epoch']
    training=json.loads((A/f'{variant}_training.json').read_text());assert training['steps']==1860
print('PASS: independent evaluation exactly reproduces both per-class IoUs, global mIoU and confusion matrices')
print('PASS: evaluated valid-pixel totals, highest-validation checkpoint selection, 20-epoch histories and 1,860 updates per model')
(A/'verification_summary.json').write_text(json.dumps({'status':'PASS','exact_accuracy_match_on_checkpoint_reevaluation':True,'valid_test_pixels':5966855,'epochs_per_variant':20,'updates_per_variant':1860},indent=2))
