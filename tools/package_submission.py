"""Build a concise project archive containing the report, code and evidence."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'23115002_submission.zip'
files=[ROOT/name for name in ['README.md','PLAN_OF_ACTION.md','SUBMISSION.md','DATA_ATTRIBUTION.md','requirements.txt','requirements-lock.txt','reproduce.sh','.gitignore','tools/package_submission.py','tools/verify_results.py']]
excluded={'annotation_audit.log','evaluation.log','reevaluation.log','verification_results.json'}
for folder in ['src','artifacts','report']:
    files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file()
        and '__pycache__' not in p.parts
        and not any(part.startswith('qa') for part in p.relative_to(ROOT).parts)
        and p.name not in excluded and not p.name.endswith('_predictions.npy'))
files=sorted(set(files))
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
with zipfile.ZipFile(OUT,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for p in files:archive.write(p,'23115002_ACV/'+str(p.relative_to(ROOT)))
    archive.writestr('23115002_ACV/SUBMISSION_SHA256.json',json.dumps(manifest,indent=2))
print(OUT,OUT.stat().st_size,'bytes',len(files),'files')
