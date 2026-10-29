"""Audit packaged artifact bytes without loading models or running predictors."""
import csv, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
registry=json.loads((ROOT/'policy-model-selector-v3/model_registry.json').read_text())
rows=[]
for m in registry['models']:
    profile=m['profile_id']; tool=m['tool_id']
    if profile not in registry['profiles']: continue
    if tool.startswith('speech_'):
        directory=ROOT/'pd-speech-baseline/artifacts/speech_extracted_v2'; name=tool.removeprefix('speech_')+'.joblib'
    elif tool.startswith('italian_'):
        directory=ROOT/'pd-speech-baseline/artifacts/italian_vowel_a_grouped_v1'; name=tool.removeprefix('italian_')+'.joblib'
    elif tool in ['spiral_hog_rf','spiral_resnet18_linear']:
        directory=ROOT/'policy-model-selector-v3/trained_tools'/profile; name=tool.removeprefix('spiral_')+'.joblib'
    else: continue
    path=directory/name; expected=m.get('artifact_sha256'); observed=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    meta=json.loads((directory/'artifacts.json').read_text()) if (directory/'artifacts.json').is_file() else {}
    metadata_hash=meta.get('model_sha256',{}).get(name)
    status='matched' if expected and expected==observed==metadata_hash else ('artifact_missing' if observed is None else 'binding_mismatch')
    rows.append(dict(profile=profile,model=m['model_id'],evaluation_id=m['evaluation_id'],expected_sha256=expected,observed_sha256=observed,metadata_sha256=metadata_hash,status=status))
print(json.dumps({'artifact_identity_only':True,'model_loads':0,'classifier_executions':0,'rows':rows},indent=2))
if any(r['status']=='binding_mismatch' for r in rows):raise SystemExit(1)
