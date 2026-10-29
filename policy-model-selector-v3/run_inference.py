"""Execute a selected policy through a registered real model adapter."""
import argparse,json,hashlib
from pathlib import Path
from policy_engine import select,dispatch,sha
p=argparse.ArgumentParser()
p.add_argument('--policy',required=True,type=Path)
p.add_argument('--image',required=True,type=Path)
p.add_argument('--profile',required=True)
p.add_argument('--models',type=Path,default=Path('trained_tools'))
p.add_argument('--registry',type=Path,default=Path(__file__).parent/'model_registry.json')
p.add_argument('--output',type=Path,default=Path('results/inference.json'))
a=p.parse_args()
if a.output.exists():raise ValueError("Output exists; choose a new output path")
policy=json.loads(a.policy.read_text());registry=json.loads(a.registry.read_text())
if policy['registry_sha256']!=sha(registry):raise ValueError('Registry changed; recompile policy')
checked=select(registry,policy['policy'],policy['request'],a.profile)
if checked.get('status')!='selected' or checked['selected_model']!=policy.get('selected_model') or checked['selected_tool']!=policy.get('selected_tool'):raise ValueError('Policy selection mismatch')
from artifact_binding import verify_image_binding
binding=verify_image_binding(registry,a.profile,checked['selected_model'],checked['selected_tool'],a.models)
from image_tools import infer
raw=a.image.read_bytes()
handlers={t:(lambda tool=t:infer(raw,tool,a.models,a.profile)) for t in ['spiral_hog_rf','spiral_resnet18_linear']}
result=dispatch(checked,a.profile,handlers)
report={'input_sha256':hashlib.sha256(raw).hexdigest(),'policy_sha256':checked['policy_sha256'],'registry_sha256':checked['registry_sha256'],'decision_reason':checked['reason'],'result':result,'artifact_binding':binding}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
