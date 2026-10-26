"""Compile a previously translated policy without an API call."""
import argparse,json
from pathlib import Path
from policy_engine import select
p=argparse.ArgumentParser()
p.add_argument('--spec',required=True,type=Path)
p.add_argument('--request-file',required=True,type=Path)
p.add_argument('--profile',required=True)
p.add_argument('--registry',type=Path,default=Path(__file__).parent/'model_registry.json')
p.add_argument('--output',type=Path,default=Path('results/compiled_policy.json'))
a=p.parse_args()
result=select(json.loads(a.registry.read_text()),json.loads(a.spec.read_text()),a.request_file.read_text(encoding='utf-8-sig').strip(),a.profile)
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
