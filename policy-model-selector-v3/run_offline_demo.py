from pathlib import Path
import json
from create_examples import examples
from policy_engine import select
root=Path(__file__).parent
registry=json.loads((root/'model_registry.json').read_text())
results=[]
for e in examples():
 r=select(registry,e['gold_spec'],e['request'],e['profile_id'])
 assert r['status']==e['expected_status'],(e['id'],r)
 assert r['selected_model']==e['expected_model'],(e['id'],r)
 results.append({'case':e['id'],'status':r['status'],'selected_model':r['selected_model']})
 print(e['id']+': '+r['status']+' -> '+str(r['selected_model']))
out=root/'verified_results';out.mkdir(exist_ok=True)
(out/'offline_demo.json').write_text(json.dumps({'type':'Hand-authored specs; deterministic engine only; no LLM tested','passed':len(results),'results':results},indent=2))
