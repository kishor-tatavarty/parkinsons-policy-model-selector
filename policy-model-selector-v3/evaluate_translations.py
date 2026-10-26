"""Run13 fixed translation cases; report semantics separately from selection."""
import argparse,json,subprocess,sys
from pathlib import Path
from policy_engine import select,sha
from schema_and_prompt import SCHEMA,PROMPT
from translate_policy import MODEL
root=Path(__file__).parent
p=argparse.ArgumentParser()
p.add_argument('--execute',action='store_true')
p.add_argument('--key-env-file',type=Path)
p.add_argument('--output',type=Path,default=Path('results/translation_benchmark'))
a=p.parse_args()
registry=json.loads((root/'model_registry.json').read_text());gold=json.loads((root/'examples/benchmark_gold.json').read_text())
def semantics(s):
 def canonical(metric,op=None,value=None):
  if metric in ('false_positive_rate','false_negative_rate'):
   metric={'false_positive_rate':'specificity','false_negative_rate':'sensitivity'}[metric]
   if op:op='max' if op=='min' else 'min';value=round(1-value,10)
  return (metric,op,value)
 return {'profile':s['profile_id'],'objective':canonical(s['primary_metric'])[0],
 'constraints':sorted(canonical(c['metric'],c['operator'],round(c['value'],10)) for c in s['metric_constraints']),
 'cpu':s['cpu_required'],'latency':s['max_latency_ms'],'cost':s['max_cost_usd_per_case'],
 'ties':[canonical(t['metric'])[0] for t in s['tie_breakers']],
 'clarification':s['requires_clarification'],'unresolved':bool(s['unresolved_requirements'])}
rows=[]
for e in gold:
 request_file=root/'examples'/e['id']/'request.txt'
 output=a.output/e['id']
 cmd=[sys.executable,str(root/'translate_policy.py'),'--request-file',str(request_file),'--profile',e['profile_id'],'--output',str(output)]
 if a.execute:cmd.append('--execute')
 if a.key_env_file:cmd.extend(['--key-env-file',str(a.key_env_file)])
 result=subprocess.run(cmd,capture_output=True,text=True)
 if result.returncode!=0:
  # Contract/validation failure is recorded; no automatic API retry.
  rows.append({'case':e['id'],'status':'failed_or_invalid','details':result.stdout.strip() or 'Translator process failed'})
  if a.execute:break
  continue
 matches=list(output.glob('*/translated_spec.json'))
 if not matches:rows.append({'case':e['id'],'status':'offline_prepared'});continue
 # Retain exact run contract by matching expected compiler configuration, not newest-file ordering.
 matching=[]
 for path in matches:
  config=json.loads((path.parent/'contract.json').read_text())['config']
  if config['prompt']==PROMPT and config['schema']==SCHEMA and config['registry_sha256']==sha(registry) and config['model']==MODEL and config['request']==e['request'] and config['profile']==e['profile_id']:
   matching.append(path)
 if len(matching)!=1:raise ValueError('Ambiguous benchmark contracts')
 spec=json.loads(matching[0].read_text());compiled=select(registry,spec,e['request'],e['profile_id'])
 rows.append({'case':e['id'],'status':'evaluated','semantic_match':semantics(spec)==semantics(e['gold_spec']),
 'selection_match':compiled['status']==e['expected_status'] and compiled['selected_model']==e['expected_model'],
 'actual_status':compiled['status'],'selected_model':compiled['selected_model']})
a.output.mkdir(parents=True,exist_ok=True)
summary={'cases':len(gold),'evaluated':sum(r['status']=='evaluated' for r in rows),
 'semantic_matches':sum(r.get('semantic_match',False) for r in rows),'selection_matches':sum(r.get('selection_match',False) for r in rows),
 'note':'13 hand-authored development cases; not independent evidence of generalization; equivalent FPR/specificity and FNR/sensitivity objectives normalized',
 'results':rows}
(a.output/'benchmark_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
