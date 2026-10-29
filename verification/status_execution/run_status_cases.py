"""No API calls. Structured route/dispatch checks, not a language benchmark."""
import argparse,csv,hashlib,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--release',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
for flag in ['kaggle-image','newhandpd-image','telephone-wav','italian-wav']:p.add_argument('--'+flag,type=Path)
a=p.parse_args();root=a.release.resolve();selector=root/'policy-model-selector-v3';here=Path(__file__).resolve().parent
if a.output.exists():raise SystemExit('Output exists; use a new directory')
a.output.mkdir(parents=True);sys.path.insert(0,str(selector));from policy_engine import select,sha
registry=json.loads((selector/'model_registry.json').read_text());results=[]
def spec(profile,metric='accuracy',quote='Maximize accuracy'):
 return dict(profile_id=profile,primary_metric=metric,primary_quote=quote,metric_constraints=[],cpu_required=None,cpu_quote=None,max_latency_ms=None,latency_quote=None,max_cost_usd_per_case=None,cost_quote=None,tie_breakers=[],requires_clarification=False,clarification_questions=[],unresolved_requirements=[])
profiles=[('kaggle','kaggle_spiral_v1',a.kaggle_image,'image','hog_rf'),('newhandpd','newhandpd_spiral_v1',a.newhandpd_image,'image','resnet18_linear'),('telephone','telephone_vowel_81_extracted_v2',a.telephone_wav,'speech','acoustic_rf'),('italian','italian_vowel_a_grouped_v1',a.italian_wav,'speech','acoustic_lr')]
for alias,profile,source,kind,winner in profiles:
 cases=[];cases.append(('selected','Maximize accuracy.',spec(profile),'selected',profile+'__'+winner))
 s=spec(profile,None,None);s.update(requires_clarification=True,clarification_questions=['Which metric should define best?']);cases.append(('clarification','Choose the best model.',s,'needs_clarification',None))
 s=spec(profile);s['metric_constraints']=[dict(metric='sensitivity',operator='min',value=1.0,quote='sensitivity at least 100%')];cases.append(('no_eligible','Maximize accuracy with sensitivity at least 100%.',s,'no_eligible_model',None))
 s=spec(profile);s.update(max_latency_ms=1,latency_quote='latency at most 1 ms');cases.append(('missing_deployment','Maximize accuracy with latency at most 1 ms.',s,'insufficient_deployment_evidence',None))
 s=spec(profile);s['metric_constraints']=[dict(metric='sensitivity',operator='min',value=.99,quote='sensitivity at least 80%')];cases.append(('invalid','Maximize accuracy with sensitivity at least 80%.',s,'invalid',None))
 if alias in ['kaggle','italian']:
  metric='sensitivity' if alias=='kaggle' else 'specificity';cases.append(('tied','Maximize '+metric+'.',spec(profile,metric,'Maximize '+metric),'tied',None))
 if alias=='kaggle':
  s=spec(profile);s['metric_constraints']=[dict(metric='specificity',operator='max',value=.85,quote='specificity at most 85%')];cases.append(('alternate_selected','Maximize accuracy with specificity at most 85%.',s,'selected',profile+'__resnet18_linear'))
 if alias=='newhandpd':cases.append(('alternate_selected','Maximize specificity.',spec(profile,'specificity','Maximize specificity'),'selected',profile+'__hog_rf'))
 for name,request,s,expected,expected_model in cases:
  case=a.output/(alias+'_'+name);case.mkdir();invalid=False
  try:decision=select(registry,s,request,profile)
  except ValueError as e:
   if expected!='invalid':raise
   invalid=True;decision={'status':'invalid','selected_model':None,'validation_error':str(e)}
  assert decision['status']==expected and decision.get('selected_model')==expected_model,(alias,name,decision)
  (case/'reference.json').write_text(json.dumps(dict(request=request,policy=s,expected_status=expected,expected_model=expected_model,reference_origin='developer-defined structured control; no LLM interpretation tested'),indent=2))
  (case/'decision.json').write_text(json.dumps(decision,indent=2))
  row=dict(case_id=alias+'_'+name,profile=profile,expected_status=expected,observed_status=decision['status'],expected_model=expected_model,selected_model=decision.get('selected_model'),decision_match=True,entrypoint_status='pending_input',predictor_invocations='',execution_pass='',evidence_kind='structured decision only',input_sha256='',error='')
  if source is not None:
   source=source.resolve()
   if not source.is_file():raise SystemExit('Input missing: '+str(source))
   row['input_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
   saved=decision
   if invalid:
    saved=select(registry,spec(profile),'Maximize accuracy.',profile);saved['request']=request;saved['policy']=s
   policy=case/'policy.json';policy.write_text(json.dumps(saved,indent=2))
   trace=case/'predictor_trace.json'
   cmd=[sys.executable,str(here/'probe_entrypoint.py'),'--kind',kind,'--entry',str(selector/('run_inference.py' if kind=='image' else 'speech_runner.py')),'--trace',str(trace),'--']
   if kind=='image':
    output=case/'prediction.json';cmd+=['--policy',str(policy.resolve()),'--profile',profile,'--models',str(selector/'trained_tools'),'--image',str(source),'--output',str(output.resolve())]
   else:
    output=case/'execution';cmd+=['--policy',str(policy.resolve()),'--profile',alias,'--input',str(source),'--output',str(output.resolve())]
   run=subprocess.run(cmd,capture_output=True,text=True);(case/'entrypoint_stdout.txt').write_text(run.stdout);(case/'entrypoint_stderr.txt').write_text(run.stderr)
   observed=json.loads(trace.read_text());calls=observed['predictor_invocations'];n=len(calls);passed=False
   if expected=='selected' and run.returncode==0 and n==1:
    if kind=='image':
     report=json.loads(output.read_text());bound=report['artifact_binding'];result=report['result'];passed=bound['model_id']==expected_model and bound['artifact_sha256']==next(m['artifact_sha256'] for m in registry['models'] if m['model_id']==expected_model) and report['input_sha256']==row['input_sha256'] and calls[0]['tool_id']==decision['selected_tool'] and result['tool_id']==decision['selected_tool']
    else:
     summary=json.loads((output/'summary.json').read_text());reports=[json.loads(x.read_text()) for x in (output/'cases').glob('*.json')];passed=summary.get('completed')==1 and summary['selected_model']==expected_model and len(reports)==1 and reports[0]['selected_model']==expected_model and calls[0]['tool_id']==decision['selected_tool']
   elif expected!='selected':
    if kind=='image':passed=n==0 and run.returncode!=0 and not output.exists()
    elif expected=='invalid':passed=n==0 and run.returncode!=0 and not (output/'summary.json').exists()
    else:passed=n==0 and run.returncode==0 and json.loads((output/'summary.json').read_text()).get('executed')==0
   row.update(entrypoint_status='passed' if passed else 'failed',predictor_invocations=n,execution_pass=bool(passed),evidence_kind='real predictor' if expected=='selected' else 'real entry-point refusal; no predictor',error='' if passed else observed.get('exception','Unexpected execution outcome'))
  results.append(row);print(row['case_id'],row['observed_status'],row['entrypoint_status'],flush=True)
with (a.output/'cases.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
summary=dict(cases=len(results),decision_matches=sum(r['decision_match'] for r in results),entrypoint_passed=sum(r['entrypoint_status']=='passed' for r in results),entrypoint_failed=sum(r['entrypoint_status']=='failed' for r in results),pending_inputs=sum(r['entrypoint_status']=='pending_input' for r in results),actual_predictor_invocations=sum(r['predictor_invocations'] for r in results if isinstance(r['predictor_invocations'],int)),api_calls=0,classifier_accuracy_evaluated=False,scope='structured selection and execution controls; no new language benchmark',registry_sha256=sha(registry),source_hashes={n:hashlib.sha256((selector/n).read_bytes()).hexdigest() for n in ['policy_engine.py','run_inference.py','speech_runner.py','artifact_binding.py','request_boundary.py']})
(a.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if summary['entrypoint_failed']:raise SystemExit(1)
