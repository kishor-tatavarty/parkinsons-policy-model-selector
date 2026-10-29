from pathlib import Path
import json,sys,tempfile,zipfile,subprocess,csv,argparse
p=argparse.ArgumentParser();p.add_argument('--archive',required=True,type=Path);p.add_argument('--output',required=True,type=Path);args=p.parse_args()
if args.output.exists():raise SystemExit('Output exists; choose a new directory')
root=Path(__file__).resolve().parent;sel=root/'policy-model-selector-v3';sys.path.insert(0,str(sel));from policy_engine import select
registry=json.loads((sel/'model_registry.json').read_text());ev=args.output;ev.mkdir(parents=True);profile='kaggle_spiral_v1';records=[]

with tempfile.TemporaryDirectory() as t,zipfile.ZipFile(args.archive) as z:
 temp=Path(t); names=[next(n for n in sorted(z.namelist()) if n.startswith('spiral/testing/'+label+'/') and n.endswith('.png')) for label in ['healthy','parkinson']]
 for model in ['hog_rf','resnet18_linear']:
  request='Maximize accuracy.' if model=='hog_rf' else 'Maximize accuracy with specificity at most 85%.'
  constraints=[] if model=='hog_rf' else [dict(metric='specificity',operator='max',value=.85,quote='specificity at most 85%')]
  spec=dict(profile_id=profile,primary_metric='accuracy',primary_quote='Maximize accuracy',metric_constraints=constraints,cpu_required=None,cpu_quote=None,max_latency_ms=None,latency_quote=None,max_cost_usd_per_case=None,cost_quote=None,tie_breakers=[],requires_clarification=False,clarification_questions=[],unresolved_requirements=[])
  policy=select(registry,spec,request,profile);assert policy['selected_model']==profile+'__'+model
  policy_path=ev/(model+'_policy.json');policy_path.write_text(json.dumps(policy,indent=2))
  for n in names:
   image=temp/Path(n).name;image.write_bytes(z.read(n));out=ev/(model+'_'+image.stem+'_execution.json')
   cmd=[sys.executable,str(sel/'run_inference.py'),'--policy',str(policy_path),'--profile',profile,'--models',str(sel/'trained_tools'),'--image',str(image),'--output',str(out)]
   r=subprocess.run(cmd,capture_output=True,text=True);assert r.returncode==0,r.stderr
   report=json.loads(out.read_text());records.append(dict(model=model,source_member=n,input_sha256=report['input_sha256'],artifact_sha256=report['artifact_binding']['artifact_sha256'],prediction=report['result']['prediction'],score=report['result']['raw_pd_score'],status='passed',real_predictor=True,api_calls=0))
   print(model,n,'passed',flush=True)
(ev/'entrypoint_tests.json').write_text(json.dumps(records,indent=2))
