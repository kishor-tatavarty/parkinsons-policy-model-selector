"""Integration tests using fake prediction adapters; never calls APIs."""
import hashlib,json,pathlib,shutil,subprocess,sys,tempfile
runner=pathlib.Path(__file__).resolve().parents[1]/'policy-model-selector-v3/speech_runner.py'
engine=runner.with_name('policy_engine.py')
with tempfile.TemporaryDirectory() as tmp:
 root=pathlib.Path(tmp);shutil.copy2(runner,root/runner.name);shutil.copy2(engine,root/'policy_engine.py');shutil.copy2(runner.with_name('request_boundary.py'),root/'request_boundary.py')
 sys.path.insert(0,str(root));from policy_engine import select
 reg={'profiles':{},'models':[]}
 for alias,profile,conf,dirname,prefix,predictor,cfg in [
 ('telephone','telephone_vowel_81_extracted_v2',(28,13,13,27),'speech_extracted_v2','speech_','predict_speech.py','speech_config.json'),
 ('italian','italian_vowel_a_grouped_v1',(40,3,6,49),'italian_vowel_a_grouped_v1','italian_','predict_italian.py','italian_config.json')]:
  reg['profiles'][profile]={'evaluation_id':'test','scope':'fixture'}
  project=root/alias;project.mkdir();artifact=project/'artifacts'/dirname/'acoustic_lr.joblib';artifact.parent.mkdir(parents=True);artifact.write_bytes(b'fixture')
  h=hashlib.sha256(b'fixture').hexdigest()
  reg['models'].append(dict(model_id=profile+'__acoustic_lr',profile_id=profile,evaluation_id='test',confusion=dict(zip(['tn','fp','fn','tp'],conf)),tool_id=prefix+'acoustic_lr',tool_version='test',artifact_sha256=h))
  (root/cfg).write_text(json.dumps(dict(profile_id=profile,speech_project=str(project),rscript='not-needed')))
  (project/predictor).write_text('''import argparse,json,hashlib\nfrom pathlib import Path\np=argparse.ArgumentParser();p.add_argument('--model');p.add_argument('--wav');p.add_argument('--out');p.add_argument('--rscript');a=p.parse_args()\nif Path(a.wav).name=='bad.wav':raise SystemExit('fixture decode failure')\nPath(a.out).write_text(json.dumps(dict(profile_id='''+repr(profile)+''',selected_tool=a.model,model_sha256='''+repr(h)+''',input_sha256=hashlib.sha256(Path(a.wav).read_bytes()).hexdigest(),prediction='healthy',raw_pd_score=.2,previously_used_for_fitting=False)))\n''')
 (root/'model_registry.json').write_text(json.dumps(reg))
 inputs=root/'inputs';inputs.mkdir();(inputs/'a.wav').write_bytes(b'fixture audio');(inputs/'b.wav').write_bytes(b'other audio')
 def run(alias,selection,out,code=0):
  r=subprocess.run([sys.executable,str(root/'speech_runner.py'),'--profile',alias,*selection,'--input',str(inputs),'--output',str(out)],capture_output=True,text=True)
  assert r.returncode==code,(r.stdout,r.stderr);return json.loads((out/'summary.json').read_text())
 for alias in ['telephone','italian']:
  s=run(alias,['--choose','acoustic_lr'],root/(alias+'_success'));assert s['completed']==2
 profile='italian_vowel_a_grouped_v1';request='Maximize accuracy with sensitivity at least 95%'
 spec=dict(profile_id=profile,primary_metric='accuracy',primary_quote='Maximize accuracy',metric_constraints=[dict(metric='sensitivity',operator='min',value=.95,quote='sensitivity at least 95%')],cpu_required=None,cpu_quote=None,max_latency_ms=None,latency_quote=None,max_cost_usd_per_case=None,cost_quote=None,tie_breakers=[],requires_clarification=False,clarification_questions=[],unresolved_requirements=[])
 policy=root/'policy.json';policy.write_text(json.dumps(select(reg,spec,request,profile)))
 s=run('italian',['--policy',str(policy)],root/'reject');assert s['executed']==0 and s['status']=='no_eligible_model'
 spec['metric_constraints']=[];policy.write_text(json.dumps(select(reg,spec,'Maximize accuracy',profile)))
 s=run('italian',['--policy',str(policy)],root/'policy_success');assert s['completed']==2
 (inputs/'bad.wav').write_bytes(b'broken')
 s=run('italian',['--choose','acoustic_lr'],root/'partial',2);assert s['completed']==2 and s['failed']==1
 print('Passed: both profile routes; policy selection; unmet-constraint no-execution; batch failure isolation.')
