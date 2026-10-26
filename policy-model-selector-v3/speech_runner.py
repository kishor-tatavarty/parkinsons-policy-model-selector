"""Unified telephone/Italian selection and batch inference. Run beside policy_engine.py."""
import argparse,csv,hashlib,json,subprocess,sys
from pathlib import Path
from policy_engine import select,sha,metrics
from request_boundary import require_requirements_only

PROFILES={
 'telephone':('speech_config.json','predict_speech.py','speech_extracted_v2','speech_'),
 'italian':('italian_config.json','predict_italian.py','italian_vowel_a_grouped_v1','italian_'),
}
MODELS=('acoustic_lr','acoustic_rf','spectrogram_resnet18_linear')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):p.write_text(json.dumps(obj,indent=2),encoding='utf-8')
def execute_case(record,wav,project,predictor,artifact_dir,rscript,destination):
    name=record['model_id'].split('__',1)[1]
    artifact=project/'artifacts'/artifact_dir/(name+'.joblib')
    if digest(artifact)!=record['artifact_sha256']:raise ValueError('Registered artifact hash mismatch')
    before=digest(wav)
    run=subprocess.run([sys.executable,str(project/predictor),'--model',name,'--wav',str(wav),'--out',str(destination),'--rscript',rscript],cwd=project,capture_output=True,text=True)
    if run.returncode:raise RuntimeError((run.stdout+'\n'+run.stderr)[-4000:])
    result=json.loads(destination.read_text())
    if (result['profile_id']!=record['profile_id'] or result['selected_tool']!=name or result['model_sha256']!=record['artifact_sha256'] or result['input_sha256']!=before or digest(wav)!=before):raise ValueError('Prediction binding mismatch')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',required=True,choices=PROFILES)
    selection=p.add_mutually_exclusive_group(required=True)
    selection.add_argument('--policy',type=Path)
    selection.add_argument('--choose',choices=MODELS)
    selection.add_argument('--request',help='Natural-language performance requirements')
    p.add_argument('--input',required=True,type=Path,help='A WAV file or directory; directories scanned recursively')
    p.add_argument('--output',required=True,type=Path,help='New output directory')
    p.add_argument('--execute',action='store_true',help='Enable API translation for --request')
    p.add_argument('--key-env-file',type=Path)
    a=p.parse_args();root=Path(__file__).resolve().parent
    config_name,predictor,artifact_dir,prefix=PROFILES[a.profile]
    config=json.loads((root/config_name).read_text());profile=config['profile_id']
    registry=json.loads((root/'model_registry.json').read_text());evaluation=registry['profiles'][profile]['evaluation_id']
    project=Path(config['speech_project']).resolve();source=a.input.resolve();out=a.output.resolve()
    if out.exists():raise ValueError('Output already exists; choose a new directory')
    files=[source] if source.is_file() and source.suffix.lower()=='.wav' else sorted(x for x in source.rglob('*') if x.is_file() and x.suffix.lower()=='.wav') if source.is_dir() else []
    if not files:raise ValueError('No WAV inputs found')
    out.mkdir(parents=True)
    candidates=[m for m in registry['models'] if m['profile_id']==profile and m['evaluation_id']==evaluation]
    if a.choose:
        matches=[m for m in candidates if m['tool_id']==prefix+a.choose]
        if len(matches)!=1:raise ValueError('Explicit choice not uniquely registered')
        record=matches[0]
        decision=dict(status='selected',selection_mode='explicit_user_choice',request=None,policy_constraints_checked=False,profile_id=profile,evaluation_id=evaluation,registry_sha256=sha(registry),selected_model=record['model_id'],selected_tool=record['tool_id'],selected_metrics=metrics(record),reason='User selected model with --choose')
    else:
        if a.request:
            request_file=out/'request.txt';request_file.write_text(a.request,encoding='utf-8')
            command=[sys.executable,str(root/'translate_policy.py'),'--request-file',str(request_file),'--profile',profile,'--output',str(out/'translation')]
            if a.execute:command.append('--execute')
            if a.key_env_file:command+=['--key-env-file',str(a.key_env_file.resolve())]
            run=subprocess.run(command,cwd=root,capture_output=True,text=True)
            (out/'translation.log').write_text(run.stdout+'\n'+run.stderr,encoding='utf-8')
            if run.returncode:raise RuntimeError('Translation failed; see translation.log')
            paths=list((out/'translation').glob('*/compiled_policy.json'))
            if not paths:
                print('Translation prepared; no inference. Add --execute and use a new output directory.');return
            if len(paths)!=1:raise ValueError('Ambiguous compiled policy')
            saved=json.loads(paths[0].read_text())
        else:saved=json.loads(a.policy.read_text(encoding='utf-8-sig'))
        require_requirements_only(saved['request'])
        if saved['registry_sha256']!=sha(registry):raise ValueError('Registry changed; recompile policy')
        decision=select(registry,saved['policy'],saved['request'],profile)
        if decision['policy_sha256']!=saved['policy_sha256'] or decision['status']!=saved['status'] or decision.get('selected_model')!=saved.get('selected_model') or decision.get('selected_tool')!=saved.get('selected_tool'):raise ValueError('Compiled policy mismatch')
        if decision['status']=='selected':
            matches=[m for m in candidates if m['model_id']==decision['selected_model']]
            if len(matches)!=1:raise ValueError('Selected model not uniquely registered')
            record=matches[0]
    write(out/'selection.json',decision)
    if decision['status']!='selected':
        write(out/'summary.json',dict(status=decision['status'],inputs=len(files),executed=0));print(json.dumps(decision,indent=2));return
    rows=[];case_dir=out/'cases';case_dir.mkdir()
    for index,wav in enumerate(files,1):
        name=f'{index:04d}_{hashlib.sha256(str(wav).encode()).hexdigest()[:12]}'
        destination=case_dir/(name+'.json')
        row=dict(input_path=str(wav),status='failed',prediction='',raw_pd_score='',previously_used_for_fitting='',error='')
        try:
            result=execute_case(record,wav,project,predictor,artifact_dir,config['rscript'],destination)
            write(destination,dict(selection_sha256=sha(decision),selected_model=record['model_id'],selected_tool=record['tool_id'],tool_version=record['tool_version'],result=result))
            row.update(status='complete',prediction=result['prediction'],raw_pd_score=result['raw_pd_score'],previously_used_for_fitting=result['previously_used_for_fitting'])
        except Exception as error:
            row['error']=str(error);write(destination,dict(status='failed',input_path=str(wav),error=str(error),selection_sha256=sha(decision)))
        rows.append(row);print(f"{index}/{len(files)} {wav.name}: {row['status']}",flush=True)
    with (out/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    completed=sum(r['status']=='complete' for r in rows)
    summary=dict(status='complete' if completed==len(rows) else 'partial' if completed else 'failed',profile_id=profile,selected_model=record['model_id'],inputs=len(rows),completed=completed,failed=len(rows)-completed,scope='Inference execution; no accuracy computed without labels')
    write(out/'summary.json',summary);print(json.dumps(summary,indent=2))
    if completed!=len(rows):raise SystemExit(2)
if __name__=='__main__':main()
