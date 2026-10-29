"""Observe real entry points; wrappers call the original predictor unchanged."""
import argparse,builtins,hashlib,json,runpy,sys,traceback
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--kind',choices=['image','speech'],required=True);p.add_argument('--entry',type=Path,required=True);p.add_argument('--trace',type=Path,required=True);p.add_argument('args',nargs=argparse.REMAINDER);a=p.parse_args()
trace={'predictor_invocations':[],'mock_predictor':False}
sys.path.insert(0,str(a.entry.resolve().parent));sys.argv=[str(a.entry)]+(a.args[1:] if a.args[:1]==['--'] else a.args)
original_import=builtins.__import__
def observe_import(name,*args,**kwargs):
 module=original_import(name,*args,**kwargs)
 if name=='image_tools' and not getattr(module,'_evidence_observed',False):
  real=module.infer
  def observe_infer(raw,tool,artifact_root,profile):
   trace['predictor_invocations'].append({'tool_id':tool,'profile_id':profile,'input_sha256':hashlib.sha256(raw).hexdigest()})
   return real(raw,tool,artifact_root,profile)
  module.infer=observe_infer;module._evidence_observed=True
 return module
try:
 if a.kind=='image':
  builtins.__import__=observe_import;runpy.run_path(str(a.entry),run_name='__main__')
 else:
  namespace=runpy.run_path(str(a.entry),run_name='observed_speech_runner');main=namespace['main'];g=main.__globals__;real=g['execute_case']
  def observe_execute(record,wav,*args,**kwargs):
   trace['predictor_invocations'].append({'tool_id':record['tool_id'],'profile_id':record['profile_id'],'input_sha256':hashlib.sha256(wav.read_bytes()).hexdigest()})
   return real(record,wav,*args,**kwargs)
  g['execute_case']=observe_execute;main()
except BaseException as e:
 trace['exception_type']=type(e).__name__;trace['exception']=str(e)
 if isinstance(e,SystemExit) and e.code in (0,None):pass
 else:traceback.print_exc();raise SystemExit(1)
finally:
 builtins.__import__=original_import;a.trace.parent.mkdir(parents=True,exist_ok=True);a.trace.write_text(json.dumps(trace,indent=2))
