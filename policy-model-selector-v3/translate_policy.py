"""LLM translates text; deterministic engine selects. Dry-run default."""
import argparse,hashlib,json,os
from pathlib import Path
import urllib.request,urllib.error
from policy_engine import select,sha
from request_boundary import require_requirements_only
from schema_and_prompt import SCHEMA,PROMPT,PROFILE_IDS
MODEL='gpt-4.1-mini-2025-04-14'

def parse(response):
 if response.get('status')!='completed':raise ValueError('Incomplete response; no retry')
 parts=[c for o in response.get('output',[]) for c in o.get('content',[])]
 if any(c.get('type')=='refusal' for c in parts):raise ValueError('Refusal; no retry')
 return json.loads(''.join(c.get('text','') for c in parts if c.get('type')=='output_text'))

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--request-file',required=True,type=Path)
 p.add_argument('--profile',required=True,choices=PROFILE_IDS)
 p.add_argument('--registry',type=Path,default=Path(__file__).parent/'model_registry.json')
 p.add_argument('--output',type=Path,default=Path('results/policy_translation'))
 p.add_argument('--execute',action='store_true')
 p.add_argument('--key-env-file',type=Path)
 a=p.parse_args()
 request=a.request_file.read_text(encoding='utf-8-sig').strip()
 if not request or len(request)>6000:raise ValueError('Request must contain1..6000 characters')
 require_requirements_only(request)
 registry=json.loads(a.registry.read_text(encoding='utf-8'))
 config={'model':MODEL,'prompt':PROMPT,'schema':SCHEMA,'request':request,'profile':a.profile,
  'registry_sha256':sha(registry),'max_output_tokens':1400,'temperature':0,
  'engine_sha256':hashlib.sha256((Path(__file__).parent/'policy_engine.py').read_bytes()).hexdigest()}
 contract=sha(config);directory=a.output/contract[:16];directory.mkdir(parents=True,exist_ok=True)
 path=directory/'response.json'
 body={'model':MODEL,'store':False,'temperature':0,'max_output_tokens':1400,
 'input':[{'role':'system','content':PROMPT},{'role':'user','content':json.dumps({'trusted_profile_id':a.profile,'request':request})}],
 'text':{'format':{'type':'json_schema','name':'model_selection_policy','strict':True,'schema':SCHEMA}}}
 contract_path=directory/'contract.json'
 data={'contract_sha256':contract,'config':config}
 if contract_path.exists() and json.loads(contract_path.read_text())!=data:raise ValueError('Contract conflict')
 contract_path.write_text(json.dumps(data,indent=2),encoding='utf-8')
 print(json.dumps({'model':MODEL,'profile':a.profile,'network_enabled':a.execute,'output':str(directory),'cached_response':path.exists()},indent=2))
 if path.exists():response=json.loads(path.read_text())
 elif not a.execute:print('Offline preparation complete. No API call made.');return
 else:
  marker=directory/'attempted.json'
  if marker.exists():raise ValueError('Previous attempt has no saved response; stop to avoid duplicate billing')
  key=os.environ.get('OPENAI_API_KEY','').strip()
  if a.key_env_file:
   for line in a.key_env_file.read_text(encoding='utf-8-sig').splitlines():
    line=line.strip().removeprefix('export ')
    if line.startswith('OPENAI_API_KEY='):key=line.split('=',1)[1].strip().strip('"\x27');break
  if not key:raise ValueError('OPENAI_API_KEY unavailable; no request sent')
  marker.write_text(json.dumps({'contract_sha256':contract}),encoding='utf-8')
  req=urllib.request.Request('https://api.openai.com/v1/responses',data=json.dumps(body).encode(),
   headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
  try:
   with urllib.request.urlopen(req,timeout=120) as reply:response=json.load(reply)
  except urllib.error.HTTPError as e:raise RuntimeError('API HTTP '+str(e.code)+';stopped without retry') from None
  except Exception:raise RuntimeError('Network/response failure; stopped without retry') from None
  path.write_text(json.dumps(response,indent=2),encoding='utf-8')
 spec=parse(response)
 (directory/'translated_spec.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
 result=select(registry,spec,request,a.profile)
 (directory/'compiled_policy.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 print(json.dumps(result,indent=2))

if __name__=='__main__':
 try:main()
 except (ValueError,RuntimeError,FileNotFoundError) as e:raise SystemExit(str(e))
