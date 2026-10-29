"""Freeze, audit references and run parser/control. No network or classifiers.
Reference-derived fields are confined to the oracle control and scorer. Parser
receives only request/profile. Results are provisional until reference review.
"""
import argparse,collections,csv,hashlib,json,math,sys,time,types
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'frozen'))
import policy_engine as engine
from rule_parser import parse

def readlines(p):
 return [json.loads(l) for l in p.read_text(encoding='utf-8-sig').splitlines() if l.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def numeric_select(reg,spec,request,profile):
 # Exact frozen engine ranking, excluding only language-specific quote/numeric
 # grounding for the ideal structured-field control. Never used for execution.
 g=dict(engine.select.__globals__);g['validate']=lambda *args:None
 fn=types.FunctionType(engine.select.__code__,g)
 return fn(reg,spec,request,profile)
def signature(s):
 if s is None:return None
 def canonical(m,op=None,v=None):
  if m in ('specificity','sensitivity'):
   m={'specificity':'false_positive_rate','sensitivity':'false_negative_rate'}[m]
   if op:op={'min':'max','max':'min'}[op];v=1-v
  return m,op,round(v,10) if v is not None else None
 return dict(profile=s['profile_id'],objective=canonical(s['primary_metric'])[0],constraints=sorted(set(canonical(c['metric'],c['operator'],c['value']) for c in s['metric_constraints'])),ties=[canonical(t['metric'])[0] for t in s['tie_breakers']],cpu=True if s['cpu_required'] is True else None,latency=s['max_latency_ms'],cost=s['max_cost_usd_per_case'],clarify=bool(s['requires_clarification']),unresolved_present=bool(s['unresolved_requirements']))
def explicit(reg,profile,model):
 m=[m for m in reg['models'] if m['profile_id']==profile and m['evaluation_id']==reg['profiles'][profile]['evaluation_id'] and m['model_id'].split('__')[-1]==model]
 if len(m)!=1:raise ValueError('Explicit candidate not unique')
 return dict(status='selected',selected_model=m[0]['model_id'])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',default='results/offline_v1');a=ap.parse_args();out=ROOT/a.output
 if out.exists():raise SystemExit('Output already exists; choose a new --output directory.')
 qs=readlines(ROOT/'inputs/requests.jsonl');refs=readlines(ROOT/'references/reference_answers.original.jsonl');qmap={q['case_id']:q for q in qs};rmap={r['case_id']:r for r in refs}
 assert len(qs)==len(qmap)==len(refs)==len(rmap)==120 and set(qmap)==set(rmap)
 assert len({q['request'] for q in qs})==120
 assert set(collections.Counter(r['request_family'] for r in refs).values())=={20}
 assert set(collections.Counter(q['profile_id'] for q in qs).values())=={60}
 files=[p for p in (ROOT/'frozen').iterdir() if p.is_file()]+[ROOT/'inputs/requests.jsonl',ROOT/'references/reference_answers.original.jsonl',ROOT/'run_offline.py']
 freeze={str(p.relative_to(ROOT)):sha(p) for p in files};frozen=ROOT/'freeze.json'
 if frozen.exists() and json.loads(frozen.read_text())['files']!=freeze:raise SystemExit('Frozen source/input changed. Preserve this version; create a separately labelled revision.')
 if not frozen.exists():write(frozen,dict(files=freeze,note='Frozen after test-set construction and initial AI-assisted inspection, before evaluated-method benchmark outputs. No retrospective pre-exposure freeze claim. Scoring is provisional; not independent human review.'))
 reg=json.loads((ROOT/'frozen/model_registry.json').read_text());audit=[];results=[]
 for q in qs:
  ref=rmap[q['case_id']];spec=ref['expected_structured_policy'];notes=[]
  for quote in ref['supporting_quotes']:
   if quote not in q['request']:notes.append('Supporting quote absent')
  if spec:
   try:engine.validate(spec,q['request'],q['profile_id'])
   except ValueError as e:notes.append('Existing validator: '+str(e))
   result=numeric_select(reg,spec,q['request'],q['profile_id'])
  else:result=explicit(reg,q['profile_id'],ref['expected_selected_model'].split('__')[-1])
  if (result['status'],result.get('selected_model'))!=(ref['expected_selection_status'],ref['expected_selected_model']):notes.append('Numerical reference outcome mismatch')
  audit.append(dict(case_id=q['case_id'],notes=notes,reference_review_status=ref['reference_review_status']))
  for arm in ['structured_control','rule_parser']:
   start=time.perf_counter()
   if arm=='structured_control':
    parsed=dict(action='explicit_choice' if spec is None else 'requirements',model=ref['expected_selected_model'].split('__')[-1] if spec is None else None,policy=spec);decision=result
   else:
    parsed=parse(q['request'],q['profile_id'])
    try:decision=explicit(reg,q['profile_id'],parsed['model']) if parsed['action']=='explicit_choice' else engine.select(reg,parsed['policy'],q['request'],q['profile_id'])
    except (ValueError,KeyError,TypeError) as e:decision=dict(status='invalid',selected_model=None,error=str(e))
   duration=(time.perf_counter()-start)*1000
   sem=(parsed.get('action')=='explicit_choice' and parsed.get('model')==ref['expected_selected_model'].split('__')[-1]) if spec is None else signature(parsed.get('policy'))==signature(spec)
   final=(decision['status'],decision.get('selected_model'))==(ref['expected_selection_status'],ref['expected_selected_model'])
   selected=decision['status']=='selected'
   results.append(dict(case_id=q['case_id'],arm=arm,parsed=parsed,decision=decision,latency_ms=duration,provisional_semantic_match=sem,provisional_outcome_match=final,incorrect_authorization=bool(selected and not(sem and final)),selection_authorized=selected,appropriate_clarification=ref['expected_selection_status']=='needs_clarification' and decision['status']=='needs_clarification'))
 out.mkdir(parents=True);write(out/'reference_audit.json',audit);write(out/'case_results.json',results)
 summary={}
 for arm in ['structured_control','rule_parser']:
  rr=[r for r in results if r['arm']==arm];summary[arm]=dict(cases=len(rr),semantic_matches=sum(r['provisional_semantic_match'] for r in rr),outcome_matches=sum(r['provisional_outcome_match'] for r in rr),authorized=sum(r['selection_authorized'] for r in rr),incorrect_authorizations=sum(r['incorrect_authorization'] for r in rr),appropriate_clarifications=sum(r['appropriate_clarification'] for r in rr),reference_clarification_cases=sum(r['expected_selection_status']=='needs_clarification' for r in refs),statuses=dict(collections.Counter(r['decision']['status'] for r in rr)))
 report=dict(status='provisional_offline_check',arms=summary,reference_flags=[r for r in audit if r['notes']],api_calls=0,classifier_executions=0,limits=['References not independently human-reviewed','Clarification scoring checks supported fields and presence of unresolved/clarification flags; it does not establish adequacy of clarification reasons','Scorer normalizes complementary metrics and CPU false/null, but does not interpret arbitrary acceptable_interpretations prose','Strict/missing requirements require human reference adjudication','120 requests observed before this execution freeze; no later tuning permitted','Timings are local single-pass execution, not controlled latency benchmark'])
 write(out/'summary.json',report);print(json.dumps(dict(status=report['status'],cases=120,reference_flags=len(report['reference_flags']),api_calls=0,classifier_executions=0,output=str(out)),indent=2))
if __name__=='__main__':main()
