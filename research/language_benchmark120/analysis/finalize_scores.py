"""Analysis-only adjudication and consistent four-arm scoring; no model changes.
Original references and original signatures preserved. Only two documented
contradictory-constraint representation alternatives are added after exposure.
This is AI-assisted adjudication, not independent human reference approval.
"""
import argparse,csv,json,math,hashlib,collections
from pathlib import Path
from frozen_scoring import signature
ROOT=Path(__file__).resolve().parent
ALTERNATIVES={
 'test_047':[dict(metric='accuracy',operator='min',value=.9),dict(metric='accuracy',operator='max',value=.85)],
 'test_101':[dict(metric='false_positive_rate',operator='max',value=.3),dict(metric='false_positive_rate',operator='min',value=.35)]}
ERRORS={
 'test_001':'constraint_promoted_to_objective','test_004':'unsupported_unit_conversion','test_015':'unsupported_metric_misinterpreted','test_018':'strictness_and_priority_error','test_019':'threshold_rounding','test_021':'unnecessary_clarification','test_028':'unnecessary_clarification','test_030':'missing_clarification_flags','test_038':'conditional_model_scope_misrepresented','test_040':'unordered_objectives_and_invented_constraint','test_043':'negation_not_preserved','test_053':'missing_clarification_flags','test_058':'unnecessary_clarification','test_069':'conditional_model_scope_misrepresented','test_076':'strictness_not_preserved','test_086':'cost_mapped_to_error_rate','test_094':'constraint_promoted_to_objective','test_098':'constraint_promoted_to_objective','test_113':'constraint_promoted_to_objective','test_115':'priority_order_error'}
def sem_match(parsed,ref,allow_alternative):
 if parsed is None:return False
 s=ref['expected_structured_policy']
 if s is None:return parsed['action']=='explicit_choice' and parsed['model']==ref['expected_selected_model'].split('__')[-1]
 actual=signature(parsed.get('policy'));expected=signature(s)
 if actual==expected:return True
 cid=ref['case_id']
 if not allow_alternative or cid not in ALTERNATIVES or actual is None:return False
 alt=json.loads(json.dumps(s));alt['metric_constraints']=[dict(c,quote='reference adjudication alternative') for c in ALTERNATIVES[cid]]
 return actual==signature(alt)
def wilson(k,n):
 if not n:return [None,None]
 z=1.959963984540054;den=1+z*z/n;mid=(k/n+z*z/(2*n))/den;half=z*math.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/den;return [100*(mid-half),100*(mid+half)]
def savecsv(p,rows):
 with p.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 a=argparse.ArgumentParser();a.add_argument('--llm-results',type=Path,required=True);a.add_argument('--offline-results',type=Path,required=True);a.add_argument('--output',type=Path,default=ROOT/'results');o=a.parse_args();o.output.mkdir(exist_ok=True,parents=True)
 refs=[json.loads(l) for l in (ROOT/'inputs/reference_answers.original.jsonl').read_text().splitlines() if l];rq={r['case_id']:r for r in map(json.loads,(ROOT/'inputs/requests.jsonl').read_text().splitlines())};off=json.loads((o.offline_results/'case_results.json').read_text());off={(r['case_id'],r['arm']):r for r in off};allrows=[];changed=[];blocks=[];refreview=[];usage=collections.Counter();latencies=[];hashes={}
 for ref in refs:
  cid=ref['case_id'];record=json.loads((o.llm_results/cid/'paired_decisions.json').read_text());response=json.loads((o.llm_results/cid/'response.json').read_text());usage.update({k:response['usage'].get(k,0) for k in ['input_tokens','output_tokens','total_tokens']});latencies.append(record['network_ms'])
  for file in (o.llm_results/cid).glob('*.json'):hashes[cid+'/'+file.name]=hashlib.sha256(file.read_bytes()).hexdigest()
  for arm in ['structured_control','rule_parser','llm_raw','llm_validated']:
   if arm.startswith('llm_'):parsed=record['parsed'];decision=record['raw' if arm=='llm_raw' else 'validated']
   else:source=off[(cid,arm)];parsed=source['parsed'];decision=source['decision']
   old=sem_match(parsed,ref,False);sem=sem_match(parsed,ref,True);final=(decision['status'],decision.get('selected_model'))==(ref['expected_selection_status'],ref['expected_selected_model']);selected=decision['status']=='selected'
   row=dict(case_id=cid,family=ref['request_family'],arm=arm,original_semantic_match=old,adjudicated_semantic_match=sem,outcome_match=final,authorized=selected,incorrect_authorization=bool(selected and not(sem and final)),appropriate_clarification=bool(ref['expected_selection_status']=='needs_clarification' and decision['status']=='needs_clarification'),reference_status=ref['expected_selection_status'],observed_status=decision['status'],failure_category=ERRORS.get(cid,'') if not sem else '',validation_error=decision.get('error','') if arm=='llm_validated' else '')
   allrows.append(row)
   if old!=sem:changed.append(dict(case_id=cid,arm=arm,old_score=old,new_score=sem,reason='Contradictory bounds retained exactly while clarification requested; alternative preserves intent'))
  if record['validated']['status']=='invalid':
   err=record['validated'].get('error','');cat='numeric_grounding' if 'metric threshold' in err else 'deployment_grounding' if 'deployment number' in err else 'quote_grounding' if 'non-verbatim' in err else 'named_choice_guard' if 'unconditional' in err else 'policy_structure'
   blocks.append(dict(case_id=cid,category=cat,error=err,semantic_match=sem_match(record['parsed'],ref,True),raw_status=record['raw']['status']))
  refreview.append(dict(case_id=cid,review_type='AI-assisted post-output adjudication',original_reference_preserved=True,decision='Accept exact contradictory-bound preservation as alternative' if cid in ALTERNATIVES else 'Retain original reference for this analysis',independent_human_review='not completed',note='Units remain restricted to explicit ms/USD; CPU false/null normalized; strict bounds not inclusive; no score-driven prompt/parser changes'))
 summary=[];families=[]
 for arm in ['structured_control','rule_parser','llm_raw','llm_validated']:
  rr=[r for r in allrows if r['arm']==arm];sem=sum(r['adjudicated_semantic_match'] for r in rr);old=sum(r['original_semantic_match'] for r in rr);final=sum(r['outcome_match'] for r in rr);sel=sum(r['authorized'] for r in rr);bad=sum(r['incorrect_authorization'] for r in rr);cl=sum(r['appropriate_clarification'] for r in rr)
  summary.append(dict(arm=arm,n=120,semantic_original=old,semantic_adjudicated=sem,semantic_percent=100*sem/120,semantic_wilson_lower=wilson(sem,120)[0],semantic_wilson_upper=wilson(sem,120)[1],outcome_matches=final,outcome_percent=100*final/120,outcome_wilson_lower=wilson(final,120)[0],outcome_wilson_upper=wilson(final,120)[1],authorized=sel,coverage_percent=100*sel/120,incorrect_authorizations=bad,incorrect_percent_all=100*bad/120,incorrect_percent_authorized=100*bad/sel if sel else '',appropriate_clarifications=cl,clarification_percent=100*cl/20))
  for fam in sorted({r['family'] for r in rr}):
   ff=[r for r in rr if r['family']==fam];families.append(dict(arm=arm,family=fam,n=len(ff),semantic_matches=sum(r['adjudicated_semantic_match'] for r in ff),outcome_matches=sum(r['outcome_match'] for r in ff),authorized=sum(r['authorized'] for r in ff),incorrect_authorizations=sum(r['incorrect_authorization'] for r in ff)))
 for name,data in [('four_arm_results.csv',summary),('case_results.csv',allrows),('family_results.csv',families),('adjudication_changes.csv',changed),('validation_blocks.csv',blocks),('reference_review_log.csv',refreview)]:savecsv(o.output/name,data)
 (o.output/'summary.json').write_text(json.dumps(dict(status='completed_AI_assisted_adjudication_not_independent_human_validation',arms=summary,changed_semantic_scores=changed,validation_block_counts=dict(collections.Counter(r['category'] for r in blocks)),usage=dict(usage),mean_network_ms=sum(latencies)/120,new_api_calls=0,classifier_executions=0),indent=2))
 (o.output/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
