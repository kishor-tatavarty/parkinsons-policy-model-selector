"""Deterministic research model selection. Standard library only."""
import hashlib,json,math,re
METRICS=('accuracy','balanced_accuracy','sensitivity','specificity','false_positive_rate','false_negative_rate')
HIGHER={'accuracy','balanced_accuracy','sensitivity','specificity'}

def sha(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()

def metrics(record):
 c=record['confusion'];tn,fp,fn,tp=(c[k] for k in ['tn','fp','fn','tp'])
 if any(type(v)!=int or v<0 for v in [tn,fp,fn,tp]) or tn+fp==0 or tp+fn==0:raise ValueError('Invalid confusion counts')
 sensitivity=tp/(tp+fn);specificity=tn/(tn+fp)
 return {'accuracy':(tp+tn)/(tp+tn+fp+fn),'balanced_accuracy':(sensitivity+specificity)/2,
  'sensitivity':sensitivity,'specificity':specificity,'false_positive_rate':1-specificity,'false_negative_rate':1-sensitivity}

def number_grounded(value,quote,percent_allowed=True):
 numbers=[float(x) for x in re.findall(r'(?<![\w.])\d+(?:\.\d+)?',quote)]
 return any(math.isclose(value,n,abs_tol=1e-9) or (percent_allowed and math.isclose(value,n/100,abs_tol=1e-9)) for n in numbers)

def validate(spec,request,profile_id):
 expected={'profile_id','primary_metric','primary_quote','metric_constraints','cpu_required','cpu_quote',
 'max_latency_ms','latency_quote','max_cost_usd_per_case','cost_quote','tie_breakers','requires_clarification',
 'clarification_questions','unresolved_requirements'}
 if set(spec)!=expected:raise ValueError('Policy fields differ from schema')
 if spec['profile_id']!=profile_id:raise ValueError('Translator changed trusted input profile')
 if spec['primary_metric'] not in (*METRICS,None):raise ValueError('Unsupported primary metric')
 if type(spec['requires_clarification'])!=bool:raise ValueError('Invalid clarification flag')
 if spec['cpu_required'] not in (True,False,None) or (spec['cpu_required'] is not None and type(spec['cpu_required'])!=bool):raise ValueError('Invalid CPU requirement')
 def quote(q):
  if not isinstance(q,str) or not q.strip() or q not in request:raise ValueError('Missing or non-verbatim requirement evidence')
 if spec['primary_metric'] is not None:quote(spec['primary_quote'])
 if not isinstance(spec['metric_constraints'],list):raise ValueError('Invalid constraint list')
 for c in spec['metric_constraints']:
  if set(c)!={'metric','operator','value','quote'} or c['metric'] not in METRICS or c['operator'] not in ('min','max'):raise ValueError('Invalid metric constraint')
  v=c['value']
  if type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1:raise ValueError('Metric threshold outside0..1')
  quote(c['quote'])
  if not number_grounded(v,c['quote']):raise ValueError('Ungrounded metric threshold')
 if spec['cpu_required'] is not None:quote(spec['cpu_quote'])
 for key,q in [('max_latency_ms','latency_quote'),('max_cost_usd_per_case','cost_quote')]:
  v=spec[key]
  if v is not None:
   if type(v) not in (int,float) or not math.isfinite(v) or v<0:raise ValueError('Invalid deployment limit')
   quote(spec[q])
   if not number_grounded(v,spec[q],False):raise ValueError('Ungrounded deployment number; use explicit ms/USD units')
 if not isinstance(spec['tie_breakers'],list):raise ValueError('Invalid tie breakers')
 seen=set()
 for t in spec['tie_breakers']:
  if set(t)!={'metric','quote'} or t['metric'] not in METRICS or t['metric']==spec['primary_metric'] or t['metric'] in seen:raise ValueError('Invalid tie breaker')
  quote(t['quote']);seen.add(t['metric'])
 for k in ['clarification_questions','unresolved_requirements']:
  if not isinstance(spec[k],list) or not all(isinstance(x,str) and x.strip() for x in spec[k]):raise ValueError('Invalid clarification text')
 if spec['requires_clarification'] and not spec['clarification_questions']:raise ValueError('Clarification requested without questions')
 return spec

def select(registry,spec,request,profile_id):
 validate(spec,request,profile_id)
 result={'policy_version':'1.0','request':request,'profile_id':profile_id,'policy':spec,
  'registry_sha256':sha(registry),'policy_sha256':sha(spec),'scope':'research selection; not a clinical deployment approval'}
 profile=registry['profiles'].get(profile_id)
 if not profile:return {**result,'status':'unsupported_profile','selected_model':None}
 if spec['requires_clarification'] or spec['unresolved_requirements'] or spec['primary_metric'] is None:
  return {**result,'status':'needs_clarification','selected_model':None,'questions':spec['clarification_questions'] or ['Which performance objective and unresolved limits should govern selection?']}
 candidates=[m for m in registry['models'] if m['profile_id']==profile_id and m['evaluation_id']==profile['evaluation_id']]
 if not candidates:return {**result,'status':'no_evaluation_evidence','selected_model':None}
 considered=[];eligible=[];unknown=[]
 for m in candidates:
  mm=metrics(m);reasons=[];missing=[]
  for c in spec['metric_constraints']:
   v=mm[c['metric']]
   if (c['operator']=='min' and v<c['value']-1e-12) or (c['operator']=='max' and v>c['value']+1e-12):
    reasons.append(c['metric']+' violates '+c['operator']+' '+str(c['value']))
  dep=m.get('deployment',{})
  if spec['cpu_required'] is True:
   if dep.get('cpu_supported') is None:missing.append('cpu_supported')
   elif dep['cpu_supported'] is not True:reasons.append('CPU execution unsupported')
  for field,limit in [('latency_ms',spec['max_latency_ms']),('cost_usd_per_case',spec['max_cost_usd_per_case'])]:
   if limit is not None:
    if dep.get(field) is None:missing.append(field)
    elif dep[field]>limit:reasons.append(field+' exceeds limit')
  considered.append({'model_id':m['model_id'],'metrics':mm,'rejected_reasons':reasons,'missing_evidence':missing})
  if not reasons and missing:unknown.append(m['model_id'])
  elif not reasons:eligible.append((m,mm))
 result.update(evaluation_id=profile['evaluation_id'],evaluation_scope=profile['scope'],considered=considered)
 # Unknown eligible candidates prevent a claim that a known candidate is best.
 if unknown:return {**result,'status':'insufficient_deployment_evidence','selected_model':None,'unknown_models':unknown}
 if not eligible:return {**result,'status':'no_eligible_model','selected_model':None}
 priority=[spec['primary_metric']]+[t['metric'] for t in spec['tie_breakers']]
 def key(pair):return tuple(pair[1][k] if k in HIGHER else -pair[1][k] for k in priority)
 winner=max(eligible,key=key)
 tied=[p[0]['model_id'] for p in eligible if all(abs(a-b)<=1e-12 for a,b in zip(key(p),key(winner)))]
 if len(tied)>1:return {**result,'status':'tied','selected_model':None,'tied_models':tied}
 m,mm=winner
 return {**result,'status':'selected','selected_model':m['model_id'],'selected_tool':m['tool_id'],
  'tool_version':m['tool_version'],'selected_metrics':mm,'ranking_metrics':priority,
  'reason':'All explicit constraints satisfied; best lexicographic rank on '+', '.join(priority)}

def dispatch(policy,profile_id,tool_handlers):
 """Caller supplies trusted, registered callable adapters. No arbitrary code execution."""
 if policy.get('status')!='selected' or policy.get('profile_id')!=profile_id:raise ValueError('No matching selected policy')
 tool=policy['selected_tool']
 if tool not in tool_handlers:raise ValueError('Tool adapter not installed: '+tool)
 return tool_handlers[tool]()
