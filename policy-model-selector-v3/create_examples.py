"""Hand-authored fixtures: engine tests, NOT measured LLM performance."""
from pathlib import Path
import json

def spec(profile,metric,quote):
 return {'profile_id':profile,'primary_metric':metric,'primary_quote':quote,'metric_constraints':[],
 'cpu_required':None,'cpu_quote':None,'max_latency_ms':None,'latency_quote':None,
 'max_cost_usd_per_case':None,'cost_quote':None,'tie_breakers':[],
 'requires_clarification':False,'clarification_questions':[],'unresolved_requirements':[]}

def examples():
 out=[]
 def add(cid,request,s,status,model=None):out.append({'id':cid,'request':request,'profile_id':s['profile_id'],'gold_spec':s,'expected_status':status,'expected_model':model})
 k='kaggle_spiral_v1';n='newhandpd_spiral_v1'
 q='Maximize accuracy.';add('kaggle_accuracy',q,spec(k,'accuracy',q),'selected',k+'__hog_rf')
 add('newhandpd_accuracy',q,spec(n,'accuracy',q),'selected',n+'__resnet18_linear')
 q='Minimize false negatives.';add('prioritize_detection',q,spec(n,'false_negative_rate',q),'selected',n+'__resnet18_linear')
 q='Minimize false positives.';add('prioritize_false_alarms',q,spec(n,'false_positive_rate',q),'selected',n+'__hog_rf')
 q='Minimize false positives while sensitivity is at least 80%.';s=spec(n,'false_positive_rate','Minimize false positives');s['metric_constraints']=[{'metric':'sensitivity','operator':'min','value':.8,'quote':'sensitivity is at least 80%'}]
 add('constrained_false_alarms',q,s,'selected',n+'__resnet18_linear')
 q='Maximize specificity with sensitivity at least 85%.';s=spec(n,'specificity','Maximize specificity');s['metric_constraints']=[{'metric':'sensitivity','operator':'min','value':.85,'quote':'sensitivity at least 85%'}]
 add('impossible_detection_floor',q,s,'no_eligible_model')
 q='Maximize balanced accuracy and run on CPU without a GPU.';s=spec(n,'balanced_accuracy','Maximize balanced accuracy');s['cpu_required']=True;s['cpu_quote']='run on CPU without a GPU'
 add('cpu_execution',q,s,'selected',n+'__resnet18_linear')
 q='Maximize accuracy with latency at most 50 ms.';s=spec(n,'accuracy','Maximize accuracy');s['max_latency_ms']=50;s['latency_quote']='latency at most 50 ms'
 add('unmeasured_latency',q,s,'insufficient_deployment_evidence')
 q='Maximize accuracy with cost at most 0.01 USD per case.';s=spec(n,'accuracy','Maximize accuracy');s['max_cost_usd_per_case']=.01;s['cost_quote']='cost at most 0.01 USD per case'
 add('unmeasured_cost',q,s,'insufficient_deployment_evidence')
 q='Choose the best model.';s=spec(n,None,None);s['requires_clarification']=True;s['clarification_questions']=['Should best mean sensitivity, specificity, accuracy or balanced accuracy?']
 add('ambiguous_best',q,s,'needs_clarification')
 q='Prioritize detection but referral capacity is limited.';s=spec(n,'sensitivity','Prioritize detection');s['requires_clarification']=True;s['unresolved_requirements']=['Acceptable false-positive or referral limit is unspecified.'];s['clarification_questions']=['What false-positive or referral limit must be satisfied?']
 add('unspecified_capacity',q,s,'needs_clarification')
 q='Maximize sensitivity.';add('speech_no_results',q,spec('speech_pending_v1','sensitivity',q),'no_evaluation_evidence')
 q='Maximize sensitivity; if tied, maximize specificity.';s=spec(k,'sensitivity','Maximize sensitivity');s['tie_breakers']=[{'metric':'specificity','quote':'if tied, maximize specificity'}]
 add('explicit_tie_breaker',q,s,'selected',k+'__hog_rf')
 return out

if __name__=='__main__':
 root=Path(__file__).parent/'examples';root.mkdir(exist_ok=True)
 fixtures=examples()
 for e in fixtures:
  d=root/e['id'];d.mkdir(exist_ok=True)
  (d/'request.txt').write_text(e['request'],encoding='utf-8')
  (d/'spec.json').write_text(json.dumps(e['gold_spec'],indent=2),encoding='utf-8')
 (root/'benchmark_gold.json').write_text(json.dumps(fixtures,indent=2),encoding='utf-8')
 print('Created',len(fixtures),'hand-authored policy cases.')
