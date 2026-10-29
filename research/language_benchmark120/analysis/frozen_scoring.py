"""Original provisional semantic-signature scoring, unchanged."""
def signature(s):
 if s is None:return None
 def canonical(m,op=None,v=None):
  if m in ('specificity','sensitivity'):
   m={'specificity':'false_positive_rate','sensitivity':'false_negative_rate'}[m]
   if op:op={'min':'max','max':'min'}[op];v=1-v
  return m,op,round(v,10) if v is not None else None
 return dict(profile=s['profile_id'],objective=canonical(s['primary_metric'])[0],constraints=sorted(set(canonical(c['metric'],c['operator'],c['value']) for c in s['metric_constraints'])),ties=[canonical(t['metric'])[0] for t in s['tie_breakers']],cpu=True if s['cpu_required'] is True else None,latency=s['max_latency_ms'],cost=s['max_cost_usd_per_case'],clarify=bool(s['requires_clarification']),unresolved_present=bool(s['unresolved_requirements']))
