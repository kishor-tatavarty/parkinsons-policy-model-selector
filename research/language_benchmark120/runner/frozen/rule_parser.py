"""Closed-vocabulary research parser. Post-exposure development baseline, not a production parser.
No reference cases, expected models or registry performance are read by this module.
Unknown words, negation, strict bounds and combined named choices require clarification.
"""
import re
from policy_engine import METRICS
VERSION='development_rules_1.0'
ALIASES={'acoustic_rf':('acoustic_rf','random forest','rf'),
 'acoustic_lr':('acoustic_lr','logistic regression','lr'),
 'spectrogram_resnet18_linear':('spectrogram_resnet18_linear','resnet18','resnet')}
MODEL='|'.join(re.escape(x) for v in ALIASES.values() for x in v)
METRIC=r'balanced[ _-]accuracy|false[ _-]positive(?:[ _-]rate|s)?|false[ _-]negative(?:[ _-]rate|s)?|accuracy|sensitivity|specificity'
M=re.compile(r'\b('+METRIC+r')\b',re.I)
THRESHOLD=re.compile(r'\b(?P<metric>'+METRIC+r')\s+(?:of\s+)?(?P<op>at least|at most|no more than|no less than)\s+(?P<value>\d+(?:\.\d+)?)\s*(?P<percent>%|percent)?',re.I)
TIE=re.compile(r'(?:then\s+)?break ties (?:with|by)\s+(?P<a>'+METRIC+r')|(?:and\s+)?use\s+(?P<b>'+METRIC+r')\s+as (?:the\s+)?tie[- ]breaker',re.I)
OBJECTIVE=re.compile(r'(?P<cue>maximize|minimize|minimizes|maximizes|highest|lowest|best|top|rank(?:ed)? by|ordered by)\s+(?:overall\s+)?(?P<metric>'+METRIC+r')\b',re.I)
# Context/scaffolding vocabulary is finite. Unconsumed words are never silently dropped.
WORDS=set('please can you i want to run use select choose execute evaluate the a an on for with using data dataset set samples recordings vowel telephone italian model models algorithm which gives find that we need identify of by ranked ordered requiring require and then stop if tied'.split())

def canonical(s):
 s=re.sub(r'[ _-]+','_',s.lower())
 return {'false_positives':'false_positive_rate','false_negatives':'false_negative_rate'}.get(s,s)
def blank(profile):
 return dict(profile_id=profile,primary_metric=None,primary_quote=None,metric_constraints=[],cpu_required=None,cpu_quote=None,max_latency_ms=None,latency_quote=None,max_cost_usd_per_case=None,cost_quote=None,tie_breakers=[],requires_clarification=False,clarification_questions=[],unresolved_requirements=[])
def clarify(spec,why,action='clarify'):
 spec['requires_clarification']=True;spec['clarification_questions']=[why]
 return dict(action=action,model=None,policy=spec,explanation=why)
def parse(request,profile):
 spec=blank(profile);text=' '.join(request.strip().split())
 # Unsupported meaning/units are not translated into a numeric default.
 if re.search(r'\b(not|never|except|unless|balancing|cost|clinical|hospitals?|fastest|sub-millisecond|latency|budget)\b',text,re.I):
  return clarify(spec,'Unsupported meaning or deployment requirement; specify a supported policy.')
 if re.search(r'\bapply\b.*\branking\b',text,re.I):return clarify(spec,'Rankings cannot be transferred across dataset profiles.')
 if re.search(r'\b(below|above|less than|greater than|exceeds?)\b|[<>]',text,re.I):
  return clarify(spec,'Strict bounds are not represented by the inclusive policy schema.','requirements' if M.search(text) else 'clarify')
 mentions=list(re.finditer(r'(?<!\w)(?:'+MODEL+r')(?!\w)',text,re.I))
 if mentions:
  if len(mentions)!=1:return clarify(spec,'Choose exactly one model.')
  if M.search(text) or re.search(r'\b(if|only|but|without|and|under|within)\b',text,re.I):return clarify(spec,'Named choice with conditions requires clarification.')
  mention=mentions[0];word=mention.group().lower();model=next(k for k,v in ALIASES.items() if word in v)
  rest=text[:mention.start()]+' '+text[mention.end():];rest=rest.replace(profile,' ')
  if any(t.lower() not in WORDS for t in re.findall(r'[A-Za-z_]+',rest)):return clarify(spec,'Named choice has unsupported wording.')
  if not re.search(r'\b(use|select|choose|run|execute|evaluate)\b',text,re.I) and text.strip('.!? ').lower()!=word:return clarify(spec,'A mention alone is not an explicit choice.')
  return dict(action='explicit_choice',model=model,policy=None,explanation='Unconditional registered model choice.')
 work=text;covered=[]
 for match in TIE.finditer(text):
  spec['tie_breakers'].append(dict(metric=canonical(match['a'] or match['b']),quote=match.group()));covered.append(match.span())
 for match in THRESHOLD.finditer(text):
  v=float(match['value']);v=v/100 if match['percent'] else v
  if not 0<=v<=1:return clarify(spec,'Metric thresholds must be fractions or explicit percentages.','requirements')
  spec['metric_constraints'].append(dict(metric=canonical(match['metric']),operator='min' if match['op'].lower() in ('at least','no less than') else 'max',value=v,quote=match.group()));covered.append(match.span())
 for st,en in covered:work=work[:st]+' '*(en-st)+work[en:]
 objectives=list(OBJECTIVE.finditer(work))
 if len(objectives)>1:return clarify(spec,'Specify a single primary objective and ordered tie breakers.','requirements')
 if objectives:
  match=objectives[0];metric=canonical(match['metric']);cue=match['cue'].lower()
  if (cue.startswith('minim') or cue=='lowest') and metric not in ('false_positive_rate','false_negative_rate'):return clarify(spec,'Requested direction is unsupported for this metric.','requirements')
  if (cue.startswith('maxim') or cue in ('highest','top')) and metric in ('false_positive_rate','false_negative_rate'):return clarify(spec,'Requested direction is unsupported for this metric.','requirements')
  spec['primary_metric']=metric;spec['primary_quote']=text[match.start():match.end()];covered.append(match.span())
 residue=text
 for st,en in covered:residue=residue[:st]+' '*(en-st)+residue[en:]
 residue=residue.replace(profile,' ')
 unknown=[t for t in re.findall(r'[A-Za-z_]+',residue) if t.lower() not in WORDS]
 action='requirements' if objectives or spec['metric_constraints'] else 'clarify'
 if unknown:return clarify(spec,'Unsupported wording: '+', '.join(sorted(set(unknown))),action)
 if spec['primary_metric'] is None:return clarify(spec,'Specify the primary ranking objective.',action)
 return dict(action='requirements',model=None,policy=spec,explanation='Supported expressions parsed without LLM assistance.')
