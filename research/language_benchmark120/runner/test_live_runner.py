import copy,json,tempfile,unittest,sys
from pathlib import Path
from unittest.mock import patch
import run_llm as runner

def policy():
 return dict(profile_id='telephone_vowel_81_extracted_v2',primary_metric='accuracy',primary_quote='Maximize accuracy',metric_constraints=[],cpu_required=None,cpu_quote=None,max_latency_ms=None,latency_quote=None,max_cost_usd_per_case=None,cost_quote=None,tie_breakers=[],requires_clarification=False,clarification_questions=[],unresolved_requirements=[])
def response(s):return {'status':'completed','output':[{'content':[{'type':'output_text','text':json.dumps(s)}]}]}
class Tests(unittest.TestCase):
 def setUp(self):self.reg=json.loads((runner.ROOT/'frozen/model_registry.json').read_text())
 def test_same_response_same_valid_choice(self):
  q={'action':'requirements','model':None,'policy':policy()};r=runner.paired_decisions(response(q),'Maximize accuracy',q['policy']['profile_id'],self.reg)
  self.assertEqual(r['raw']['status'],'selected');self.assertEqual(r['validated']['status'],'selected');self.assertEqual(r['raw']['selected_model'],r['validated']['selected_model'])
 def test_invented_threshold_blocked_only_validated(self):
  s=policy();s['metric_constraints']=[dict(metric='sensitivity',operator='min',value=.65,quote='sensitivity at least 60%')]
  r=runner.paired_decisions(response(dict(action='requirements',model=None,policy=s)),'Maximize accuracy with sensitivity at least 60%',s['profile_id'],self.reg)
  self.assertEqual(r['raw']['status'],'selected');self.assertEqual(r['validated']['status'],'invalid')
 def test_conditional_named_choice_blocked(self):
  q=dict(action='explicit_choice',model='acoustic_lr',policy=None);r=runner.paired_decisions(response(q),'Use LR only if sensitivity is at least 80%','telephone_vowel_81_extracted_v2',self.reg)
  self.assertEqual(r['raw']['status'],'selected');self.assertEqual(r['validated']['status'],'invalid')
 def test_refusal_both_invalid(self):
  r=runner.paired_decisions({'status':'completed','output':[{'content':[{'type':'refusal'}]}]},'x','telephone_vowel_81_extracted_v2',self.reg)
  self.assertEqual(r['raw']['status'],'invalid');self.assertEqual(r['validated']['status'],'invalid')
 def test_schema_rejects_mixed_route(self):
  with self.assertRaises(ValueError):runner.parse_response(response(dict(action='explicit_choice',model='acoustic_lr',policy=policy())))
 def test_env_loader_does_not_need_network(self):
  with tempfile.TemporaryDirectory() as t,patch.dict(runner.os.environ,{},clear=True):
   p=Path(t)/'.env';p.write_text('OPENAI_API_KEY="fixture-not-a-real-key"\n');self.assertEqual(runner.load_key(p),'fixture-not-a-real-key')
 def test_dryrun_no_network_no_references(self):
  with tempfile.TemporaryDirectory() as t,patch.object(sys,'argv',['run_llm.py','--output',t]),patch.object(runner.urllib.request,'urlopen',side_effect=AssertionError('No network')) as net:
   runner.main();net.assert_not_called();self.assertTrue((Path(t)/'contract.json').exists())
if __name__=='__main__':unittest.main()
