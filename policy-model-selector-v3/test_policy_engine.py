import copy,json,unittest
from pathlib import Path
from create_examples import examples,spec
from policy_engine import select,dispatch,metrics
class EngineTests(unittest.TestCase):
 def setUp(self):self.registry=json.loads((Path(__file__).parent/'model_registry.json').read_text())
 def test_all_gold_engine_cases(self):
  for e in examples():
   with self.subTest(e['id']):
    r=select(self.registry,e['gold_spec'],e['request'],e['profile_id'])
    self.assertEqual(r['status'],e['expected_status']);self.assertEqual(r['selected_model'],e['expected_model'])
 def test_no_cross_profile_ranking(self):
  e=examples()[0];self.registry['models'][2]['confusion']={'tn':140,'fp':0,'fn':0,'tp':104}
  r=select(self.registry,e['gold_spec'],e['request'],e['profile_id'])
  self.assertEqual(r['selected_model'],'kaggle_spiral_v1__hog_rf')
 def test_wrong_evaluation_ignored(self):
  e=examples()[0];self.registry['models'][0]['evaluation_id']='other'
  r=select(self.registry,e['gold_spec'],e['request'],e['profile_id'])
  self.assertEqual(r['selected_model'],'kaggle_spiral_v1__resnet18_linear')
 def test_ungrounded_threshold_rejected(self):
  e=copy.deepcopy(examples()[4]);e['gold_spec']['metric_constraints'][0]['value']=.9
  with self.assertRaises(ValueError):select(self.registry,e['gold_spec'],e['request'],e['profile_id'])
 def test_changed_profile_rejected(self):
  e=copy.deepcopy(examples()[0]);e['gold_spec']['profile_id']='newhandpd_spiral_v1'
  with self.assertRaises(ValueError):select(self.registry,e['gold_spec'],e['request'],e['profile_id'])
 def test_tie_without_policy_is_not_arbitrary(self):
  r=self.registry; r['models'][1]['confusion']=dict(r['models'][0]['confusion'])
  e=examples()[0];result=select(r,e['gold_spec'],e['request'],e['profile_id'])
  self.assertEqual(result['status'],'tied');self.assertIsNone(result['selected_model'])
 def test_equal_accuracy_error_objective(self):
  p='kaggle_spiral_v1';r=self.registry
  r['models'][0]['confusion']={'tn':90,'fp':10,'fn':30,'tp':70}
  r['models'][1]['confusion']={'tn':70,'fp':30,'fn':10,'tp':90}
  q='Minimize false positives.';a=select(r,spec(p,'false_positive_rate',q),q,p)
  q='Minimize false negatives.';b=select(r,spec(p,'false_negative_rate',q),q,p)
  self.assertEqual(a['selected_model'],p+'__hog_rf');self.assertEqual(b['selected_model'],p+'__resnet18_linear')
 def test_dispatch_only_registered_tool(self):
  e=examples()[0];r=select(self.registry,e['gold_spec'],e['request'],e['profile_id'])
  self.assertEqual(dispatch(r,e['profile_id'],{'spiral_hog_rf':lambda:'fixture_result'}),'fixture_result')
  with self.assertRaises(ValueError):dispatch(r,'speech_pending_v1',{})
  with self.assertRaises(ValueError):dispatch(r,e['profile_id'],{})
 def test_confusion_counts_reject_invalid_values(self):
  with self.assertRaises(ValueError):metrics({'confusion':{'tn':-1,'fp':1,'fn':2,'tp':3}})
 def test_missing_evidence_not_zero(self):
  e=examples()[7];r=select(self.registry,e['gold_spec'],e['request'],e['profile_id'])
  self.assertEqual(r['status'],'insufficient_deployment_evidence')
if __name__=='__main__':unittest.main()
