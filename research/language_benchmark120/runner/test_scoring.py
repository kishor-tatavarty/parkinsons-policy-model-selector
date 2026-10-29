import unittest,copy
from run_offline import signature
from rule_parser import blank
class ScoringTests(unittest.TestCase):
 def test_complementary_constraint(self):
  a=blank('p');a['primary_metric']='specificity';a['metric_constraints']=[dict(metric='sensitivity',operator='min',value=.8,quote='a')]
  b=copy.deepcopy(a);b['primary_metric']='false_positive_rate';b['metric_constraints']=[dict(metric='false_negative_rate',operator='max',value=.2,quote='b')]
  self.assertEqual(signature(a),signature(b))
 def test_omitted_constraint_does_not_pass(self):
  a=blank('p');a['metric_constraints']=[dict(metric='sensitivity',operator='min',value=.8,quote='a')];b=copy.deepcopy(a);b['metric_constraints']=[];self.assertNotEqual(signature(a),signature(b))
 def test_ordered_ties_not_reordered(self):
  a=blank('p');a['tie_breakers']=[dict(metric='accuracy',quote='a'),dict(metric='sensitivity',quote='b')];b=copy.deepcopy(a);b['tie_breakers'].reverse();self.assertNotEqual(signature(a),signature(b))
 def test_no_cpu_requirement_equivalence(self):
  a=blank('p');b=copy.deepcopy(a);b['cpu_required']=False;self.assertEqual(signature(a),signature(b));b['cpu_required']=True;self.assertNotEqual(signature(a),signature(b))
if __name__=='__main__':unittest.main()
