import unittest,tempfile,json,hashlib,subprocess,sys
from pathlib import Path
from artifact_binding import verify_image_binding
ROOT=Path(__file__).parent
class Tests(unittest.TestCase):
 def fixture(self,root):
  p='newhandpd_spiral_v1';d=root/p;d.mkdir();blob=b'fixture';(d/'hog_rf.joblib').write_bytes(blob);h=hashlib.sha256(blob).hexdigest();meta={'profile_id':p,'model_sha256':{'hog_rf.joblib':h},'fit_images':196};(d/'artifacts.json').write_text(json.dumps(meta));r={'profiles':{p:{'evaluation_id':'e'}},'models':[{'model_id':p+'__hog_rf','profile_id':p,'evaluation_id':'e','tool_id':'spiral_hog_rf','artifact_sha256':h}]};return p,d,meta,r
 def test_valid_binding(self):
  with tempfile.TemporaryDirectory() as t:
   p,d,m,r=self.fixture(Path(t));self.assertEqual(verify_image_binding(r,p,p+'__hog_rf','spiral_hog_rf',t)['fit_images'],196)
 def test_modified_artifact_and_metadata_refused(self):
  with tempfile.TemporaryDirectory() as t:
   p,d,m,r=self.fixture(Path(t));(d/'hog_rf.joblib').write_bytes(b'replacement');m['model_sha256']['hog_rf.joblib']=hashlib.sha256(b'replacement').hexdigest();(d/'artifacts.json').write_text(json.dumps(m))
   with self.assertRaises(ValueError):verify_image_binding(r,p,p+'__hog_rf','spiral_hog_rf',t)
 def test_missing_binding(self):
  with tempfile.TemporaryDirectory() as t:
   p,d,m,r=self.fixture(Path(t));r['models'][0].pop('artifact_sha256')
   with self.assertRaises(ValueError):verify_image_binding(r,p,p+'__hog_rf','spiral_hog_rf',t)
 def test_incompatible_profile_and_evaluation(self):
  with tempfile.TemporaryDirectory() as t:
   p,d,m,r=self.fixture(Path(t));r['models'][0]['evaluation_id']='other'
   with self.assertRaises(ValueError):verify_image_binding(r,p,p+'__hog_rf','spiral_hog_rf',t)
 def test_real_entrypoint_refuses_before_adapter_import(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);p,d,m,r=self.fixture(root);r['models'][0].update(confusion={'tn':2,'fp':0,'fn':0,'tp':2},tool_version='fixture');r['profiles'][p]['scope']='fixture'
   sys.path.insert(0,str(ROOT));from policy_engine import select
   spec=dict(profile_id=p,primary_metric='accuracy',primary_quote='Maximize accuracy',metric_constraints=[],cpu_required=None,cpu_quote=None,max_latency_ms=None,latency_quote=None,max_cost_usd_per_case=None,cost_quote=None,tie_breakers=[],requires_clarification=False,clarification_questions=[],unresolved_requirements=[])
   (root/'registry.json').write_text(json.dumps(r));(root/'policy.json').write_text(json.dumps(select(r,spec,'Maximize accuracy',p)));(d/'hog_rf.joblib').write_bytes(b'tampered')
   run=subprocess.run([sys.executable,str(ROOT/'run_inference.py'),'--policy',str(root/'policy.json'),'--profile',p,'--registry',str(root/'registry.json'),'--models',str(root),'--image',str(root/'missing.png'),'--output',str(root/'output.json')],capture_output=True,text=True)
   self.assertNotEqual(run.returncode,0);self.assertIn('hashes differ',run.stderr);self.assertNotIn('cv2',run.stderr);self.assertFalse((root/'output.json').exists())
if __name__=="__main__":unittest.main()
