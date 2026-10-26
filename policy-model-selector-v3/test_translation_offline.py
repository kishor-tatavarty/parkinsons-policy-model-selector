import io,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stdout
from create_examples import examples
import translate_policy
class TranslationTests(unittest.TestCase):
 def test_mocked_api_to_compiled_policy_and_cache(self):
  e=examples()[4]
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);request=root/'request.txt';request.write_text(e['request'])
   args=['translate_policy.py','--request-file',str(request),'--profile',e['profile_id'],'--output',str(root/'out'),'--execute']
   response={'status':'completed','output':[{'type':'message','content':[{'type':'output_text','text':json.dumps(e['gold_spec'])}]}]}
   def fake(req,timeout):
    body=json.loads(req.data)
    self.assertNotIn('confusion',json.dumps(body));self.assertFalse(body['store'])
    return io.BytesIO(json.dumps(response).encode())
   with patch.dict(os.environ,{'OPENAI_API_KEY':'synthetic-test-value'}),patch('urllib.request.urlopen',side_effect=fake) as net,patch.object(sys,'argv',args),redirect_stdout(io.StringIO()):
    translate_policy.main();translate_policy.main()
    self.assertEqual(net.call_count,1)
   compiled=json.loads(next((root/'out').glob('*/compiled_policy.json')).read_text())
   self.assertEqual(compiled['selected_model'],e['expected_model'])
 def test_incomplete_translation_rejected(self):
  with self.assertRaises(ValueError):translate_policy.parse({'status':'incomplete'})
if __name__=='__main__':unittest.main()
