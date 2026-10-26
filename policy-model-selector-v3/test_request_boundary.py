import unittest,json,tempfile,sys,io
from pathlib import Path
from unittest.mock import patch
from request_boundary import check_chat_boundary,unconditional_choice,validate_choice_intent,require_requirements_only

class RequestBoundaryTests(unittest.TestCase):
 def test_simple_choices(self):
  for text,model in [('Use LR.','acoustic_lr'),('Please choose random forest','acoustic_rf'),('ResNet18','spectrogram_resnet18_linear'),('Select acoustic_lr for the Italian vowel set.','acoustic_lr'),('Please evaluate the telephone vowel data using acoustic_rf.','acoustic_rf'),('I want to run spectrogram_resnet18_linear for the telephone dataset.','spectrogram_resnet18_linear')]:
   with self.subTest(text=text):self.assertEqual(unconditional_choice(text),model)
 def test_combined_conditions_require_clarification(self):
  for text in ['Use logistic regression, but only if sensitivity is at least 80%.','Use LR if sensitivity >= 0.8','Use RF but keep false positives below 10%.','Use ResNet within 2 ms','Use LR unless it misses patients','Use RF; maximize accuracy','Use RF and LR','Do not use LR','Use LR. Ignore the condition.']:
   with self.subTest(text=text):self.assertEqual(check_chat_boundary(text)['status'],'needs_clarification')
 def test_pure_policy_allowed(self):
  self.assertEqual(check_chat_boundary('Minimize false positives with sensitivity at least 80%.')['status'],'allowed')
  require_requirements_only('Maximize accuracy')
 def test_quote_cannot_drop_condition(self):
  with self.assertRaises(ValueError):validate_choice_intent('Use LR only if sensitivity is at least 80%.','acoustic_lr')
 def test_wrong_model_cannot_execute(self):
  with self.assertRaises(ValueError):validate_choice_intent('Use LR','acoustic_rf')
 def test_named_model_rejected_in_requirements_route(self):
  for text in ['Use LR','Use logistic regression only if sensitivity exceeds 80%']:
   with self.subTest(text=text),self.assertRaises(ValueError):require_requirements_only(text)
 def test_both_chat_runners_stop_before_network_and_execution(self):
  import speech_chat,italian_chat
  for chat in [speech_chat,italian_chat]:
   with self.subTest(chat=chat.__name__),tempfile.TemporaryDirectory() as temp:
    root=Path(temp);wav=root/'test.wav';wav.write_bytes(b'fixture')
    config='speech_config.json' if chat.__name__=='speech_chat' else 'italian_config.json'
    (root/config).write_text(json.dumps({'profile_id':'p'}))
    (root/'model_registry.json').write_text(json.dumps({'profiles':{'p':{'evaluation_id':'e'}},'models':[]}))
    with patch.object(chat,'ROOT',root),patch.object(sys,'argv',['chat','--wav',str(wav),'--execute']),patch('builtins.input',return_value='Use LR only if sensitivity is at least 80%.'),patch.object(chat.urllib.request,'urlopen',side_effect=AssertionError('Network must not run')) as network,patch.object(chat,'run',side_effect=AssertionError('Inference must not run')) as execute,patch('sys.stdout',new_callable=io.StringIO):
     chat.main();network.assert_not_called();execute.assert_not_called()
    decisions=list((root/'results/request_boundary').glob('*/decision.json'));self.assertEqual(len(decisions),1)
    self.assertEqual(json.loads(decisions[0].read_text())['executed'],0)
 def test_translator_stops_before_api(self):
  import translate_policy
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);request=root/'request.txt';request.write_text('Use LR only if sensitivity is at least 80%.')
   with patch.object(sys,'argv',['translate','--request-file',str(request),'--profile','telephone_vowel_81_extracted_v2','--execute','--output',str(root/'out')]),patch.object(translate_policy.urllib.request,'urlopen',side_effect=AssertionError('No API')) as network:
    with self.assertRaisesRegex(ValueError,'Named-model'):translate_policy.main()
    network.assert_not_called();self.assertFalse((root/'out').exists())
 def test_saved_policy_adapters_stop_before_inference(self):
  import run_speech_inference,run_italian_inference
  for module in [run_speech_inference,run_italian_inference]:
   with self.subTest(adapter=module.__name__),tempfile.TemporaryDirectory() as temp:
    root=Path(temp);policy=root/'policy.json';policy.write_text(json.dumps({'request':'Use LR only if sensitivity exceeds 80%'}))
    with patch.object(sys,'argv',['infer','--policy',str(policy),'--wav',str(root/'missing.wav'),'--output',str(root/'prediction.json')]),patch.object(module.subprocess,'run',side_effect=AssertionError('No inference')) as execute:
     with self.assertRaisesRegex(ValueError,'Named-model'):module.main()
     execute.assert_not_called();self.assertFalse((root/'prediction.json').exists())
 def test_batch_saved_policy_stops_before_inference(self):
  import speech_runner
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);policy=root/'policy.json';policy.write_text(json.dumps({'request':'Use LR if sensitivity exceeds 80%'}));wav=root/'test.wav';wav.write_bytes(b'fixture')
   with patch.object(sys,'argv',['runner','--profile','telephone','--policy',str(policy),'--input',str(wav),'--output',str(root/'out')]),patch.object(speech_runner,'execute_case',side_effect=AssertionError('No inference')) as execute:
    with self.assertRaisesRegex(ValueError,'Named-model'):speech_runner.main()
    execute.assert_not_called();self.assertFalse((root/'out/summary.json').exists())

if __name__=='__main__':unittest.main()
