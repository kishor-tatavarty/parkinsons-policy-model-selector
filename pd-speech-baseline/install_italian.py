"""Install Italian adapters alongside existing telephone adapters. No API calls."""
from pathlib import Path
import argparse,json,ast

p=argparse.ArgumentParser()
p.add_argument('--speech-project',type=Path,default=Path(__file__).resolve().parent)
p.add_argument('--selector',type=Path,default=Path(__file__).resolve().parents[1]/'policy-model-selector-v3')
a=p.parse_args();speech=a.speech_project;selector=a.selector

def replace_once(text,old,new):
    if text.count(old)!=1:raise ValueError('Expected one source anchor: '+old[:100])
    return text.replace(old,new)

source=(speech/'predict_speech.py').read_text(encoding='utf-8-sig')
source=source.replace('import io','import io\nimport math\nfrom scipy.signal import resample_poly')
start=source.index('    rate, audio = wavfile.read(io.BytesIO(raw))')
end=source.index('\n    artifacts = ',start)
source=source[:start]+'''    rate, audio = wavfile.read(io.BytesIO(raw))
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise SystemExit("Expected finite mono audio.")
    if audio.dtype == np.int16 and rate in (16000, 44100):
        divisor = math.gcd(rate, 8000)
        audio = resample_poly(audio.astype(np.float64) / 32768.0,
                              8000 // divisor, rate // divisor).astype(np.float32)
    elif rate == 8000 and audio.dtype == np.float32:
        audio = audio.copy()
    else:
        raise SystemExit("Use original mono PCM16 at 16/44.1kHz or prepared float32 at 8kHz.")
    if len(audio) < 12000 or not np.isfinite(audio).all():
        raise SystemExit("Prepared audio must be finite and at least 1.5 seconds.")
    buffer = io.BytesIO()
    wavfile.write(buffer, 8000, audio)
    prepared_raw = buffer.getvalue()
''' +source[end:]
source=source.replace('artifacts/speech_extracted_v2','artifacts/italian_vowel_a_grouped_v1')
source=source.replace('extract_speech_features.R','extract_italian_features.R')
source=source.replace('data/extracted','data/italian_prepared_v1')
source=source.replace('results/extraction_v1','results/italian_extraction_v1')
source=replace_once(source,'(inputs / "case.wav").write_bytes(raw)','(inputs / "case.wav").write_bytes(prepared_raw)')
start=source.index('    # Detect reuse of a recording included in final model fitting.')
end=source.index('\n    result = ',start)
source=source[:start]+'''    prepared_hash = hashlib.sha256(prepared_raw).hexdigest()
    training_hashes = set(meta["training_prepared_sha256"])
''' +source[end:]
source=replace_once(source,'"previously_used_for_fitting": audio_hash in training_hashes','"prepared_sha256": prepared_hash,\n        "previously_used_for_fitting": prepared_hash in training_hashes')
outputs={speech/'predict_italian.py':source}

config=json.loads((selector/'speech_config.json').read_text())
config.update(profile_id='italian_vowel_a_grouped_v1',speech_project=str(speech.resolve()))
for old,new in [('run_speech_inference.py','run_italian_inference.py'),('review_speech.py','review_italian.py'),('speech_chat.py','italian_chat.py')]:
    text=(selector/old).read_text(encoding='utf-8-sig')
    for before,after in [('speech_config.json','italian_config.json'),('predict_speech.py','predict_italian.py'),('artifacts/speech_extracted_v2','artifacts/italian_vowel_a_grouped_v1'),('speech_spectrogram_resnet18_linear','italian_spectrogram_resnet18_linear'),('speech_acoustic_rf','italian_acoustic_rf'),('speech_acoustic_lr','italian_acoustic_lr'),('review_speech.py','review_italian.py'),('run_speech_inference.py','run_italian_inference.py'),('results/speech_chat','results/italian_chat')]:text=text.replace(before,after)
    if old=='review_speech.py':
        text=text.replace('same 81 recordings','same 98 recordings')
        text=text.replace('RF has the highest observed accuracy in this comparison.','LR has the highest observed accuracy in this comparison.')
        text=text.replace('No candidate reaches 80% PD detection.','All three candidates reach 80% PD detection in this comparison.')
    outputs[selector/new]=text
for path,text in outputs.items():
    if path.exists():raise SystemExit('Refusing to overwrite '+str(path))
    ast.parse(text,filename=str(path))
config_path=selector/'italian_config.json'
if config_path.exists():raise SystemExit('Italian config already exists')
# Validate the registry before creating files.
registry=json.loads((selector/'model_registry.json').read_text())
assert config['profile_id'] in registry['profiles']
meta=json.loads((speech/'artifacts/italian_vowel_a_grouped_v1/artifacts.json').read_text())
for model in registry['models']:
    if model['profile_id']==config['profile_id']:
        name=model['tool_id'].removeprefix('italian_')+'.joblib'
        assert model['artifact_sha256']==meta['model_sha256'][name]
for path,text in outputs.items():path.write_text(text,encoding='utf-8');print('Created:',path)
config_path.write_text(json.dumps(config,indent=2),encoding='utf-8')
print('Installed. Existing telephone scripts unchanged. No API calls made.')
