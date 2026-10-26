"""Configure local adapter paths after extracting this release."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--rscript',required=True,type=Path);a=p.parse_args()
if not a.rscript.is_file():raise SystemExit('Rscript executable not found')
root=Path(__file__).resolve().parent
for name in ['speech_config.json','italian_config.json']:
 path=root/'policy-model-selector-v3'/name
 config=json.loads(path.read_text());config.update(speech_project=str(root/'pd-speech-baseline'),rscript=str(a.rscript.resolve()))
 path.write_text(json.dumps(config,indent=2),encoding='utf-8');print('Configured:',name)
