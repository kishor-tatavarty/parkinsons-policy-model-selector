"""Fit actual tool artifacts using supplied training folder only; no evaluation."""
import argparse,hashlib,json,zipfile
from pathlib import Path
import joblib,numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from image_tools import hog_feature,DeepExtractor
p=argparse.ArgumentParser()
p.add_argument('--zip',type=Path,required=True)
p.add_argument('--profile',choices=['kaggle_spiral_v1','newhandpd_spiral_v1'],required=True)
p.add_argument('--output',type=Path,default=Path('trained_tools'))
a=p.parse_args()
root=a.output/a.profile
if (root/'artifacts.json').exists():raise ValueError('Existing artifacts found; use another --output directory to preserve versions')
with zipfile.ZipFile(a.zip) as z:
 prefix='spiral/training/' if any(n.startswith('spiral/training/') for n in z.namelist()) else 'drawings/spiral/training/'
 names=sorted(n for n in z.namelist() if n.startswith(prefix) and n.lower().endswith('.png'))
 expected=72 if a.profile=='kaggle_spiral_v1' else 196
 if len(names)!=expected:raise ValueError('Training count does not match registered profile')
 y=np.array([int(n.split('/')[-2]=='parkinson') for n in names]);raws=[z.read(n) for n in names]
 if any(n.split('/')[-2] not in ['healthy','parkinson'] for n in names):raise ValueError('Unknown label')
 if len(set(hashlib.sha256(b).hexdigest() for b in raws))!=len(raws):raise ValueError('Duplicate training bytes')
print('Extracting features and fitting tools on',len(raws),'training images only.',flush=True)
x=np.array([hog_feature(r) for r in raws]);deep=DeepExtractor().extract(raws)
rf=RandomForestClassifier(n_estimators=100,random_state=1,n_jobs=1).fit(x,y)
head=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=20261021)).fit(deep,y)
root.mkdir(parents=True,exist_ok=True)
joblib.dump(rf,root/'hog_rf.joblib');joblib.dump(head,root/'resnet18_linear.joblib')
m={'profile_id':a.profile,'tool_version':'1.0','fit_images':len(raws),'training_only':True,
 'source_zip_sha256':hashlib.sha256(a.zip.read_bytes()).hexdigest(),
 'model_sha256':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ['hog_rf.joblib','resnet18_linear.joblib']},
 'training_image_sha256':[hashlib.sha256(r).hexdigest() for r in raws],
 'note':'Fit artifacts for inference; registry metrics describe prior protocol, not new evaluation of these saved files.'}
(root/'artifacts.json').write_text(json.dumps(m,indent=2),encoding='utf-8');print(json.dumps(m|{'training_image_sha256':'recorded in artifacts.json'},indent=2))
