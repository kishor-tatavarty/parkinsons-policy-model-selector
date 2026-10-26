"""Fixed exploratory speech comparison. No API calls or hyperparameter search."""
import argparse, csv, hashlib, io, json, platform, zipfile
from pathlib import Path
import numpy as np
import openpyxl
from scipy.io import wavfile
from scipy.signal import spectrogram
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score

SEED=20261021

def sha(b): return hashlib.sha256(b).hexdigest()
def metrics(y,p):
    tn,fp,fn,tp=map(int,confusion_matrix(y,p>=.5,labels=[0,1]).ravel())
    return dict(n=len(y),correct=tn+tp,tn=tn,fp=fp,fn=fn,tp=tp,accuracy=(tn+tp)/len(y),sensitivity=tp/(tp+fn),specificity=tn/(tn+fp),false_positive_rate=fp/(tn+fp),balanced_accuracy=(tp/(tp+fn)+tn/(tn+fp))/2,roc_auc=float(roc_auc_score(y,p)))

def read_data(root):
    book=openpyxl.load_workbook(root/'Demographics_age_sex.xlsx',read_only=True,data_only=True)
    meta={r[0]:r[1:] for r in list(book['Parselmouth'].values)[1:] if r[0]}
    features={}; names=[]
    for sheet in book:
        if sheet.title=='Parselmouth': continue
        rows=list(sheet.values); names.extend(sheet.title+':'+str(c) for c in rows[0][1:-1])
        for row in rows[1:]:
            if not row[0]:continue
            if row[0] not in meta or row[-1]!=meta[row[0]][0]:raise ValueError('Feature label/ID mismatch')
            features.setdefault(row[0],[]).extend(float(v) for v in row[1:-1])
    records=[]; waves=[]; hashes=set()
    for archive,label in [('HC_AH.zip','HC'),('PD_AH.zip','PwPD')]:
        with zipfile.ZipFile(root/archive) as z:
            for member in sorted(z.namelist()):
                if not member.lower().endswith('.wav'):continue
                raw=z.read(member); sid=Path(member).stem
                if sid not in meta or meta[sid][0]!=label: raise ValueError('Audio metadata mismatch')
                sr,x=wavfile.read(io.BytesIO(raw)); x=np.asarray(x,dtype=np.float32)
                if sr!=8000 or x.ndim!=1 or not np.isfinite(x).all():raise ValueError('Unexpected audio format')
                pcmhash=sha(x.tobytes())
                if pcmhash in hashes:raise ValueError('Duplicate waveform: inspect before evaluation')
                hashes.add(pcmhash)
                records.append(dict(sample_id=sid,label=label,age=meta[sid][1],sex=meta[sid][2],sample_rate=sr,duration_seconds=len(x)/sr,peak=float(np.max(np.abs(x))),wav_sha256=sha(raw)))
                waves.append(x)
    ids=[r['sample_id'] for r in records]
    if len(set(ids))!=len(ids) or set(ids)!=set(meta):raise ValueError('Sample ID coverage mismatch')
    X=np.array([features[sid] for sid in ids])
    if X.shape!=(len(ids),len(names)) or not np.isfinite(X).all():raise ValueError('Invalid supplied features')
    return records,waves,X,names

def embeddings(waves):
    import torch, torchvision
    from torchvision.models import resnet18,ResNet18_Weights
    torch.set_num_threads(2)
    net=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1);net.fc=torch.nn.Identity();net.eval()
    images=[]
    for x in waves:
        # Fixed 32 ms Hann windows, 50% overlap. Whole recording; no trimming or label-based decisions.
        _,_,power=spectrogram(x,fs=8000,window='hann',nperseg=256,noverlap=128,nfft=256,scaling='spectrum',mode='psd')
        db=10*np.log10(np.maximum(power,1e-12)); db=np.clip(db-db.max(),-80,0)
        pixels=np.rint((db+80)*255/80).astype('uint8')[::-1].copy()
        im=Image.fromarray(pixels).resize((224,224),Image.Resampling.BILINEAR).convert('RGB')
        images.append(np.asarray(im).copy().transpose(2,0,1)/255.)
    mean=torch.tensor([.485,.456,.406])[None,:,None,None];std=torch.tensor([.229,.224,.225])[None,:,None,None]
    result=[]
    with torch.inference_mode():
        for i in range(0,len(images),8):
            batch=torch.tensor(np.array(images[i:i+8]),dtype=torch.float32)
            result.append(net((batch-mean)/std).numpy())
    return np.concatenate(result),dict(torch=torch.__version__,torchvision=torchvision.__version__)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,required=True);ap.add_argument('--out',type=Path,default=Path('results/speech_v1'));args=ap.parse_args()
    if args.out.exists():raise SystemExit('Output already exists; use a new --out directory to preserve results.')
    records,waves,X,names=read_data(args.data_dir)
    print('Matched',len(records),'recordings;',X.shape[1],'supplied features. Extracting frozen ResNet18 embeddings...',flush=True)
    Z,versions=embeddings(waves); y=np.array([r['label']=='PwPD' for r in records],dtype=int)
    folds=list(StratifiedKFold(5,shuffle=True,random_state=SEED).split(X,y)); fold_id=np.zeros(len(y),dtype=int)
    for k,(_,te) in enumerate(folds):fold_id[te]=k
    specs=[('acoustic_lr',X,lambda:make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=SEED))),('acoustic_rf',X,lambda:RandomForestClassifier(n_estimators=300,random_state=SEED,n_jobs=1)),('spectrogram_resnet18_linear',Z,lambda:make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=SEED)))]
    scores={}; summary=[]
    for name,features,build in specs:
        p=np.zeros(len(y)); per_fold=[]
        for k,(tr,te) in enumerate(folds):
            model=build();model.fit(features[tr],y[tr]);p[te]=model.predict_proba(features[te])[:,1];per_fold.append(dict(fold=k,**metrics(y[te],p[te])))
        scores[name]=p;summary.append(dict(model=name,**metrics(y,p),folds=per_fold));print(name,metrics(y,p),flush=True)
    args.out.mkdir(parents=True)
    report=dict(profile_id='telephone_vowel_81_v1',evaluation_id='telephone_vowel_81_fixed5fold_v1',scope='exploratory Sample-ID-level out-of-fold comparison; no independent external test',seed=SEED,recordings=len(y),healthy=int(sum(y==0)),parkinson=int(sum(y==1)),feature_count=X.shape[1],feature_names=names,demographics_used_for_prediction=False,training_preprocessing='StandardScaler fitted inside each fold; RF uses supplied features directly',spectrogram='8kHz; full recording; Hann256 hop128 PSD; relative dB clipped -80..0; grayscale stretch224; ImageNet ResNet18 frozen/eval; 512 embeddings; scaled logistic head',threshold=.5,hyperparameter_search=False,versions=dict(python=platform.python_version(),**versions),input_sha256={p.name:sha(p.read_bytes()) for p in [args.data_dir/'Demographics_age_sex.xlsx',args.data_dir/'HC_AH.zip',args.data_dir/'PD_AH.zip']},metrics=summary)
    (args.out/'comparison.json').write_text(json.dumps(report,indent=2))
    with (args.out/'predictions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['sample_id','actual','fold']+list(scores));writer.writeheader()
        for i,r in enumerate(records):writer.writerow(dict(sample_id=r['sample_id'],actual=int(y[i]),fold=int(fold_id[i]),**{n:float(v[i]) for n,v in scores.items()}))
    with (args.out/'audit.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    print('Saved:',args.out)
if __name__=='__main__':main()
