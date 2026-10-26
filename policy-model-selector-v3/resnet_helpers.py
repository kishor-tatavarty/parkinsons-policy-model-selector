"""Frozen ImageNet ResNet18 + linear classification head; training CV only."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score

def prepare_image(raw):
    # White composite prevents transparent pixels becoming black background.
    im=Image.open(io.BytesIO(raw)).convert('RGBA')
    bg=Image.new('RGBA',im.size,'white');bg.alpha_composite(im)
    return ImageOps.pad(bg.convert('RGB'),(224,224),method=Image.Resampling.BILINEAR,color='white')

def metrics(y,p):
    pred=(p>.5).astype(int)
    tn,fp,fn,tp=map(int,confusion_matrix(y,pred,labels=[0,1]).ravel())
    return {'n':len(y),'correct':tn+tp,'tn':tn,'fp':fp,'fn':fn,'tp':tp,
            'accuracy':float(np.mean(pred==y)),'sensitivity':tp/(tp+fn),
            'specificity':tn/(tn+fp),'roc_auc':float(roc_auc_score(y,p))}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--zip',required=True,type=Path)
    p.add_argument('--stage',choices=['smoke','cv'],default='smoke')
    p.add_argument('--output',type=Path,default=Path('results/resnet18_transfer'))
    a=p.parse_args()
    import torch
    import torchvision
    from torchvision.models import resnet18, ResNet18_Weights
    from torchvision.transforms import Compose,ToTensor,Normalize
    torch.manual_seed(20261021);torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    with zipfile.ZipFile(a.zip) as z:
        prefix='spiral/' if any(n.startswith('spiral/training/') for n in z.namelist()) else 'drawings/spiral/'
        names=sorted(n for n in z.namelist() if n.startswith(prefix+'training/') and n.lower().endswith('.png'))
        if len(names)!=72:raise ValueError('Expected 72 training spirals')
        if a.stage=='smoke':
            names=[next(n for n in names if '/healthy/' in n),next(n for n in names if '/parkinson/' in n)]
        rows=[{'member':n,'actual':n.split('/')[-2], 'sha256':hashlib.sha256(z.read(n)).hexdigest()} for n in names]
        images=[prepare_image(z.read(n)) for n in names]
    print('Loading ImageNet ResNet18 weights (first run downloads about 45 MB).',flush=True)
    weights=ResNet18_Weights.IMAGENET1K_V1
    model=resnet18(weights=weights,progress=True)
    model.fc=torch.nn.Identity()
    model.eval()
    for param in model.parameters():param.requires_grad_(False)
    transform=Compose([ToTensor(),Normalize([.485,.456,.406],[.229,.224,.225])])
    chunks=[]
    with torch.inference_mode():
        for start in range(0,len(images),8):
            batch=torch.stack([transform(im) for im in images[start:start+8]])
            chunks.append(model(batch).numpy())
            print('Features:', min(start+8,len(images)),'/',len(images),flush=True)
    x=np.concatenate(chunks)
    assert x.shape==(len(images),512) and np.isfinite(x).all()
    a.output.mkdir(parents=True,exist_ok=True)
    report={'stage':a.stage,'weights':'ResNet18_Weights.IMAGENET1K_V1','backbone':'frozen; eval mode',
            'head':'StandardScaler + LogisticRegression(C=1,max_iter=2000); fitted inside each fold',
            'preprocessing':'white composite; aspect-preserving pad to224; ImageNet normalize; no crop or augmentation',
            'test_images_decoded':False,'feature_shape':list(x.shape),
            'torch':torch.__version__,'torchvision':torchvision.__version__,
            'archive_sha256':hashlib.sha256(a.zip.read_bytes()).hexdigest()}
    if a.stage=='cv':
        y=np.array([int(r['actual']=='parkinson') for r in rows])
        assert y.sum()==36
        folds=list(StratifiedKFold(n_splits=6,shuffle=True,random_state=20261021).split(x,y))
        model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=20261021))
        scores=cross_val_predict(model,x,y,cv=folds,method='predict_proba')[:,1]
        for fold,(_,valid) in enumerate(folds):
            for i in valid:rows[i].update(fold=fold+1,predicted='parkinson' if scores[i]>.5 else 'healthy',pd_score=float(scores[i]))
        report.update(fold_seed=20261021,metrics=metrics(y,scores),scope='exploratory image-level training CV')
        with (a.output/'out_of_fold_predictions.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    else: report['status']='Feature extraction works; classifier not evaluated'
    (a.output/(a.stage+'_summary.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
