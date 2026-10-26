"""Actual image-pipeline adapters, separate from policy translation."""
import hashlib,json
from pathlib import Path
import cv2,joblib,numpy as np
from skimage.feature import hog
from resnet_helpers import prepare_image

def hog_feature(raw):
 im=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
 if im is None:raise ValueError('Unreadable image')
 gray=cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),(200,200))
 binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
 return hog(binary,orientations=9,pixels_per_cell=(10,10),cells_per_block=(2,2),transform_sqrt=True,block_norm='L1')

class DeepExtractor:
 def __init__(self):
  import torch
  from torchvision.models import resnet18,ResNet18_Weights
  from torchvision.transforms import Compose,ToTensor,Normalize
  self.torch=torch;torch.set_num_threads(2);torch.manual_seed(20261021)
  torch.use_deterministic_algorithms(True)
  self.net=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
  self.net.fc=torch.nn.Identity();self.net.eval()
  for p in self.net.parameters():p.requires_grad_(False)
  self.transform=Compose([ToTensor(),Normalize([.485,.456,.406],[.229,.224,.225])])
 def extract(self,raws):
  chunks=[]
  with self.torch.inference_mode():
   for start in range(0,len(raws),8):
    batch=self.torch.stack([self.transform(prepare_image(r)) for r in raws[start:start+8]])
    chunks.append(self.net(batch).numpy())
  return np.concatenate(chunks)

def infer(raw,tool_id,artifact_directory,profile_id):
 directory=Path(artifact_directory)/profile_id
 metadata=json.loads((directory/'artifacts.json').read_text())
 if metadata['profile_id']!=profile_id:raise ValueError('Artifact profile mismatch')
 name={'spiral_hog_rf':'hog_rf.joblib','spiral_resnet18_linear':'resnet18_linear.joblib'}.get(tool_id)
 if name is None:raise ValueError('Unsupported image tool')
 path=directory/name
 if hashlib.sha256(path.read_bytes()).hexdigest()!=metadata['model_sha256'][name]:raise ValueError('Artifact hash differs')
 # Only load artifacts produced locally by fit_tools.py; joblib is not safe for untrusted files.
 model=joblib.load(path)
 x=hog_feature(raw).reshape(1,-1) if tool_id=='spiral_hog_rf' else DeepExtractor().extract([raw])
 score=float(model.predict_proba(x)[0,1])
 return {'tool_id':tool_id,'tool_version':'1.0','profile_id':profile_id,'artifact_sha256':metadata['model_sha256'][name],
 'prediction':'parkinson' if score>.5 else 'healthy','raw_pd_score':score,
 'score_note':'Uncalibrated model score, not clinical disease probability','fit_images':metadata['fit_images']}
