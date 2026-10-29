"""Preflight registry-to-artifact binding. Standard library; never deserializes."""
import hashlib,json
from pathlib import Path
TOOLS={'spiral_hog_rf':'hog_rf.joblib','spiral_resnet18_linear':'resnet18_linear.joblib'}
def verify_image_binding(registry,profile,model_id,tool,artifact_root):
 candidates=[m for m in registry['models'] if m['model_id']==model_id and m['profile_id']==profile and m['evaluation_id']==registry['profiles'][profile]['evaluation_id'] and m['tool_id']==tool]
 if len(candidates)!=1:raise ValueError('No unique compatible registered candidate')
 expected=candidates[0].get('artifact_sha256')
 if not expected:raise ValueError('No registry artifact binding; execution unavailable')
 if tool not in TOOLS:raise ValueError('Unsupported image tool')
 directory=Path(artifact_root)/profile
 metadata=json.loads((directory/'artifacts.json').read_text())
 if metadata.get('profile_id')!=profile:raise ValueError('Artifact profile mismatch')
 path=directory/TOOLS[tool];observed=hashlib.sha256(path.read_bytes()).hexdigest()
 if metadata.get('model_sha256',{}).get(path.name)!=expected or observed!=expected:raise ValueError('Registry, metadata and artifact hashes differ')
 return dict(profile_id=profile,model_id=model_id,evaluation_id=candidates[0]['evaluation_id'],artifact_sha256=observed,fit_images=metadata.get('fit_images'),scope='byte identity; no clinical or evaluation validity assertion')
