"""Verify curated repository bytes. Run before configuring local paths."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
manifest=json.loads((root/'REPOSITORY_SHA256.json').read_text())
failures=[name for name,expected in manifest.items() if not (root/name).is_file() or hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected]
print('Files checked:',len(manifest),'Mismatches:',len(failures))
for name in failures:print(name)
if failures:raise SystemExit(1)
print('File identity only; no assertion of clinical validity or source authentication.')
