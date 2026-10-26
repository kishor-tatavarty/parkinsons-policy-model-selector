from pathlib import Path
import csv
import hashlib
import json
import joblib
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from run_speech import read_data, embeddings, SEED

out = Path("artifacts/speech_extracted_v2")
if out.exists():
    raise SystemExit("Artifact directory already exists; refusing to overwrite.")

records, waves, _, _ = read_data(Path("data"))
feature_path = Path("results/extraction_v1/extracted_features.csv")

with feature_path.open(newline="") as f:
    reader = csv.DictReader(f)
    names = reader.fieldnames[1:]
    rows = list(reader)

by_id = {r["sample_id"]: r for r in rows}
ids = [r["sample_id"] for r in records]
assert len(by_id) == len(rows), "Duplicate feature IDs"
assert set(by_id) == set(ids), "Feature/audio ID mismatch"
assert len(names) == 80

X = np.array([[float(by_id[sid][c]) for c in names] for sid in ids])
assert np.isfinite(X).all()
y = np.array([r["label"] == "PwPD" for r in records], dtype=int)

Z, versions = embeddings(waves)
models = {
    "acoustic_lr": (
        make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1, max_iter=2000, random_state=SEED)
        ), X
    ),
    "acoustic_rf": (
        RandomForestClassifier(
            n_estimators=300, random_state=SEED, n_jobs=1
        ), X
    ),
    "spectrogram_resnet18_linear": (
        make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1, max_iter=2000, random_state=SEED)
        ), Z
    ),
}

extractor = Path("extract_speech_features.R")
assert extractor.is_file(), "Extraction script missing"

out.mkdir(parents=True)
hashes = {}
for name, (model, features) in models.items():
    model.fit(features, y)
    path = out / f"{name}.joblib"
    joblib.dump(model, path)
    hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    print("Saved:", path)

metadata = {
    "profile_id": "telephone_vowel_81_extracted_v2",
    "fit_recordings": len(y),
    "healthy": int(sum(y == 0)),
    "parkinson": int(sum(y == 1)),
    "feature_names": names,
    "feature_csv_sha256": hashlib.sha256(feature_path.read_bytes()).hexdigest(),
    "extractor_sha256": hashlib.sha256(extractor.read_bytes()).hexdigest(),
    "model_sha256": hashes,
    "training_sample_ids": ids,
    "classification_threshold": 0.5,
    "versions": versions,
    "note": "Final inference models fitted on all 81 recordings. Performance evidence remains the earlier out-of-fold evaluation."
}
(out / "artifacts.json").write_text(json.dumps(metadata, indent=2))
print("Complete: three fitted speech pipelines saved.")
