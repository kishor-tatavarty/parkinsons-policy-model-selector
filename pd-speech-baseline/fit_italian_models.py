from pathlib import Path
import csv
import hashlib
import json
import joblib
import numpy as np
from scipy.io import wavfile
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from run_speech import embeddings, SEED

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)

out = Path("artifacts/italian_vowel_a_grouped_v1")
if out.exists():
    raise SystemExit("Italian artifacts already exist; refusing to overwrite.")

report_path = Path("results/italian_comparison_v1/comparison.json")
report = json.loads(report_path.read_text())
feature_path = Path("results/italian_extraction_v1/extracted_features.csv")
manifest_path = Path("data/italian_prepared_v1/manifest.csv")

assert digest(feature_path) == report["feature_sha256"]
assert digest(manifest_path) == report["manifest_sha256"]

fields, rows = read_csv(feature_path)
_, manifest = read_csv(manifest_path)
by_id = {r["sample_id"]: r for r in manifest}
assert len(by_id) == len(manifest)
assert len({r["sample_id"] for r in rows}) == len(rows)

records = [by_id[r["sample_id"]] for r in rows]
names = fields[1:]
assert len(names) == 80
X = np.array([[float(r[c]) for c in names] for r in rows])
assert np.isfinite(X).all()
y = np.array([r["label"] == "parkinson" for r in records], dtype=int)
assert len(y) == 98

waves = []
for r in records:
    path = Path("data/italian_prepared_v1") / (r["sample_id"] + ".wav")
    assert digest(path) == r["prepared_sha256"]
    rate, audio = wavfile.read(path)
    assert rate == 8000 and audio.ndim == 1 and audio.dtype == np.float32
    waves.append(audio)

Z, versions = embeddings(waves)
linear = lambda: make_pipeline(
    StandardScaler(),
    LogisticRegression(C=1, max_iter=2000, random_state=SEED)
)
models = {
    "acoustic_lr": (linear(), X),
    "acoustic_rf": (
        RandomForestClassifier(
            n_estimators=300, random_state=SEED, n_jobs=1
        ), X
    ),
    "spectrogram_resnet18_linear": (linear(), Z),
}

out.mkdir(parents=True)
hashes = {}
for name, (model, data) in models.items():
    model.fit(data, y)
    path = out / (name + ".joblib")
    joblib.dump(model, path)
    hashes[path.name] = digest(path)
    print("Saved:", path)

metadata = {
    "profile_id": report["profile_id"],
    "evaluation_id": report["evaluation_id"],
    "fit_recordings": len(y),
    "feature_names": names,
    "feature_sha256": digest(feature_path),
    "manifest_sha256": digest(manifest_path),
    "extractor_sha256": digest(Path("extract_italian_features.R")),
    "model_sha256": hashes,
    "training_prepared_sha256": [r["prepared_sha256"] for r in records],
    "classification_threshold": 0.5,
    "versions": versions,
    "note": "Final models fitted on all 98 retained recordings. Performance evidence is the preceding grouped OOF evaluation."
}
(out / "artifacts.json").write_text(json.dumps(metadata, indent=2))
print("Complete: three Italian speech pipelines saved.")
