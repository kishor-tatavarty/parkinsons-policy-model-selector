from pathlib import Path
import csv
import hashlib
import json
import re
import unicodedata
import numpy as np
from scipy.io import wavfile
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from run_speech import embeddings, metrics, SEED

def load_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)

def normalized_name(folder):
    name = folder.replace("\\", "/").split("/")[-1]
    name = unicodedata.normalize("NFKD", name)
    return re.sub(r"[^a-z0-9]", "", name.encode("ascii", "ignore").decode().lower())

def main():
    out = Path("results/italian_comparison_v1")
    if out.exists():
        raise SystemExit("Comparison output already exists.")

    feature_path = Path("results/italian_extraction_v1/extracted_features.csv")
    manifest_path = Path("data/italian_prepared_v1/manifest.csv")
    fields, features = load_csv(feature_path)
    _, manifest = load_csv(manifest_path)
    by_id = {r["sample_id"]: r for r in manifest}
    assert len(by_id) == len(manifest)
    assert len({r["sample_id"] for r in features}) == len(features)

    names = fields[1:]
    assert len(names) == 80 and len(features) == 98
    records = [by_id[r["sample_id"]] for r in features]
    X = np.array([[float(r[c]) for c in names] for r in features])
    assert np.isfinite(X).all()
    y = np.array([r["label"] == "parkinson" for r in records], dtype=int)
    assert (int(sum(y == 0)), int(sum(y == 1))) == (43, 55)

    # Conservative grouping: matching folder names across sessions are merged.
    groups = [
        r["label"] + "__" + normalized_name(r["recording_folder"])
        for r in records
    ]
    assert all(g.split("__", 1)[1] for g in groups)

    # Merge groups linked by duplicate prepared-waveform hashes.
    parent = {g: g for g in groups}
    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g
    seen = {}
    for r, g in zip(records, groups):
        h = r["prepared_sha256"]
        if h in seen:
            previous_group, previous_label = seen[h]
            assert previous_label == r["label"], "Duplicate audio across labels"
            parent[find(g)] = find(previous_group)
        else:
            seen[h] = (g, r["label"])
    groups = np.array([find(g) for g in groups])

    folds = list(StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=SEED
    ).split(X, y, groups))
    fold_ids = np.full(len(y), -1, dtype=int)
    for k, (tr, te) in enumerate(folds):
        assert not set(groups[tr]) & set(groups[te])
        assert len(set(y[tr])) == len(set(y[te])) == 2
        assert (fold_ids[te] == -1).all()
        fold_ids[te] = k
    assert (fold_ids >= 0).all()

    print("Recordings:", len(y), "Conservative groups:", len(set(groups)), flush=True)
    print("Extracting frozen ResNet18 embeddings...", flush=True)
    waves = []
    for r in records:
        path = Path("data/italian_prepared_v1") / (r["sample_id"] + ".wav")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == r["prepared_sha256"]
        rate, audio = wavfile.read(path)
        assert rate == 8000 and audio.ndim == 1 and audio.dtype == np.float32
        waves.append(audio)
    Z, versions = embeddings(waves)

    linear = lambda: make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1, max_iter=2000, random_state=SEED)
    )
    candidates = [
        ("acoustic_lr", X, linear),
        ("acoustic_rf", X, lambda: RandomForestClassifier(
            n_estimators=300, random_state=SEED, n_jobs=1
        )),
        ("spectrogram_resnet18_linear", Z, linear),
    ]
    results = []
    scores = {}
    for name, data, build in candidates:
        probabilities = np.zeros(len(y))
        fold_results = []
        for k, (tr, te) in enumerate(folds):
            model = build()
            model.fit(data[tr], y[tr])
            positive = list(model.classes_).index(1)
            probabilities[te] = model.predict_proba(data[te])[:, positive]
            fold_results.append({
                "fold": k, **metrics(y[te], probabilities[te])
            })
        scores[name] = probabilities
        pooled = metrics(y, probabilities)
        results.append({"model": name, **pooled, "folds": fold_results})
        print(name, json.dumps(pooled), flush=True)

    out.mkdir(parents=True)
    report = {
        "profile_id": "italian_vowel_a_grouped_v1",
        "evaluation_id": "italian_a98_grouped5fold_v1",
        "scope": "Exploratory grouped five-fold OOF; 98 recordings; no external test",
        "seed": SEED,
        "recordings": len(y),
        "healthy": int(sum(y == 0)),
        "parkinson": int(sum(y == 1)),
        "conservative_groups": len(set(groups)),
        "grouping": "Normalized participant-folder names merged across sessions; duplicate prepared-file hashes linked. Identity not independently verified.",
        "excluded_sample_ids": sorted(set(by_id) - {r["sample_id"] for r in features}),
        "preprocessing": "Mono PCM16 scaled and polyphase-resampled to 8kHz float32. R spectral extractor unchanged; ResNet uses whole prepared recording.",
        "classification_threshold": 0.5,
        "hyperparameter_search": False,
        "feature_names": names,
        "feature_sha256": hashlib.sha256(feature_path.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "versions": versions,
        "metrics": results,
    }
    (out / "comparison.json").write_text(json.dumps(report, indent=2))
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sample_id", "actual", "group", "fold", *scores
        ])
        writer.writeheader()
        for i, r in enumerate(records):
            writer.writerow({
                "sample_id": r["sample_id"],
                "actual": int(y[i]),
                "group": groups[i],
                "fold": int(fold_ids[i]),
                **{name: float(values[i]) for name, values in scores.items()},
            })
    print("Saved:", out)

if __name__ == "__main__":
    main()
