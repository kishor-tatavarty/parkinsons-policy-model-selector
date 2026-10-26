import argparse
import csv
import hashlib
import io
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import joblib
import numpy as np
from scipy.io import wavfile
from run_speech import embeddings

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=[
        "acoustic_lr", "acoustic_rf", "spectrogram_resnet18_linear"
    ])
    parser.add_argument("--wav", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--rscript", default=
        "Rscript")
    args = parser.parse_args()

    if args.out.exists():
        raise SystemExit("Output exists; choose a new --out filename.")

    raw = args.wav.read_bytes()
    rate, audio = wavfile.read(io.BytesIO(raw))
    if rate != 8000 or audio.ndim != 1:
        raise SystemExit("This profile requires mono 8000 Hz WAV audio.")
    if len(audio) / rate < 1.5 or not np.isfinite(audio).all():
        raise SystemExit("Audio must be finite and at least 1.5 seconds.")

    artifacts = Path("artifacts/speech_extracted_v2")
    meta = json.loads((artifacts / "artifacts.json").read_text())
    model_path = artifacts / (args.model + ".joblib")
    if digest(model_path) != meta["model_sha256"][model_path.name]:
        raise SystemExit("Model hash mismatch.")

    if args.model == "spectrogram_resnet18_linear":
        if audio.dtype != np.float32:
            raise SystemExit(
                "ResNet v2 expects float32 WAV, matching its training input. "
                "Integer PCM support needs explicit normalization."
            )
        features, _ = embeddings([audio])
    else:
        extractor = Path("extract_speech_features.R").resolve()
        if digest(extractor) != meta["extractor_sha256"]:
            raise SystemExit("Extractor hash mismatch.")

        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            inputs = work / "data/extracted"
            inputs.mkdir(parents=True)
            (inputs / "case.wav").write_bytes(raw)
            shutil.copy2(extractor, work / "extract_speech_features.R")

            run = subprocess.run(
                [args.rscript, "extract_speech_features.R"],
                cwd=work, capture_output=True, text=True
            )
            if run.returncode:
                raise SystemExit(run.stdout + "\n" + run.stderr)

            csv_path = work / "results/extraction_v1/extracted_features.csv"
            if not csv_path.exists():
                raise SystemExit("Extractor produced no feature CSV.")
            with csv_path.open(newline="") as f:
                reader = csv.DictReader(f)
                columns = reader.fieldnames
                rows = list(reader)

            expected = ["sample_id"] + meta["feature_names"]
            if columns != expected or len(rows) != 1:
                raise SystemExit("Unexpected feature schema or row count.")
            features = np.array([[
                float(rows[0][name]) for name in meta["feature_names"]
            ]])

    if not np.isfinite(features).all():
        raise SystemExit("Nonfinite inference features.")

    model = joblib.load(model_path)
    positive = list(model.classes_).index(1)
    score = float(model.predict_proba(features)[0, positive])
    audio_hash = hashlib.sha256(raw).hexdigest()

    # Detect reuse of a recording included in final model fitting.
    with Path("results/speech_extracted_v2/audit.csv").open(newline="") as f:
        training_hashes = {row["wav_sha256"] for row in csv.DictReader(f)}

    result = {
        "profile_id": meta["profile_id"],
        "selected_tool": args.model,
        "input_file": args.wav.name,
        "input_sha256": audio_hash,
        "model_sha256": digest(model_path),
        "prediction": "parkinson" if score >= 0.5 else "healthy",
        "raw_pd_score": score,
        "score_note": "Uncalibrated classifier score.",
        "previously_used_for_fitting": audio_hash in training_hashes
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
