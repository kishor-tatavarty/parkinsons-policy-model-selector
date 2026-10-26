import argparse
import csv
import hashlib
import io
import math
from scipy.signal import resample_poly
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
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise SystemExit("Expected finite mono audio.")
    if audio.dtype == np.int16 and rate in (16000, 44100):
        divisor = math.gcd(rate, 8000)
        audio = resample_poly(audio.astype(np.float64) / 32768.0,
                              8000 // divisor, rate // divisor).astype(np.float32)
    elif rate == 8000 and audio.dtype == np.float32:
        audio = audio.copy()
    else:
        raise SystemExit("Use original mono PCM16 at 16/44.1kHz or prepared float32 at 8kHz.")
    if len(audio) < 12000 or not np.isfinite(audio).all():
        raise SystemExit("Prepared audio must be finite and at least 1.5 seconds.")
    buffer = io.BytesIO()
    wavfile.write(buffer, 8000, audio)
    prepared_raw = buffer.getvalue()

    artifacts = Path("artifacts/italian_vowel_a_grouped_v1")
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
        extractor = Path("extract_italian_features.R").resolve()
        if digest(extractor) != meta["extractor_sha256"]:
            raise SystemExit("Extractor hash mismatch.")

        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            inputs = work / "data/italian_prepared_v1"
            inputs.mkdir(parents=True)
            (inputs / "case.wav").write_bytes(prepared_raw)
            shutil.copy2(extractor, work / "extract_italian_features.R")

            run = subprocess.run(
                [args.rscript, "extract_italian_features.R"],
                cwd=work, capture_output=True, text=True
            )
            if run.returncode:
                raise SystemExit(run.stdout + "\n" + run.stderr)

            csv_path = work / "results/italian_extraction_v1/extracted_features.csv"
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

    prepared_hash = hashlib.sha256(prepared_raw).hexdigest()
    training_hashes = set(meta["training_prepared_sha256"])

    result = {
        "profile_id": meta["profile_id"],
        "selected_tool": args.model,
        "input_file": args.wav.name,
        "input_sha256": audio_hash,
        "model_sha256": digest(model_path),
        "prediction": "parkinson" if score >= 0.5 else "healthy",
        "raw_pd_score": score,
        "score_note": "Uncalibrated classifier score.",
        "prepared_sha256": prepared_hash,
        "previously_used_for_fitting": prepared_hash in training_hashes
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
