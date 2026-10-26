import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from policy_engine import metrics, sha

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wav", type=Path)
    p.add_argument("--choose", choices=[
        "acoustic_rf", "acoustic_lr", "spectrogram_resnet18_linear"
    ])
    p.add_argument("--output", type=Path)
    a = p.parse_args()

    root = Path(__file__).resolve().parent
    registry = json.loads((root / "model_registry.json").read_text())
    config = json.loads((root / "speech_config.json").read_text())
    profile = config["profile_id"]
    evaluation = registry["profiles"][profile]["evaluation_id"]
    candidates = [
        m for m in registry["models"]
        if m["profile_id"] == profile and m["evaluation_id"] == evaluation
    ]
    mapping = {
        "acoustic_rf": "speech_acoustic_rf",
        "acoustic_lr": "speech_acoustic_lr",
        "spectrogram_resnet18_linear": "speech_spectrogram_resnet18_linear",
    }

    print("Speech comparison: same 81 recordings, fixed five-fold evaluation.")
    for name, tool in mapping.items():
        record = next(m for m in candidates if m["tool_id"] == tool)
        values = metrics(record)
        print(
            f"{name}: accuracy {values['accuracy']:.2%}; "
            f"PD detection {values['sensitivity']:.2%}; "
            f"healthy correctly classified {values['specificity']:.2%}"
        )

    print("\nRF has the highest observed accuracy in this comparison.")
    print("No candidate reaches 80% PD detection.")
    print("This summary uses registered evidence; it makes no LLM call.")

    if not a.choose:
        print("\nUse --choose MODEL --wav PATH --output PATH to execute your choice.")
        return
    if not a.wav or not a.output:
        raise ValueError("--choose requires --wav and --output.")
    if a.output.exists():
        raise ValueError("Output exists; choose a new filename.")

    wav = a.wav.resolve()
    input_hash = digest(wav)
    tool = mapping[a.choose]
    record = next(m for m in candidates if m["tool_id"] == tool)
    project = Path(config["speech_project"])
    artifact = project / "artifacts/speech_extracted_v2" / (a.choose + ".joblib")
    if digest(artifact) != record["artifact_sha256"]:
        raise ValueError("Registered artifact hash mismatch.")

    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp) / "prediction.json"
        run = subprocess.run([
            sys.executable, str(project / "predict_speech.py"),
            "--model", a.choose, "--wav", str(wav),
            "--out", str(output), "--rscript", config["rscript"],
        ], cwd=project, capture_output=True, text=True)
        if run.returncode:
            raise RuntimeError(run.stdout + "\n" + run.stderr)
        result = json.loads(output.read_text())

    if (
        result["profile_id"] != profile
        or result["selected_tool"] != a.choose
        or result["model_sha256"] != record["artifact_sha256"]
        or result["input_sha256"] != input_hash
        or digest(wav) != input_hash
    ):
        raise ValueError("Inference binding mismatch.")

    result["adapter_model"] = result["selected_tool"]
    result["selected_tool"] = tool
    result["tool_version"] = record["tool_version"]
    report = {
        "selection_mode": "explicit_user_choice",
        "selection_source": "CLI --choose",
        "profile_id": profile,
        "evaluation_id": evaluation,
        "registry_sha256": sha(registry),
        "selected_model": record["model_id"],
        "selected_metrics": metrics(record),
        "policy_constraints_checked": False,
        "decision_reason": "User explicitly selected this registered model.",
        "result": result,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n" + json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
