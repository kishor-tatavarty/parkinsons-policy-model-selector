from request_boundary import require_requirements_only
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from policy_engine import select, dispatch, sha

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True, type=Path)
    p.add_argument("--wav", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()

    root = Path(__file__).resolve().parent
    config = json.loads((root / "speech_config.json").read_text())
    registry = json.loads((root / "model_registry.json").read_text())
    policy = json.loads(a.policy.read_text(encoding="utf-8-sig"))
    require_requirements_only(policy['request'])
    profile = config["profile_id"]
    wav = a.wav.resolve()

    if a.output.exists():
        raise ValueError("Output already exists.")
    if policy["registry_sha256"] != sha(registry):
        raise ValueError("Registry changed; recompile policy.")

    checked = select(registry, policy["policy"], policy["request"], profile)
    if (
        checked["status"] != "selected"
        or checked["selected_model"] != policy.get("selected_model")
        or checked["selected_tool"] != policy.get("selected_tool")
        or checked["policy_sha256"] != policy.get("policy_sha256")
    ):
        raise ValueError("Policy selection mismatch.")

    record = next(
        m for m in registry["models"]
        if m["model_id"] == checked["selected_model"]
    )
    project = Path(config["speech_project"])
    mapping = {
        "speech_acoustic_rf": "acoustic_rf",
        "speech_acoustic_lr": "acoustic_lr",
        "speech_spectrogram_resnet18_linear": "spectrogram_resnet18_linear",
    }

    def execute(tool):
        model_name = mapping[tool]
        artifact = (
            project / "artifacts/speech_extracted_v2"
            / (model_name + ".joblib")
        )
        if digest(artifact) != record["artifact_sha256"]:
            raise ValueError("Artifact differs from registered model.")

        with tempfile.TemporaryDirectory() as temp:
            result_path = Path(temp) / "prediction.json"
            command = [
                sys.executable, str(project / "predict_speech.py"),
                "--model", model_name,
                "--wav", str(wav),
                "--out", str(result_path),
                "--rscript", config["rscript"],
            ]
            run = subprocess.run(
                command, cwd=project, capture_output=True, text=True
            )
            if run.returncode:
                raise RuntimeError(run.stdout + "\n" + run.stderr)
            result = json.loads(result_path.read_text())

        if (
            result["profile_id"] != profile
            or result["selected_tool"] != model_name
            or result["model_sha256"] != record["artifact_sha256"]
            or result["input_sha256"] != digest(wav)
        ):
            raise ValueError("Inference result binding mismatch.")

        result["adapter_model"] = result["selected_tool"]
        result["selected_tool"] = tool
        result["tool_version"] = record["tool_version"]
        return result

    handlers = {
        tool: (lambda tool=tool: execute(tool))
        for tool in mapping
    }
    result = dispatch(checked, profile, handlers)
    report = {
        "policy_sha256": checked["policy_sha256"],
        "registry_sha256": checked["registry_sha256"],
        "selected_model": checked["selected_model"],
        "decision_reason": checked["reason"],
        "result": result,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
