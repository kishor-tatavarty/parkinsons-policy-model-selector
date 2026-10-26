import argparse
import json
import os
import urllib.request
from pathlib import Path
from policy_engine import metrics, sha
from translate_policy import MODEL

p = argparse.ArgumentParser()
p.add_argument("--execute", action="store_true")
p.add_argument("--key-env-file", type=Path)
a = p.parse_args()

root = Path(__file__).resolve().parent
registry = json.loads((root / "model_registry.json").read_text())
config = json.loads((root / "speech_config.json").read_text())
profile = config["profile_id"]
evaluation = registry["profiles"][profile]
records = [
    m for m in registry["models"]
    if m["profile_id"] == profile
    and m["evaluation_id"] == evaluation["evaluation_id"]
]
evidence = {
    "profile_id": profile,
    "scope": evaluation["scope"],
    "models": [{
        "model_id": m["model_id"],
        "tool_id": m["tool_id"],
        "confusion": m["confusion"],
        "metrics": metrics(m),
    } for m in records],
}
prompt = """Explain this speech-model comparison in plain English.
Use only the supplied evidence. Keep the response under 200 words.
Compare accuracy, Parkinson-case detection and false alarms.
Identify the highest observed accuracy, conditional on accuracy being the user's
priority. Do not claim universal superiority or use model scores as clinical
probabilities. Explain that these are cross-validation results.
Give two concrete choices: highest accuracy, and a sensitivity floor of 80%.
If no model meets that floor, say so. Finish by asking whether the user wants
the accuracy-leading model, a different named model, or another requirement.
Do not claim a tool was executed, a user choice was made, or new tests were run.
No executable commands or code."""
body = {
    "model": MODEL, "store": False, "temperature": 0,
    "max_output_tokens": 700,
    "input": [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(evidence)},
    ],
}
contract = {"body": body, "registry_sha256": sha(registry)}
directory = root / "results/speech_explanation" / sha(contract)[:16]
directory.mkdir(parents=True, exist_ok=True)
(directory / "contract.json").write_text(
    json.dumps(contract, indent=2), encoding="utf-8"
)
response_path = directory / "response.json"

if response_path.exists():
    response = json.loads(response_path.read_text())
elif not a.execute:
    print("Prepared explanation request. No API call made.")
    print("Run again with --execute to request the explanation.")
    raise SystemExit()
else:
    marker = directory / "attempted.json"
    if marker.exists():
        raise SystemExit("Prior attempt has no saved response; stopped.")
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if a.key_env_file:
        for line in a.key_env_file.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip().removeprefix("export ")
            if line.startswith("OPENAI_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("\"'")
                break
    if not key:
        raise SystemExit("API key unavailable; no request sent.")
    marker.write_text(json.dumps({"contract_sha256": sha(contract)}))
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + key,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as reply:
            response = json.load(reply)
    except Exception:
        raise SystemExit("API/network failure; stopped without retry.")
    response_path.write_text(json.dumps(response, indent=2), encoding="utf-8")

if response.get("status") != "completed":
    raise SystemExit("Incomplete response.")
parts = [
    c for item in response.get("output", [])
    for c in item.get("content", [])
]
if any(c.get("type") == "refusal" for c in parts):
    raise SystemExit("Model refused the request.")
text = "\n".join(
    c["text"] for c in parts if c.get("type") == "output_text"
)
if not text.strip():
    raise SystemExit("No explanation returned.")
(directory / "explanation.txt").write_text(text, encoding="utf-8")
print(text)
print("\nSaved:", directory)
