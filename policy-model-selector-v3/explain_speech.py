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
rankings = {}
for metric in ("accuracy", "sensitivity", "false_positive_rate"):
    ordered = sorted(
        evidence["models"],
        key=lambda m: m["metrics"][metric],
        reverse=(metric != "false_positive_rate"),
    )
    rankings[metric] = [
        {"tool_id": m["tool_id"], "value": m["metrics"][metric]}
        for m in ordered
    ]

evidence["verified_rankings_best_to_worst"] = rankings
evidence["eligible_for_sensitivity_80_percent"] = [
    m["tool_id"] for m in evidence["models"]
    if m["metrics"]["sensitivity"] >= 0.8
]

print("VERIFIED RESULTS")
for m in evidence["models"]:
    v = m["metrics"]
    print(
        f"{m['tool_id']}: accuracy {v['accuracy']:.2%}; "
        f"sensitivity {v['sensitivity']:.2%}; "
        f"false-positive rate {v['false_positive_rate']:.2%}"
    )
for metric, ranking in rankings.items():
    print(metric + " (best to worst): " +
          " > ".join(item["tool_id"] for item in ranking))

prompt = """Explain the supplied speech-model results in under 150 words.
Use the verified rankings as authoritative. Accuracy and sensitivity have
different rankings: do not conflate them. The false-positive-rate ranking
orders lower values first. Refer to the verified numerical table rather than
repeating numerical values. Explain the practical trade-off in plain English.
Recommend the accuracy-leading model only conditional on accuracy being the
user's priority. Use the supplied eligibility list when discussing the
sensitivity requirement. If it is empty, say no candidate meets that requirement.
These are cross-validation results, not an independent test.
Finish by asking for a model choice or another requirement.
Do not claim execution or user approval. No code or commands."""

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
