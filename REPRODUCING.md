# Reproduction requirements

This public revision contains source code and aggregate results. It deliberately excludes model binaries and individual input/reference/response/participant records. Full numerical reproduction requires the corresponding author release packages; those inputs must be placed under the directories expected by the scripts. Do not mistake absent inputs for reproduced experiments.

Use the offline runtime checks in README first. Models are not deserialized by verify_artifact_bindings.py; it reports missing files until trusted artifacts are supplied. The status runner accepts --release, --output, --kaggle-image, --newhandpd-image, --telephone-wav and --italian-wav; only profiles with supplied compatible inputs execute. It refuses an existing output directory. Speech inference also requires configured R. It makes no API calls.

The bootstrap script accepts --inputs, --output, --replicates10000 (use a space between the flag and value) and --seed20261029 (likewise). It requires normalized observation/group/prediction CSVs from the author package. The language scorer accepts --llm-results, --offline-results and --output and requires its original references under analysis/inputs. The frozen language runner needs its original requests and references before it can run. Supplying --execute to run_llm.py enables paid new generations; it is not required to score saved outputs.

The source and aggregate tables do not establish new independent predictive validation, clinical readiness, or general superiority of an LLM over every parser. Original and post-output scoring results are retained in four_arm_results.csv.
