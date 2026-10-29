# Parkinson's Policy Model Selector

Research prototype for LLM-assisted policy-guided selection and execution in separate Parkinson's speech and spiral workflows. Research use; not a clinical diagnostic system.

## Public revision, 29 October 2026

Original project code and associated author-created documentation are available under the [MIT License](LICENSE). The supplied/adapted UAMS R code retains its [Apache 2.0 attribution](THIRD_PARTY_NOTICES.md). Datasets, dependencies and pretrained weights retain their own terms.

This update includes:
- Deterministic selector, prompts/schema, request guards, speech and image execution runners.
- Registry hash bindings for all ten classifiers and pre-load image verification checks.
- [Model-level verification table](ARTIFACT_VERIFICATION.csv): expected/observed model hashes, evaluation, preprocessing, fitting scope and recorded test status.
- [Status execution runner](verification/status_execution/run_status_cases.py), with an observer that calls the real predictor rather than substituting a mock.
- [Aggregate 120-request benchmark results](research/language_benchmark120/analysis/RESULTS_AND_INTERPRETATION.txt) and evaluation/generation code.
- [Bootstrap intervals and paired differences](research/statistics/results) plus analysis code.

Serialized model binaries, raw recordings/images, participant-linked manifests/predictions and individual benchmark logs/input/reference bundles are not included. Automatic publication review blocked the broader release. See [publication scope](docs/PUBLICATION_SCOPE.md). Supply trusted, hash-matching artifacts for inference. The recorded verification table describes the author release, not model files installed by this public checkout.

## What was verified

The author release contains ten hash-bound classifiers. Kaggle's recovered ResNet head and reconstructed RF reproduce all 30 earlier test predictions and scores. NewHandPD's original models reproduce all 48 earlier test labels; their fitting scope is 196 images, separate from the OOF244 pipeline evaluation. Four real policy-to-execution checks passed per spiral profile. The supplied Kaggle test had already been inspected; NewHandPD reconstructed groups do not independently establish participant identity.

The status-control run had 13 passed image entry-point checks (four predictor invocations, nine refusals), and 11 speech checks pending local WAV inputs. No independent speech validation is claimed. Fixture tests are separate from genuine image inference.

## Language benchmark summary

| Arm | Policy semantics | Final outcome |
| --- | --- | --- |
| Reference-informed structured control | 120/120 (100.00%) | 120/120 (100.00%) |
| Frozen narrow rule parser | 0/120 (0.00%) | 20/120 (16.67%) |
| LLM raw | 100/120 (83.33%) | 111/120 (92.50%) |
| Same LLM plus deterministic validation | 100/120 (83.33%) | 91/120 (75.83%) |

Requests were Claude-generated; this is not an independent human or clinician benchmark. Two contradictory-bound representations received a disclosed post-output AI-assisted scoring adjustment; original LLM semantic agreement was 98/120. Validation reduced flagged wrong authorizations from eight to four while reducing selections from 80 to 64. It did not repair interpretation. The parser clarified every request, so this does not establish superiority over all parsers. No classifier was executed by this benchmark.

## Offline checks

```powershell
python verify_repository.py
python verify_artifact_bindings.py
python tests/test_batch_runner.py
Push-Location policy-model-selector-v3
python -m unittest test_policy_engine test_request_boundary test_artifact_binding -v
Pop-Location
```

The artifact audit reports absent model files until separately supplied. Unit and batch tests use fixtures, require no API calls and do not establish predictive accuracy. The public runtime passed 25 unit tests and the batch fixtures before publication.

Speech setup uses `pd-speech-baseline/requirements.txt`, R, tuneR, gsignal, seewave and signal. Configure local R with `python configure_paths.py --rscript "PATH_TO_RSCRIPT_EXECUTABLE"`. Image replay used `requirements-image.txt`, Python3.12.14 and scikit-learn1.8.0; serialized models are version-sensitive. ResNet may download ImageNet weights on first use. Never commit keys or local environment files.

Bootstrap intervals are conditional on saved predictions/folds, without retraining. Telephone used 81 Sample IDs, Italian 46 reconstructed groups, NewHandPD59 connected groups; Kaggle used image-level resampling. These intervals neither guarantee future performance nor establish statistical superiority. Selection still uses point estimates; uncertainty-aware eligibility is future work.

Training/evaluation remain separate scripts. Named choices are distinguished from policy recommendations. No patient-level speech/spiral fusion, automatic training on arbitrary uploads or continuing clarification conversation is implemented. Original public release remains in Git history at `1f7512b5a8db43d87af9f61c7e19b12a126c4b96`.
