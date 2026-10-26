# Parkinson's Policy Model Selector

Research prototype for LLM-assisted, policy-guided pipeline selection and execution in separate Parkinson's speech and spiral-drawing workflows.

## Release scope

This initial code release contains the deterministic selector, request-boundary checks, prompts and schemas, speech runners, pipeline scripts and offline tests.

Source recordings, drawings, extracted feature tables, fitted models, archived response logs and manuscript evidence packages are not included. The checked-in registry contains recorded evaluation estimates; it does not supply the fitted artifacts needed for inference.

Training and evaluation run through separate scripts. The conversational interface interprets requests against registered evidence. Deterministic code checks constraints and ranks eligible models. Installed adapters execute fitted pipelines.

Speech and spiral inputs are separate workflows, without patient-level multimodal fusion. Spiral execution requires separately supplied compatible artifacts.

## Offline checks

From the repository root:

```powershell
python tests/test_batch_runner.py
Push-Location policy-model-selector-v3
python -m unittest test_policy_engine test_request_boundary -v
Pop-Location
```

These checks use synthetic fixtures and mocked execution. They do not make API calls, fit models or measure classification accuracy.

The public copy passed 20 policy-engine and request-boundary tests and the batch integration fixtures during release preparation.

## Dependencies

The supplied dependency list is:

```powershell
python -m pip install -r pd-speech-baseline/requirements.txt
```

It records major dependencies, not a complete environment lockfile. Model compatibility also depends on the versions used during fitting. Imports may require additional dependencies.

Speech extraction requires R and the packages tuneR, gsignal, seewave and signal. ImageNet ResNet18 weights are not bundled and may download on first use.

## Configure local paths

From the repository root, supply your installed Rscript executable:

```powershell
python configure_paths.py --rscript "PATH_TO_RSCRIPT_EXECUTABLE"
```

This replaces the placeholder paths in the telephone and Italian configurations. Do not commit your local configuration changes.

Inference additionally requires compatible fitted artifacts, their fitting metadata and matching registry hashes. The repository alone does not provide a ready-to-run audio classifier.

## Selection and execution

The historical module directory is named policy-model-selector-v3 for compatibility. The source was prepared from PD_Framework_Code_v5 in Revision Package v4.

Natural-language requirements use a restricted policy schema. Selection can return clarification, no eligible model, insufficient evidence or an unresolved tie.

A named-model choice is recorded separately from a performance-based recommendation. Conditional named choices are rejected by a conservative guard; valid paraphrases outside its grammar may also be rejected.

Live language requests require separately configured API access. Offline tests require no credentials. Never commit API keys or environment files.

The interface does not automatically train all algorithms on an uploaded dataset or maintain a continuing clarification conversation.

## Evidence and limitations

Preferred pipelines differed across the evaluated datasets. Nested selection did not consistently improve predictive performance. Interpretation and explanation errors were observed.

Historical live language evaluation, cached replay, synthetic fixtures and real-input demonstrations are different forms of evidence. The checks above do not reproduce the full manuscript experiments.

Recorded estimates do not guarantee performance on future inputs. This prototype is intended for research and is not validated for clinical diagnosis or deployment.

## Public-copy changes

Compared with the archived code release:

- Installation defaults use the repository directory layout.
- Predictor Rscript defaults use PATH; --rscript can override them.
- Configuration files contain portable placeholders.
- The batch fixture copies its required request_boundary module.
- Adapted R files carry modification notices.

Archived source hashes do not describe these modified public copies.

## Attribution and licensing

See THIRD_PARTY_NOTICES.md and LICENSES/Apache-2.0.txt for the supplied R code. Original project code has not yet been assigned a licence. Public availability does not itself grant unrestricted reuse.

Datasets, dependencies and pretrained weights retain their original terms.
