# parkinsons-policy-model-selector

LLM-assisted, policy-guided model selection and execution for Parkinson’s speech and spiral-drawing analysis, with dataset-specific evaluation records, deterministic constraint checks, and traceable pipeline execution. Research prototype.

## Release status

The public release is being prepared. Implementation files, evaluation evidence, and reproduction instructions have **not yet been uploaded**. This README describes the study, not an executable release.

## Workflow

1. Train and evaluate compatible candidate pipelines using separate scripts, then register their results under a dataset profile.
2. Present registered comparisons and accept researcher requirements or an explicit model choice.
3. Use an LLM to translate natural-language requirements into a restricted structured policy. Deterministic code checks constraints and ranks eligible candidates.
4. Report clarification needs, ties, insufficient evidence, or no eligible model when appropriate.
5. Invoke a registered fitted-pipeline adapter and retain decision and execution records where implemented.

Speech and spiral drawings use separate workflows. The project does not fuse paired speech and drawing observations into a patient-level multimodal model.

## Evidence and limitations

The study evaluates candidate pipelines, numerical selection policies, synthetic language requests, and selected execution paths. Preferred pipelines differed across datasets. Adaptive selection did not consistently improve prediction, and language interpretation and explanation errors were observed.

Historical live language results, cached regression results, simulated execution fixtures, and real-input demonstrations are distinct forms of evidence. Recorded performance estimates do not guarantee performance on future inputs. This research prototype is not validated for clinical diagnosis or clinical deployment.

## Planned release materials

The archived study packages will be inspected before publishing available implementation files, deterministic tests, prompts and schemas, shareable language records, predictions, provenance records, and reproduction instructions. Only inspected materials will be added. Dependency information and commands will identify which results can be reproduced and which require separately obtained inputs or model artifacts.

## Source data and credentials

Obtain source datasets under their original access and licence conditions. This repository does not grant permission to redistribute recordings or drawings. Do not upload credentials, raw source datasets, private local paths, or unrestricted dumps of local working directories. The `.gitignore` is a convenience filter, not a substitute for inspection.

## Licensing and attribution

A project licence has not yet been assigned. Third-party code must retain its original notices and licence conditions. The adapted R spectral-feature extractor requires source attribution and preservation of its Apache License notice. Dependencies and pretrained weights remain subject to their respective terms.
