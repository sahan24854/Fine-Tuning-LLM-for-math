# Notebook 07: inference-path check and balanced-data preparation

## Purpose and scope

Notebook 06's lower learning rate produced 3/50 correct answers, compared with the first adapter's 6/50. Its repeated original-model baseline differs on two questions from Notebook 05, both still wrong. Do not interpret that trial as proof of a general learning-rate effect. Preserve both adapters and all raw outputs.

Notebook 07 prepares a limited data-mixture hypothesis and investigates inference-path reproducibility. It does not perform fine-tuning, choose a winning adapter, or run the historical test benchmark.

## Run on Kaggle

Import [Notebook 07](../notebooks/07_inference_check_and_balanced_data.ipynb). Enable Internet and a T4 GPU, attach only the **original saved Notebook 01 output**, then Save Version / Run All. One GPU is used. Earlier model weights are unnecessary. Allow roughly 30–50 minutes, with substantial variation from setup and generation length. There are 200 validation predictions and zero optimizer updates.

The first cell fetches GitHub `main` and records its commit. Each Run All creates a fresh output directory. Individual preparation/diagnostic scripts refuse nonempty destinations so that completed evidence is not overwritten. Do not repeat completed expensive cells without choosing a fresh destination. If GPU work fails after CPU preparation, the prepared data remains on disk; preserve it before restarting Kaggle.

## Data intervention

The variant contains **5,046 original GSM8K rows plus 1,680 synthetic rows**, keeping all 6,726 training examples in total. Every original demonstration is retained unchanged. The retained non-demonstration GSM8K rows are selected by a predeclared deterministic hash rule, without reading their evaluation performance.

The six skills have 280 training rows each: addition, subtraction, equal groups, exact division, multiplication followed by addition, and multiplication followed by subtraction. Every row has a declared arithmetic structure, checked steps, a numeric final label and short worked equations. This first mixture covers nonnegative integer arithmetic, not percentages, averages, conversions, algebra or every observed failure. Its 25% replacement fraction is an initial design choice, not an empirical optimum.

Two training wording templates per skill and one separate diagnostic template are declared in code. The 120 synthetic diagnostic rows (20 per skill) use held-out templates and per-skill numeric tuples. Generated questions are filtered against normalized exact questions from all original splits and the original training set. This does not establish semantic or pretraining decontamination. Held-out templates within the same constructed families are limited development checks, not an untouched broad mathematics benchmark.

The intervention changes data coverage and response style, and removes some GSM8K exposures. Therefore a later improvement cannot be attributed to one of those components separately without further ablations. Equal rows/epochs/effective batch size give equal optimizer-update counts, **not equal supervised-token exposure**. Both per-epoch response-token totals are reported. No examples are silently truncated.

The original validation/test files, resolved configuration, tokenizer and demonstration files remain unchanged. The variant has a new manifest fingerprint and parent identity. The old audit file is retained as historical parent-bundle information; the new mixture's authoritative counts and token audit are in `data/summary.json`. The training loader will compute a fresh training audit when this variant is used. Keep the original and variant distinct; historical comparison scripts reject a different bundle identity by design.

## Inference diagnostic

Load the pinned base model once and generate on the fixed 50 Notebook 05 three-shot validation prompts in four conditions:

1. Direct base.
2. Direct base repeated.
3. Same base with an untrained zero-initialized LoRA adapter.
4. Same wrapped model with its adapter disabled.

Prompt order, batches of four, greedy decoding, 512-token budget, boundary rules and hardware stay fixed. Record raw answers, stop reasons, generated lengths, scores, EOS/padding metadata, source hashes and aggregate comparisons. No model parameters are updated. The zero-adapter initialization is checked before use.

`reproducibility_gate_passed` requires matching completions, raw completions, stop reasons and generated lengths across these paths. A mismatch exposes a path-dependent output but does not identify its numerical root cause. Matching this probe does not prove every historical Trainer or saved-adapter path is identical. Review all four files if the gate fails; do not train another candidate to compensate for an evaluation discrepancy.

## Outputs and next experiment

`data/` includes the new `bundle/`, `synthetic_train.jsonl`, `synthetic_diagnostic.jsonl`, one example per training template in `review_sample.jsonl`, the mixture summary and checksums. `inference_paths/` contains four prediction/metric JSON files, the comparison summary, identity and checksums. Keep the whole ZIP.

After review, the next training experiment should compare fresh original-data and mixed-data adapters using the same inference procedure, `2e-4` reference learning rate, seed, three epochs, effective batch 32, attention-only rank-16 LoRA and loss-based checkpoint selection. Returning to `2e-4` defines a reference configuration; the small earlier score difference does not prove it optimal. The old first adapter is useful historical context, not a substitute for a matched control when making stronger causal claims. Do not add curriculum ordering, LoRA capacity changes, extra epochs or different selection rules in this same experiment.

Compare original validation performance, per-skill synthetic development performance, formatting and regressions. These development results guide iteration; reserve an additional uninspected evaluation for final claims. Notebook 08 training is intentionally deferred until the diagnostic and data sample are reviewed, to avoid spending GPU time on an unresolved comparison.
