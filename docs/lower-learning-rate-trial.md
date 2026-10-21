# Notebook 06: controlled learning-rate trial with epoch monitoring

## Question

Does a fresh adapter trained at `5e-5` preserve more correct behaviour than the first `2e-4` adapter? This is a hypothesis. The earlier run established better formatting but mixed maths performance; it did not establish excessive learning rate as the cause.

## Run on Kaggle

Import [Notebook 06](../notebooks/06_lower_lr_with_epoch_validation.ipynb), enable Internet and a T4, and attach the complete saved outputs of Notebook 01 and Notebook 05. Folder and ZIP discovery are supported. If multiple different artifacts are attached, set `MATH_BUNDLE_PATH` or `MATH_VALIDATION_PATH` to the intended input in the launcher. Keep the original bundle.

Run All / Save Version with outputs. The launcher fetches shared GitHub code and records the commit. No Notebook 03 weights are needed: training starts from the original pinned base; the first adapter is compared using verified Notebook 05 predictions. Do not continue from the first or smoke adapter.

Allow roughly 60–90 minutes on one T4, not a guarantee. Earlier training took about 48 minutes; this run adds five 50-question generation observations and setup. Only 250 full-run validation predictions and six smoke predictions are generated. No test inference is requested.

## Fixed factors and changed factors

- **Optimization change:** learning rate `2e-4 → 5e-5`, new training identity.
- **Instrumentation:** generate three-shot answers for the same 50 Notebook 05 validation IDs before training, at each epoch end, and after loading the loss-selected checkpoint. Raw completions, scorer results, prompt hashes, paired changes, and aggregate metrics are saved.
- **Unchanged:** original 6,726 training/747 validation rows, seed 42, three epochs, micro-batch four/accumulation eight, rank 16/alpha 32/dropout .05, attention targets, weight decay, linear schedule and warmup ratio, sequence ceiling, response-only objective, and lowest validation token-NLL checkpoint selection.
- **Unchanged evaluator:** same three demonstrations, greedy generation, 512-token cap, boundary stopping and numeric scorer. The input verifier checks old evaluator source hashes and reviewed first-adapter identity.

Monitoring restores Python, NumPy, CPU/CUDA RNG state, model mode, cache configuration, and tokenizer padding after generation, including errors. A real tiny-model test compares adapter weights with and without monitoring across epochs. The old adapter's epoch-level generated answers do not exist; this trial cannot compare its epoch trajectory with the old trajectory. Only selected-adapter comparisons are paired against Notebook 05.

Epoch metrics never affect training, early stopping, or checkpoint selection. The final `selected` observation is deliberately generated again even when the last epoch won. Monitoring timing includes initialization/batching and is not a latency benchmark. Resume is supported within the same unchanged environment from completed checkpoints; uncheckpointed observations may be recomputed.

## Outputs and review

`full/generated_validation/` contains `identity.json`, `base.json`, `epoch-step-*.json`, and `selected.json`, including all predictions. `full/summary.json` includes their metrics. `full/adapter` is the selected adapter. `comparison/summary.json`, `paired_predictions.json`, and `report.html` compare original, first-adapter, and candidate answers. Complete result manifests protect files; retain the whole ZIP.

Review correctness, parseability, gains/regressions, repetition, and cap hits together. The repeated original-model completions should match Notebook 05; nonzero `base_completion_differences` calls for investigation. Runtime checks are enforced, but this is not a multi-seed causal study. One question changes accuracy by two percentage points; a tiny gain is inconclusive. These validation questions already informed development and loss selection, so they are not an untouched final holdout.

Do not automatically run the old test pilot to pick settings. Preserve earlier official scores. If learning-rate changes do not address the setup/calculation weaknesses, the next separate experiment should add checked, balanced training examples with held-out wording templates. Dataset changes, LoRA capacity changes, and generated-accuracy checkpoint selection remain separate future experiments.

## Recover the original Notebook 06 final-cell metadata error

The initial comparison mistakenly compared the full Notebook 05 inference identity
against Notebook 06's smaller training-runtime identity. `model_eos_ids` (and other
inference-only fields) was absent, not proven different. The corrected comparison
requires identical core runtime field sets and values. It explicitly reports that
legacy training artifacts do not independently establish EOS/padding/adapter-runtime
equality; model revision, original tokenizer bundle, frozen sources, scoring,
prompts, and saved prediction checks remain enforced.

In the still-running Kaggle session, fetch `main` and replace only
`scripts/compare_training_trial.py` from `FETCH_HEAD`, then rerun the final cell.
Do not rerun training or replace its source files: the saved training identity
pins their hashes. This repair changes report generation only and preserves the
existing adapter, predictions, and manifests. If the session has ended, retain or
attach its saved output before starting recovery; the old `/kaggle/working` path
may no longer exist. The comparison requires no GPU or new model generation.
