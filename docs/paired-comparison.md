# Notebook 04: original model versus selected LoRA adapter

Import [Notebook 04](../notebooks/04_paired_comparison.ipynb) into a new Kaggle notebook:

```text
https://github.com/sahan24854/Fine-Tuning-LLM-for-math/blob/main/notebooks/04_paired_comparison.ipynb
```

Enable Internet and select a T4 GPU. Attach all three **saved outputs** using Add Input:

| Input | Required content |
|---|---|
| Original Notebook 01 | Data manifest, splits, demonstrations, tokenizer, configuration |
| Completed Notebook 02c | Pilot selection, predictions, identity, summary, result manifest |
| Completed Notebook 03 | Full-run adapter and its identity, checkpoints, summary, result manifest |

Attach saved files/folders or their output ZIPs, not only `.ipynb` files. Notebook 02b is a diagnostic and cannot replace 02c. The training artifact must include the `full/` result; its sibling smoke result is ignored.

Choose Run All, then Save Version with outputs. This notebook generates **300 new adapter predictions**, reuses all saved baseline answers, and performs no training. The original executed notebooks and scoring logic remain preserved.

## What is verified before loading weights

The standard-library verification cell checks:

- Complete, checksum-verified baseline and full-training artifacts, with no writes to original inputs.
- The original bundle/model revision and saved demonstrations/context window.
- Exact equality of the saved pilot selection with the original deterministic plan, including all question/prompt/reference hashes. Recomputing the plan verifies the saved selection; it does not replace it.
- All original baseline prediction scores and group metrics.
- Byte-identical frozen evaluator source files against the baseline identity: prompts, tokenization helpers, generation, stopping, scoring, and selection.
- Full training on the original training/validation IDs, with the declared validation-loss checkpoint selection.
- Adapter base revision, LoRA settings, manifested files, and exported weights matching the selected checkpoint's weights.

Discovery follows Kaggle symlinked input folders and supports standalone ZIPs. If one folder and its ZIP are both attached, the folder is preferred. Different candidate runs require an explicit choice rather than silently choosing the newest one. Remove extra inputs or set `MATH_BUNDLE_PATH`, `MATH_BASELINE_PATH`, and/or `MATH_TRAINING_PATH` to the intended folder or ZIP before the discovery cell. For training, a containing saved-output folder is accepted; discovery locates its full-run result.

ZIP extraction rejects escaping paths/symlinks and limits uncompressed size to 2 GB and file count to 20,000. Normal adapter-only training outputs fit within that limit.

## Inference and fairness

The isolated environment uses the existing pinned PyTorch 2.7.1 CUDA 11.8 / Transformers 4.57.6 stack plus PEFT 0.17.1. The full new package fingerprint is recorded. The script requires the baseline's recorded core runtime settings to match: GPU/framework/precision/attention/determinism, EOS/padding, and stopping configuration. It does not require every installed package fingerprint to equal the older environment, because adapter inference adds packages. These checks support consistent settings; they do not establish a controlled performance benchmark or bitwise portability across every environment change.

The original bundle tokenizer and pinned base revision are loaded, then the full adapter is applied, frozen and unmerged, with FP32 adapter parameters. Base generation settings are checked after adapter loading. Greedy decoding, batch four, 512 new-token cap, original job order, frozen three-shot prompts, and per-row answer-boundary stopping remain fixed. No new test answers or reference solutions are included in prompts.

If the script reports a runtime mismatch, inspect the named difference and use the original T4/core package settings. Do not edit saved identities to bypass the check. A genuinely different runtime requires a separately documented comparison experiment.

## Reading the comparison

Each dataset is reported separately. The report includes:

- Original and tuned official numeric accuracy, with invalid answers included as incorrect.
- Exact counts of improved answers, regressions, both correct, and both incorrect.
- Absolute accuracy change in percentage points, plus a paired bootstrap 95% interval.
- Parseability and strict numeric-format rates, cap hits, token totals, and descriptive generation time in the saved summary.
- Expandable original/tuned raw answers, references, extracted values, stopping reasons, and scored target text.

The bootstrap resamples matched question pairs 10,000 times with seed 42. Arithmetic resamples within each original number-range/operation stratum. The reported interval is the 2.5th–97.5th percentile of accuracy differences. In `summary.json`, accuracy and interval values are proportions; CSV/HTML change and interval columns explicitly use percentage points. Marginal Wilson intervals are also retained in each model's group metrics. These are descriptive uncertainty estimates and do not correct for development reuse, unknown pretraining contamination, or multiple comparisons.

While interrupted/incomplete, metrics compare only questions with both saved baseline and tuned answers. Final bootstrap statistics are deferred until all adapter predictions are present. The report never compares a full 100-question original score with a partial tuned subset.

## Supplementary arithmetic audit

The official scorer still requires the reviewed labelled answer. A separate, versioned exploratory rule recognizes a complete target containing only:

```text
a operator b = signed_integer
```

Its operands/operator must match the source question `What is a operator b?` exactly. The rule handles the original positive-integer questions with ASCII `+`, `-`, `*`, and `/`. It rejects generation-cap targets and targets with extra prose, extra equations, or labels. It compares the eligible right side with the verified reference; it does not search arbitrary text for the reference or pick an arbitrary last number.

The answer-evidence count is the union of official correct answers and these additional correct equations. The exact same rule is applied to both models, with paired evidence improvements and regressions reported separately. On the actual reviewed baseline it reproduces **4 official arithmetic answers plus 47 additional correct equations**, giving 51 outputs with clear correct-answer evidence.

This audit was designed after inspecting baseline outputs. It is exploratory, is not a replacement official score, and does not exhaust all possible correct answer forms or establish pure mathematical reasoning. Unlisted units such as `6 more kids` remain subject to the frozen official parser. Improved official accuracy alongside improved parseability can partly reflect learning the required format; inspect paired outputs before attributing gains to maths.

## Saved files and resume

Keep the complete comparison ZIP, including:

- `predictions.jsonl`: tuned predictions in the familiar record format.
- `paired_predictions.jsonl`: original and tuned record together for each key, plus arithmetic audit details where relevant.
- `summary.json`, `metrics.csv`, and `report.html`.
- `selection.json`, comparison configuration, environment, and identity.
- Baseline predictions and source-manifest/training-summary snapshots.
- `result_manifest.json`: checksums for completed comparison files.

The result manifest verifies the comparison's snapshots; the source training/baseline manifests inside it describe the separately retained original artifacts. The comparison ZIP does not include the adapter weights, so keep the original training ZIP too.

Predictions flush after every batch. Rerun the generation cell in the same session to resume. An interrupted final JSONL fragment is removed only from this new writable output; saved inputs are never repaired or truncated. An interrupted batch is regenerated with the original composition and completed records are not appended twice. Changed identities are refused, and completed results are verified rather than regenerated.

After completion, share `summary.json` and `paired_predictions.jsonl` or push the executed notebook to GitHub. The next step is reviewing the actual gains and failures and writing the project findings. Do not select another checkpoint or change scoring based on these test results. An additional untouched evaluation is needed for stronger final claims.

## Local commands

```bash
python scripts/compare_pilot.py --bundle /path/to/data \
  --baseline /path/to/pilot --training /path/to/training/full \
  --output artifacts/comparison --verify-only
.venv-training/bin/python scripts/compare_pilot.py --bundle /path/to/data \
  --baseline /path/to/pilot --training /path/to/training/full \
  --output artifacts/comparison
```

Verification requires no ML packages or model weights and does not create or modify the output directory. CPU mode is available for local fixtures, but a real CUDA baseline requires the matching CUDA runtime. Tiny-model tests exercise the adapter-loading CLI, artifact rejection, paired statistics, discovery, audit, and interrupted-run recovery. Actual SmolLM2 adapter accuracy remains to be measured in Kaggle.
