# Notebook 05: validation error analysis

The first run improved formatting but did not consistently improve generated word-problem accuracy. Notebook 05 collects development evidence before choosing a second training experiment.

Import [Notebook 05](../notebooks/05_validation_error_analysis.ipynb) into a new Kaggle notebook:

```text
https://github.com/sahan24854/Fine-Tuning-LLM-for-math/blob/main/notebooks/05_validation_error_analysis.ipynb
```

Enable Internet and select a T4. Attach **only the original Notebook 01 saved output and the completed Notebook 03 saved output**. The original full adapter is discovered automatically, including ZIP/symlinked folders. Notebook 02c/04 outputs are not required. Run All and save the version with outputs.

## Scope

Select 50 original validation IDs by SHA-256 ranking of seed and ID. Selection is independent of correctness and source-file order. Use the same questions for all four conditions:

| Model | Prompt | Predictions |
|---|---|---:|
| Original base | Zero-shot | 50 |
| Original base | Frozen three-shot | 50 |
| Selected first adapter | Zero-shot | 50 |
| Selected first adapter | Frozen three-shot | 50 |

There are **200 new predictions**, not a full benchmark. Each model runs in its own process, releasing GPU memory between stages. Greedy decoding, batch four, the reviewed 512-token cap, answer-boundary stopping, and official scoring remain fixed. No training runs.

These examples were excluded from gradient training, but the full validation split was used for teacher-forced checkpoint selection. This is development diagnosis, not an untouched holdout or independent final test. Notebook 05 does not evaluate the pilot/test sets or choose another checkpoint using test outcomes. The original 300-question comparison remains preserved.

The experiment can reveal whether the first adapter behaves differently with the zero-shot format used in training versus the three-shot evaluation format. It cannot establish a universal best prompting method from fifty development examples.

## Verification and saved evidence

Verify the original data bundle and full training result, base revision, training/validation IDs, selected adapter weights, and checksums before generation. The original base/adapter runtime and new package fingerprints must match within this diagnostic. Changes to code, inputs, configuration, or runtime are refused during resume.

Configuration is [configs/validation_analysis.json](../configs/validation_analysis.json). The diagnostic is capped at 50 questions and two declared prompt modes; it cannot accidentally run all 747 validation questions. The reviewed scoring configuration is pinned by checksum.

The report contains:

- Accuracy, parseability, cap hits, repetition, generated tokens, and descriptive generation time for each model/mode.
- Paired base/adapter improvements and regressions within each prompt mode.
- Question, reference worked solution, original answer, and adapter answer together.
- Raw generated text, scored target, extracted answer, and stop reason.
- Literal numerical-equality review flags and a blank human-review worksheet.

Reference validation solutions are displayed only for review. They are never passed to the generator. Three-shot prompts use the original training demonstrations.

## What the automatic checker does

Only entire lines consisting of a binary equality between literal signed integers/decimals are checked, for example:

```text
52 + 18 = 60        -> numeric equality mismatch
165 - 65 = 100      -> exact equality
0.5 * 4 = 2        -> exact equality
9 / 0 = 0          -> undefined division
```

The implementation uses exact rational arithmetic and does not execute generated code. It accepts the declared arithmetic symbols but does not handle narrative calculations, variables, chained expressions, general fractions, scientific notation, or units. Absence of a flag does not establish correct reasoning. A flagged equality may involve rounding, quotation of an incorrect calculation, or an actual error; human review determines its meaning.

These flags do not change official answer extraction or scores. A correct final number can coexist with a flawed solution. The script does not automatically label semantic reasoning errors or claim a causal diagnosis.

## Reviewing the errors

Annotate a **copy** of `review_template.csv`, preserving the generated template and checksum-protected outputs. Useful categories include:

- Wrong operation or misread quantity.
- Arithmetic error.
- Output-format failure.
- Repetition or lack of progress.
- Correct final answer with a reasoning concern.
- Ambiguous or other.

Inspect both improved and regressed answers. Prefer representative failures over a few striking examples. Compare zero-shot and three-shot behavior on the same questions before attributing a change to prompt formatting.

Possible next experiments depend on this review. Verified calculation errors may motivate additional checked arithmetic training examples; wrong-operation errors may motivate more diverse elementary word problems; prompting differences may motivate a declared prompt-format study. Regressions may motivate a smaller learning rate or a more conservative adaptation experiment. Change one main factor at a time where practical, start from the declared initialization, and use validation evidence for selection. None of those changes is automatically implemented by this notebook.

Keep the current test scores fixed. Additional training/data/scoring experiments require new identities and honest development labels, with additional untouched evaluation reserved for stronger final claims.

## Outputs and resume

The output folder includes:

- `base/` and `tuned/`: each model's 100 predictions, environment, identity, and completion manifest.
- `analysis_identity.json` and `selection.json`: fixed inputs, sample, prompts, and code fingerprints.
- `paired_predictions.jsonl`: 100 question/mode records, each containing both model answers and the reference solution; this represents 200 predictions total.
- `summary.json`, `metrics.csv`, `report.html`, and `review_template.csv`.
- Training-summary snapshot and final checksum manifest.

Predictions flush every batch. Rerun an interrupted model stage in the same session to resume; an interrupted batch is regenerated with its original composition without appending duplicates. Incomplete JSONL tail fragments are removed only from this new writable output. Original attached artifacts remain read-only. A completed result is checksum-verified on rerun. Do not edit the generated worksheet in place; use a separate annotation copy.

Share `summary.json` and `paired_predictions.jsonl`, or push the executed notebook, for review. Then define a small controlled second training experiment. The report is an intermediate diagnostic, not a claim that the original project goal has been achieved.

## Local checks

```bash
python scripts/analyze_validation.py --bundle /path/to/original-data \
  --training /path/to/training/full --output artifacts/validation --stage verify
.venv-training/bin/python scripts/analyze_validation.py --bundle /path/to/original-data \
  --training /path/to/training/full --output artifacts/validation --stage base
# Repeat with --stage tuned, then --stage report.
```

The verification stage needs no model weights or ML imports. CPU mode is available for tiny local fixtures. The tests exercise real tiny-model base/adapter inference in both modes, validation-only selection, literal flags, paired-key alignment, resume, identity checks, and report integrity. The actual SmolLM2/T4 results must be produced in Kaggle.
