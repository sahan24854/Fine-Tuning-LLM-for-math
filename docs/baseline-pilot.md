# Notebook 02c: three-shot baseline pilot

Import [Notebook 02c](../notebooks/02c_baseline_pilot.ipynb) into a **new Kaggle notebook**:

```text
https://github.com/sahan24854/Fine-Tuning-LLM-for-math/blob/main/notebooks/02c_baseline_pilot.ipynb
```

Enable Internet, select a GPU (one T4 is sufficient), and use Add Input to attach **the original Notebook 01 saved output**. Do not regenerate the data bundle. You do not need to attach Notebook 02 or 02b. Run All, then Save Version with outputs. The launcher fetches shared code from GitHub and prints its exact commit.

## What runs

1. Verify the data bundle and freeze the question selection, before loading weights.
2. Install the isolated inference environment and check CUDA execution.
3. Run CPU contract tests for the reviewed v2 evaluator and pilot.
4. Evaluate the untouched SmolLM2-360M revision on 300 questions using three worked training examples in every prompt.
5. Display a report and create a ZIP under `/kaggle/working`.

This notebook evaluates the base model; the fine-tuning notebook is a later milestone. The second T4, if present, is not used automatically.

## Frozen selection and measurement

The configuration is [configs/pilot_v2.json](../configs/pilot_v2.json). The reviewed [v2 diagnostic protocol](../configs/protocol_v2_diagnostic.json) is pinned by file checksum. Its validation-only runner remains unchanged; the separate pilot runner supplies the test-subset scope.

| Dataset | Questions | Selection |
|---|---:|---|
| GSM8K test | 100 | Lowest SHA-256 ranks of seed, split, and ID |
| SVAMP | 100 | Same ranking, independent of answers or model results |
| Arithmetic | 100 | Ten per small/large number range × five operations, ranked within each group |

Seed 42 is fixed. Ranking does not depend on source-file ordering or model correctness. The arithmetic operations are addition, subtraction, multiplication, exact division, and two-step word problems. Hash ranking is deterministic sampling; it does not guarantee that 100 questions represent every form of difficulty in a dataset.

`selection.json` records all selected IDs, prompt/question/reference hashes, base model revision, original bundle fingerprint, demonstrations hash, scoring configuration, and a comparison-contract fingerprint. It is written before downloading model weights. A later adapter evaluation must reuse this selection and check the comparison contract, while recording its adapter artifact separately.

Generation uses the existing greedy FP16 eager runtime, batch size four, and 512-token safety cap. It stops per row at the first completed final-answer line, before a new question, at EOS, or at the cap. Numeric normalization is unchanged from the reviewed v2 diagnostic. No reference answer is passed to the model. An incomplete final numeric prefix at the cap is not credited.

## Results and interpretation

The report shows each dataset separately: accuracy over all questions, descriptive 95% Wilson intervals, normalized parseability, strict numeric format rate, cap-hit count, stop reasons, generated tokens, and raw predictions. Arithmetic breakdowns are also included in `summary.json`. Invalid outputs remain incorrect in the accuracy denominator. Numeric final-answer correctness does not establish correct intermediate reasoning or unit handling.

The [review of Notebook 02b](protocol-v2-review-2026-10-20.md) found that answer-boundary stopping preserves a previously lost correct answer. It also found genuine wrong calculations and repetition. Its ten-question three-shot accuracy is not an estimate for this pilot. Do not increase the token budget or change scoring in response to pilot scores; that would require a new comparison protocol and fresh base-model results.

The original full test results have already been inspected, so label this run a **development comparison**, not an untouched final holdout. Training and checkpoint selection should use training/validation data. Reserve an additional untouched evaluation for stronger final claims. Public benchmark data may also have appeared in model pretraining; no claim of pretraining decontamination is made.

Timing includes per-token stopping checks and first-call initialization, with no warmup. It is descriptive generation time, not per-request or end-to-end latency. The notebook does not promise a completion time based on the small diagnostic.

## Saved output and interruption

Keep the complete ZIP, including:

- `selection.json`, `identity.json`, `source_data_manifest.json`, and `environment.txt`;
- `predictions.jsonl`, `summary.json`, `metrics.csv`, and `report.html`;
- `result_manifest.json`, which checksums the completed files.

Every batch is flushed to disk. Rerun the generation cell in the same session to resume. An interrupted final JSONL fragment is discarded, complete records are validated, and an incomplete batch is regenerated with its original composition without appending duplicates. Changed code/package/runtime/bundle/protocol identities are refused; use a new output directory for a different experiment. A completed result is checksum-verified on resume.

Local commands:

```bash
python scripts/run_pilot.py --bundle /path/to/original-bundle \
  --output artifacts/pilot-v2 --plan-only
.venv-baseline/bin/python scripts/run_pilot.py --bundle /path/to/original-bundle \
  --output artifacts/pilot-v2
```

The plan-only command needs no ML packages or model weights. CPU execution (`--device cpu`) is provided for development tests; the real baseline should use the Kaggle GPU environment.

After completion, inspect/share `summary.json` and `predictions.jsonl`. The next implementation milestone is LoRA training with validation-based checkpoint selection, followed by paired evaluation using the saved pilot contract. The old 5,638-prediction protocol-v1 run remains preserved for the audit trail.
