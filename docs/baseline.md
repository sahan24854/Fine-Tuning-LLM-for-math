# Notebook 02: baseline evaluation

The baseline measures the untouched `HuggingFaceTB/SmolLM2-360M` model using the exact model revision, tokenizer, data, and demonstration examples saved by Notebook 01. It does not train an adapter. The original bundle remains unchanged.

## Kaggle steps

1. Save a completed version of Notebook 01 **with its outputs**.
2. Create a new Kaggle notebook and import this GitHub URL:

   ```text
   https://github.com/sahan24854/Fine-Tuning-LLM-for-math/blob/main/notebooks/02_baseline.ipynb
   ```

3. Use **Add Input** to attach the saved Notebook 01 output. Attach one data bundle. If you downloaded its ZIP, upload it as a Kaggle dataset and attach that instead.
4. Enable **Internet** and choose a **GPU** accelerator. The evaluator uses the first GPU, even if two are allocated.
5. Run through the smoke-test cell to inspect the setup and approximate runtime. **Run All also runs the full benchmark**; there is no separate confirmation prompt.
6. Save the completed notebook version with outputs, and retain the resulting baseline ZIP alongside the original data bundle.

No source-code copying is necessary. Like Notebook 01, the launcher fetches shared code from GitHub. Creating/attaching Kaggle notebooks happens in your signed-in browser.

Bundle discovery follows symlinked input folders (including links to mounts outside `/kaggle/input`) and prevents traversal loops. Renamed ZIP files are recognized by their manifest. If discovery fails, the error lists visible input paths and rejected manifests. After a shared-code fix, rerun the first code cell to fetch the latest commit, then rerun the input-discovery cell; notebook re-import is not needed.

The JSONL reader uses physical newline boundaries. GSM8K includes a question with a literal Unicode line separator (`U+2028`) inside a valid JSON string; using `str.splitlines()` incorrectly splits that record. If an older code revision reports `JSONDecodeError: Unterminated string` after bundle hash verification, fetch the latest code and rerun discovery and the smoke test. Keep the original data bundle. In the same live session, the installed environment from the setup cell can be reused; after restarting, run setup again.

## What runs

The notebook verifies the data bundle before installing an isolated runtime. The setup uses PyTorch 2.7.1 from the official CUDA 11.8 index and the pinned tokenizer stack from Notebook 01. It runs a CUDA matrix multiplication before downloading model weights. The model then runs in FP16 with eager attention, greedy decoding, and deterministic PyTorch algorithms. TF32 is disabled. These choices are recorded for the later tuned-model comparison.

The smoke test evaluates 20 validation examples in each prompt mode (40 predictions). It is a functional and throughput check; it is not a benchmark score and does not need to achieve a particular accuracy to pass. An unscored validation warmup precedes measurement.

The first completed run exposed a protocol problem: all outputs hit the generation cap and almost all were rejected by the strict format parser. Read [the audit](baseline-audit-2026-10-20.md) before interpreting its accuracy. The full CLI now blocks when a smoke group has at least 50% invalid or at least 80% length-limited answers. These engineering alarms require investigation of validation completions; increasing the cap alone is not a justified fix. Existing scores are preserved.

The full run evaluates:

| Benchmark | Questions | Modes | Predictions |
|---|---:|---:|---:|
| GSM8K test | 1,319 | 2 | 2,638 |
| SVAMP | 1,000 | 2 | 2,000 |
| Arithmetic | 500 | 2 | 1,000 |
| Total | 2,819 | 2 | 5,638 |

Zero-shot and three-shot use the same questions. Three-shot demonstrations come from training only. The generation budget remains 512 new tokens. Reference answers never enter the generation call. Padding is on the left, with an attention mask, and only generated tokens are decoded and scored.

## Reading the results

- `predictions.jsonl`: every saved completion, extracted numeric answer, reference answer, correctness, finish reason, token counts, and amortized generation time.
- `summary.json`: counts, accuracy, Wilson 95% intervals, invalid-answer and token-limit counts, and arithmetic breakdowns.
- `metrics.csv`: a compact table for spreadsheets.
- `report.html`: an offline report with accuracy bars; also displayed inside Kaggle.
- `identity.json`: frozen run identity, including bundle hash, model revision, code hashes, package fingerprint, GPU/runtime details, and generation settings.
- `environment.txt`: full resolved package versions.
- `result_manifest.json`: completion status, file hashes, and peak allocated CUDA memory for the completing process. Earlier interrupted processes are not included in that memory statistic.

Primary accuracy requires exactly one parseable `Final answer:` line. Formatting failures are incorrect and also counted separately. A token-limit hit remains in the denominator; it can still be correct if a valid final answer was produced. This measures final-answer correctness under the specified output contract, not the validity of every reasoning step.

The baseline's Wilson intervals are not the paired bootstrap confidence interval for improvement. That comparison requires aligned baseline and fine-tuned predictions and will be implemented later. Do not merge the three benchmarks into one headline accuracy.

Generation timing is synchronized with the GPU and excludes model loading, warmup, tokenization, and scoring. Batch time is divided across examples, so it represents throughput rather than single-request latency. The smoke estimate can differ from actual full-run time.

## Resume behavior

Each batch is flushed to disk. Rerunning the evaluation cell reuses validated predictions. A partially written final line is discarded and regenerated; malformed complete records cause an error. If a batch was partly saved, the original batch is regenerated and only missing predictions are appended, preserving its padding and composition.

The evaluator refuses to combine runs with changed data, code, package versions, hardware, batch size, or settings. Completed runs preserve their original timing/memory reports on rerun. Run directories use a data fingerprint, code commit, and batch size.

Within a live Kaggle session, rerun the interrupted cell. After a session ends, you must retain/download its partial output directory and attach that output to the next notebook session. The launcher copies a matching run directory from `/kaggle/input` into `/kaggle/working`. Use the same code revision (`MATH_PROJECT_REF`) and GPU type. If packages changed, recreate the original environment from its recorded versions or start a fresh run. Unsaved `/kaggle/working` contents are not guaranteed to survive a session ending.

Optional environment settings are `MATH_PROJECT_REF` (code revision), `MATH_BUNDLE_PATH` (specific data directory or ZIP), and `MATH_BATCH_SIZE` (default 4). Changing batch size starts a different run and requires a new smoke test. Normal first runs need none of these settings.

## Local validation and limits

```bash
python scripts/setup_environment.py --environment /tmp/math-baseline-test \
  --profile baseline --torch-variant cpu
/tmp/math-baseline-test/bin/python -m unittest discover -s tests -v
```

The tests use a tiny randomly initialized Llama model to exercise actual generation, padding/EOS handling, the smoke/full CLI flow, scoring, checksums, and interrupted-run recovery without network model downloads. They establish pipeline behavior, not SmolLM2 accuracy. GPU execution and the real model benchmark must be validated in Kaggle.

The data-only tests can run without PyTorch; inference tests are explicitly skipped in that case. Keep validation results and completed benchmark results distinct.
