# Audit of the first completed SmolLM2 baseline

Source: the executed `notebooks/02_baseline.ipynb` committed to GitHub at `e3f8264`, running evaluator revision `c865dde`. Model revision: `f8027fd0eaeea54caa13c31d31b9fdc459c38b49`.

## Confirmed observations

| Run / mode | Predictions | Strictly invalid | Generation cap reached |
|---|---:|---:|---:|
| Validation zero-shot | 20 | 20 | 20 |
| Validation three-shot | 20 | 19 | 20 |
| GSM8K zero-shot | 1,319 | 1,318 | 1,319 |
| GSM8K three-shot | 1,319 | 1,261 | 1,319 |
| SVAMP zero-shot | 1,000 | 1,000 | 1,000 |
| SVAMP three-shot | 1,000 | 968 | 1,000 |
| Arithmetic zero-shot | 500 | 500 | 500 |
| Arithmetic three-shot | 500 | 483 | 500 |

In the test run, 5,530/5,638 predictions (98.1%) were invalid under the strict format contract. Only 108 were parseable. All 5,638 reached the configured generation cap. The displayed nonzero accuracies correspond to 2 arithmetic, 2 GSM8K, and 7 SVAMP answers. The notebook records about 11 hours 40 minutes for the full-evaluation cell.

The source data counts and completion counts are consistent. Successful process completion and checksum verification do not establish a scientifically adequate evaluation protocol.

## Confirmed faults in the evaluation design

1. **Smoke acceptance was too weak.** A completed process was sufficient to proceed. The observed 40/40 capped and 39/40 invalid validation outputs should have triggered investigation before the full benchmark.
2. **Strict scoring is highly sensitive to output format.** It requires one and only one complete `Final answer: <number>` line. Missing labels, repeated labels, terminal prose/punctuation, and other formats are invalid even when a correct numeric solution might be present. These scores cannot be interpreted as a pure measure of mathematical capability.
3. **Termination is not aligned with the target answer.** Generation stops at configured EOS or 512 new tokens. There is no target-answer-boundary or next-question stop. A base completion model can continue into more text/problems. A correct first target answer followed by another labelled answer is rejected by the current strict parser. This mechanism is confirmed in a regression test; its incidence in the actual outputs is not yet known.
4. **Format learning can confound the planned training comparison.** Training explicitly teaches the required format. A large increase in strict accuracy could therefore include increased format compliance, not only mathematical improvement. The later report must distinguish these.

## What the cap proves, and what it does not

`finish_reason=length` means that no configured EOS occurred within the generated budget. It does not establish that the target solution still needed more reasoning tokens. Outputs may instead have finished and continued, repeated themselves, or ignored the requested format.

The context-fit audit is separate: the longest measured input was 573 tokens, so input plus 512 generated tokens fits the 2,048-token configured context. That arithmetic remains correct. It does not prove that 512 generated tokens is a sufficient stopping budget for this model/prompt combination.

The current evidence cannot establish whether a larger generation budget would improve correctness. It could simply increase runtime and produce more irrelevant continuations. Do not launch another full run just by raising the cap.

## Evidence still needed

The GitHub notebook contains logs and aggregate HTML but does not contain `predictions.jsonl`. Inspect the saved baseline ZIP, especially `smoke/predictions.jsonl` and `full/predictions.jsonl`. Check original questions, early labelled answers, repeated question/answer blocks, endings, and EOS/generation configuration. Frozen model metadata could not be independently fetched from this cloud instance because its Hugging Face destination is blocked.

Use validation completions for revised prompt/stopping decisions. Any reinterpretation of test answers after inspecting this run is exploratory and must be labelled as such; do not overwrite the original metric or silently report a more permissive score as the original benchmark.

## Safeguards and CPU-only diagnostics

The full CLI now blocks on smoke groups with at least 50% invalid outputs or at least 80% capped outputs. These are engineering alarm thresholds, not a statistical certificate, and low numeric accuracy alone does not fail the check.

The existing generation and scoring rules were not changed. Completed data and results remain untouched. Audit an attached result directory without GPU inference:

```bash
python scripts/audit_baseline.py --run /path/to/baseline/smoke \
  --output /path/to/new-smoke-audit
```

Run the same command against `full` for post-hoc test diagnostics. The tool checks original result hashes, recomputes strict scores, counts missing/multiple/unparseable labels and next-question continuations, and saves stable completion samples. First-problem first-label counts are explicitly exploratory diagnostics, not replacement primary accuracy.

The next experiment should be a small validation-only diagnostic with verified answer extraction and termination. A versioned revised protocol can then be frozen and applied consistently to baseline and tuned models. The full benchmark should wait until that protocol passes a meaningful smoke review.

## Follow-up: supplied raw predictions

The user subsequently supplied all 40 validation smoke records and a pasted excerpt of the full predictions: 313 complete GSM8K zero-shot records, followed by a truncated record. This excerpt does not cover the other full-run modes/datasets. No original result manifests were supplied with these pasted files.

| Supplied group | Records | Capped at 512 | Strictly invalid | No final label | Multiple final labels | Next-question marker |
|---|---:|---:|---:|---:|---:|---:|
| Validation zero-shot | 20 | 20 | 20 | 18 | 2 | 11 |
| Validation three-shot | 20 | 20 | 19 | 5 | 14 | 15 |
| GSM8K zero-shot excerpt | 313 | 313 | 313 | 292 | 18 | 177 |

The next-question counts detect newline-prefixed `Question:` markers. Some outputs introduce new questions inline, so these counts are not exhaustive. All 353 complete supplied records reproduce their stored strict scores. All reference answers match the pinned GSM8K source by ID, and all 333 zero-shot prompt hashes match reconstructed prompts. Three-shot hashes were not reconstructed without the saved demonstrations/tokenizer bundle.

The anticipated scoring/termination failure is now observed in real validation data: item `gsm8k/train/07224` in three-shot mode computes `200 + 250 + 100 = 550`, writes `Final answer: 550`, then continues with repeated unrelated bank-account problems and more final-answer lines. The original strict parser marks the record invalid. The correct target answer was already present before the cap.

Real mathematical errors are also visible before continuation. A $30 manicure plus a 30% tip is answered as $9 rather than the $39 total. A question involving three weekly 200 kg collections followed by a week at half that amount is answered as 300 rather than 900 kg. Other outputs enumerate empty steps or repeat facts/calculations instead of progressing toward a solution.

These observations do not support raising the token budget as the primary remedy. They support a small validation-only protocol revision addressing answer boundaries, strict-format confounding, and repetition. They do not prove that no individual question could benefit from more tokens. The one initially correct labelled validation answer is an exploratory diagnostic finding, not an automatically revised benchmark accuracy.
