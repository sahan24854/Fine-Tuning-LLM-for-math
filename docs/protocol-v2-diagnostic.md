# Protocol v2 validation diagnostic

Import [`notebooks/02b_protocol_v2_diagnostic.ipynb`](../notebooks/02b_protocol_v2_diagnostic.ipynb) into a **new Kaggle notebook**. Enable Internet, select a GPU, and attach Notebook 01's original saved output through Add Input. Run All, save the version with outputs, and inspect/share the diagnostic report and predictions. The old full baseline is not rerun.

The default is **10 original validation questions in both prompting modes: 20 scored predictions**. The runner has no full/test-benchmark mode and refuses configurations over 20 validation questions. There is no training step. The selected questions are the first ten IDs in the already-frozen validation split order; no correctness-based selection is performed.

## Controlled changes

The base model/revision, tokenizer, prompts, frozen three-shot demonstrations, FP16 eager runtime, greedy decoding, batch size 4, and 512-token safety cap are retained. Stopping and answer extraction change. A score change here cannot be attributed to learned maths: no model weights were updated.

Protocol configuration lives in `configs/protocol_v2_diagnostic.json`. It is included in the run identity and output. Future changes require a new protocol/run identity; do not overwrite the completed v1 baseline or its data bundle.

## Stopping rules

Stopping is independent per batch row and applies only to **new generated tokens**, never prompt demonstrations:

1. Stop after the first `Final answer:` line is terminated by a newline. This applies even if the value is malformed, which is reported instead of allowing the model to try later unrelated answers.
2. Stop before a whitespace-prefixed `Question:` or `### Question:` header, including an inline header. This prevents an invented later question from supplying the target answer. A literal header mentioned in prose could trigger this rule, so raw-output review remains necessary.
3. Stop at the pinned model's EOS. An answer line ending at EOS can be parsed without a newline.
4. Stop at the original 512-token cap. If the final label has no newline, EOS, or next-question boundary, it is classified incomplete; the next digit may have been cut off.

The criterion waits for an actual boundary. It must not stop at `5` while a model is still generating `550`. Tests exercise this case using real Hugging Face batched generation and verify that different rows stop at different steps.

A tokenizer token can include text beyond the detected character boundary. The target completion is cut to the boundary for scoring, and the full decoded text through that stopping token is retained as `raw_completion`. Generated token counts include that token. Padding added while other rows finish is excluded from the stopped row's text/count.

## Extraction and metrics

Only the first labelled value inside the target completion is eligible. There is no search for the reference, arbitrary last-number extraction, or recovery from a later invented problem.

The versioned normalization accepts signed integers, decimals, exact fractions, correctly grouped commas, the listed currency prefixes/unit words, paired lightweight wrappers such as `\\boxed{39}`, and a trailing period. It does not evaluate expressions, accept alternative answers, round, apply numeric tolerance, or convert/validate units. Multiple numeric values in a label remain unparseable.

Each prompting mode reports:

- **Strict numeric format rate:** fraction whose isolated final label is directly numeric under the existing canonical-number parser.
- **Normalized parse rate:** fraction with a numeric value after the declared formatting normalization.
- **Numeric accuracy over all questions:** exact rational equality with reference; unparseable answers remain incorrect in the denominator.
- **Accuracy among parseable outputs:** an additional, explicitly selection-biased diagnostic; it must not replace the all-question metric.
- Stop-reason counts, token counts, repetition flags, and amortized generation time.

For example, `$9` normalizes to `9`, but remains wrong when the reference is `39`. Formatting improvements are not treated as mathematical improvements. The v1 strict extraction on the same target text is retained for debugging; it is not the original v1 benchmark score because stopping changed.

Engineering alarms flag groups with at least 50% unparseable targets or at least 50% generation-cap stops. Low numeric accuracy alone is not a runtime failure. **Every completed diagnostic still requires human review**, even without alarms. The result manifest is labelled `diagnostic_complete_review_required`; there is no automatic route from completion to a full benchmark.

## Reviewing the output

Open `report.html` or the displayed Kaggle report. Expand each validation result and inspect the original question, reference, normalized value, stop reason, scored target, and raw text. Check for premature boundaries, complete numbers, lost correct answers, actual wrong calculations, and repetitive nonanswers.

Keep `predictions.jsonl`, `summary.json`, `metrics.csv`, `identity.json`, `protocol.json`, and `result_manifest.json`. The identity records model/tokenizer EOS and padding IDs, source bundle hash, code/package fingerprints, and actual runtime. A separate ZIP preserves the diagnostic. The original data and results are never modified.

Results are saved after each batch. In the same session, rerun the diagnostic cell with the same settings to resume. Changed code/environment/configuration requires a new output directory; hashes prevent silently combining experiments. Retain outputs before ending a Kaggle session.

Generation timing includes custom stopping checks and first-call initialization; there is no warmup. This small diagnostic is not a latency benchmark. Peak allocated CUDA memory covers the completing process, not earlier interrupted processes.

## Validation and next decision

The notebook runs CPU contract tests before downloading the real model's weights. Local tests include deterministic scripted continuations through a real tiny Llama `generate` call, randomly initialized model execution, the real CLI, normalization edge cases, validation-only scope, and immutable result resume. They do not establish actual SmolLM2 accuracy or CUDA performance; that is what the small Kaggle run must check.

After reviewing the 20 outputs, decide whether termination/extraction is reliable enough to freeze a revised baseline protocol. If it is, a later explicitly configured pilot can evaluate a fixed, balanced subset before LoRA training. If it is not, adjust only using validation evidence and repeat a small diagnostic. Do not launch the old 5,638-prediction run or increase the token cap automatically.
