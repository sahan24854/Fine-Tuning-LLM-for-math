# Review of Notebook 02b predictions

Reviewed the 20 complete records supplied on 2026-10-20. Uploaded file SHA-256: `c363c9d1bd8131691ac9d8e65fcefd57ece2121058907afa663ba91023cd0cd2`.

## Results

These are ten validation questions evaluated twice, not twenty distinct questions or a final benchmark.

| Metric | Zero-shot | Three-shot |
|---|---:|---:|
| Predictions | 10 | 10 |
| Correct, all questions | 0/10 | 1/10 |
| Normalized numeric answer available | 1/10 | 7/10 |
| Strict numeric format | 1/10 | 4/10 |
| Stopped at final-answer line | 1 | 7 |
| Stopped before a new question | 5 | 0 |
| Reached 512-token cap | 4 | 3 |
| Generated tokens | 2,718 | 1,940 |

All twenty questions and references match the pinned source data. Recomputed scores match all twenty saved scores. Prompt hashes match the corresponding original smoke records, and every new raw completion is an exact prefix of its original continuation. The saved stop reasons and target slices agree with the frozen v2 boundary rules. These checks support the stopping/extraction implementation for the observed outputs; they do not prove it correct for every possible continuation.

The supplied predictions omit the run identity and result manifests, so archive integrity and exact package/hardware identity were not independently verified.

## What changed

The money-counting question (`gsm8k/train/07224`) now stops after `Final answer: 550` and correctly earns credit. In the original smoke output, it continued into unrelated questions and the strict parser rejected it. This is a corrected measurement, not a trained improvement.

Currency normalization also behaves as intended: the manicure answer `$9` becomes numeric `9`, but remains incorrect against reference `39`. A parseable answer is not automatically a correct answer. Six of the seven parseable three-shot answers are wrong.

Thirteen predictions stop before the cap. Total output is 4,658 tokens versus 10,240 for the same twenty original smoke predictions: approximately 54.5% fewer tokens. This is not a controlled wall-clock speed benchmark; batch composition and runtime conditions can affect timing.

All seven cap-limited outputs show repetition or lack of progress, including empty numbered steps, repeated statements, or repeated equations. These records do not justify a blanket increase in the token cap. Longer generation might change individual outcomes, but the evidence does not establish that truncation is the main remaining cause of low accuracy.

## Decision and next experiment

The observed stopping and extraction behavior is suitable to freeze for a small pilot. Zero-shot still has 90% unparseable targets and triggers the diagnostic's quality alarm. Three-shot has 70% parseable targets and 30% cap hits, with one correct result. Ten questions are too few to estimate general maths performance reliably.

Use three-shot as the proposed primary comparison and retain zero-shot as a separate secondary diagnostic. This choice is based on validation behavior and must be declared before evaluating a new test subset. Preserve both earlier runs and their protocol identities.

The next implementation should select a deterministic pilot of 100 GSM8K, 100 SVAMP, and 100 arithmetic test questions using IDs and a fixed seed, without selecting by model correctness. Freeze that subset, the demonstrations, scoring, decoding, and token budget before execution. Run the untouched base model once, then evaluate the later LoRA model on precisely the same questions and settings. This is a proposed pilot, not an already executed or automatically authorized full benchmark.

Report accuracy separately per dataset, answer parseability, cap-hit rate, generated tokens, confidence intervals, and paired before/after outcomes. Include invalid answers in accuracy denominators. Training and checkpoint selection must use training/validation data, never these test outcomes. Since the original full test results have already been inspected, describe this as a development comparison rather than claiming a previously unseen final holdout; reserve an additional untouched evaluation for a stronger final claim.

No inference or scoring rules were changed during this review, and no training occurred.
