# Review of completed Notebook 05 — 2026-10-21

Reviewed executed notebook commit `b6c5538`, running code `1ebdf5824cdf391f689cd0211a052f2fef591d9e`. All 200 predictions completed: 50 original validation questions × two models × two prompting modes. The notebook source matches the published launcher; CPU contract tests passed; there are no notebook error outputs; final checksum verification and ZIP creation completed.

## Independent checks

All 200 embedded scores, parsed values, stop boundaries, repetition flags, numeric-equality checks, group counts, token totals, and paired base/adapter transitions were recomputed and agree with the saved summary. All 50 question IDs, deterministic selection order, reference answers, and worked solutions match checksum-verified source data. The notebook reports the original bundle, selected first adapter (`checkpoint-633`), and expected core runtime. Full artifact files are not present in GitHub, so this review independently verifies the embedded evidence rather than reloading the actual saved archives outside Kaggle.

These questions were held out from gradient training but used for teacher-forced validation-loss checkpoint selection. They are development data, not an untouched final holdout. Preserve the previous test results and avoid using the already-inspected 300 test questions to choose training settings.

## Results

| Model / prompting | Correct | Parseable | Cap hits | Repetition flagged |
|---|---:|---:|---:|---:|
| Original / zero-shot | 0/50 (0%) | 3/50 | 13 | 17 |
| Original / three-shot | 5/50 (10%) | 31/50 | 9 | 9 |
| First adapter / zero-shot | 2/50 (4%) | 44/50 | 6 | 3 |
| First adapter / three-shot | 6/50 (12%) | 48/50 | 2 | 2 |

Three-shot still helps in this sample, including after adaptation. For the adapter, switching from zero-shot to three-shot yields five newly correct and one newly incorrect answer: a net four-question gain. The evidence does not support switching to zero-shot as a remedy for the weak word-problem results. It also does not prove which prompting method is best across all tasks.

Within three-shot prompting, adaptation yields five improvements and four regressions; one answer remains correct under both models. The net gain is only one question. A separately recomputed descriptive paired bootstrap interval for this +2 percentage-point change is approximately **−10 to +14 percentage points**, using 10,000 resamples and seed 42. This development sample does not establish a reliable overall word-problem improvement.

Formatting is much better: the adapter gives 48 parseable three-shot answers, but 42 of those are numerically wrong. Only two outputs in that condition hit the cap, and both repeat. Raising the output cap cannot fix the already completed wrong answers and is not the supported general intervention.

## Manual error review

All 50 adapter three-shot answers were manually reviewed. [The annotations](validation-review-2026-10-21.csv) identify source IDs, categories, and concise review notes. Categories describe observed outputs; they are subjective, not a separately validated reasoning benchmark or a causal explanation of training failure.

| Primary review category | Outputs |
|---|---:|
| Wrong operation or quantity | 27 |
| Mixed arithmetic and reasoning errors | 13 |
| Arithmetic error with an otherwise appropriate setup | 2 |
| Repetition without an answer | 2 |
| Correct final answer | 6 |

The review is not an exhaustive taxonomy: multiple errors can occur in one solution. Some officially correct outputs also have narrative issues. For example, the turtle-total answer is correct but names the wrong person in an intermediate sentence. The game-purchase example matches the reference's two-purchase interpretation, while its wording permits ambiguity. Correct final numbers do not establish fully correct explanations.

Representative failures:

- **Average, `gsm8k/train/01474`:** the correct calculation is `(10 + 8 + 12) / 3 = 10`. The adapter instead takes differences and concludes `10 / 2 = 5`.
- **Money transfer, `gsm8k/train/06728`:** it mixes the brothers' balances and also states `42 - 21 - 3 = 17`. The intended final balance is `17 + 21 - 3 = 35`.
- **Stuffed animals, `gsm8k/train/05242`:** it correctly reaches 12 and 36, then states `12 + 36 = 50` instead of 48.
- **Revenue, `gsm8k/train/05570`:** it correctly computes 90 and 100, then reports their sum as 290 instead of 190.
- **Keys, `gsm8k/train/04522`:** it computes 24 apartments but omits three keys per lock; the reference is 72.

These show both wrong mathematical setup and incorrect execution of an appropriate calculation. Missing labels alone cannot explain the adapter's remaining failures.

## Why the automatic equality flags were all zero

The conservative checker recognizes only an entire line consisting of one literal equality. Across all 200 outputs, only **two lines** matched that grammar; both were exact equalities. Most calculations occur inside prose, use `x` for multiplication, include units, or contain multiple operations.

Zero automatic flags therefore does not mean zero arithmetic errors. The manual review above exposes errors outside the checker's limited coverage. Official scores were not changed, and no reference-search or last-number fallback was introduced. A future richer audit would need a separate declared rule and offline validation; this run should not be silently rescored.

## Proposed controlled second experiment

A reasonable next hypothesis to test is whether gentler adaptation preserves earlier correct behavior while still improving formatting. This is a hypothesis, not a finding that learning rate caused the failures or that a lower value will solve them.

Declare one new candidate before training:

- Start from the exact original base model with a fresh adapter, not the first adapter's weights.
- Change learning rate from `2e-4` to **`5e-5`**.
- Keep the original training data, split, rank/alpha/dropout, three epochs, effective batch size, sequence budget, optimizer, scheduler shape, and validation-loss checkpoint selection unchanged.
- Record a new training identity and preserve the first adapter.
- Compare the new selected adapter on the fixed validation questions, primarily three-shot, against the already-saved original/first-adapter outputs. Report generated-answer accuracy, formatting, and regressions along with loss.
- Choose whether further work is justified using validation evidence. Do not select a candidate based on repeated pilot-test reruns. Reserve additional untouched evaluation for stronger final claims.

This changes one main training factor and permits a clearer comparison than simultaneously changing learning rate, datasets, and prompts. If the conservative update does not improve the observed weaknesses, a later separate experiment can investigate checked training-data coverage or model capacity. The second experiment has not yet been implemented or run by this review.
