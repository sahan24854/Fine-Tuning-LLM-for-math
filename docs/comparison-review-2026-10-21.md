# Review of completed Notebook 04 — 2026-10-21

Reviewed executed notebook commit `3977553`, using code `f2cbfd25f98bd425532d615f7733ce1f1c2e3644`. All 300 adapter predictions completed. The notebook source matches the published launcher, the CPU contract tests passed, and there are no notebook error outputs. The final cell displayed the report, completed checksum verification, and created the output ZIP.

## Verification

The embedded HTML report contains both original and tuned raw outputs for all 300 unique question keys. Independently recomputed checks confirm all 600 embedded score/parse/boundary results, all dataset correct/parseable/capped/token counts, all paired transitions, and all paired-bootstrap intervals. Questions and reference values match the checksum-verified source data. All original outputs also match the previously supplied baseline predictions.

The input-verification cell records the original comparison contract `b34b624c4aa58b70efe3c71d011075b8cc8f6f815631556e1553abe8f4050051` and selected adapter `checkpoint-633`. It reports successful checksum, saved-selection, base-revision, selected-weight, and frozen-evaluator checks. The real model loaded with the expected T4/PyTorch/Transformers/PEFT runtime, and the runtime guards permitted inference.

This review independently checks the embedded outputs. The full source/result archives and actual adapter are not committed in the notebook, so this is not an independent revalidation of their file contents outside the reported Kaggle checks. Retain the original and comparison ZIPs.

## Official numeric-answer results

| Dataset | Original | LoRA | Change | Newly correct | Previously correct, now wrong | Paired bootstrap 95% interval |
|---|---:|---:|---:|---:|---:|---:|
| gsm8k_test | 5/100 | 8/100 | +3 pp | 8 | 5 | -4 to +10 pp |
| svamp_test | 20/100 | 12/100 | -8 pp | 5 | 13 | -16 to +0 pp |
| arithmetic_test | 4/100 | 54/100 | +50 pp | 51 | 1 | +44 to +56 pp |

Official correctness includes unparseable answers as incorrect. The GSM8K increase is small and its interval includes zero; this pilot does not establish a convincing word-problem gain. SVAMP declines in the sample, with more regressions than improvements. All five originally correct GSM8K questions become incorrect, while eight previously incorrect questions become correct. The model has not consistently improved on elementary word problems.

## Formatting and stopping clearly improved

| Dataset | Original parseable | LoRA parseable | Original cap hits | LoRA cap hits |
|---|---:|---:|---:|---:|
| GSM8K | 75/100 | 95/100 | 16 | 5 |
| SVAMP | 84/100 | 98/100 | 9 | 2 |
| Arithmetic | 23/100 | 97/100 | 8 | 1 |

The adapter supplies usable labelled numeric answers much more consistently. Cap-limited outputs fall from 33/300 to 8/300. It also ends at EOS frequently instead of continuing into another problem. These are useful behavioral changes, but format/termination improvements are not sufficient evidence of better mathematical reasoning.

## Arithmetic: most of the official jump is a measurement-format effect

The frozen official arithmetic score rises from 4/100 to 54/100. The symmetric exploratory equation audit additionally recognizes 47 original correct unlabelled equations and no additional tuned equations. Under that narrow answer-evidence rule, the comparison is **51/100 to 54/100**, with six evidence improvements and three regressions. This is not a new official benchmark score or an exhaustive measure of mathematical ability. The audit was designed after inspecting the original baseline.

| Arithmetic stratum | Original correct-answer evidence | LoRA correct-answer evidence |
|---|---:|---:|
| small/add | 10/10 | 10/10 |
| small/subtract | 10/10 | 10/10 |
| small/multiply | 7/10 | 9/10 |
| small/divide | 7/10 | 9/10 |
| small/two_step | 1/10 | 2/10 |
| large/add | 8/10 | 8/10 |
| large/subtract | 8/10 | 6/10 |
| large/multiply | 0/10 | 0/10 |
| large/divide | 0/10 | 0/10 |
| large/two_step | 0/10 | 0/10 |

The small-number multiplication/division samples improve modestly under the evidence rule; small addition/subtraction were already correct. Large subtraction regresses. Neither model shows correct-answer evidence under this audit for the sampled large multiplication, large division, or large two-step questions. Each stratum has only ten questions, so these observations are diagnostic rather than broad capability guarantees.

## Examples confirm both gains and real regressions

- GSM8K app-count question: original computes `52 + 18 = 60`; adapter correctly computes `52 + 18 = 70` and answers 70. This is a genuine corrected calculation in the observed output.
- GSM8K water-balloon question: adapter correctly uses `10 * 30 - 12 = 288`; original ignores the pack count and answers 18.
- GSM8K laptop discount: original correctly answers 800 for a 20% discount on 1000; adapter writes `1000 - .2 * 1000 = 9980` and answers 9980. This is a genuine regression, not a parsing issue.
- SVAMP subtraction: original correctly computes `521 - 174 = 347`; adapter uses incorrect operations/calculations and answers 231.

The lower teacher-forced validation loss from Notebook 03 did not translate into consistent generated word-problem improvements. It measures prediction of supplied correct solution tokens; it does not directly optimize the exact correctness of a complete generated solution. This run alone cannot determine whether learning rate, data coverage, objective, model capacity, or another factor is the principal cause.

## Timing and next decision

Adapter generation time sums to about 834.6 seconds (13.9 minutes); original generation time was about 941.1 seconds (15.7 minutes). Output tokens fall from 29,095 to 17,963. These are descriptive figures across separate runs, including per-token stop checks and first-call effects; they are not a controlled speed or end-to-end latency benchmark.

The first train/evaluate experiment is complete and reproducible. Its honest conclusion is successful training and much stronger format compliance, with mixed mathematical outcomes and no consistent word-problem improvement. It does not yet satisfy a claim of broadly improved elementary word-problem ability.

The next useful work is to record this experiment, inspect failure patterns on the validation split, and design a controlled second experiment using validation evidence. Any new training settings or data mixture should be declared in a new run, with checkpoint selection independent of these test results. Avoid blindly extending training or tuning to these 300 already-inspected questions. Reserve an additional untouched evaluation for stronger claims. Preserve the current adapter, all raw outputs, original scorer, and exploratory audit.
