# Review of Notebook 06 artifacts — 2026-10-22

Reviewed uploaded training metadata and all six generated-validation files from code `928f82f47d3963025152d47d9c1c0e774d7f4be8`. The seven JSON training artifacts and six generated-validation files match their saved manifest checksums. The browser-saved HTML was transformed during saving, but its embedded summary and generated metrics agree with the JSON files. Actual adapter and checkpoint weight files were not uploaded; their bytes were not independently reloaded in this review.

All 250 predictions were rescored independently. Prompt/reference bindings, raw/target boundaries, stop reasons, repetition flags, metric totals and pairs agree with the saved observations. Final-epoch and selected predictions agree exactly apart from elapsed time.

| Condition | Correct / 50 | Parseable / 50 | Cap hits |
| --- | ---: | ---: | ---: |
| Repeated original | 5 | 31 | 10 |
| Epoch 1 | 1 | 48 | 1 |
| Epoch 2 | 3 | 45 | 4 |
| Epoch 3 / selected | 3 | 47 | 3 |

Training completed 633 updates. Validation response NLL decreases from 1.039555 to 0.760041, with the final checkpoint selected by the declared rule. The first adapter's selected NLL was 0.707868 and its same diagnostic score was 6/50. Candidate/first-adapter pairs are one improvement, four regressions, two both correct and 43 both wrong. This one-seed, repeatedly inspected 50-question development comparison is not evidence of a general learning-rate law.

The repeated original baseline differs on two questions: `gsm8k/train/00157` (beads) and `gsm8k/train/06055` (pages read). Both are incorrect in both runs; none of the 50 correctness labels changes. The beads continuation becomes repetitive and cap-limited, accounting for the ninth-to-tenth cap-hit difference. Checksums verify the difference, not its numerical cause.

Examples show persistent mathematical errors: `12 + 36 = 50`, `90 + 100 = 290`, omitted keys-per-lock multiplication, and subtraction of weekend revenues that should be added. The weekly pull-up solution improves to `5 * 2 * 7 = 70`. The elevator, snack budget, turtle total and trash-can questions regress against the first adapter. Some correct final answers still contain narrative issues or match an ambiguous reference interpretation.

Preserve both adapters. The next step is the [Notebook 07 diagnostic and checked-data preparation](balanced-data-preparation.md), with no new optimizer changes or automatic training. A fresh matched data-mixture/control comparison remains a future experiment after review. Do not silently replace earlier test scores or claim the data intervention is already successful.
