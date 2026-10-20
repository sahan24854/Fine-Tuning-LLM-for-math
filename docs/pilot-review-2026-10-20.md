# Review of executed Notebook 02c

Reviewed the notebook uploaded in commit `d31892f`, executed with code commit `80439bf16091bf607eddfc152d02d1d47b01e780`. This is an untouched-base-model evaluation, not a training run. The model revision is `f8027fd0eaeea54caa13c31d31b9fdc459c38b49` and the comparison contract is `b34b624c4aa58b70efe3c71d011075b8cc8f6f815631556e1553abe8f4050051`.

## Execution and checks

The notebook completed 300/300 predictions, displayed the report, and created its output ZIP. The launcher's final checksum check completed without an error output. The notebook source matches the published launcher. CUDA execution and the stopping/scoring/pilot contract tests succeeded. The `wrapt` startup warning and `torch_dtype` deprecation warning did not prevent inference.

The embedded HTML report contains all 300 unique prediction targets, raw completions, references, and score details. Review checks established:

- All questions and reference answers match the checksum-verified source datasets.
- All selected IDs and their order match the declared deterministic selection algorithm, including arithmetic stratification.
- Recomputed answer, correctness, and parse-status fields match all 300 saved records.
- All target slices and stopping reasons are consistent with the reviewed boundary rules.
- Dataset counts, correct/parseable/capped counts, and generated-token totals agree between the embedded report and printed summary.

The original ZIP, identity, selection, and result-manifest files are not committed alongside the notebook. This review verifies the embedded evidence; it does not independently verify the complete saved archive or reconstruct its runtime/package fingerprint. Retain the Kaggle ZIP and original Notebook 01 bundle for comparison.

## Official results under the frozen answer contract

| Dataset | Correct | Accuracy | Descriptive 95% Wilson interval | Parseable answers | Cap hits |
|---|---:|---:|---:|---:|---:|
| GSM8K | 5/100 | 5% | 2.2–11.2% | 75/100 | 16/100 |
| SVAMP | 20/100 | 20% | 13.3–28.9% | 84/100 | 9/100 |
| Arithmetic | 4/100 | 4% | 1.6–9.8% | 23/100 | 8/100 |

There are 33 generation-cap hits across 300 predictions (11%). This differs from the original protocol-v1 run's 100% cap-hit rate. Summed batch generation time is approximately 941 seconds (15.7 minutes), excluding installation, model loading, and other notebook work; it is not total notebook runtime or a controlled speed comparison.

These are development results on previously inspected test sets. Their confidence intervals do not address development reuse, pretraining contamination, or the arithmetic sample's stratification. Final-answer accuracy does not establish correctness of every reasoning step.

## Main finding: correct unlabelled arithmetic is scored invalid

The scorer requires a `Final answer:` label. Many arithmetic continuations instead give a correct equation and then start a new question:

```text
Question: What is 18 + 9?
Solution: 18 + 9 = 27

Question:
```

Stopping before the new question is appropriate. The saved target still contains the correct equation, but extraction returns no labelled answer. Thus the 4% arithmetic headline cannot be interpreted as "the model can solve only 4% of these arithmetic problems."

A deliberately narrow exploratory audit found **47 additional correct unlabelled equations**. It accepts only a whole target consisting of one equation `a operator b = integer`, requires the left side to match the question's operands and operator exactly, and compares the right side to the reference. It does not search for the reference in arbitrary text or choose a last number from a multi-step solution.

| Arithmetic group | Correct unlabelled equations additionally identified |
|---|---:|
| Small addition | 10/10 |
| Small subtraction | 10/10 |
| Small multiplication | 7/10 |
| Small division | 4/10 |
| Large addition | 8/10 |
| Large subtraction | 8/10 |
| Other groups | 0 identified by this narrow rule |

Together with the four officially credited outputs, at least **51 of these 100 outputs contain a correct answer in a clearly interpretable form**. This is an exploratory lower bound on answer evidence in this sample, not a replacement official score, an independently established model accuracy, or a predeclared benchmark metric. More complicated unlabelled outputs were not counted by this rule.

This is a measurement limitation in the project design: the ten-question diagnostic examined GSM8K validation questions and did not expose the arithmetic format pattern. The original official results remain correct under their declared contract, but that contract combines mathematical correctness and compliance with an answer format. Fine-tuning may improve either component.

The unit whitelist also rejects some numeric labels with unlisted wording. For example, SVAMP `chal-520` produces `Final answer: 6 more kids`, with reference `6`, and is marked unparseable. Other rejected labels contain genuinely wrong numbers or ambiguous expressions. Do not treat all invalid answers as hidden correct solutions.

## Genuine maths errors also remain

Examples include `9 * 17 = 163` (reference 153), `518 + 561 = 1089` (reference 1079), and `750 - 546 = 105` (reference 204). Word-problem failures include incorrect operations and quantities. Conversely, some GSM8K solutions correctly compute both the intermediate steps and final answer, including a 20% discount on a $1000 laptop yielding $800.

The data therefore show both format failures and mathematical failures. Raising the output cap cannot repair a missing label in an already completed correct equation, and is not supported as the general remedy.

## Next comparison

Keep the frozen contract scores and complete raw outputs. Before interpreting a fine-tuning gain as improved mathematical ability, report answer-format compliance separately and include a documented, identical arithmetic audit on both base and tuned outputs. Any supplementary rule designed after inspecting these results must be identified as exploratory, versioned, and applied symmetrically; it must not silently replace the official score.

LoRA training can use the original training split with response-only loss and validation-based checkpoint selection. Reuse the saved 300-question pilot contract for its development comparison. Improvement in the official metric alone may reflect better label formatting, so inspect paired raw outputs and the separate format/maths evidence. Reserve additional untouched questions for a stronger final assessment. No additional baseline GPU run is required merely to document these findings.
