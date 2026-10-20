# Maths transfer learning with SmolLM2

Measure `HuggingFaceTB/SmolLM2-360M` before and after LoRA fine-tuning on maths problems. The new experiment starts with **[Notebook 01: data and protocol](notebooks/01_data_and_protocol.ipynb)**. The original Qwen/Kaggle notebooks remain as historical experiments.

**Current step: [Notebook 04: paired comparison](notebooks/04_paired_comparison.ipynb).** Import it into a new Kaggle notebook, enable Internet and a T4, and attach the saved outputs from Notebook 01, Notebook 02c, and Notebook 03. It verifies the original artifacts, generates 300 adapter answers on the saved pilot questions, and compares accuracy, formatting, paired changes, and exploratory arithmetic evidence. See [the comparison guide](docs/paired-comparison.md).

The executed [Notebook 03](notebooks/03_lora_training.ipynb) completed all 633 updates. [Its training review](docs/training-review-2026-10-20.md) records validation loss decreasing from 1.040 to 0.708; this is not yet evidence of a generated-answer accuracy gain. Keep the full training ZIP and selected adapter.

The executed [Notebook 02c: three-shot pilot](notebooks/02c_baseline_pilot.ipynb) is preserved with outputs. Its 300 predictions are the development baseline for the later paired comparison. Read [the pilot review](docs/pilot-review-2026-10-20.md): missing answer labels hide many correct arithmetic equations, so improved format compliance must be distinguished from improved maths. Keep its complete Kaggle output ZIP.

[Notebook 02b](notebooks/02b_protocol_v2_diagnostic.ipynb) remains the small validation diagnostic. [Its reviewed predictions](docs/protocol-v2-review-2026-10-20.md) support the stopping fix while showing remaining model errors. The pilot is a development comparison on previously inspected test sets, not an untouched final holdout.

The executed [Notebook 02: protocol-v1 baseline](notebooks/02_baseline.ipynb) remains preserved as an audit trail. Do not rerun that full benchmark to validate protocol v2.

**Evaluation audit:** the first full run had 100% length-limited outputs and 98.1% strict-format invalid answers. Read [the findings](docs/baseline-audit-2026-10-20.md) before interpreting the scores or starting training. A completed run is not sufficient evidence of valid measurement. The CLI now checks smoke health before permitting a full run; a CPU-only tool can audit saved predictions without repeating inference.

## Run Notebook 01 on Kaggle

1. Sign into Kaggle and create a notebook at <https://www.kaggle.com/code/new>.
2. Use **File → Import Notebook**, select the GitHub/URL import option, and import:

   ```text
   https://github.com/sahan24854/Fine-Tuning-LLM-for-math/blob/main/notebooks/01_data_and_protocol.ipynb
   ```

   If the dialog asks for a repository and file separately, use `sahan24854/Fine-Tuning-LLM-for-math` and `notebooks/01_data_and_protocol.ipynb` on `main`.
3. Enable **Internet** in notebook settings. Leave the accelerator at **None**; this step uses CPU only. Kaggle may require account verification before enabling Internet.
4. Choose **Run All**. The notebook fetches the current GitHub code, creates an isolated environment, downloads the public data and model tokenizer, audits the data, and creates a ZIP under `/kaggle/working`.
5. **Save Version → Save & Run All** and retain the outputs. Attach this saved notebook output to the later baseline notebook using Kaggle's Add Input flow.

This public repository requires no GitHub token to read. Public model/tokenizer downloads require no Hugging Face token. Creating a Kaggle notebook and authorizing any account connection happen in your signed-in browser; this repository does not contain account credentials.

## How GitHub updates reach Kaggle

GitHub is the source of truth. Edit shared Python code and `configs/data_protocol.json` in GitHub or in a local checkout, commit, and push. On the next **Run All**, the launcher fetches `main` into a fresh temporary checkout and records its exact commit.

An imported Kaggle notebook is a copy, not a continuously synchronized Git repository. The launcher removes the need to copy/paste changed shared code. If the launcher's own cells change, re-import it. Avoid editing shared logic inside Kaggle or saving the launcher over the GitHub source. Optional Kaggle “Save to GitHub” integration is a separate one-way export feature, not required for this workflow.

Every run has an immutable data bundle. **Use the same bundle for baseline and adapted-model evaluation**; do not prepare new splits between them. To repeat an older preparation, use the manifest's code commit as `MATH_PROJECT_REF` and the saved `config.resolved.json` as the configuration. See [the detailed workflow](docs/kaggle-workflow.md).

## What the first notebook teaches

| Step | Why it matters |
|---|---|
| Inspect the protocol | Agree on the task and scoring before seeing model results |
| Download pinned, checksum-verified data | Know exactly which examples were used |
| Split and check overlap | Keep training, model selection, and final evaluation separate |
| Generate arithmetic probes | Measure specific arithmetic operations and larger-number stress cases |
| Audit tokens | Avoid truncating solutions or overflowing evaluation context |
| Freeze demonstrations and model revision | Give the original and adapted models comparable inputs |
| Save a verified bundle | Make later results reproducible and inspectable |

### Data

- **GSM8K:** 7,473 official training and 1,319 test examples. The preparation reserves 10% of the cleaned training pool for validation. Training examples exceeding the token budget are excluded and listed; official test examples are never dropped.
- **SVAMP:** all 1,000 challenge examples, evaluation only.
- **Generated arithmetic:** 500 evaluation-only examples, covering addition, subtraction, multiplication, exact division, and two-step word problems in two number ranges.

The protocol removes normalized exact question overlap from the training pool and reports duplicates within evaluation sets. It does **not** establish semantic decontamination or absence from pretraining. Report each benchmark separately. Test answers are retained for later scoring but are not shown in notebook previews or passed to the model.

### Outputs

`train.jsonl`, `validation.jsonl`, `gsm8k_test.jsonl`, `svamp_test.jsonl`, `arithmetic_test.jsonl`, `demonstrations.json`, `excluded_training.jsonl`, `audit.json`, `config.requested.json`, `config.resolved.json`, tokenizer files, `environment.txt`, and `manifest.json`.

Training rows store separate `prompt` and `completion` strings. Later SFT code must concatenate their separately encoded token IDs, mask prompt labels and padding with `-100`, and append EOS as defined in `training_contract`. Do not apply whole-sequence loss or silently truncate examples. This is a base model experiment, so the protocol uses a plain text format rather than an assumed chat template.

Notebook 01 produces **data-ready** artifacts. It does not evaluate model accuracy, train LoRA, choose hyperparameters, or claim that maths performance improved. Those are later milestones.

## Develop locally

Python 3.10+ is required; The environment setup supports system Python installations without `ensurepip`, including Kaggle Python 3.13. Use an environment with Internet access to GitHub/raw.githubusercontent.com, PyPI, and huggingface.co.

```bash
python scripts/setup_environment.py --environment .venv-data
.venv-data/bin/python -m pip check
python -m unittest discover -s tests -v
.venv-data/bin/python scripts/prepare_data.py \
  --output artifacts/data-run-001 --cache .cache/math-sources
.venv-data/bin/python scripts/verify_bundle.py artifacts/data-run-001
```

For dataset-only validation when Hugging Face is unavailable:

```bash
python scripts/prepare_data.py --audit-sources-only \
  --output artifacts/source-audit-001 --cache .cache/math-sources
```

The source-only command is intentionally **not** a substitute for the full tokenizer audit and does not create a baseline-ready manifest. Commands refuse to overwrite a nonempty output directory.

## References

- [SmolLM2-360M model card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M)
- [GSM8K source and dataset documentation](https://github.com/openai/grade-school-math)
- [SVAMP source and dataset documentation](https://github.com/arkilpatel/SVAMP)
- [PEFT LoRA guide](https://huggingface.co/docs/peft/developer_guides/lora)
- [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)

Consult the original model/dataset licenses before redistribution. Downloaded datasets and generated run artifacts are not committed to this repository.
