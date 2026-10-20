# Notebook 03: LoRA maths training

Import [Notebook 03](../notebooks/03_lora_training.ipynb) into a **new Kaggle notebook**:

```text
https://github.com/sahan24854/Fine-Tuning-LLM-for-math/blob/main/notebooks/03_lora_training.ipynb
```

Enable Internet and select a T4 GPU. Attach the **original saved Notebook 01 output** using Add Input. Run All and save the version with outputs. The pilot output is not needed for training; retain it for the later paired evaluation. One GPU is used even when Kaggle supplies two T4s.

## What the model learns

LoRA adds small trainable matrices to the attention projections. The original SmolLM2 weights remain frozen. This first experiment uses ordinary LoRA with an FP16 base and FP32 adapter parameters, not quantization. Only `q_proj`, `k_proj`, `v_proj`, and `o_proj` receive adapters. The checkpoint will contain the learned adapter, not a standalone copy of the full base model.

For each example, the original bundle supplies a problem and worked solution. The input is:

```text
[prompt with problem][worked solution and Final answer: number][EOS]
```

The prompt and response are tokenized separately, following Notebook 01's explicit contract. Prompt labels and padding are `-100`; solution tokens, final-answer tokens, and appended EOS are supervised. Prompt text still provides context. EOS supervision teaches the model to end its answer. There is no packing or silent truncation. All saved training token counts must match the tokenizer, and overlength training or validation rows stop the run rather than disappearing from the experiment.

The full run uses the original filtered training split (6,726 rows in the reviewed bundle) and original validation split (747 rows). It does not use GSM8K test, SVAMP, or arithmetic examples in the loss or checkpoint selection. Model revision and source-bundle hashes are recorded.

## First-run configuration

Settings live in [configs/lora_training.json](../configs/lora_training.json):

| Setting | Value |
|---|---|
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Epochs | 3 |
| Learning rate | 0.0002 |
| Optimizer | AdamW |
| Micro-batch / accumulation | 4 / 8 |
| Effective batch | 32, except final partial updates |
| Sequence ceiling | 1024, matching the original bundle |
| Scheduler / warmup | Linear / 3% of optimizer updates |
| Weight decay / gradient clipping | 0.01 / 1.0 |
| Memory setting | Gradient checkpointing, non-reentrant |
| Checkpointing | Every 100 optimizer updates and epoch ends |
| Selection | Lowest validation response-token negative log likelihood |
| Recovery state | Best checkpoint plus recent checkpoints, with optimizer/scheduler/RNG |

These are declared starting settings, not tuned or guaranteed optimal hyperparameters. For 6,726 examples, the plan is 211 optimizer updates per epoch and 633 overall. The audit calculates the count from the actual bundle.

The teacher-forced objective uses exact summed next-token cross-entropy divided by the number of supervised response tokens. Gradient accumulation normalizes over supervised tokens across its micro-batches. Validation loss is also token-weighted and excludes prompts/padding. Validation uses the zero-shot text format used in training; it does not generate answers or estimate three-shot pilot accuracy.

## Notebook flow

1. Fetch and record the GitHub code commit; verify Notebook 01's bundle.
2. Create an isolated training environment with the known PyTorch 2.7.1 CUDA 11.8 build, Transformers 4.57.6, PEFT 0.17.1, and Accelerate 1.10.1. Check CUDA execution.
3. Run CPU contract tests with tiny local models.
4. Audit all training/validation sequences without loading pretrained weights.
5. Run a two-update smoke experiment on the first 64 training rows and 16 validation rows. Require finite loss/adapter parameters, an actual LoRA parameter change, and completed artifacts.
6. Start the full experiment from a fresh adapter over the original model. The smoke adapter is not used as initialization.
7. Select the lowest-validation-loss checkpoint, export its adapter/tokenizer, display the report, and create a ZIP.

Run All intentionally proceeds from a successful smoke check into full training. A failed smoke check blocks the full CLI. The smoke and full stages must have matching bundle/config/code/runtime/package identities. Low mathematical accuracy is not tested or used as an engineering gate.

The base-model validation loss is saved before training. The report shows it alongside selected validation loss and checkpoint history. If training loss decreases while validation loss rises, that suggests overfitting under this objective. Lower validation loss does not prove improved generated maths accuracy, and this run does not require loss to improve in order to report an honest result.

## Saved output and recovery

Outputs are under `/kaggle/working/lora-v1-<bundle>-<config>-<commit>/`:

- `audit/data_audit.json`: original data IDs, token counts, and planned scope.
- `smoke/`: the engineering check and its adapter, labelled as smoke.
- `full/adapter/`: the selected **full-run** adapter for later evaluation.
- `full/tokenizer/`: saved tokenizer needed to reproduce encoding.
- `full/checkpoints/`: standard Transformers recovery state with per-checkpoint checksums.
- `full/summary.json`, `training_history.json`, `report.html`, `data_audit.json`, `base_validation.json`, configuration, environment, source manifest, identity, and result checksums.

Keep the complete ZIP and original Notebook 01 bundle. The adapter configuration records the base model revision; loading the adapter later also requires loading that exact base revision. Do not substitute the smoke adapter or assume the latest checkpoint is the selected best checkpoint.

If interrupted in the same Kaggle session, rerun the full-training cell. The code verifies identity and resumes the latest completed checkpoint with optimizer, scheduler, and random state. Uncheckpointed updates must be repeated. The selected best checkpoint is also verified. A partial checkpoint without its completion manifest is refused rather than silently trusted. Changed code, configuration, data, packages, Python version, or runtime require a new run directory. Exact recovery was tested on a tiny CPU model; the real GPU smoke check validates the T4 path. This is not a promise of identical results across different hardware/software.

The default setup preserves Kaggle's kernel environment. Cosmetic `torch_dtype` warnings and the image's nonfatal `sitecustomize`/missing-`wrapt` warning may remain; CUDA/tests/training must still succeed. Training speed and peak memory have not been measured on the real SmolLM2/T4 combination in this cloud workspace. If the actual smoke check encounters a GPU or memory issue, retain its full error rather than changing the protocol midway through a run.

## What comes after training

Paired evaluation is the next notebook. It must attach the full-run adapter, original bundle, and saved 300-question pilot selection. Reuse the same base revision, questions, prompts, demonstrations, generation budget, stopping, and official scoring.

The [pilot review](pilot-review-2026-10-20.md) identified 47 additional correct arithmetic equations rejected for missing `Final answer:` labels. Fine-tuning on labelled solutions will likely improve output formatting; an official-score increase alone cannot establish how much mathematical ability improved. Report format compliance separately and apply the same documented supplementary arithmetic audit to both models, identifying it as exploratory. Do not change only the tuned-model scorer or select checkpoints using these test answers. Reserve additional untouched evaluation for a stronger final claim.

## Local development

```bash
python scripts/setup_environment.py --environment .venv-training --profile training --torch-variant cpu
.venv-training/bin/python -m unittest discover -s tests -p test_training.py -v
.venv-training/bin/python scripts/train_lora.py --bundle /path/to/original-bundle \
  --output artifacts/training-audit --stage full --audit-only
.venv-training/bin/python scripts/train_lora.py --bundle /path/to/original-bundle \
  --output artifacts/training-smoke --stage smoke --device cpu
```

CPU mode is for development checks. Use the Kaggle GPU notebook for the real experiment. For full training, pass `--stage full --smoke-output /path/to/matching-smoke-output` and a separate output directory. No training results, adapter files, or benchmark gains are claimed before that run completes.
