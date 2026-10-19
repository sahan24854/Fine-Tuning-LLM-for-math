# GitHub → Kaggle workflow

## Notebook 01, in plain language

This notebook prepares the experiment's inputs and rules. It downloads real labelled maths problems, separates the examples used for learning from those used to choose a model and those used to evaluate it, and records all decisions. No GPU is necessary because it downloads only a tokenizer, not SmolLM2's weights.

The tokenizer converts text to token IDs. We measure those lengths because a model has a finite context window. The notebook excludes long training solutions with a recorded reason and fails if any evaluation prompt would overflow; it never silently chops off a correct answer.

The validation set is for checkpoint/hyperparameter selection. The GSM8K test set, SVAMP, and generated arithmetic set are for reporting final performance. Although the first baseline will be evaluated on the test sets, do not use its individual errors to repeatedly tune prompts or training settings; use validation for those decisions.

## One-time connection

Import `notebooks/01_data_and_protocol.ipynb` from this public GitHub repository using Kaggle's notebook import dialog. No account-to-account OAuth connection or personal access token is needed for public read access. Enable Internet, select CPU, and run all cells. Exact menu labels can vary with Kaggle UI changes; look for notebook import from GitHub/URL.

The first cell fetches `main` into a new checkout on every run. The remaining cells run the latest `scripts/prepare_data.py` and use the latest `configs/data_protocol.json`. The checkout and isolated dependencies are temporary; output files live in `/kaggle/working/data-protocol-<run-id>` and an adjacent ZIP.

The loop is:

1. Update shared Python/configuration on GitHub.
2. Run all cells in the existing Kaggle launcher.
3. Save the Kaggle version with outputs.
4. Keep that bundle fixed for a baseline/fine-tuned comparison.

Kaggle import is a snapshot of the notebook cells. It does not watch GitHub or rerun experiments automatically. The launcher fetches changes only when you run it. A future change to the launcher itself requires re-importing; normal data/logic/config changes do not.

## Reproducibility

For a comparison already started, reuse its original bundle. For a new experiment, rerun Notebook 01 to create a new bundle. The source dataset URLs use fixed commits and their byte hashes are checked. `main` for the model is resolved once to a full commit; later notebooks must load the model at `manifest.json` → `model.revision` and must not independently resolve `main`.

To regenerate a prior bundle locally, check out its `code_commit`, install the recorded package versions in an isolated environment, and pass the saved configuration:

```bash
python scripts/prepare_data.py \
  --config /path/to/prior-bundle/config.resolved.json \
  --output artifacts/reproduced-run --cache .cache/math-sources
python scripts/verify_bundle.py artifacts/reproduced-run
```

To fetch an older code commit in the Kaggle launcher, set `MATH_PROJECT_REF` in the session environment before the first cell. It defaults to `main` for everyday updates. Reusing saved artifacts is simpler than regenerating a completed comparison. Timestamps and environment metadata can differ even when data and prompts are identical.

## Preparing the next notebook

Save a Kaggle version of Notebook 01 with its outputs. In the future baseline notebook, use **Add Input** to select that saved notebook output (or upload the saved bundle as a private Kaggle dataset). Read it from Kaggle's read-only `/kaggle/input` mount, verify all hashes using `scripts/verify_bundle.py`, then load the pinned model revision and run evaluation. Do not recreate the split or reselect demonstrations.

We will build the actual baseline, training, and comparison notebooks next. They are not included in this milestone.

## Troubleshooting

- **`ensurepip` failed in an older Cell 2:** re-import the updated GitHub notebook and run all cells. Setup now lives in `scripts/setup_environment.py`, creates the environment with `--without-pip`, and bootstraps a checksum-verified pip wheel without changing the notebook kernel.

- **GitHub file not found:** ensure the new notebook has been pushed to `main`; use its full GitHub URL in the import dialog.
- **Internet disabled / DNS / connection errors:** enable Kaggle Internet access and complete any account verification Kaggle requests. Retain TLS and checksum verification.
- **Hugging Face fails:** the tokenizer and model metadata must be reachable. Public access normally needs no token. The notebook stops rather than estimating token lengths.
- **Output directory exists:** choose a fresh run directory. The notebook generates unique run IDs automatically.
- **No model accuracy displayed:** expected for Notebook 01. A completed preparation manifest means data is ready, not that the model has been evaluated.
- **Training samples excluded:** inspect `excluded_training.jsonl` and `audit.json`. Adjust the token limit only in a new protocol run, before baseline evaluation; never change the dataset halfway through a comparison.
- **Context budget error:** revise demonstration count/size or context budget in GitHub, within the actual model's supported context length, and create a new bundle.
