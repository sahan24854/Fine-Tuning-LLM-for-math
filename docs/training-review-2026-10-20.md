# Review of completed Notebook 03

Reviewed executed notebook commit `4970eea`, using training code `25f438582ec7f2d7de6affd7f574faad5714ffeb`. Its source cells match the published launcher and all eight code cells executed without notebook error outputs.

## Checks and result

- Original model revision: `f8027fd0eaeea54caa13c31d31b9fdc459c38b49`.
- Original data bundle: `b9d3419aa71419626d274843f88dc6190f5828877b20a1cc67e5249e760b5c71`.
- All 6,726 training IDs and 747 validation IDs match the checksum-verified source split, including order. No test IDs occur in these lists.
- Maximum audited sequence lengths: 488 training tokens and 447 validation tokens, below the 1,024-token ceiling. There was no need for truncation.
- CPU training contract tests and the two-update real-model smoke check passed.
- Full run completed three epochs and all 633 planned optimizer updates, with effective batch size 32 except partial epoch-end updates.
- Trainable LoRA parameters: 3,276,800, approximately 0.90% of the base-plus-adapter parameter count. The training code restricts trainable parameters to LoRA matrices.
- All printed training losses, validation losses, and gradient norms are finite.
- The selected checkpoint is `checkpoint-633`; its loss equals the minimum observed checkpoint validation loss and the re-evaluated selected-model loss.

| Point | Validation response-token negative log likelihood |
|---|---:|
| Original base model | 1.039555 |
| End of epoch 1 | 0.735921 |
| End of epoch 2 | 0.713643 |
| End of epoch 3 / selected checkpoint | 0.707868 |

All nine recorded checkpoint validation losses decrease. The final loss is approximately 31.9% below the base-model value. Improvements become smaller over time; there is no rising-validation-loss pattern in this run. This does not establish the absence of overfitting under other measures or the usefulness of additional epochs.

The full Trainer call reports 2,856.8 seconds, approximately 47.6 minutes. This includes its internal training/evaluation/checkpoint work, but excludes notebook setup, the smoke experiment, and separate pre/post-training evaluation calls. Peak process CUDA allocation is approximately 1.96 GiB; it is not total GPU memory occupancy.

The final cell completed checksum verification and printed the selected adapter path and output ZIP:

```text
/kaggle/working/lora-v1-b9d3419aa714-1100d832-25f43858/full/adapter
/kaggle/working/lora-v1-b9d3419aa714-1100d832-25f43858.zip
```

The image's missing-`wrapt` startup warning and `torch_dtype` deprecation warning did not prevent completion. The pad-token alignment message is consistent with the shared loader assigning EOS as padding when no pad token is present; it does not show that new vocabulary entries were added.

## What is established

The evidence supports a successful engineering run and improved prediction of correct validation solution tokens under teacher forcing. A 31.9% loss reduction is not a 31.9% accuracy gain. It does not measure freely generated answers, prove improved reasoning, or separate formatting gains from mathematical gains.

The notebook prints checksummed artifact completion, but the full adapter/identity/archive are not included in the GitHub notebook. This review therefore does not independently reload the real trained weights or verify the saved archive itself. Keep the complete Kaggle output ZIP, including `full/adapter`, and the original Notebook 01 and 02c outputs. The tiny-model local tests establish reload behavior for the implementation; they do not replace testing the actual saved SmolLM2 adapter in the next evaluation.

## Next step

Proceed to paired evaluation of the selected full-run adapter on the original 300-question pilot. Reuse the original bundle, selection, model revision, prompts/demonstrations, generation settings, stopping, and official scorer. Report dataset accuracy, parseability, cap hits, and paired improvements/regressions. Apply the documented supplementary arithmetic analysis symmetrically to both models and label it exploratory, because the baseline contains correct equations rejected for missing final-answer labels. Do not train more or select a different checkpoint based on these test results.
