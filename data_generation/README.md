# Corrected benchmark generation

This directory prepares the 14 public tabular datasets and creates MCAR, MAR,
and MNAR masks.

## Protocol

For every dataset and repetition:

1. The original supervised target and non-feature identifiers are excluded.
2. Rows are split into a 70% training candidate and a 30% test set. Up to 15%
   of the training candidate is reserved for validation.
3. MAR/MNAR standardization, coefficient scaling, and intercept calibration use
   training rows only.
4. The fixed mechanism is applied to train, validation, and test. Test
   values and test missing rates do not participate in mechanism selection.
5. Each partition receives `round(rho * N * D)` missing logical cells. MCAR uses
   uniform weights; MAR/MNAR use their logistic probabilities as relative
   weights for sampling without replacement.
6. Numerical normalization uses observed training cells only.
7. A categorical variable is one logical feature during masking. Its complete
   binary code is masked or observed together.

MAR keeps its sampled predictor variables fully observed. MNAR allows the
missingness of a target variable to depend on variables that can themselves be
missing, which makes the mechanism non-ignorable from the imputer's viewpoint.

## One-command generation on Windows

```powershell
powershell -ExecutionPolicy Bypass -File data_generation/generate_benchmark.ps1 `
  -PythonExe python `
  -DataRoot data `
  -Rates 30 `
  -PrepareBase
```

Use `-OverwriteMasks` only when intentionally replacing generated masks.

## Direct commands

Prepare base files and create MNAR:

```bash
python data_generation/generate_mnar.py \
  --output-root data \
  --prepare-base \
  --mask-types t2 \
  --missing-rate 0.30
```

Create MCAR and MAR from the prepared base files:

```bash
python data_generation/generate_mcar_mar.py \
  --data-root data \
  --mask-types mcar mar \
  --missing-rate 0.30
```

Both commands accept `--datasets` and `--splits` to restrict generation.

## Outputs

Each dataset stores:

```text
data/<dataset>/masks/rate30/<mechanism>/
  train_mask_<split>.npy
  valid_mask_<split>.npy
  test_mask_<split>.npy
  mechanism_<split>.json

data/<dataset>/normalized/rate30/<mechanism>/
  split_<split>.npz
  split_<split>_train.npz
  split_<split>_valid.npz
  split_<split>_test.npz
```

`mechanism_<split>.json` records feature-local indices, train-fitted mechanism
parameters, expected and realized feature rates, seeds, and protocol version.

## Legacy compatibility

`normalize_diffputer_preprocessed.py` can read both corrected feature-only masks
and legacy full-table masks. New masks are always feature-only. Legacy masks may
include the excluded supervised label and should not be used for claims based on
the corrected protocol.
