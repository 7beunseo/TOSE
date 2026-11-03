# TOSE

Official implementation of **TOSE: Target-Conditional Observed-Set Encoder for
Mixed-Type Tabular Imputation**.

TOSE represents the currently observed variables as a set, contextualizes them
with an observed-set Transformer, and retrieves that memory separately for each
reconstruction target. Numerical variables are reconstructed with continuous
heads and categorical variables with per-variable class heads.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
.venv/Scripts/activate
python -m pip install -r requirements.txt
```

On Linux or macOS, activate the environment with `source .venv/bin/activate`.

## Data preparation

The public generator creates feature-only MCAR, MAR, and MNAR benchmarks.
The supervised target column is never included in either the model input or the
missingness mechanism.

On Windows, generate the 30% benchmark with:

```powershell
powershell -ExecutionPolicy Bypass -File data_generation/generate_benchmark.ps1 `
  -PythonExe python `
  -Rates 30 `
  -PrepareBase
```

To generate selected datasets or repetitions:

```powershell
powershell -ExecutionPolicy Bypass -File data_generation/generate_benchmark.ps1 `
  -PythonExe python `
  -Datasets adult,bean,breast `
  -Splits 0,1,2 `
  -Rates 30 `
  -PrepareBase
```

See [data_generation/README.md](data_generation/README.md) for the protocol and
direct Python commands.

## Training

Run full TOSE on one mechanism and one or more repetitions:

```bash
python run_tose_benchmark.py \
  --datasets adult bean breast \
  --splits 0 1 2 3 4 5 6 7 8 9 \
  --data-root data \
  --mask-type MCAR \
  --ratio 30 \
  --ablation-mode full \
  --run-tag tose_full \
  --skip-completed
```

Use `MCAR`, `MAR`, or `MNAR` as the missingness type.

Default full-model settings use embedding width 64, four heads, two
observed-set blocks, six target-conditional blocks, batch size 128, at most 600
epochs, and validation-only early stopping with patience 45. The masked-modeling
objective uses 25% auxiliary remasking and the fixed loss
`L_observed + 0.6 * L_remasked`.

## Corrected missingness protocol

This release corrects two issues found while auditing the research pipeline:

1. Earlier mask-generation code appended the original supervised target to the
   logical feature table. In MAR, that excluded label could be selected as a
   missingness predictor, and dropping it later could also shift the effective
   feature missing rate away from 30%.
2. Earlier MAR/MNAR calibration used concatenated train and test covariates.
   The public generator fits standardization, coefficients, and intercepts on
   training rows only and never uses test rates to select a mechanism.

The corrected generator samples the nearest feasible exact number of missing
logical cells independently in train, validation, and test. Logistic
probabilities are retained as relative sampling weights. Therefore a requested
rate `rho` differs from the realized partition rate by at most half a logical
cell divided by the number of cells in that partition.

These corrections change the generated masks. Results produced with archived
legacy masks are not numerically identical to results from this public
generator. Using the corrected protocol requires regenerating masks and
rerunning the affected experiments.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests check target exclusion, feature-only mask mapping, exact partition
rates, and MAR predictor observability.

## Repository layout

- `model.py`: TOSE architecture and ablation paths.
- `trainer.py`, `losses.py`, `metrics.py`: training and evaluation.
- `dataset.py`: train/validation/test loading.
- `data_generation/`: dataset preparation and corrected missingness generation.
- `tests/`: smoke and data-protocol tests.

## Reproducibility notes

- Normalization statistics are estimated from observed training cells
  only and then applied unchanged to validation and test.
- Checkpoints are selected using validation data only.
- Categorical masks operate at the original logical-variable level; all binary
  code bits belonging to one categorical variable share the same mask.
- Repetition summaries use the sample standard deviation (`ddof=1`).
