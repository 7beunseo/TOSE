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
objective uses 25% auxiliary remasking. The relative loss weight is
`alpha = 0.625`:

```text
L_normalized = 0.625 * L_observed + 0.375 * L_remasked
```

These coefficients are the default values used directly by the implementation.

## Matched-depth structural ablations

The matched-depth structural-ablation protocol uses the same parameter count, embedding
width, attention-head count, observed-set depth, target-conditional depth,
prediction heads, loss, optimizer, remasking policy, and training schedule as
full TOSE. With the default configuration, every variant has two observed-set
blocks and six target-conditional blocks. Only the source-weighting path named
in the final column is changed.

| CLI mode | Paper label | Observed-set blocks | Target blocks | Controlled change |
|---|---|---:|---:|---|
| `full` | Full TOSE | 2 | 6 | None |
| `static_column_attention` | w/o Sample-Adaptive Weighting | 2 | 6 | Attention scores use target/source column identity and do not depend on row values |
| `shared_target_query` | w/o Target-Specific Weighting | 2 | 6 | The mean of the existing query seeds is passed as one shared query to every target-to-source attention layer; no parameter is added |
| `uniform_source_aggregation` | w/o Learned Source Weighting | 2 | 6 | Every currently observed source receives weight `1/K` |

Run all four configurations under MCAR, MAR, and MNAR with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_matched_depth_structural_ablations.ps1 `
  -PythonExe python `
  -DataRoot data `
  -Ratio 30
```

Each `report.json` records `observed_set_block_count` and
`tose_target_block_count`. The structural-ablation tests also fail if any of
the four default configurations no longer uses the same `2+6` depth.

## Baseline reproduction

The `baselines/` directory contains the adapters used for ReMasker, PMAE,
CACTI, GAIN, MIWAE, DSAN, DeepIFSAC, MICE, and MissForest. The adapters pin the
official neural-model repositories to recorded commits and apply the common
TOSE split, missingness mask, validation-only checkpoint selection, categorical
decoding, and metric protocol. Appendix B.4-B.7 settings are executable defaults
and are checked by automated tests.

```powershell
powershell -ExecutionPolicy Bypass -File baselines/bootstrap_official_sources.ps1
powershell -ExecutionPolicy Bypass -File baselines/run_appendix_baseline.ps1 `
  -Method ReMasker -Mechanism MCAR -Ratio 30
```

See [baselines/README.md](baselines/README.md) for exact settings, official
source commits, CACTI embedding preparation, R dependencies, and the scope of
each mixed-type adaptation.

## Missingness protocol

The benchmark generator constructs feature-only MCAR, MAR, and MNAR masks. The
original supervised target is excluded from the imputation table and does not
participate in the missingness mechanism.

For MAR and MNAR, standardization statistics, coefficients, and intercepts are
estimated using training rows only. The resulting mechanism is fixed before it
is applied to validation and test, and neither partition is used to select its
parameters. MAR predictor variables remain observed by construction.

The generator samples the nearest feasible exact number of missing logical
cells independently in train, validation, and test. Logistic probabilities are
retained as relative sampling weights. Consequently, a requested rate `rho`
differs from the realized partition rate by at most half a logical cell divided
by the number of cells in that partition.

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
- `data_generation/`: dataset preparation and missingness generation.
- `scripts/run_matched_depth_structural_ablations.ps1`: controlled structural
  ablation runner using the full-model depth for every variant.
- `baselines/`: pinned official baseline sources, protocol adapters, and
  Appendix B.4-B.7 configuration checks.
- `tests/`: smoke and data-protocol tests.

## Reproducibility notes

- The row-level train/validation/test partition is fixed across repetitions.
  Repetition indices denote independently generated missingness masks and
  training seeds, not independently redrawn row partitions.
- Normalization statistics are estimated from observed training cells
  only and then applied unchanged to validation and test.
- Checkpoints are selected using validation data only.
- Categorical masks operate at the original logical-variable level; all binary
  code bits belonging to one categorical variable share the same mask.
- Categorical domains are treated as dataset schema. Their labels are read from
  the complete released table, but test category frequencies and values are not
  used to estimate normalization or model parameters.
- Repetition summaries use the sample standard deviation (`ddof=1`).
