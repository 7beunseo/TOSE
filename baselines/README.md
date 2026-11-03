# Baseline reproduction

This directory reproduces the nine comparison methods under the same prepared
train/validation/test rows, MCAR/MAR/MNAR masks, validation-only model
selection, and final evaluation code used for TOSE.

The neural adapters do not vendor third-party model repositories. Run
`bootstrap_official_sources.ps1` to clone the recorded official commits into
the ignored `.baseline_sources/` directory. This preserves the authors'
architecture code and makes the TOSE-specific protocol changes explicit in the
adapter files. `official_sources.json` is the machine-readable provenance
record.

## Shared training protocol

Neural baselines train for at most 600 epochs and use validation-only early
stopping with patience 45 and minimum improvement 0.0002. Test values and test
metrics are not used for checkpoint selection. MICE and MissForest use at most
10 imputation iterations instead of neural epochs.

| Method | Optimizer | Learning rate | Weight decay | Batch size |
|---|---|---:|---:|---:|
| ReMasker | AdamW, betas=(0.9, 0.95) | 1e-3 | 0.01 | 64 |
| PMAE | AdamW, betas=(0.9, 0.95) | 1e-3 | 0.01 | 64 |
| CACTI | AdamW, betas=(0.9, 0.95) | 1e-3 | 1e-3 | 128 |
| GAIN | Adam | 1e-3 | 0 | 128 |
| MIWAE | Adam | 1e-3 | 0 | 64 |
| DSAN | Adam | 3e-3 | 1e-5 | 128 |
| DeepIFSAC | AdamW | 1e-4 | 1e-2 | 128 |

The complete architecture, masking, sampling, and categorical-representation
settings live in `settings.py`. `tests/test_baseline_settings.py` prevents the
paper-critical values from drifting away from Appendix B.4-B.7.

## Official source setup

```powershell
python -m pip install -r requirements-baselines.txt
powershell -ExecutionPolicy Bypass -File baselines/bootstrap_official_sources.ps1
python baselines/verify_protocol.py
```

The bootstrap script checks out detached commits and never copies checkpoints
or result files into this repository. Third-party source code remains governed
by its original repository and license.

For MICE and MissForest, install the recorded R package versions:

```powershell
Rscript baselines/install_r_packages.R
```

## Running a method

The public mechanism names are `MCAR`, `MAR`, and `MNAR`.

```powershell
powershell -ExecutionPolicy Bypass -File baselines/run_appendix_baseline.ps1 `
  -Method PMAE `
  -Mechanism MAR `
  -Ratio 30 `
  -Splits 0,1,2,3,4,5,6,7,8,9
```

Valid method names are `ReMasker`, `PMAE`, `CACTI`, `GAIN`, `MIWAE`, `DSAN`,
`DeepIFSAC`, `MICE`, and `MissForest`. The launcher passes the Appendix values
explicitly, so local defaults from an upstream repository cannot silently
change the experiment.

Add `-DryRun` to print the fully resolved command without starting training.

CACTI additionally requires its column-context embeddings. Prepare them once:

```powershell
$env:PYTHONPATH = ".baseline_sources/cacti"
python baselines/CACTI/prepare_fair_column_embeddings.py `
  --data-root data `
  --output-root .baseline_assets/cacti_embeddings
```

## Mixed-type handling

| Method | Categorical input and prediction |
|---|---|
| ReMasker | One-hot groups; group-wise argmax |
| PMAE | Integer scalar with train-observed Min-Max scaling; inverse scaling and nearest-integer decoding |
| CACTI | Ordinal scalar token with Min-Max scaling; inverse scaling, rounding, and clipping |
| GAIN | Integer scalar with Min-Max scaling; inverse scaling and rounding |
| MIWAE | Training-vocabulary one-hot groups; group-wise argmax over Student-t imputations |
| DSAN | Per-column category embeddings; sigmoid for binary and argmax for multiclass variables |
| DeepIFSAC | Native per-column embeddings and cross-entropy heads; argmax decoding |
| MICE | R factors with pmm/logreg/polyreg/polr |
| MissForest | R factors with classification-forest prediction |

These wrappers are mixed-type benchmark adaptations where an original method
does not natively expose the required categorical output. They preserve the
official model architecture while making encoding and decoding explicit. All
methods are scored only after predictions are decoded to original logical
variables.

## Method-specific notes

- ReMasker uses mask ratio 0.5, four heads, MLP ratio 4, cosine scheduling,
  warm-up 40, and minimum learning rate 1e-5.
- PMAE uses proportional masking with `a=0.05` and `b=0.5`, four heads, MLP
  ratio 4, cosine scheduling, warm-up 40, and minimum learning rate 1e-5.
- CACTI uses width 64, encoder/decoder depth 10/4, eight heads, median-truncated
  copy masking at ratio 0.9, warm-up 50, minimum learning rate 5e-6, and
  gradient clipping 5.
- GAIN uses two hidden layers in each network, hint rate 0.9, alpha 100, and
  input noise sampled from U(0, 0.01).
- MIWAE uses two 128-unit hidden layers, a Student-t observation model, latent
  dimension `ceil(input_dim / 2)`, K=20 training samples, 1,000 validation
  imputation samples, and 10,000 final samples.
- DSAN uses representation width 32, eight heads, two hidden layers, dropout
  0.5, and 30% denoising corruption.
- DeepIFSAC uses width 32, depth 6, eight-head col-row attention, dropout 0.1,
  categorical/continuous CutMix rates 0.3/0.3, temperature 0.7, contrastive
  weight 0.5, and categorical/numerical denoising weights 1/10.
- MICE uses `m=1`, at most 10 iterations, and variable-specific
  pmm/logreg/polyreg/polr updates with `nnet.MaxNWts=100000`.
- MissForest uses at most 10 iterations, 100 trees,
  `mtry=floor(sqrt(p))`, replacement sampling, and sample fraction 1.
