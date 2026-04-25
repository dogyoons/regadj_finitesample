# Finite-sample regression adjustment experiments

This repository contains the Python code used for the simulation studies in the paper **"Design-Based Finite-Sample Analysis for Regression Adjustment"**. The code implements oracle-setting Monte Carlo experiments for finite-sample confidence intervals under complete randomization, with a focus on difference-in-means (DiM) and arm-wise OLS regression adjustment (OLS--RA).

## Repository contents

- `ols_primitives.py`  
  Finite-population data-generation utilities and basic ATE estimators, including DiM, arm-wise OLS--RA, and related helpers.

- `finite_swap_module.py`  
  Swap-sensitivity, branch-aware geometry, and Monte Carlo utilities for the oracle quantities entering the finite-sample bounds.

- `run_experiments_finite.py`  
  Experiment drivers for the manuscript simulations, together with a lightweight CLI.

## What the code computes

The manuscript studies finite-sample confidence intervals based on oracle quantities
\[
V^\star,\qquad R^\star,\qquad B^\star.
\]

The code supports two main experiment types:

- **DiM experiments (Experiment 1).**  
  For DiM, the remaining-pool oracle quantities \(V^\star\) and \(R^\star\) admit closed forms under the reveal law used in the paper, so the finite-sample radius is computed from those exact oracle quantities.

- **OLS--RA experiments (Experiments 2 and 3).**  
  For OLS--RA, the oracle quantities are approximated by Monte Carlo estimators
  \[
  \widehat V^\star,\qquad \widehat R^\star,\qquad \widehat B^\star.
  \]
  The current implementation uses the **direct Monte Carlo estimate** of \(\widehat R^\star\); the optional one-sided UCB correction is implemented in the helper module but is not used in the reported manuscript tables.

The code also records several diagnostics used in the appendix discussion, including sampled predictable-quadratic-variation surrogates and sampled raw-swap ranges.

## Requirements

- Python 3.10 or newer
- NumPy
- SciPy

Install the runtime dependencies with:

```bash
pip install numpy scipy
```

## Quick start

### Command line

In the current version, the CLI path is set up for the OLS--RA experiment driver (`exp2`). A small smoke test is:

```bash
python run_experiments_finite.py \
  --which exp2 \
  --outdir ./results_smoke \
  --ns 25 \
  --gammas 0.25 \
  --R 1 \
  --Nassign 5 \
  --BS 3 \
  --Bpair 3 \
  --Bi 2 \
  --Bcond 2 \
  --Bj 2 \
  --delta 0.05
```

This writes CSV outputs to `./results_smoke`.

### Programmatic use

The repository is organized so that the experiment driver reuses the two helper modules. All experiment functions are callable directly from Python.

Example (Experiment 1 / DiM):

```python
from pathlib import Path
import run_experiments_finite as ref

grid = ref.ExpGrid(n=50, rho=0.3, gammas=(0.0,), seed=12345)
mc = ref.MCConfig(B_S=30, B_pair=30, Bi=10, Bcond=10, Bj=10, N_assign=500, delta=0.05)

ref.exp1_dim_validity_multi_delta(
    grid=grid,
    mc=mc,
    outdir=Path("./results_exp1"),
    n_grid=(10, 20, 40, 80),
    gamma_grid=(0.0,),
    R_outer=2,
    agg_width="mean",
    save_full=True,
)
```

Example (Experiment 3 / strong-signal OLS--RA):

```python
from pathlib import Path
import run_experiments_finite as ref

grid = ref.ExpGrid(n=50, rho=0.3, gammas=(0.25, 0.75, 1.25), seed=12345)
mc = ref.MCConfig(B_S=30, B_pair=30, Bi=10, Bcond=10, Bj=10, N_assign=200, delta=0.05)

ref.exp3_strong_signal_ra_bo(
    grid=grid,
    mc=mc,
    outdir=Path("./results_exp3"),
    ns=(50,),
    gammas=(0.25, 0.75, 1.25),
    thetas=(1.0, 2.0, 4.0),
    deltas=(0.05,),
    R=5,
)
```

## Output files

Depending on the experiment, the driver writes some or all of the following CSV files:

- `...__raw.csv` — per-assignment outputs
- `...__rep.csv` — per-outer-replicate summaries
- `...__summary.csv` — summaries aggregated across outer replicates
- `...__diagnostics.csv` — diagnostic summaries used in the appendix discussion
- `..._long__...csv` — optional detailed per-assignment logs for Experiment 1

## Notes

- Reveal orders are permutations of treated **positions** `0, ..., n1-1`, not permutations of all unit indices.
- `generate_worst_case_eps(...)` is intended for classical, non-interpolating settings. In nearly interpolating regimes it may raise a `ValueError` if no nontrivial centered residual direction exists.
- The code is written to stay close to the manuscript definitions and formulas; clarity and manuscript compatibility are prioritized over aggressive optimization.

## Suggested smoke checks

```bash
python -m py_compile ols_primitives.py finite_swap_module.py run_experiments_finite.py
python run_experiments_finite.py --which exp2 --outdir ./results_smoke --ns 25 --gammas 0.25 --R 1 --Nassign 5 --BS 3 --Bpair 3 --Bi 2 --Bcond 2 --Bj 2 --delta 0.05
```
