# Finite-sample regression adjustment experiments

This repository contains the Python code used for the simulation studies in the paper on finite-sample analysis of regression adjustment in randomized experiments.

## Files

- `ols_primitives.py`  
  Data-generation utilities and basic ATE estimators (DiM, arm-wise OLS-RA, Lei–Ding debiased RA).
- `finite_swap_module.py`  
  Finite-swap geometry, swap-sensitivity calculations, and Monte Carlo estimators of the quantities used in the finite-sample bounds.
- `run_experiments_finite.py`  
  Experiment driver for the manuscript simulations.

## Requirements

- Python 3.10 or newer
- NumPy
- SciPy

Install the runtime dependencies with:

```bash
pip install numpy scipy
```

## Quick start

Run the default CLI experiment (`exp3`) with a small smoke-test configuration:

```bash
python run_experiments_finite.py \
  --which exp3 \
  --outdir ./results \
  --n 20 \
  --gammas 0.25 \
  --thetas 1.0 \
  --Nassign 3 \
  --BS 3 \
  --Bpair 3 \
  --Bi 2 \
  --Bcond 2 \
  --delta 0.05
```

The script writes CSV outputs to `./results`.

## Programmatic use

The repository is organized so that the experiment driver reuses the two helper modules.
The current CLI dispatches `exp3` directly, while the other experiments remain callable from Python.

Example:

```python
from pathlib import Path
import run_experiments_finite as ref

grid = ref.ExpGrid(n=50, rho=0.3, gammas=(0.25, 0.75, 1.25), seed=12345)
mc = ref.MCConfig(B_S=100, B_pair=50, Bi=10, Bcond=10, N_assign=50, delta=0.05)

ref.exp3_strong_signal_ra_bo(
    grid=grid,
    mc=mc,
    outdir=Path("./results"),
    ns=(50,),
    gammas=(0.25, 0.75, 1.25),
    thetas=(1.0, 2.0, 4.0),
    deltas=(0.05,),
    R=5,
)
```

## Notes

- The reveal order is always a permutation of treated **positions** `0, ..., n1-1`, not a permutation of all unit indices.
- The helper `generate_worst_case_eps(...)` is intended for classical, non-interpolating settings. In nearly interpolating regimes it may raise a `ValueError` if no nontrivial centered residual direction exists.
- The scripts are written to favor clarity and manuscript alignment over aggressive optimization.

## Suggested smoke tests

```bash
python -m py_compile ols_primitives.py finite_swap_module.py run_experiments_finite.py
python ols_primitives.py
python run_experiments_finite.py --which exp3 --outdir ./results_smoke --n 20 --gammas 0.25 --thetas 1.0 --Nassign 3 --BS 3 --Bpair 3 --Bi 2 --Bcond 2 --delta 0.05
```
