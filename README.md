# Finite-sample regression adjustment experiments

This repository contains the Python code used for the simulation studies in the paper **"Design-Based Finite-Sample Analysis for Regression Adjustment"**. The code implements oracle-setting Monte Carlo experiments for finite-sample confidence intervals under complete randomization, with a focus on difference-in-means (DiM) and arm-wise OLS regression adjustment (OLS--RA).

## Repository contents

- `ols_primitives.py`  
  Finite-population data-generation utilities and basic ATE estimators.

- `finite_swap_module.py`  
  Geometry, swap-sensitivity, and Monte Carlo utilities for the oracle quantities entering the finite-sample bounds.

- `run_experiments_finite.py`  
  Experiment drivers and a lightweight command-line interface.

## What the code computes

The manuscript studies finite-sample confidence intervals based on the oracle quantities $V^\star,\qquad R^\star,\qquad B^\star$.

The code supports the following experiment types.

### 1. DiM experiments
For DiM, the remaining-pool oracle quantities $V^\star$ and $R^\star$ admit closed forms under the reveal law used in the paper, and $B^\star=0$.  
Accordingly, the DiM finite-sample radius is computed from exact oracle quantities rather than Monte Carlo approximations.

### 2. OLS--RA experiments
For OLS--RA, the oracle quantities are approximated by Monte Carlo estimators $\widehat V^\star,\qquad \widehat R^\star,\qquad \widehat B^\star$.
The current implementation uses the **direct Monte Carlo estimate** of $\widehat R^\star$.  
An optional one-sided UCB correction is implemented in the helper code, but it is not used in the manuscript tables.

### 3. Diagnostic quantities
The code also records diagnostic quantities used in the appendix discussion, including

- $V_{\mathrm{PQV}}$: the pre-de-noising predictable-quadratic-variation diagnostic,
- $R_{\mathrm{swap}}$: the sampled raw-swap range diagnostic,
- $B_{\mathrm{emp}}$: empirical absolute bias over sampled assignments.

These are diagnostics tied to the implemented Monte Carlo pipeline; they are not alternative oracle parameters.

## Requirements

- Python 3.10 or newer
- NumPy
- SciPy

Install the runtime dependencies with:

```bash
pip install numpy scipy
