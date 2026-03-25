"""
ols_primitives.py

Author:     Dogyoon Song
Created:    2025-09-25
Revised:    2026-03-25

Purpose:
    Finite-population data-generation utilities and basic ATE estimators used by
    the manuscript experiments.

Contents:
    1. Covariate, residual, and potential-outcome generators
    2. OLS / DiM / OLS-RA / Lei-Ding estimation primitives
    3. A small executable smoke test

Maintenance notes:
    - Keep experiment-specific orchestration in run_experiments_finite.py.
    - Keep generator helpers numerically simple and side-effect free whenever possible.
    - Keep returned arrays C-contiguous and float64 where that matters downstream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np



# =====================================================================
# 1. Data generation utilities
# =====================================================================

## --------------------------------------------------------------------
## 1-A. Data generation primitives
## --------------------------------------------------------------------

def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(None if seed is None else int(seed))


def generate_X(n: int, p: int, dist: str, rng: np.random.Generator,
    spike: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """
    Generate an (n x p) covariate matrix X, then column-center it and
    rescale each column to have Euclidean norm sqrt(n).
    Input params:
        * dist in {"gauss","t2","t1"}.
        * Optional 'spike' = (fraction, scale).
    Returns C-contiguous float64 array.
    """
    if dist == "gauss":
        Z = rng.standard_normal((n, p))
    elif dist == "t2":
        Z = rng.standard_t(df=2.0, size=(n, p))
    elif dist == "t1":
        Z = rng.standard_t(df=1.0, size=(n, p))
    else:
        raise ValueError(f"Unknown dist_X: {dist}")
    Z -= Z.mean(axis=0, keepdims=True)

    if spike is not None and spike[0] > 0 and spike[1] != 1.0:
        k = max(1, int(round(spike[0] * n)))
        idx = np.argsort(np.linalg.norm(Z, axis=1))[-k:]
        Z[idx, :] *= spike[1]

    col_norms = np.linalg.norm(Z, axis=0, keepdims=True) + 1e-12
    X = (Z * (math.sqrt(n) / col_norms)).astype(np.float64, copy=False)

    return np.ascontiguousarray(X)


def generate_typical_eps(n: int, dist: str, rng: np.random.Generator) -> np.ndarray:
    """
    Generate typical residuals, i.e., an (n x 1) 'noise' variable instance
    """
    if dist == "gauss":
        rng_out = rng.standard_normal(n)
    elif dist == "t2":
        rng_out = rng.standard_t(df=2.0, size=n)
    elif dist == "t1":
        rng_out = rng.standard_t(df=1.0, size=n)
    else:
        raise ValueError(f"Unknown eps dist: {dist}")

    rng_out -= rng_out.mean()    # enforce 1^T eps = 0 in finite sample
    return rng_out


def generate_worst_case_eps(X: np.ndarray, scale: float = 3.0,
                            rng: Optional[np.random.Generator] = None
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct worst-case residuals aligned with leverage:
        eps^(0) = eps   and     eps^(1) = scale * eps,
    where eps is proportional to (I - H)(h - mean(h)·1), with
        H = X (X^T X)^+ X^T
    and h = diag(H).

    This construction assumes a nontrivial residual direction orthogonal to both
    the intercept and the columns of X. In nearly interpolating regimes that
    direction can be trivial; in that case we raise a ValueError rather than
    silently returning an invalid vector.
    """
    n = X.shape[0]
    X = np.ascontiguousarray(X, dtype=np.float64)

    XtX_pinv = np.linalg.pinv(X.T @ X)
    XM = X @ XtX_pinv
    h = np.einsum("ij,ij->i", X, XM, optimize=True)

    # Use the actual average leverage, not p/n, so the construction remains
    # correct when X is rank-deficient.
    v = h - float(np.mean(h)) * np.ones(n, dtype=np.float64)
    e = v - X @ (XtX_pinv @ (X.T @ v))
    norm_e = float(np.linalg.norm(e))

    if norm_e < 1e-12:
        rng = rng or np.random.default_rng()
        e = rng.standard_normal(n)
        e -= e.mean()
        e = e - X @ (XtX_pinv @ (X.T @ e))
        norm_e = float(np.linalg.norm(e))
        if norm_e < 1e-12:
            raise ValueError(
                "generate_worst_case_eps could not find a nonzero residual direction "
                "orthogonal to both the intercept and X. This typically occurs in "
                "interpolating regimes where rank(X) is close to n-1."
            )

    eps0 = math.sqrt(n) * (e / norm_e)
    eps1 = float(scale) * eps0
    return eps1, eps0


def complete_randomization_assignments(n: int, n1: int, N: int,
                                       rng: np.random.Generator) -> np.ndarray:
    """
    Return an (N x n) matrix of arm indicators (rows), with |S1|=n1 per row.
        * n:    population size
        * N:    # instances
    """
    T = np.zeros((N, n), dtype=np.int8)
    for j in range(N):
        idx = rng.choice(n, size=n1, replace=False)
        T[j, idx] = 1

    return T

## --------------------------------------------------------------------
## 1-B. Potential outcome generators
## --------------------------------------------------------------------

def make_potential_outcomes(X: np.ndarray, beta1: np.ndarray, beta0: np.ndarray,
                            eps1: np.ndarray, eps0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    X:      n x p matrix
    beta:   p x 1 vector
    y:      n x 1 vector
    """
    y1 = X @ beta1 + eps1
    y0 = X @ beta0 + eps0
    return y1, y0


def random_unit_beta(p: int, rng: np.random.Generator) -> np.ndarray:
    """
    Draw a random direction on the unit sphere S^{p-1}.
    """
    v = rng.standard_normal(p)
    nrm = float(np.linalg.norm(v)) + 1e-12
    return v / nrm


def make_potential_outcomes_with_signal(
        X: np.ndarray,
        snr: float,
        rng: np.random.Generator,
        eps_type: str = "typical",  # "typical" or "worst"
        eps_dist: str = "gauss",    # "gauss", "t2", or "t1"
        eps_scale: float = 3.0,     # only used for "worst"
        same_eps: bool = True,
        eps_ratio: float = 1.0,     # ratio of eps1 / eps0; only used for "typical"
        betas_dir: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        same_beta: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Typical-case generator with controlled SNR:
        y^{(a)} = X beta^{(a)} + eps^{(a)},
        ||beta^{(a)}||^2 = snr, and eps^{(a)} is drawn from the requested noise model.

    Returns (y1, y0, beta1, beta0).

    If betas_dir is provided, it must be a pair (beta1_dir, beta0_dir) of
    direction vectors; each is then scaled to the requested signal-to-noise ratio.
    """
    n, p = X.shape
    if betas_dir is None:
        beta0_dir = random_unit_beta(p, rng)
        beta1_dir = beta0_dir.copy() if same_beta else (random_unit_beta(p, rng))
    else:
        beta1_dir, beta0_dir = betas_dir

    beta0 = math.sqrt(max(0.0, float(snr))) * beta0_dir
    beta1 = math.sqrt(max(0.0, float(snr))) * beta1_dir

    if eps_type == "typical":
        eps0 = generate_typical_eps(n, eps_dist, rng)
        eps1 = eps0.copy() if same_eps else (float(eps_ratio) * generate_typical_eps(n, eps_dist, rng))
    else:
        eps1, eps0 = generate_worst_case_eps(X, eps_scale, rng)

    y1, y0 = make_potential_outcomes(X, beta1, beta0, eps1, eps0)
    return y1, y0, beta1, beta0


## --------------------------------------------------------------------
## 1-C. Potential outcomes class
## --------------------------------------------------------------------

class PotentialOutcomes:
    """
    Class that contains the potential outcomes data of a finite population of size n
        X:          n x p matrix
        y_treated:  n x 1 vector
        y_control:  n x 1 vector
        obs_indic:  N x n vector in {0,1}^{N x n}, with each row sum being equal to n1
    """

    def __init__(self, X: np.ndarray, y_treated: np.ndarray, y_control: np.ndarray, obs_indicators: np.ndarray):
        self.X = X
        self.y_treated = y_treated
        self.y_control = y_control
        self.obs_indicators = obs_indicators
        self.covariate_trim = False
        self.delta = 0.0

    def trim_covariates(self, delta: float) -> None:
        """
        Columnwise winsorization at quantiles [delta/2, 1-delta/2], followed by
        re-centering and columnwise re-scaling to norm sqrt(n).
        Mutates self.X in place; does not return a value.
        """
        if delta <= 0.0:
            self.covariate_trim = False
            self.delta = 0.0
        else:
            n, p = self.X.shape
            X = np.array(self.X, dtype=np.float64, copy=True, order='C')
            lo = delta / 2.0
            hi = 1.0 - lo
            for j in range(p):
                qlo = np.quantile(X[:, j], lo)
                qhi = np.quantile(X[:, j], hi)
                X[:, j] = np.clip(X[:, j], qlo, qhi)
            X -= X.mean(axis=0, keepdims=True)
            col_norms = np.linalg.norm(X, axis=0, keepdims=True) + 1e-12
            X *= (math.sqrt(n) / col_norms)

            self.covariate_trim = True
            self.delta = delta
            self.X = np.ascontiguousarray(X, dtype=np.float64)

    def get_X(self):
        return self.X

    def get_y(self):
        return self.y_treated, self.y_control

    def get_T(self):
        return self.obs_indicators



# =====================================================================
# 2. OLS and ATE primitives
# =====================================================================

## --------------------------------------------------------------------
## 2-A. OLS with intercept
## --------------------------------------------------------------------

@dataclass
class OLSArmFit:
    mu_hat:     float       # intercept
    beta_hat:   np.ndarray  # slope
    resid:      np.ndarray  # residual
    n_arm:      int         # size of the arm
    xbar:       np.ndarray  # covariate mean


def _ols_with_intercept(y: np.ndarray, X: np.ndarray) -> OLSArmFit:
    m, p = X.shape
    Z = np.c_[np.ones(m, dtype=np.float64), np.asarray(X, dtype=np.float64, order='C')]
    y = np.asarray(y, dtype=np.float64, order='C')
    coef, *_ = np.linalg.lstsq(Z, y, rcond=None)
    mu_hat = float(coef[0]); beta_hat = coef[1:]
    resid = y - Z @ coef
    xbar = X.mean(axis=0)
    return OLSArmFit(mu_hat=mu_hat, beta_hat=beta_hat, resid=resid, n_arm=m, xbar=xbar)


## --------------------------------------------------------------------
## 2-B. Basic ATE estimators
## --------------------------------------------------------------------

def true_tau(y1: np.ndarray, y0: np.ndarray) -> float:
    return float(np.mean(y1) - np.mean(y0))

    
def difference_in_means(y_obs: np.ndarray, T: np.ndarray) -> float:
    """
    y_obs:  observed outcomes;  y_obs[ i ] = y^{(1)}[ i ] iff T[ i ] = 1
    T:      treatment indicators
    """
    return float(y_obs[T == 1].mean() - y_obs[T == 0].mean())


def ols_ra(y_obs: np.ndarray, T: np.ndarray, X: np.ndarray
           ) -> Tuple[float, Dict[int, OLSArmFit], Tuple[np.ndarray, np.ndarray]]:
    """
    Return (tau_hat, fits_by_arm, (idx1, idx0)).
        * tau_hat:      ATE estimate
        * fits_by_arm:  OLSArmFit1, OLSArmFit0
        * idx1/idx0:    unit indices treated (1) / controlled (0)
    """
    idx1 = np.nonzero(T == 1)[0]
    idx0 = np.nonzero(T == 0)[0]
    X1 = X[idx1, :]; y1 = y_obs[idx1]
    X0 = X[idx0, :]; y0 = y_obs[idx0]
    fit1 = _ols_with_intercept(y1, X1)
    fit0 = _ols_with_intercept(y0, X0)
    tau_hat = fit1.mu_hat - fit0.mu_hat
    return tau_hat, {1: fit1, 0: fit0}, (idx1, idx0)


## --------------------------------------------------------------------
## 2-C. Degree-0 bias correction in related work (by Lei-Ding, Biometrika 2021)
## --------------------------------------------------------------------

def _leverage_diag_population(X: np.ndarray) -> np.ndarray:
    """
    Return leverage scores h_i = x_i^T (X^T X)^+ x_i without materializing
    the n×n hat matrix.
    Uses a fast path only when X^T X happens to be close to n I_p.
    """
    X = np.asarray(X, dtype=np.float64, order='C')
    n, p = X.shape

    # Fast path only when X^T X happens to be close to n I_p.
    # This is not guaranteed by generate_X(), which rescales columns but does not
    # orthogonalize them. In that special case, h_i = ||x_i||^2 / n.
    XtX = X.T @ X
    if np.allclose(XtX, n * np.eye(p), rtol=1e-5, atol=1e-7):
        g = np.einsum('ij,ij->i', X, X, optimize=True)  # row-wise ||x_i||^2
        return g / float(n)

    # General path: still avoid H = X X^+
    # Use (X^T X)^+ which is p×p, then h_i = x_i^T M x_i where M = (X^T X)^+
    M = np.linalg.pinv(XtX)  # p×p
    XM = X @ M               # n×p
    return np.einsum('ij,ij->i', X, XM, optimize=True)  # row-wise dot


def lei_ding_debiased_ra(y_obs: np.ndarray, T: np.ndarray, X: np.ndarray,
                         h: Optional[np.ndarray] = None
                        ) -> Tuple[float, float, Dict[str, Any]]:
    """
    Lei–Ding debiased regression adjustment.
    Implements (using population leverage scores) the debiased estimator:
        tau_ra_db = tau_ra - ( n1/n0 * Delta0 - n0/n1 * Delta1 ),
    where Delta_a = (1/n_a) * sum_{i in S_a} h_i * e_{a,i} and h_i = leverage_i.

    Returns:
        tau_ra_db : float
            Lei–Ding debiased OLS-RA estimate.
        tau_ra : float
            The vanilla armwise OLS-RA estimate (intercept difference).
        info : dict
            Diagnostics with keys: {'Delta1','Delta0','h','idx1','idx0'}.
    """
    # Armwise OLS with intercept to get tau_ra and residuals
    tau_ra, fits, (idx1, idx0) = ols_ra(y_obs, T, X)
    e1 = fits[1].resid 
    e0 = fits[0].resid
    n1 = len(e1)
    n0 = len(e0)

    # Population leverages h_i = x_i^T (X^T X)^{-1} x_i
    if h is None:
        h = _leverage_diag_population(X)
        
    Delta1 = float(np.dot(h[idx1], e1) / float(n1))
    Delta0 = float(np.dot(h[idx0], e0) / float(n0))

    # Bias term and debiased estimator (match Lei–Ding R code)
    bias = (n1 / float(n0)) * Delta0 - (n0 / float(n1)) * Delta1
    tau_ra_db = float(tau_ra - bias)

    info = {"Delta1": Delta1, "Delta0": Delta0, "h": h, "idx1": idx1, "idx0": idx0}
    return tau_ra_db, tau_ra, info


# =====================================================================
# 3. Executable smoke test
# =====================================================================

def main():
    n = 100
    p = 2
    n1 = 30
    N = 5
    delta = 0.2

    rng = _rng(1)
    dist_X = "gauss"
    snr = 1.0
    eps_type = "typical"

    X = generate_X(n, p, dist=dist_X, rng=rng, spike=None)
    y1, y0, beta1, beta0 = make_potential_outcomes_with_signal(X, snr, rng, eps_type)
    T_mat = complete_randomization_assignments(n, n1, N, rng)

    PO = PotentialOutcomes(X, y1, y0, T_mat)
    PO.trim_covariates(delta)
    X_used = PO.get_X()

    T = T_mat[0, :]
    y_obs = T * y1 + (1 - T) * y0

    tau_hat, _, _ = ols_ra(y_obs, T, X_used)
    tau_ra_db, tau_ra, info = lei_ding_debiased_ra(y_obs, T, X_used)

    print("True ATE =", np.mean(y1) - np.mean(y0))
    print("DiM estimate =", difference_in_means(y_obs, T))
    print("OLS-RA estimate =", tau_hat)
    print("Lei_Ding estimate =", tau_ra_db, tau_ra)


if __name__ == "__main__":
    main()