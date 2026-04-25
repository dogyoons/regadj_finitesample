"""
run_experiments_finite.py

Author:     Dogyoon Song
Created:    2025-09-27
Revised:    2026-04-25

Purpose:
    Monte Carlo drivers for the manuscript experiments on finite-sample
    confidence intervals based on finite-swap concentration.

Contents:
    1. Lightweight configuration dataclasses and helper utilities
    2. Experiment drivers exp1, exp2, exp3, and exp4
    3. A small CLI that dispatches exp1, exp2, and exp3

Maintenance notes:
    - Reuse ols_primitives.py and finite_swap_module.py rather than duplicating logic.
    - Reveal orders are permutations of treated positions 0, ..., n1 - 1.
    - Keep experiment-specific file naming stable because downstream analysis reads the CSV outputs.
"""

import argparse
import csv
import glob
import importlib
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from math import sqrt
from scipy.special import erfcinv
import numpy as np

# ---------------------------------------------------------------------
# Robust loaders for the two modules (filenames may carry prefixes)
# ---------------------------------------------------------------------

def _load_module_by_pattern(modname: str, pattern: str):
    """
    Try a normal import first. If that fails, search for matching files and load
    the best candidate deterministically.

    Priority:
      1) same-version sibling of this script
         (e.g. '(Mar25_v1) finite_swap_module.py'),
      2) exact basename (e.g. 'finite_swap_module.py'),
      3) same directory as this script,
      4) most recently modified file.
    """
    try:
        return importlib.import_module(modname)
    except Exception:
        pass

    roots = [os.getcwd(), os.path.dirname(__file__)]
    candidates = []
    for r in roots:
        candidates.extend(glob.glob(os.path.join(r, pattern)))

    if not candidates:
        raise ImportError(f"Could not import '{modname}' and no file matched pattern='{pattern}'.")

    exact_name = f"{modname}.py"
    sibling_name = os.path.basename(__file__).replace("run_experiments_finite.py", exact_name)

    candidates = sorted(
        {os.path.abspath(path) for path in candidates},
        key=lambda path: (
            os.path.basename(path) != sibling_name,
            os.path.basename(path) != exact_name,
            os.path.dirname(path) != os.path.dirname(__file__),
            -os.path.getmtime(path),
            path,
        ),
    )

    for path in candidates:
        spec = importlib.util.spec_from_file_location(modname, path)
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        sys.modules[modname] = mod  # required so sub-imports can find it
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        return mod

    raise ImportError(f"Could not import '{modname}' from any matched file.")


# Load the two modules (handles names like 'finite_swap_module.py')
fsm = _load_module_by_pattern("finite_swap_module", "*finite_swap_module.py")
op  = _load_module_by_pattern("ols_primitives", "*ols_primitives.py")


# ---------------------------------------------------------------------
# Mini-helper function
# ---------------------------------------------------------------------

def _z_from_delta(delta: float) -> float:
    """
    Two-sided Gaussian critical value z_{1-delta/2}.
    Uses scipy.special.erfcinv so all experiments handle arbitrary delta values
    consistently without hard-coded lookup tables.
    """
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must lie in (0,1); got {delta}.")
    return float(sqrt(2.0) * erfcinv(delta))

# ---------------------------------------------------------------------
# Config dataclasses (kept minimal and explicit)
# ---------------------------------------------------------------------

@dataclass
class MCConfig:
    # For B* (Appx04-Alg MCBias): number of assignments and pairs per assignment
    B_S: int = 30
    B_pair: int = 30
    lambda_mode: str = "max"  # 'gap'|'ratio'|'max'
    # For (V*,R*) (Appx04-Alg MCVarRange)
    Bi: int = 10
    Bcond: int = 10
    Bj: int =   10      # 0 enumerates all admissible controls J; positive values sample Bj controls
    # Monte Carlo over assignments for coverage summaries
    N_assign: int = 500
    # nominal level for CI
    delta: float = 0.05

@dataclass
class ExpGrid:
    n: int = 50
    rho: float = 0.3  # treatment fraction
    gammas: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
    seed: int = 12345
    # For exp2
    SNRs: Tuple[float, ...] = (0.0, 1.0, 2.0, 4.0, 8.0)
    align_modes: Tuple[str, ...] = ("random", "aligned")  # beta alignment with X

# ---------------------------------------------------------------------
# Small helpers (data generation that *reuses* ols_primitives)
# ---------------------------------------------------------------------

def _make_X(n: int, p: int, rng: np.random.Generator, dist: str = "gauss") -> np.ndarray:
    """
    Wrapper around ols_primitives.generate_X(..., dist=dist).
    The returned X is column-centered and each column is scaled to norm sqrt(n);
    it is not orthogonalized to satisfy X^T X = n I_p in general.
    """
    return op.generate_X(n, p, dist=dist, rng=rng)

def _make_residuals(X: np.ndarray, regime: str, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Typical regime: independent N(0,1) residuals in both arms.
    Worst-case regime: use ols_primitives.generate_worst_case_eps, scale treated by 3x.

    Note: the worst-case construction is intended for classical, non-interpolating
    designs and may raise a ValueError if no nontrivial centered residual direction exists.
    """
    n, p = X.shape
    if regime == "typical":
        e0 = op.generate_typical_eps(n, dist="gauss", rng=rng)
        e1 = op.generate_typical_eps(n, dist="gauss", rng=rng)
        return e1.astype(np.float64), e0.astype(np.float64)
    elif regime == "worst":
        eps1, eps0 = op.generate_worst_case_eps(X, scale=3.0, rng=rng)
        return eps1.astype(np.float64), eps0.astype(np.float64)
    else:
        raise ValueError("regime must be 'typical' or 'worst'")

def _make_signal_pair(X: np.ndarray, SNR: float, align: str, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (y1,y0) with the same linear signal in both arms (τ=0) plus independent noise.
      align='random'   : beta is a random unit vector
      align='aligned'  : beta aligns with the top right singular vector (largest σ)
    """
    n, p = X.shape
    if align == "random":
        beta_dir = op.random_unit_beta(p, rng)
    elif align == "aligned":
        # compute top right singular vector of X
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        beta_dir = Vt[0, :].astype(np.float64)
        beta_dir /= (np.linalg.norm(beta_dir) + 1e-12)
    else:
        raise ValueError("align must be 'random' or 'aligned'")
    beta = math.sqrt(max(0.0, float(SNR))) * beta_dir
    mu = X @ beta
    e0 = op.generate_typical_eps(n, dist="gauss", rng=rng).astype(np.float64)
    e1 = op.generate_typical_eps(n, dist="gauss", rng=rng).astype(np.float64)
    y0 = mu + e0
    y1 = mu + e1
    return y1, y0

# ---------------------------------------------------------------------
# Exp 1: DiM — FS vs Wald (Gaussian X), multi-δ, percentile envelopes,
#         exact remaining-pool oracle quantities (V*, R*), and
#         terminal-assignment diagnostics (V_emp, R_emp).
#         NOTE: DiM does not use X; we vary n only (γ-grid collapses to {0.0}).
# ---------------------------------------------------------------------
def exp1_dim_validity_multi_delta(
    grid: ExpGrid,
    mc: MCConfig,
    outdir: Path,
    deltas: Tuple[float, ...] = (0.01, 0.05, 0.10),
    perc_levels: Tuple[float, ...] = (0.005, 0.025, 0.05),
    n_grid: Tuple[int, ...] = (25, 100, 400),      # main-text cells; extend in appendix
    gamma_grid: Tuple[float, ...] = (0.0,),        # DiM ignores X; fix γ=0 ⇒ p=1
    x_dist: str = "gauss",
    R_outer: int = 1,                              # outer replications of (X,y)
    agg_width: str = "median",                     # "median" or "mean" across assignments (inner)
    save_full: bool = False,                       # write per-assignment "long" CSV
    run_tag: Optional[str] = None,
) -> Path:
    assert agg_width in ("median", "mean")
    rng0 = np.random.default_rng(grid.seed + 1101)
    outdir.mkdir(parents=True, exist_ok=True)

    def _neyman_var_dim(S1: np.ndarray, y1: np.ndarray, y0: np.ndarray) -> float:
        n = len(y1)
        mask = np.zeros(n, dtype=bool); mask[S1] = True
        obs = np.where(mask, y1, y0)
        yT, yC = obs[mask], obs[~mask]
        sT2 = float(np.var(yT, ddof=1)) if len(yT) > 1 else 0.0
        sC2 = float(np.var(yC, ddof=1)) if len(yC) > 1 else 0.0
        nT, nC = max(len(yT), 1), max(len(yC), 1)
        return sT2 / nT + sC2 / nC

    # Closed-form terminal-assignment diagnostic for DiM.
    # This is not the same object as the oracle (V*, R*) under the remaining-pool reveal law;
    # it is an empirical surrogate used for Table 2 diagnostics.
    def _R_emp_dim(
        S1: np.ndarray, Pi: np.ndarray, y1: np.ndarray, y0: np.ndarray,
        n: int, n1: int, return_mu: bool = False
    ) -> Union[float, Tuple[float, np.ndarray]]:
        n0 = n - n1
        mask = np.zeros(n, dtype=bool); mask[S1] = True
        S0 = np.where(~mask)[0]
        if S0.size == 0 or S1.size == 0:
            return (0.0, np.zeros(n1, dtype=float)) if return_mu else 0.0

        # RB mean over controls for each treated i  (μ_i in the order of S1, length n1)
        y1_S0_mean = float(np.mean(y1[S0])); y0_S0_mean = float(np.mean(y0[S0]))
        mu = (y1_S0_mean - y1[S1]) / n1 - (y0[S1] - y0_S0_mean) / n0  # shape (n1,)

        # α_t table and reveal positions *in S1 indexing* (Π are positions 0..n1-1)
        alpha = np.array([fsm._alpha_t(n, n1, t) for t in range(1, n1+1)], dtype=float)
        pos_in_reveal = np.empty(n1, dtype=int)
        pos_in_reveal[Pi] = np.arange(1, n1+1, dtype=int)  # position of each S1-indexed unit

        # max over i∈S1 of α_{pos(i)} * |μ_i|
        vals = np.abs(mu) * alpha[pos_in_reveal - 1]
        R_emp = float(np.max(vals))
        return (R_emp, mu) if return_mu else R_emp

    def _dim_exact_vr(
        S1: np.ndarray,
        Pi: np.ndarray,
        y1: np.ndarray,
        y0: np.ndarray,
        n: int,
        n1: int,
    ) -> Tuple[float, float]:
        """
        Exact oracle (V*, R*) for DiM under the corrected remaining-pool law.

        For DiM, write
            a_i = y1_i / n1 + y0_i / n0.
        For a revealed past S_past and remaining pool R, the one-step drift is
            zeta_t(i) = E_J[a_J - a_i | i, S_past]
                      = (sum_{u in R} a_u - |R| a_i) / (|R|-1),
        with i uniform on R. This avoids generic swap Monte Carlo entirely.
        """
        n0 = n - n1
        score = y1 / float(n1) + y0 / float(n0)

        available = np.ones(n, dtype=bool)
        Vsum = 0.0
        Rmax = 0.0

        for t, pos in enumerate(Pi):
            m = int(np.sum(available))
            if m <= 1:
                break

            vals = score[available]
            mean_vals = float(np.mean(vals))
            zeta = (m / float(m - 1)) * (mean_vals - vals)

            alpha_t = fsm._alpha_t(n, n1, t + 1)
            Vsum += (alpha_t ** 2) * float(np.var(zeta, ddof=0))
            Rmax = max(Rmax, alpha_t * float(np.max(np.abs(zeta))))

            # Reveal the actual treated unit for the next step.
            available[int(S1[int(pos)])] = False

        return float(Vsum), float(Rmax)

    # filename tag
    def _tag(vals, prefix): return prefix + "-".join(str(v).replace(".", "p") for v in vals)
    default_tag = "_".join([f"X{x_dist}", _tag(n_grid, "n"), _tag(gamma_grid, "g"), f"seed{grid.seed}"])
    tag = (run_tag if (run_tag is not None and len(run_tag) > 0) else default_tag)

    # optional long file
    long_writer = None
    if save_full:
        long_fields = (["outer_rep","n","gamma","p","assign_idx","tau_true","V","R","R_emp","V_emp"]
                       + [f"rad_fs_d{int(100*d):03d}" for d in deltas]
                       + [f"rad_wald_d{int(100*d):03d}" for d in deltas]
                       + ["err"])
        long_tag = f"{tag}_R{int(R_outer)}_N{int(mc.N_assign)}_Bi{int(mc.Bi)}_Bcond{int(mc.Bcond)}"
        lf = open(outdir / f"exp1_dim_validity_long__{long_tag}.csv", "w", newline="")
        long_writer = csv.DictWriter(lf, fieldnames=long_fields)
        long_writer.writeheader()

    outer_rows: Dict[Tuple[int,float], List[Dict[str, Any]]] = {}

    for r in range(int(R_outer)):
        rng = np.random.default_rng(rng0.integers(0, 2**63-1))
        for n in n_grid:
            n1 = int(round(grid.rho * n))
            for gamma in gamma_grid:  # single value γ=0.0 by default
                p = int(math.ceil(n ** float(gamma)))  # = 1 when γ=0
                X = _make_X(n, p, rng, dist=x_dist)
                y1_res, y0_res = _make_residuals(X, regime="typical", rng=rng)
                y1 = y1_res.copy()
                y0 = y0_res.copy()

                tau_true = float(np.mean(y1) - np.mean(y0))

                V_list: List[float] = []; R_list: List[float] = []
                R_emp_list: List[float] = []; V_emp_list: List[float] = []
                err_list: List[float] = []
                cov_fs = {d: [] for d in deltas}; cov_wald = {d: [] for d in deltas}
                rad_fs = {d: [] for d in deltas}; rad_wald = {d: [] for d in deltas}

                for a in range(int(mc.N_assign)):
                    S1 = rng.choice(n, size=n1, replace=False).astype(int)
                    Pi = fsm.make_random_reveal_order(n1, rng)

                    # DiM admits an exact closed-form computation of the oracle
                    # (V*, R*) under the remaining-pool reveal law.
                    V, R = fsm.estimate_VR_DIM_exact(S1, X, y1, y0, Pi, rng=None)
                    V = float(V)
                    R = float(R)
                    V_list.append(V)
                    R_list.append(R)

                    # Separately compute terminal-assignment empirical diagnostics.
                    # These are useful benchmarks but are not identical to (V*, R*).
                    R_emp, mu = _R_emp_dim(S1, Pi, y1, y0, n=n, n1=n1, return_mu=True)
                    R_emp = float(R_emp)

                    mu_ord = mu[Pi]
                    m = int(mu_ord.size)
                    k_tail = np.arange(m, 0, -1, dtype=float)
                    s1_tail = np.cumsum(mu_ord[::-1])[::-1]
                    s2_tail = np.cumsum((mu_ord * mu_ord)[::-1])[::-1]
                    mean_tail = s1_tail / k_tail
                    var_tail = np.maximum(s2_tail / k_tail - mean_tail * mean_tail, 0.0)

                    alpha_vec = np.array(
                        [fsm._alpha_t(n, n1, tt) for tt in range(1, n1 + 1)],
                        dtype=float,
                    )
                    V_emp = float(np.sum((alpha_vec ** 2) * var_tail))

                    R_emp_list.append(R_emp)
                    V_emp_list.append(V_emp)

                    err = float(fsm.tau_hat_DIM(S1, y1, y0) - tau_true)
                    err_list.append(err)
                    ney = _neyman_var_dim(S1, y1, y0)

                    for d in deltas:
                        L = float(np.log(2.0 / d))
                        rfs = math.sqrt(2.0 * V * L) + (R / 3.0) * L
                        z = _z_from_delta(d); rwd = z * math.sqrt(max(ney, 0.0))
                        rad_fs[d].append(rfs); rad_wald[d].append(rwd)
                        cov_fs[d].append(float(abs(err) <= rfs))
                        cov_wald[d].append(float(abs(err) <= rwd))

                    if long_writer is not None:
                        rec_long = {"outer_rep": int(r), "n": int(n), "gamma": float(gamma), "p": int(p),
                                    "assign_idx": int(a), "tau_true": tau_true,
                                    "V": V, "R": R, "R_emp": R_emp, "V_emp": V_emp, "err": err}
                        for d in deltas:
                            rec_long[f"rad_fs_d{int(100*d):03d}"] = rad_fs[d][-1]
                            rec_long[f"rad_wald_d{int(100*d):03d}"] = rad_wald[d][-1]
                        long_writer.writerow(rec_long)

                V_arr = np.asarray(V_list, dtype=float)
                R_arr = np.asarray(R_list, dtype=float)
                R_emp_arr = np.asarray(R_emp_list, dtype=float)
                V_emp_arr = np.asarray(V_emp_list, dtype=float)
                err_arr = np.asarray(err_list, dtype=float)

                def _agg(v: np.ndarray) -> float:
                    return float(np.median(v)) if agg_width == "median" else float(np.mean(v))

                rec: Dict[str, Any] = {
                    "n": int(n), "gamma": float(gamma), "p": int(p),
                    "x_dist": x_dist, "Bi": int(mc.Bi), "Bcond": int(mc.Bcond), "N_assign": int(mc.N_assign),
                    "tau_true": tau_true,
                    "V_med": float(np.median(V_arr)), "V_mean": float(np.mean(V_arr)),
                    "V_med_inner_var": float(np.var(V_arr, ddof=0)),
                    "R_med": float(np.median(R_arr)), "R_mean": float(np.mean(R_arr)),
                    "R_med_inner_var": float(np.var(R_arr, ddof=0)),
                    "R_emp_med": float(np.median(R_emp_arr)),
                    "R_emp_inner_var": float(np.var(R_emp_arr, ddof=0)),
                    "V_emp_med": float(np.median(V_emp_arr)),
                    "V_emp_inner_var": float(np.var(V_emp_arr, ddof=0)),
                    "err_mean": float(np.mean(err_arr)), "err_sd": float(np.std(err_arr, ddof=1)),
                }
                for d in deltas:
                    rfs = np.asarray(rad_fs[d], dtype=float)
                    rwd = np.asarray(rad_wald[d], dtype=float)
                    wfs = 2.0 * rfs
                    wwd = 2.0 * rwd
                    rec[f"cov_fs_d{int(100*d):03d}"]         = float(np.mean(cov_fs[d]))      # mean across assignments
                    rec[f"cov_wald_d{int(100*d):03d}"]       = float(np.mean(cov_wald[d]))
                    rec[f"width_fs_med_d{int(100*d):03d}"]   = _agg(wfs)
                    rec[f"width_wald_med_d{int(100*d):03d}"] = _agg(wwd)
                    tag_d = f"d{int(100*d):03d}"
                    rec[f"width_fs_mean_{tag_d}"]  = float(np.mean(wfs))
                    rec[f"width_wald_mean_{tag_d}"] = float(np.mean(wwd))
                    rec[f"width_fs_var_{tag_d}"] = float(np.var(wfs, ddof=0))
                    rec[f"width_wald_var_{tag_d}"] = float(np.var(wwd, ddof=0))
                    rec[f"width_fs_p10_d{int(100*d):03d}"]   = float(np.quantile(wfs, 0.10))
                    rec[f"width_fs_p90_d{int(100*d):03d}"]   = float(np.quantile(wfs, 0.90))
                    rec[f"width_wald_p10_d{int(100*d):03d}"] = float(np.quantile(wwd, 0.10))
                    rec[f"width_wald_p90_d{int(100*d):03d}"] = float(np.quantile(wwd, 0.90))
                for q in perc_levels:
                    qlo = float(np.quantile(err_arr, q)); qhi = float(np.quantile(err_arr, 1.0-q))
                    rec[f"qlo_{int(1000*q):03d}"] = qlo; rec[f"qhi_{int(1000*q):03d}"] = qhi
                    rec[f"qwidth_{int(1000*q):03d}"] = qhi - qlo

                outer_rows.setdefault((n, float(gamma)), []).append(rec)

        print(f"[INFO] exp1-dim outer replicate {r+1}/{R_outer} done.")

    # aggregate across outer reps (coverage → mean; others → median if agg_width == 'median')
    rows: List[Dict[str, Any]] = []
    
    for (n, gamma), lst in sorted(outer_rows.items()):
        base = {"n": int(n), "gamma": float(gamma), "p": int(lst[0]["p"])}
        keys = [k for k in lst[0].keys() if k not in ("n", "gamma", "p")]

        def _outer_agg(key: str, vals: List[float]) -> float:
            if key.startswith("cov_"):
                return float(np.mean(vals))
            if (
                key.startswith("width_")
                or key.startswith("qwidth_")
                or key in {"V_med", "V_emp_med", "R_med", "R_emp_med"}
            ):
                return float(np.median(vals))
            return float(np.mean(vals))

        for k in keys:
            vals = [r[k] for r in lst if k in r]

            if all(isinstance(v, (int, float, np.integer, np.floating)) for v in vals):
                fvals = [float(v) for v in vals]
                base[k] = _outer_agg(k, fvals)

                if (
                    k.startswith("cov_")
                    or k.startswith("qwidth_")
                    or k in {"V_med", "V_emp_med", "R_med", "R_emp_med"}
                ):
                    base[f"{k}_outer_var"] = float(np.var(fvals, ddof=0))
            else:
                # keep the first (all reps should have the same metadata value)
                base[k] = vals[0]

        rows.append(base)

    # write summary CSV
    def _tag(vals, prefix): return prefix + "-".join(str(v).replace(".", "p") for v in vals)
    full_tag = (run_tag if run_tag else "_".join([f"X{x_dist}",
                                                  _tag(n_grid, "n"),
                                                  _tag(gamma_grid, "g"),
                                                  f"seed{grid.seed}",
                                                  f"R{R_outer}",
                                                  f"aw{agg_width[0]}"]))
    out_csv = outdir / f"exp1_dim_validity_multi__{full_tag}.csv"
    if rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    if save_full:
        lf.close()

    return out_csv


# ---------------------------------------------------------------------
# Experiment 2 (main text): OLS–RA finite-sample CI vs DiM Wald (Neyman)
# Multi-n, multi-γ, multiple δ, with R potential-outcome replicates.
# We report CI WIDTHS (not radii), coverage, percentile bands, and inner variance.
# ---------------------------------------------------------------------

def _fmt_hms(secs: float) -> str:
    """format seconds as HH:MM:SS (floor)"""
    secs = max(0.0, float(secs))
    return time.strftime("%H:%M:%S", time.gmtime(secs))


def _tag_list(prefix, vals, fmt="{:g}"):
    if isinstance(vals, (list, tuple)):
        s = "-".join(fmt.format(v) for v in vals)
    else:
        s = fmt.format(vals)
    return f"{prefix}{s}"

def _make_exp2_stem(ns, gammas, deltas, R, N, rho, Bi, Bcond, BS, Bpair, Bj=0):
    n_tag  = _tag_list("n", ns, "{:d}")
    g_tag  = _tag_list("g", gammas, "{:.2f}")
    d_tag  = _tag_list("d", deltas, "{:.2f}")
    budg   = f"Bi{Bi}_Bcond{Bcond}_Bj{Bj}_BS{BS}_Bpair{Bpair}"
    return f"exp2_ra_finite_ci__{n_tag}__{g_tag}__{d_tag}__R{int(R)}_N{int(N)}_rho{rho:.2f}__{budg}"


def _wald_neyman_width_dim(y1, y0, S1, delta):
    """
    Two-sided Wald WIDTH for DiM under realized assignment S1:
    width = 2 * z_{1-delta/2} * sqrt( s1^2/n1 + s0^2/n0 ).
    """
    n = int(y1.size); n1 = int(len(S1)); n0 = n - n1
    S1 = np.asarray(S1, dtype=int)
    S0 = np.setdiff1d(np.arange(n), S1, assume_unique=False)
    s1 = float(np.var(y1[S1], ddof=1)) if n1 > 1 else 0.0
    s0 = float(np.var(y0[S0], ddof=1)) if n0 > 1 else 0.0
    se = math.sqrt(max(s1 / max(n1, 1) + s0 / max(n0, 1), 0.0))
    z = _z_from_delta(float(delta))
    return float(2.0 * z * se)  # WIDTH (not radius)


def _wald_neyman_se_dim(y1, y0, S1):
    n  = int(y1.size)
    n1 = int(len(S1)); n0 = n - n1
    S1 = np.asarray(S1, dtype=int)
    S0 = np.setdiff1d(np.arange(n), S1, assume_unique=False)
    s1 = float(np.var(y1[S1], ddof=1)) if n1 > 1 else 0.0
    s0 = float(np.var(y0[S0], ddof=1)) if n0 > 1 else 0.0
    se = math.sqrt(max(s1 / max(n1, 1) + s0 / max(n0, 1), 0.0))
    return se


def exp2_ra_finite_ci(
    grid, mc, outdir,
    ns=(16,),                    # you can set (16,25,49) etc.
    gammas=(0.25,0.50,0.75,1.00,1.25,1.50),
    deltas=(0.05,),
    R=50,
    x_dist="gauss",
):
    """
    Experiment 2: RA finite-sample CIs vs DiM Wald.
    Writes:
      - one per-replicate CSV (rows: replicates within (n,gamma))
      - one across-replicate summary CSV (rows: (n,gamma))
    Filenames include the configuration tags to avoid overlap across runs.
    """
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    # ----- filename stem with settings -----
    stem = _make_exp2_stem(
        ns=tuple(int(x) for x in ns),
        gammas=tuple(float(g) for g in gammas),
        deltas=tuple(float(d) for d in deltas),
        R=R, N=int(mc.N_assign), rho=float(grid.rho),
        Bi=int(mc.Bi), Bcond=int(mc.Bcond), BS=int(mc.B_S),
        Bpair=int(mc.B_pair), Bj=int(mc.Bj)
    )
    rep_path = outdir / f"{stem}__rep.csv"
    sum_path = outdir / f"{stem}__summary.csv"
    diag_path = outdir / f"{stem}__diagnostics.csv"   # optional; comment out if not desired
    raw_path = outdir / f"{stem}__raw.csv"
    if raw_path.exists():
        raw_path.unlink()
    raw_header_written = False

    raw_base_cols = [
        "n", "gamma", "p", "rep", "assign", "n1",
        "B_RA", "Vhat_RA", "Rhat_RA", "Rswap_RA", "Vpqv_RA",
        "tau_resid_RA", "tau_resid_DIM", "se_DIM"
    ]
    raw_delta_cols = []
    for d in deltas:
        raw_delta_cols += [
            f"width_RA@{d}", f"covered_RA@{d}",
            f"width_DIM@{d}", f"covered_DIM@{d}"
        ]
    raw_fieldnames = raw_base_cols + raw_delta_cols

    # --- progress timers ---
    t0_all = time.time()

    # ---- main computation (unchanged core logic) ----
    rep_rows = []     # replicate-level
    groups = {}       # (n,gamma,p) -> list of replicate dicts for aggregation

    N = int(mc.N_assign)
    Bi, Bcond, Bj, BS, Bpair = (
        max(2, int(mc.Bi)),
        int(mc.Bcond),
        int(mc.Bj),
        int(mc.B_S),
        int(mc.B_pair),
    )
    rng0 = np.random.default_rng(grid.seed + 2202)

    for n in ns:
        for gamma in gammas:
            p = int(np.ceil(float(n) ** float(gamma)))
            rngX = np.random.default_rng(rng0.integers(2**61 - 1))
            X = _make_X(int(n), int(p), rngX, dist=x_dist)

            # per-cell timers
            t_cell_start = time.time()
            ema_rep = None; alpha = 0.3

            for rep in range(int(R)):
                t_rep_start = time.time()
                rng = np.random.default_rng(rng0.integers(2**61 - 1))

                y1, y0 = _make_signal_pair(X, SNR=1.0, align="random", rng=rng)
                tau_true = float(np.mean(y1) - np.mean(y0))
                n1 = int(round(grid.rho * n))

                # B* once per instance (RA)
                t_b0 = time.perf_counter()
                B_RA, EG_RA, Varf_RA = fsm.estimate_Bstar(
                    S1_size=n1, X=X, y1=y1, y0=y0, BS=BS, Bpair=Bpair, rng=rng,
                    method="RA", branch="auto", lambda_mode=str(mc.lambda_mode)
                )
                t_bstar = time.perf_counter() - t_b0

                # assignments + reveal orders
                S_list, Pi_list = [], []
                for _ in range(N):
                    S1 = rng.choice(int(n), size=n1, replace=False).astype(int)
                    S_list.append(S1)
                    Pi_list.append(fsm.make_random_reveal_order(n1, rng))

                # per-assignment (V*,R*) and residuals
                Vhat, Rhat = np.empty(N), np.empty(N)
                resid_RA, resid_DIM = np.empty(N), np.empty(N)
                Remp = np.empty(N)
                Vpqv = np.empty(N)

                # per-delta width/cov storage
                stats_RA = {d: {"w": np.empty(N), "c": np.empty(N, int)} for d in deltas}
                stats_DI = {d: {"w": np.empty(N), "c": np.empty(N, int)} for d in deltas}

                t_vr_total = 0.0
                for k in range(N):
                    S1, Pi = S_list[k], Pi_list[k]
                    t_v0 = time.perf_counter()
                    Vra, Rra, Vpq, Rswap = fsm.estimate_VR_for_assignment(
                        S1, X, y1, y0, Pi,
                        Bi=Bi, Bcond=Bcond, Bj=Bj,
                        rng=rng, method="RA", branch="auto",
                        return_rswap=True,
                    )
                    t_vr_total += time.perf_counter() - t_v0
                    Vhat[k], Rhat[k] = float(Vra), float(Rra)
                    Remp[k] = float(Rswap)   # sampled raw-swap range diagnostic
                    Vpqv[k] = float(Vpq)

                    th_ra  = float(fsm.tau_hat_RA(S1, X, y1, y0))
                    th_dim = float(fsm.tau_hat_DIM(S1, y1, y0))
                    resid_RA[k]  = th_ra  - tau_true
                    resid_DIM[k] = th_dim - tau_true

                    # compute DiM SE once; then widths scale with z
                    se_dim = _wald_neyman_se_dim(y1, y0, S1)

                    for d in deltas:
                        L = float(np.log(2.0 / float(d)))
                        w_ra  = 2.0 * (math.sqrt(2.0 * Vra * L) + (Rra / 3.0) * L + float(B_RA))
                        z     = _z_from_delta(float(d))
                        w_dim = 2.0 * z * se_dim
                        stats_RA[d]["w"][k], stats_DI[d]["w"][k] = w_ra, w_dim
                        stats_RA[d]["c"][k] = int(abs(resid_RA[k])  <= 0.5 * w_ra)
                        stats_DI[d]["c"][k] = int(abs(resid_DIM[k]) <= 0.5 * w_dim)

                # ---- write per-assignment RAW rows for this replicate ----
                assign_rows = []
                for k in range(N):
                    row = {
                        "n": int(n), "gamma": float(gamma), "p": int(p),
                        "rep": int(rep), "assign": int(k), "n1": int(n1),
                        "B_RA": float(B_RA),
                        "Vhat_RA": float(Vhat[k]), "Rhat_RA": float(Rhat[k]),
                        "Rswap_RA": float(Remp[k]), "Vpqv_RA": float(Vpqv[k]),
                        "tau_resid_RA": float(resid_RA[k]),
                        "tau_resid_DIM": float(resid_DIM[k]),
                        "se_DIM": float(_wald_neyman_se_dim(y1, y0, S_list[k])),
                    }
                    for d in deltas:
                        row[f"width_RA@{d}"]   = float(stats_RA[d]["w"][k])
                        row[f"covered_RA@{d}"] = int(stats_RA[d]["c"][k])
                        row[f"width_DIM@{d}"]  = float(stats_DI[d]["w"][k])
                        row[f"covered_DIM@{d}"]= int(stats_DI[d]["c"][k])
                    assign_rows.append(row)

                with open(raw_path, "a", newline="") as f_raw:
                    w_raw = csv.DictWriter(f_raw, fieldnames=raw_fieldnames)
                    if not raw_header_written:
                        w_raw.writeheader()
                        raw_header_written = True
                    w_raw.writerows(assign_rows)

                # replicate-level aggregates (unchanged)
                EmpVar_RA  = float(np.var(resid_RA, ddof=0))
                EmpVar_DIM = float(np.var(resid_DIM, ddof=0))
                IPR_RA     = float(np.quantile(resid_RA, 0.975) - np.quantile(resid_RA, 0.025))
                IPR_DIM    = float(np.quantile(resid_DIM, 0.975) - np.quantile(resid_DIM, 0.025))
                Bemp_abs   = float(abs(np.mean(resid_RA)))

                rec = {
                    "n": int(n), "gamma": float(gamma), "p": int(p), "rep": int(rep),
                    "N_assign": int(N), "rho": float(grid.rho),
                    "Bi": int(Bi), "Bcond": int(Bcond), "Bj": int(Bj),
                    "BS": int(BS), "Bpair": int(Bpair),
                    "B_RA": float(B_RA), "EGamma_RA": float(EG_RA), "Varf_RA": float(Varf_RA),
                    "EmpVar_RA": EmpVar_RA, "EmpVar_DIM": EmpVar_DIM, "IPR_RA": IPR_RA, "IPR_DIM": IPR_DIM,
                    "Bemp_absmean_RA": Bemp_abs,
                    "Rswap_RA_med": float(np.median(Remp)),
                    "Rswap_RA_var": float(np.var(Remp, ddof=0)),
                    "Vpqv_med_assign": float(np.median(Vpqv)),
                    "Vhat_med_assign": float(np.median(Vhat)),
                    "Rhat_med_assign": float(np.median(Rhat)),
                    "t_bstar": float(t_bstar),
                    "t_vr_total": float(t_vr_total),
                    "t_vr_per_assign": float(t_vr_total / max(N, 1)),
                }
                for d in deltas:
                    wra, wdi = stats_RA[d]["w"], stats_DI[d]["w"]
                    cra, cdi = stats_RA[d]["c"], stats_DI[d]["c"]
                    rec[f"mean_width_RA@{d}"]  = float(np.mean(wra))
                    rec[f"var_width_RA@{d}"]   = float(np.var(wra, ddof=0))
                    rec[f"mean_width_DIM@{d}"] = float(np.mean(wdi))
                    rec[f"var_width_DIM@{d}"]  = float(np.var(wdi, ddof=0))
                    rec[f"mean_cov_RA@{d}"]    = float(np.mean(cra))
                    rec[f"var_cov_RA@{d}"]     = float(np.var(cra, ddof=0))
                    rec[f"mean_cov_DIM@{d}"]   = float(np.mean(cdi))
                    rec[f"var_cov_DIM@{d}"]    = float(np.var(cdi, ddof=0))

                rep_rows.append(rec)
                groups.setdefault((int(n), float(gamma), int(p)), []).append(rec)

                # progress / ETA print (unchanged)
                dt_rep = time.time() - t_rep_start
                ema_rep = dt_rep if (ema_rep is None) else (0.3 * dt_rep + 0.7 * ema_rep)
                reps_done, reps_left = rep + 1, int(R) - (rep + 1)
                eta_cell  = ema_rep * reps_left
                elapsed_cell = time.time() - t_cell_start
                elapsed_all  = time.time() - t0_all
                print(
                    f"[exp2] n={int(n):4d}, gamma={float(gamma):.2f} | "
                    f"rep {reps_done:>2d}/{int(R):<2d} "
                    f"t_rep={_fmt_hms(dt_rep)}  avg={_fmt_hms(ema_rep)}  "
                    f"elapsed(cell)={_fmt_hms(elapsed_cell)}  ETA(cell)={_fmt_hms(eta_cell)}  "
                    f"elapsed(total)={_fmt_hms(elapsed_all)}",
                    flush=True
                )

    # ----- write per-replicate CSV (unchanged) -----
    if rep_rows:
        with open(rep_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rep_rows[0].keys()))
            w.writeheader(); w.writerows(rep_rows)

    # ----- aggregate across replicates (summary CSV) -----
    sum_rows = []
    for (n, gamma, p), rows in groups.items():
        row = {"n": n, "gamma": gamma, "p": p, "R": len(rows)}
        EmpVar_RA_all  = np.array([r["EmpVar_RA"] for r in rows], dtype=float)
        Remp_all = np.array([r["Rswap_RA_med"] for r in rows], dtype=float)
        IPR_RA_all     = np.array([r["IPR_RA"] for r in rows], dtype=float)
        IPR_DIM_all    = np.array([r["IPR_DIM"]   for r in rows], dtype=float)
        Bemp_abs_all   = np.array([r["Bemp_absmean_RA"] for r in rows], dtype=float)
        B_RA_all       = np.array([r["B_RA"] for r in rows], dtype=float)

        row["EmpVar_RA_med"] = float(np.median(EmpVar_RA_all))
        row["Rswap_RA_med"] = float(np.median(Remp_all))
        row["Rswap_RA_var"] = float(np.var(Remp_all, ddof=0))
        row["IPR_RA_med"]    = float(np.median(IPR_RA_all))
        row["IPR_RA_var"]    = float(np.var(IPR_RA_all, ddof=0))
        row["IPR_DIM_med"]   = float(np.median(IPR_DIM_all))
        row["IPR_DIM_var"]   = float(np.var(IPR_DIM_all, ddof=0))
        row["Bemp_absmean_RA_med"] = float(np.median(Bemp_abs_all))
        row["B_RA_med"] = float(np.median(B_RA_all))

        for d in deltas:
            mw_ra  = np.array([r[f"mean_width_RA@{d}"]  for r in rows], dtype=float)
            mw_di  = np.array([r[f"mean_width_DIM@{d}"] for r in rows], dtype=float)
            vw_ra  = np.array([r[f"var_width_RA@{d}"]   for r in rows], dtype=float)
            vw_di  = np.array([r[f"var_width_DIM@{d}"]  for r in rows], dtype=float)
            mc_ra  = np.array([r[f"mean_cov_RA@{d}"]    for r in rows], dtype=float)
            mc_di  = np.array([r[f"mean_cov_DIM@{d}"]   for r in rows], dtype=float)

            row[f"width_RA_medOfMeans@{d}"]  = float(np.median(mw_ra))
            row[f"width_DIM_medOfMeans@{d}"] = float(np.median(mw_di))
            row[f"width_RA_paren@{d}"]       = float(np.median(vw_ra))
            row[f"width_DIM_paren@{d}"]      = float(np.median(vw_di))
            row[f"cov_RA_medOfMeans@{d}"]    = float(np.median(mc_ra))
            row[f"cov_DIM_medOfMeans@{d}"]   = float(np.median(mc_di))
            row[f"cov_RA_paren@{d}"]         = float(np.var(mc_ra, ddof=0))
            row[f"cov_DIM_paren@{d}"]        = float(np.var(mc_di, ddof=0))
        sum_rows.append(row)

    if sum_rows:
        with open(sum_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sum_rows[0].keys()))
            w.writeheader(); w.writerows(sum_rows)

    # optional diagnostics CSV (unchanged)
    diag_rows = []
    for (n, gamma, p), rows in groups.items():
        diag_rows.append({
            "n": n, "gamma": gamma, "p": p, "R": len(rows),
            "Vhat_med_med": float(np.median([r["Vhat_med_assign"] for r in rows])),
            "Vhat_med_var": float(np.var([r["Vhat_med_assign"] for r in rows], ddof=0)),
            "Rhat_med_med": float(np.median([r["Rhat_med_assign"] for r in rows])),
            "Rhat_med_var": float(np.var([r["Rhat_med_assign"] for r in rows], ddof=0)),
            "Bhat_med": float(np.median([r["B_RA"] for r in rows])),
            "Bhat_var": float(np.var([r["B_RA"] for r in rows], ddof=0)),
            "Bemp_absmean_RA_med": float(np.median([r["Bemp_absmean_RA"] for r in rows])),
            "Bemp_absmean_RA_var": float(np.var([r["Bemp_absmean_RA"] for r in rows], ddof=0)),
            "EmpVar_RA_med": float(np.median([r["EmpVar_RA"] for r in rows])),
            "EmpVar_RA_var": float(np.var([r["EmpVar_RA"] for r in rows], ddof=0)),
            "Rswap_RA_med": float(np.median([r["Rswap_RA_med"] for r in rows])),
            "Rswap_RA_var": float(np.var([r["Rswap_RA_med"] for r in rows], ddof=0)),
            "Vpqv_med_med": float(np.median([r["Vpqv_med_assign"] for r in rows])),
            "Vpqv_med_var": float(np.var([r["Vpqv_med_assign"] for r in rows], ddof=0)),
        })
    if diag_rows:
        with open(diag_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(diag_rows[0].keys()))
            w.writeheader(); w.writerows(diag_rows)

    print(f"[INFO] exp2 | wrote:\n  {raw_path.name}\n  {rep_path.name}\n  {sum_path.name}\n  {diag_path.name}")
    return rep_path, sum_path, diag_path, raw_path




# ---------------------------------------------------------------------
# EXPERIMENT 3 (optional): B* decomposition diagnostics across γ
# ---------------------------------------------------------------------


def _make_exp3_stem(ns, gammas, thetas, deltas, R, N, rho, Bi, Bcond, BS, Bpair, Bj=0):
    n_tag  = _tag_list("n", ns, "{:d}")
    g_tag  = _tag_list("g", gammas, "{:.2f}")
    t_tag  = _tag_list("t", thetas, "{:.2f}")
    d_tag  = _tag_list("d", deltas, "{:.2f}")
    budg   = f"Bi{Bi}_Bcond{Bcond}_Bj{Bj}_BS{BS}_Bpair{Bpair}"
    return f"exp3_strong_signal__{n_tag}__{g_tag}__{t_tag}__{d_tag}__R{int(R)}_N{int(N)}_rho{rho:.2f}__{budg}"


def exp3_strong_signal_ra_bo(
    grid: ExpGrid,
    mc: MCConfig,
    outdir: Path,
    ns: Tuple[int, ...] = (25, 50, 100),
    gammas: Tuple[float, ...] = (0.25, 0.75, 1.25),
    thetas: Tuple[float, ...] = (1.0, 2.0, 4.0),
    deltas: Tuple[float, ...] = (0.05,),
    R: int = 50,
    x_dist: str = "gauss",
) -> Tuple[Path, Path, Path]:
    """
    Experiment 3: effect of stronger linear signal (θ ∈ {1,2,4}) on RA finite-sample widths.
    Records both standard RA–FS widths and the bias-omitted variant; compares to DiM Wald.
    Data format follows Experiment 2 with minimal additional fields.
    """
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    stem = _make_exp3_stem(
        ns=tuple(int(x) for x in ns),
        gammas=tuple(float(g) for g in gammas),
        thetas=tuple(float(t) for t in thetas),
        deltas=tuple(float(d) for d in deltas),
        R=R, N=int(mc.N_assign), rho=float(grid.rho),
        Bi=int(mc.Bi), Bcond=int(mc.Bcond), BS=int(mc.B_S),
        Bpair=int(mc.B_pair), Bj=int(mc.Bj)
    )
    rep_path = outdir / f"{stem}__rep.csv"
    sum_path = outdir / f"{stem}__summary.csv"
    raw_path = outdir / f"{stem}__raw.csv"
    diag_path = outdir / f"{stem}__diagnostics.csv"
    if raw_path.exists():
        raw_path.unlink()
    raw_header_written = False

    # --- raw schema = Exp.2 + (theta) + RAbo fields
    raw_base_cols = [
        "n", "gamma", "p", "theta", "rep", "assign", "n1",
        "B_RA", "Vhat_RA", "Rhat_RA", "Rswap_RA", "Vpqv_RA",
        "tau_resid_RA", "tau_resid_DIM", "se_DIM"
    ]
    raw_delta_cols = []
    for d in deltas:
        raw_delta_cols += [
            f"width_RA@{d}",   f"covered_RA@{d}",     # standard RA–FS (as in Exp.2)
            f"width_RAbo@{d}", f"covered_RAbo@{d}",   # new: bias-omitted RA
            f"width_DIM@{d}",  f"covered_DIM@{d}",
        ]
    raw_fieldnames = raw_base_cols + raw_delta_cols

    # budgets / constants
    N = int(mc.N_assign)
    Bi, Bcond, Bj, BS, Bpair = (
        max(2, int(mc.Bi)), int(mc.Bcond), int(mc.Bj), int(mc.B_S), int(mc.B_pair)
    )
    rng0 = np.random.default_rng(grid.seed + 3303)

    rep_rows = []
    # --- progress timers (for total elapsed / ETA) ---
    t0_all = time.time()
    groups: Dict[Tuple[int, float, int, float], list] = {}  # (n,γ,p,θ) → [rep dicts]

    for n in ns:
        n = int(n)
        n1 = int(round(grid.rho * n))
        for gamma in gammas:
            gamma = float(gamma)
            p = int(math.ceil(n ** gamma))
            for rep in range(int(R)):
                t_rep_start = time.time()
                rng = np.random.default_rng(rng0.integers(0, 2**63-1))
                # generate X once per (n,γ,rep)
                X = _make_X(n, p, rng, dist=x_dist)
                # per-cell timers (EMA over reps, like Exp 2)
                t_cell_start = time.time()
                ema_rep = None
                alpha = 0.30

                for theta in thetas:
                    theta = float(theta)
                    # y^(a) = θ X β* + ε^(a), with random unit β*: use SNR=θ^2, align='random'
                    y1, y0 = _make_signal_pair(X, SNR=theta * theta, align="random", rng=rng)
                    tau_true = float(np.mean(y1) - np.mean(y0))

                    # one B* (RA) per (X,y)
                    B_RA, EG_RA, Varf_RA = fsm.estimate_Bstar(
                        n1, X, y1, y0, BS=BS, Bpair=Bpair, rng=rng,
                        method="RA", branch="auto", lambda_mode=mc.lambda_mode
                    )

                    # pre-sample assignments and reveal orders
                    S_list, Pi_list = [], []
                    for _ in range(N):
                        S1 = rng.choice(n, size=n1, replace=False).astype(int)
                        S_list.append(S1)
                        Pi_list.append(fsm.make_random_reveal_order(n1, rng))

                    # arrays per assignment
                    Vhat = np.empty(N); Rhat = np.empty(N)
                    Remp = np.empty(N); Vpqv = np.empty(N)
                    resid_RA = np.empty(N); resid_DIM = np.empty(N)

                    # widths & coverage containers
                    stats_RA   = {d: {"w": np.empty(N), "c": np.empty(N, int)} for d in deltas}
                    stats_RAbo = {d: {"w": np.empty(N), "c": np.empty(N, int)} for d in deltas}
                    stats_DIM  = {d: {"w": np.empty(N), "c": np.empty(N, int)} for d in deltas}

                    for k in range(N):
                        # assignment-level heartbeat (every ~10% of N)
                        if k == 0:
                            t_chunk = time.time()
                            ema_chunk = None
                            k_step = max(1, N // 10)


                        S1, Pi = S_list[k], Pi_list[k]
                        Vra, Rra, Vpq = fsm.estimate_VR_for_assignment(
                            S1, X, y1, y0, Pi,
                            Bi=Bi, Bcond=Bcond, Bj=Bj,
                            rng=rng, method="RA", branch="auto"
                        )
                        Vra = float(Vra); Rra = float(Rra)
                        Vhat[k], Rhat[k], Vpqv[k] = Vra, Rra, float(Vpq)
                        Remp[k] = float(
                            fsm.compute_R_emp_RA(
                                S1, X, y1, y0, Pi,
                                Bcond=Bcond, rng=rng, branch="auto",
                                Bi_emp=Bi, Bj=Bj
                            )
                        )

                        th_ra  = float(fsm.tau_hat_RA(S1, X, y1, y0))
                        th_dim = float(fsm.tau_hat_DIM(S1, y1, y0))
                        resid_RA[k]  = th_ra  - tau_true
                        resid_DIM[k] = th_dim - tau_true

                        se_dim = _wald_neyman_se_dim(y1, y0, S1)

                        for d in deltas:
                            L = float(np.log(2.0 / float(d)))
                            # standard RA–FS WIDTH (as in Exp.2)
                            w_ra_fs = 2.0 * (math.sqrt(max(2.0 * Vra * L, 0.0)) + (Rra / 3.0) * L + float(B_RA))

                            # bias-omitted RA WIDTH
                            w_ra_bo = 2.0 * (math.sqrt(max(2.0 * Vra * L, 0.0)) + (Rra / 3.0) * L)
                            # DiM Wald WIDTH
                            z = _z_from_delta(float(d))
                            w_dim = 2.0 * z * se_dim

                            stats_RA[d]["w"][k]   = w_ra_fs
                            stats_RAbo[d]["w"][k] = w_ra_bo
                            stats_DIM[d]["w"][k]  = w_dim
                            stats_RA[d]["c"][k]   = int(abs(resid_RA[k])  <= 0.5 * w_ra_fs)
                            stats_RAbo[d]["c"][k] = int(abs(resid_RA[k])  <= 0.5 * w_ra_bo)
                            stats_DIM[d]["c"][k]  = int(abs(resid_DIM[k]) <= 0.5 * w_dim)

                        # progress heartbeat
                        if ((k + 1) % k_step == 0) or (k + 1 == N):
                            dt_chunk = time.time() - t_chunk
                            ema_chunk = dt_chunk if (ema_chunk is None) else (0.30 * dt_chunk + 0.70 * ema_chunk)
                            assigns_left = N - (k + 1)
                            # estimate ETA for remaining assignments based on EMA per chunk
                            eta_assign = (assigns_left / k_step) * (ema_chunk if ema_chunk is not None else dt_chunk)
                            pct = 100.0 * (k + 1) / float(N)
                            print(
                                f"[exp3] n={int(n):4d}, γ={float(gamma):.2f}, θ={float(theta):.2f} | "
                                f"assign {k + 1:>4d}/{int(N):<4d} ({pct:5.1f}%)  "
                                f"chunk={_fmt_hms(dt_chunk)}  ETA(assign)={_fmt_hms(eta_assign)}",
                                flush=True
                            )
                            t_chunk = time.time()



                    # ---- write per-assignment RAW rows ----
                    assign_rows = []
                    for k in range(N):
                        row = {
                            "n": int(n), "gamma": float(gamma), "p": int(p), "theta": float(theta),
                            "rep": int(rep), "assign": int(k), "n1": int(n1),
                            "B_RA": float(B_RA),
                            "Vhat_RA": float(Vhat[k]), "Rhat_RA": float(Rhat[k]),
                            "Rswap_RA": float(Remp[k]), "Vpqv_RA": float(Vpqv[k]),
                            "tau_resid_RA": float(resid_RA[k]),
                            "tau_resid_DIM": float(resid_DIM[k]),
                            "se_DIM": float(_wald_neyman_se_dim(y1, y0, S_list[k])),
                        }
                        for d in deltas:
                            row[f"width_RA@{d}"]    = float(stats_RA[d]["w"][k])
                            row[f"covered_RA@{d}"]  = int(stats_RA[d]["c"][k])
                            row[f"width_RAbo@{d}"]  = float(stats_RAbo[d]["w"][k])
                            row[f"covered_RAbo@{d}"]= int(stats_RAbo[d]["c"][k])
                            row[f"width_DIM@{d}"]   = float(stats_DIM[d]["w"][k])
                            row[f"covered_DIM@{d}"] = int(stats_DIM[d]["c"][k])
                        assign_rows.append(row)

                    with open(raw_path, "a", newline="") as f_raw:
                        w_raw = csv.DictWriter(f_raw, fieldnames=raw_fieldnames)
                        if not raw_header_written:
                            w_raw.writeheader(); raw_header_written = True
                        w_raw.writerows(assign_rows)

                    # replicate-level aggregates (same conventions as Exp.2)
                    EmpVar_RA  = float(np.var(resid_RA, ddof=0))
                    EmpVar_DIM = float(np.var(resid_DIM, ddof=0))
                    IPR_RA     = float(np.quantile(resid_RA, 0.975) - np.quantile(resid_RA, 0.025))
                    IPR_DIM    = float(np.quantile(resid_DIM, 0.975) - np.quantile(resid_DIM, 0.025))
                    Bemp_abs   = float(abs(np.mean(resid_RA)))  # empirical |mean bias| for table diagnostics

                    rec = {
                        "n": int(n), "gamma": float(gamma), "p": int(p), "theta": float(theta), "rep": int(rep),
                        "N_assign": int(N), "rho": float(grid.rho),
                        "Bi": int(Bi), "Bcond": int(Bcond), "BS": int(BS), "Bpair": int(Bpair),
                        "B_RA": float(B_RA), "EGamma_RA": float(EG_RA), "Varf_RA": float(Varf_RA),
                        "EmpVar_RA": EmpVar_RA, "EmpVar_DIM": EmpVar_DIM,
                        "IPR_RA": IPR_RA, "IPR_DIM": IPR_DIM,
                        "Bemp_absmean_RA": Bemp_abs,
                        "Rswap_RA_med": float(np.median(Remp)), "Rswap_RA_var": float(np.var(Remp, ddof=0)),
                        "Vpqv_med_assign": float(np.median(Vpqv)),
                        "Vhat_med_assign": float(np.median(Vhat)),
                        "Rhat_med_assign": float(np.median(Rhat)),
                    }
                    for d in deltas:
                        wra   = stats_RA[d]["w"];   cra   = stats_RA[d]["c"]
                        wrabo = stats_RAbo[d]["w"]; crabo = stats_RAbo[d]["c"]
                        wdi   = stats_DIM[d]["w"];  cdi   = stats_DIM[d]["c"]
                        rec[f"mean_width_RA@{d}"]    = float(np.mean(wra))
                        rec[f"var_width_RA@{d}"]     = float(np.var(wra, ddof=0))
                        rec[f"mean_width_RAbo@{d}"]  = float(np.mean(wrabo))
                        rec[f"var_width_RAbo@{d}"]   = float(np.var(wrabo, ddof=0))
                        rec[f"mean_width_DIM@{d}"]   = float(np.mean(wdi))
                        rec[f"var_width_DIM@{d}"]    = float(np.var(wdi, ddof=0))
                        rec[f"mean_cov_RA@{d}"]      = float(np.mean(cra))
                        rec[f"var_cov_RA@{d}"]       = float(np.var(cra, ddof=0))
                        rec[f"mean_cov_RAbo@{d}"]    = float(np.mean(crabo))
                        rec[f"var_cov_RAbo@{d}"]     = float(np.var(crabo, ddof=0))
                        rec[f"mean_cov_DIM@{d}"]     = float(np.mean(cdi))
                        rec[f"var_cov_DIM@{d}"]      = float(np.var(cdi, ddof=0))

                    rep_rows.append(rec)
                    groups.setdefault((int(n), float(gamma), int(p), float(theta)), []).append(rec)

                # --- per-replication progress / ETA (after finishing all θ for this rep) ---
                dt_rep = time.time() - t_rep_start
                ema_rep = dt_rep if (ema_rep is None) else (alpha * dt_rep + (1.0 - alpha) * ema_rep)
                reps_done = rep + 1
                reps_left = int(R) - reps_done
                eta_cell = ema_rep * reps_left
                elapsed_cell = time.time() - t_cell_start
                elapsed_all = time.time() - t0_all
                print(
                    f"[exp3] n={int(n):4d}, gamma={float(gamma):.2f} | "
                    f"rep {reps_done:>2d}/{int(R):<2d} "
                    f"t_rep={_fmt_hms(dt_rep)}  avg={_fmt_hms(ema_rep)}  "
                    f"elapsed(cell)={_fmt_hms(elapsed_cell)}  ETA(cell)={_fmt_hms(eta_cell)}  "
                    f"elapsed(total)={_fmt_hms(elapsed_all)}",
                    flush=True
                )

    # ----- write replicate CSV -----
    if rep_rows:
        with open(rep_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rep_rows[0].keys()))
            w.writeheader(); w.writerows(rep_rows)

    # ----- summary across replicates (median of means, as in Exp.2) -----
    sum_rows = []
    for (n, gamma, p, theta), rows in groups.items():
        row = {"n": n, "gamma": gamma, "p": p, "theta": theta, "R": len(rows)}
        EmpVar_RA_all  = np.array([r["EmpVar_RA"] for r in rows], dtype=float)
        IPR_RA_all     = np.array([r["IPR_RA"] for r in rows], dtype=float)
        IPR_DIM_all    = np.array([r["IPR_DIM"] for r in rows], dtype=float)
        Remp_all       = np.array([r["Rswap_RA_med"] for r in rows], dtype=float)
        Bemp_abs_all   = np.array([r["Bemp_absmean_RA"] for r in rows], dtype=float)

        row["EmpVar_RA_med"]  = float(np.median(EmpVar_RA_all))
        row["Rswap_RA_med"]    = float(np.median(Remp_all))
        row["Rswap_RA_var"]    = float(np.var(Remp_all, ddof=0))
        row["IPR_RA_med"]     = float(np.median(IPR_RA_all))
        row["IPR_RA_var"]     = float(np.var(IPR_RA_all, ddof=0))
        row["IPR_DIM_med"]    = float(np.median(IPR_DIM_all))
        row["IPR_DIM_var"]    = float(np.var(IPR_DIM_all, ddof=0))
        row["Bemp_absmean_RA_med"] = float(np.median(Bemp_abs_all))
        row["B_RA_med"] = float(np.median(np.array([r["B_RA"] for r in rows], dtype=float)))

        for d in deltas:
            mw_ra    = np.array([r[f"mean_width_RA@{d}"]    for r in rows], dtype=float)
            mw_rabo  = np.array([r[f"mean_width_RAbo@{d}"]  for r in rows], dtype=float)
            mw_dim   = np.array([r[f"mean_width_DIM@{d}"]   for r in rows], dtype=float)
            mc_ra    = np.array([r[f"mean_cov_RA@{d}"]      for r in rows], dtype=float)
            mc_rabo  = np.array([r[f"mean_cov_RAbo@{d}"]    for r in rows], dtype=float)
            mc_dim   = np.array([r[f"mean_cov_DIM@{d}"]     for r in rows], dtype=float)
            row[f"width_RA_medOfMeans@{d}"]    = float(np.median(mw_ra))
            row[f"width_RAbo_medOfMeans@{d}"]  = float(np.median(mw_rabo))
            row[f"width_DIM_medOfMeans@{d}"]   = float(np.median(mw_dim))
            row[f"cov_RA_medOfMeans@{d}"]      = float(np.median(mc_ra))
            row[f"cov_RAbo_medOfMeans@{d}"]    = float(np.median(mc_rabo))
            row[f"cov_DIM_medOfMeans@{d}"]     = float(np.median(mc_dim))
            # parentheses used in tables in Exp.2: keep consistent
            row[f"width_paren@{d}"]            = float(np.median(EmpVar_RA_all))
            row[f"cov_RA_paren@{d}"]           = float(np.var(mc_ra, ddof=0))
            row[f"cov_DIM_paren@{d}"]          = float(np.var(mc_dim, ddof=0))
        sum_rows.append(row)

    if sum_rows:
        with open(sum_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sum_rows[0].keys()))
            w.writeheader(); w.writerows(sum_rows)

    # ----- diagnostics (optional; mirrors Exp.2) -----
    diag_rows = []
    for (n, gamma, p, theta), rows in groups.items():
        diag_rows.append({
            "n": n, "gamma": gamma, "p": p, "theta": theta, "R": len(rows),
            "Vhat_med_med": float(np.median([r["Vhat_med_assign"] for r in rows])),
            "Vhat_med_var": float(np.var([r["Vhat_med_assign"] for r in rows], ddof=0)),
            "Rhat_med_med": float(np.median([r["Rhat_med_assign"] for r in rows])),
            "Rhat_med_var": float(np.var([r["Rhat_med_assign"] for r in rows], ddof=0)),
            "Bhat_med":     float(np.median([r["B_RA"] for r in rows])),
            "Bhat_var":     float(np.var([r["B_RA"] for r in rows], ddof=0)),
            "EmpVar_RA_med": float(np.median([r["EmpVar_RA"] for r in rows])),
            "EmpVar_RA_var": float(np.var([r["EmpVar_RA"] for r in rows], ddof=0)),
            "Rswap_RA_med":   float(np.median([r["Rswap_RA_med"] for r in rows])),
            "Rswap_RA_var":   float(np.var([r["Rswap_RA_med"] for r in rows], ddof=0)),
            "Vpqv_med_med":  float(np.median([r["Vpqv_med_assign"] for r in rows])),
            "Vpqv_med_var":  float(np.var([r["Vpqv_med_assign"] for r in rows], ddof=0)),
        })
    if diag_rows:
        with open(diag_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(diag_rows[0].keys()))
            w.writeheader(); w.writerows(diag_rows)

    print(f"[INFO] exp3 | wrote:\n  {raw_path.name}\n  {rep_path.name}\n  {sum_path.name}\n  {diag_path.name}")
    return rep_path, sum_path, diag_path


# ---------------------------------------------------------------------
# EXPERIMENT 4 (optional): Detection power under nonzero τ
# ---------------------------------------------------------------------

def _make_tau_shift(y1: np.ndarray, y0: np.ndarray, tau: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (y1', y0') with true τ = tau by shifting treated outcomes.
    """
    y1p = y1 + float(tau)
    return y1p, y0

def exp4_power_vs_alt(grid: ExpGrid, mc: MCConfig, outdir: Path,
                      tau_vals: Iterable[float] = (0.0, 0.3, 0.6, 1.0),
                      apply_ucb: bool = False,
                      eta_ucb: Optional[float] = None,
                      dmax_ucb: Optional[float] = None) -> Path:
    """
    For a fixed design (γ=1.0) and typical residuals, evaluate, for τ in tau_vals,
    the probability that the two-sided CI around τ̂ excludes 0 (as a crude power proxy).
    """
    rng = np.random.default_rng(grid.seed + 404)
    n = int(grid.n); n1 = int(round(grid.rho * n))
    p = int(math.ceil(n ** 1.0))
    X = _make_X(n, p, rng)
    y1_base, y0_base = _make_residuals(X, regime="typical", rng=rng)

    outdir.mkdir(parents=True, exist_ok=True)
    rows: list[Dict[str, Any]] = []
    Ldelta = float(np.log(2.0 / mc.delta))

    for tau in tau_vals:
        y1, y0 = _make_tau_shift(y1_base, y0_base, tau=tau)
        B_RA, EG_RA, Var_RA = fsm.estimate_Bstar(n1, X, y1, y0, BS=mc.B_S, Bpair=mc.B_pair,
                                                 rng=rng, method="RA", branch="auto", lambda_mode=mc.lambda_mode)

        excl0 = []           # indicator: 0 is outside the CI centered at τ̂
        cov_tau = []         # indicator: |τ̂ - τ| ≤ radius (should be ≥ 1 - δ if sharp)
        t_list = []          # record τ̂ values
        rads_plain = []      # per-assignment plain radius
        rads_eff = []        # per-assignment effective radius (UCB if applied)

        for _ in range(int(mc.N_assign)):
            S1 = rng.choice(n, size=n1, replace=False).astype(int)
            Pi = fsm.make_random_reveal_order(n1, rng)
            V, R, _ = fsm.estimate_VR_for_assignment(S1, X, y1, y0, Pi,
                                                  Bi=int(mc.Bi), Bcond=int(mc.Bcond),
                                                  rng=rng, method="RA", branch="auto")
            t = fsm.tau_hat_RA(S1, X, y1, y0)
            rad_plain = math.sqrt(2.0 * V * Ldelta) + (R / 3.0) * Ldelta + B_RA

            if apply_ucb:
                V_ucb, R_ucb, _ = fsm.estimate_VR_for_assignment(
                    S1, X, y1, y0, Pi,
                    Bi=int(mc.Bi), Bcond=int(mc.Bcond),
                    rng=rng, method="RA", branch="auto",
                    apply_ucb=True,
                    eta=(eta_ucb if eta_ucb is not None else mc.delta),
                    delta_max=dmax_ucb,  # reserved for future use; currently ignored in fsm
                )
                rad_eff = math.sqrt(2.0 * V_ucb * Ldelta) + (R_ucb / 3.0) * Ldelta + B_RA
            else:
                rad_eff = rad_plain

            # record per assignment
            rads_plain.append(float(rad_plain))
            rads_eff.append(float(rad_eff))
            t_list.append(float(t))
            excl0.append(float(abs(t - 0.0) > rad_eff))         # power proxy
            cov_tau.append(float(abs(t - float(tau)) <= rad_eff))  # coverage at τ

        excl0 = np.asarray(excl0, dtype=float)
        cov_tau = np.asarray(cov_tau, dtype=float)
        t_arr = np.asarray(t_list, dtype=float)
        rows.append({
            "tau": float(tau),
            "exclude0_rate": float(np.mean(excl0)),
            "cov_tau": float(np.mean(cov_tau)),
            "tau_hat_mean": float(np.mean(t_arr)),
            "tau_hat_sd": float(np.std(t_arr, ddof=1)),
            "B_RA": float(B_RA), "EGamma_RA": float(EG_RA), "Var_RA": float(Var_RA),
            "rad_plain_med": float(np.median(rads_plain)),
            "rad_ucb_med": float(np.median(rads_eff)),
        })
        print(f"[INFO] exp4 | τ={tau:.2f} | P[0 outside CI]≈{np.mean(excl0):.3f} | "
              f"cov@τ≈{np.mean(cov_tau):.3f} | rad_med≈{np.median(rads_eff):.3e}")

    out_csv = outdir / "exp4_power_vs_alt.csv"
    if rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    return out_csv


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def _parse_which(s: str) -> Tuple[str, ...]:
    s = (s or "").strip()
    if not s:
        return ("exp1",)
    return tuple(x.strip().lower() for x in s.split(",") if x.strip())


def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Run finite-swap Monte Carlo experiments (RA/DiM).")
    ap.add_argument("--which", type=str, default="exp1",
                    help="Comma-separated subset of {exp1,exp2,exp3}.")
    ap.add_argument("--ns", type=str, default="",
                    help="Comma-separated n values. Defaults are experiment-specific.")
    ap.add_argument("--outdir", type=str, default="./results", help="Output directory.")
    ap.add_argument("--seed", type=int, default=12345, help="Base RNG seed.")
    ap.add_argument("--n", type=int, default=50, help="Population size n.")
    ap.add_argument("--rho", type=float, default=0.3, help="Treatment fraction.")
    ap.add_argument("--gammas", type=str, default="",
                    help="Comma-separated gamma values for p=ceil(n^gamma). If omitted, each experiment uses its default grid.")
    ap.add_argument("--snrs", type=str, default="0.0,1.0,2.0",#,4.0,8.0,16.0",
                    help="Comma-separated SNR values retained for the programmatic Experiment 2 interface.")
    ap.add_argument("--thetas", type=str, default="1.0,2.0,4.0",
                    help="Comma-separated θ values for Experiment 3 (strong signal).")

    ap.add_argument("--R", type=int, default=20,
                    help="# outer finite-population replicates.")
    ap.add_argument("--Nassign", type=int, default=500, help="# assignments per design for coverage.")
    ap.add_argument("--BS", type=int, default=30, help="# assignments for B* (MC)")
    ap.add_argument("--Bpair", type=int, default=30, help="# (i,j) pairs per assignment for B* (MC)")
    ap.add_argument("--Bi", type=int, default=10, help="# candidate i's per step for (V,R) MC")
    ap.add_argument("--Bcond", type=int, default=10, help="# proxied completions T per i for (V,R) MC")
    ap.add_argument("--Bj", type=int, default=10,
                    help="# admissible controls J sampled per completion for RA MCVarRange; 0 enumerates all.")
    ap.add_argument("--delta", type=float, default=0.05, help="CI nominal miscoverage.")
    ap.add_argument("--apply_ucb", action="store_true",
                    help="Retained for the programmatic Experiment 4 interface; not used by the current CLI path.")
    ap.add_argument("--eta", type=float, default=None,
                    help="Retained for the programmatic Experiment 4 interface; not used by the current CLI path.")
    ap.add_argument("--delta_max", type=float, default=None,
                    help="Reserved for future envelope-based refinements; currently unused.")
    # ap.add_argument("--which", type=str, default="exp1",
    #             help="Comma-separated subset of {exp1,exp2,exp3,exp4}")
    args = ap.parse_args(list(argv) if argv is not None else None)

    outdir = Path(args.outdir)
    gammas = tuple(float(x) for x in args.gammas.split(",")) if args.gammas else ()
    ns_arg = tuple(int(x) for x in args.ns.split(",")) if getattr(args, "ns", "") else ()    
    snrs = tuple(float(x) for x in args.snrs.split(",")) if args.snrs else ()
    thetas = tuple(float(x) for x in args.thetas.split(",")) if getattr(args, "thetas", None) else ()
    which = set(_parse_which(args.which))
    supported = {"exp1", "exp2", "exp3"}
    unsupported = sorted(which - supported)
    if unsupported:
        raise ValueError(
            f"Unsupported experiment(s): {unsupported}. "
            f"Currently supported through the CLI: {sorted(supported)}."
        )

    grid = ExpGrid(n=int(args.n), rho=float(args.rho), gammas=gammas, SNRs=snrs, seed=int(args.seed))
    mc   = MCConfig(B_S=int(args.BS), B_pair=int(args.Bpair),
                    Bi=int(args.Bi), Bcond=int(args.Bcond), Bj=int(args.Bj),
                    N_assign=int(args.Nassign), delta=float(args.delta))
    
    apply_ucb  = bool(getattr(args, "apply_ucb", False))
    eta_ucb    = float(args.eta) if getattr(args, "eta", None) is not None else None
    dmax_ucb   = float(args.delta_max) if getattr(args, "delta_max", None) is not None else None


    # Run selected experiments

    if "exp1" in which:
        exp1_dim_validity_multi_delta(
            grid, mc, outdir,
            n_grid=ns_arg if ns_arg else (10, 20, 40, 80, 160, 320, 640),
            gamma_grid=(0.0,),
            R_outer=int(args.R),
            agg_width="mean",
            save_full=True,
        )

    if "exp2" in which:
        exp2_ra_finite_ci(
            grid, mc, outdir,
            ns=ns_arg if ns_arg else (50,),
            # gammas=gammas if gammas else (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5),
            gammas=gammas if gammas else (1.5,),
            deltas=(float(args.delta),),
            R=int(args.R),
            x_dist="gauss",
        )

    if "exp3" in which:
        exp3_strong_signal_ra_bo(
            grid, mc, outdir,
            ns=ns_arg if ns_arg else (int(args.n),),
            gammas=gammas if gammas else (0.25, 0.75, 1.25),
            thetas=thetas if thetas else (1.0, 2.0, 4.0),
            deltas=(float(args.delta),),
            R=int(args.R),
            x_dist="gauss",
        )


if __name__ == "__main__":
    main()
