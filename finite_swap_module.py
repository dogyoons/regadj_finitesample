"""
finite_swap_module.py

Author:     Dogyoon Song
Created:    2025-09-26
Revised:    2026-04-25

Purpose:
    Swap-sensitivity, geometry, and Monte Carlo utilities for the finite-sample
    bounds studied in the manuscript.

Contents:
    1. Geometry helpers for the M-branch and K-branch
    2. Exact swap-sensitivity utilities for RA and DiM
    3. Monte Carlo estimators of B*, R*, and V*
    4. Self-check / smoke-test routines

Maintenance notes:
    - Reuse estimators from ols_primitives.py when available.
    - The M-branch uses exact rank-one update formulas when the normal matrix is invertible.
    - The K-branch falls back to exact rebuilds whenever an update would be unstable.
"""


from __future__ import annotations

import itertools
import math
import time
from statistics import NormalDist
from typing import Literal, Optional, Tuple
import numpy as np

# Robust loader for optional ols_primitives, even when the filename is non-standard (e.g., "(Nov5_v1) ols_primitives.py")
def _load_ols_primitives_module():
    try:
        import ols_primitives as _op  # type: ignore
        return _op
    except Exception:
        pass

    import glob
    import importlib.util
    import os
    import sys

    # Search near this file and cwd for any "*ols_primitives.py".
    # If multiple prefixed copies exist, prefer the sibling with the same prefix.
    roots = []
    try:
        here = os.path.dirname(__file__)
        roots.append(here)
        roots.append(os.path.dirname(here))
    except Exception:
        pass
    roots.append(os.getcwd())

    candidates = []
    for r in roots:
        try:
            candidates.extend(glob.glob(os.path.join(r, "*ols_primitives.py")))
        except Exception:
            continue
    if not candidates:
        return None

    sibling_name = os.path.basename(__file__).replace("finite_swap_module.py", "ols_primitives.py")

    candidates = sorted(
        {os.path.abspath(path) for path in candidates},
        key=lambda path: (
            os.path.basename(path) != sibling_name,
            os.path.basename(path) != "ols_primitives.py",
            os.path.dirname(path) != os.path.dirname(__file__),
            -os.path.getmtime(path),
            path,
        ),
    )

    path = candidates[0]
    spec = importlib.util.spec_from_file_location("ols_primitives", path)
    if spec is None or spec.loader is None:
        return None

    mod = importlib.util.module_from_spec(spec)
    # Insert into sys.modules before execution for dataclass/type resolution.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


# Try to load optional helpers
try:
    _ols_mod = _load_ols_primitives_module()
except Exception:
    _ols_mod = None

if _ols_mod is not None:
    _ols_with_intercept_mbranch = getattr(_ols_mod, "_ols_with_intercept", None)
    _true_tau_population = getattr(_ols_mod, "true_tau", None)
else:
    _ols_with_intercept_mbranch = None
    _true_tau_population = None

_EPS = 1e-12    # Global tolerance threshold for numerical stability
_STD_NORMAL = NormalDist()


# =====================================================================
# 1. PART Helpers
# =====================================================================

## --------------------------------------------------------------------
## 1-A. Basic utilities
## --------------------------------------------------------------------

def _safe_pinv_sym(A: np.ndarray) -> np.ndarray:
    """
    Symmetric Moore–Penrose pseudoinverse with eigenvalue clipping.
    Falls back to np.linalg.pinv if eigh fails.
    """
    try:
        w, V = np.linalg.eigh(A)
        w_inv = np.where(w > _EPS, 1.0 / w, 0.0)
        return (V * w_inv) @ V.T
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A)


def _augment_design_with_intercept(X: np.ndarray) -> np.ndarray:
    """
    Augment design with an unpenalized intercept column.
    Z = [1, X] has shape (n, p+1).
    """
    n = int(X.shape[0])
    return np.column_stack([np.ones(n, dtype=float), X])


def _indices_complement(S1: np.ndarray, n: int) -> np.ndarray:
    S1 = np.asarray(S1, dtype=int).reshape(-1)
    mask = np.zeros(n, dtype=bool)
    mask[S1] = True
    return np.where(~mask)[0]


def _validate_S1(S1: np.ndarray, n: int) -> np.ndarray:
    S1 = np.asarray(S1, dtype=int).reshape(-1)
    if S1.ndim != 1:
        raise ValueError("S1 must be a 1D index array.")
    if np.any(S1 < 0) or np.any(S1 >= n):
        raise ValueError("S1 indices out of range.")
    if np.unique(S1).size != S1.size:
        raise ValueError("S1 contains duplicate indices.")
    return S1

## --------------------------------------------------------------------
## 1-B. Q-geometry helpers
## --------------------------------------------------------------------
# Branch meanings:
#   'M': classical regime (p < n). Geometry via augmented design hat Z=[1, X].
#   'K': interpolating regime (n <= p). Geometry via kernel K = X X^T.
# A small factory selects the branch; concrete classes are fully separate.

class MGeometry:
    """
    M-branch (classical) geometry for a given arm design X (shape: n×p, typically n > p).

    Provides two mathematically equivalent intercept computations:
      • mu_ols(y): standard OLS intercept (minimizes ||y - α1 - Xβ||_2).
         - By default calls ols_primitives._ols_with_intercept(y, X) if present,
           otherwise solves least squares on Z = [1, X].
      • mu_quotient(y): Proposition 1 quotient
           μ̂ = (1^T M y) / (1^T M 1),  with  M = I - X (X^T X)^+ X^T,
         implemented via a thin-QR projection (no M formed).

    Notes:
      - This class holds only X (and its shape). No Z, no inverses are stored here.
      - Part 2 caches will manage factorizations for no-refit updates.
    """
    __slots__ = ("X", "n", "p")

    def __init__(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("MGeometry: X must be 2D (n×p).")
        self.X = X
        self.n, self.p = X.shape

    @classmethod
    def from_X(cls, X: np.ndarray) -> "MGeometry":
        return cls(X)

    # ---- standard OLS intercept (definition) ----
    def mu_ols(self, y: np.ndarray) -> float:
        """
        OLS intercept via a direct fit of y on Z = [1, X].
        Uses ols_primitives._ols_with_intercept if available; otherwise lstsq(Z, y).
        """
        y = np.asarray(y, dtype=float)
        if y.shape[0] != self.n:
            raise ValueError("mu_ols: y length must equal n.")
        if _ols_with_intercept_mbranch is not None:
            fit = _ols_with_intercept_mbranch(y, self.X)
            if hasattr(fit, "mu_hat"):
                return float(getattr(fit, "mu_hat"))

        # Fallback: least squares on augmented design
        Z = _augment_design_with_intercept(self.X)
        beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
        return float(beta[0])

    # ---- quotient per Proposition 1 (Q = I - P_col(X)) ----
    def mu_quotient(self, y: np.ndarray) -> float:
        """
        Proposition 1 quotient:
            hat{mu} = (1^T M y) / (1^T M 1),
        with M = I - X (X^T X)^+ X^T = I - Q Q^T, where Q is an orthonormal
        basis for col(X).

        Implemented without forming M explicitly.
        """
        y = np.asarray(y, dtype=float)
        if y.shape[0] != self.n:
            raise ValueError("mu_quotient: y length must equal n.")
        X = self.X
        ones = np.ones(self.n, dtype=float)

        # Use a rank-revealing basis for col(X). Reduced QR is not sufficient when
        # X is rank-deficient, because it may include extra orthonormal directions
        # outside col(X). Thin SVD gives an exact basis after truncation.
        if self.p == 0:
            # No covariates; quotient reduces to the sample mean.
            return float(np.mean(y))

        U, s, _ = np.linalg.svd(X, full_matrices=False)
        keep = s > _EPS
        if not np.any(keep):
            return float(np.mean(y))
        Q = U[:, keep]

        z1 = Q.T @ ones
        zy = Q.T @ y
        denom = float(self.n - z1 @ z1)            # = 1^T (I - QQ^T) 1
        if denom <= _EPS:
            # 1 is (nearly) in col(X); revert to the definition (OLS) path.
            return self.mu_ols(y)
        num = float(ones @ y - z1 @ zy)            # = 1^T (I - QQ^T) y
        return num / denom


class KGeometry:
    """
    K-branch (interpolating) geometry for a given arm design X (shape: n×p, typically n <= p).

    Provides two intercept computations which coincide in this branch:
      • mu_ols(y): definition-faithful *primal* computation via thin SVD of X:
            minimize ||β||_2 subject to y = α·1 + Xβ,
        which reduces to α = argmin_α ||Σ^{-1} U^T (y - α1)||_2^2
        with X = U Σ V^T (thin SVD).
      • mu_quotient(y): Proposition 1 dual quotient
           hat{mu} = (1^T K^{-1} y) / (1^T K^{-1} 1),  with  K = X X^T,
        implemented by solve or pseudoinverse in the sample space (n×n).

    Implementation detail:
      - These two routes are mathematically equivalent but numerically independent,
        which enables meaningful cross-checks of the manuscript’s Proposition 1.
      - We do not store K^{-1} here. Instead, Part 2’s no-refit caches will maintain and
        update K^{-1} under insertions/deletions via Schur complements.
    """
    __slots__ = ("X", "n", "p")

    def __init__(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("KGeometry: X must be 2D (n×p).")
        self.X = X
        self.n, self.p = X.shape

    @classmethod
    def from_X(cls, X: np.ndarray) -> "KGeometry":
        return cls(X)

    def mu_ols(self, y: np.ndarray) -> float:
        """
        Primal (definition-faithful) computation via thin SVD X=U Σ V^T:
          minimize ||β||_2 subject to y = α·1 + Xβ
          => α = argmin_α ||Σ^{-1} U^T (y - α1)||_2^2
          => with a = Σ^{-1} U^T y, b = Σ^{-1} U^T 1, α = (b·a)/(b·b) if ||b||>0.

        Robust fallback (when b≈0): enforce the nullspace constraint
          (I - U U^T)(y - α1) = 0  ⇒  α = (h^T y)/(h^T 1), with h := (I - U U^T)1.
        If also h^T 1≈0 (pathological), use mean(y) as a last-resort stabilizer.
        """
        y = np.asarray(y, dtype=float)
        if y.shape[0] != self.n:
            raise ValueError("mu_ols (K): y length must equal n.")

        if self.n == 0:
            return 0.0

        # Thin SVD (n×p, n <= p): retain only nonzero singular directions.
        U, s, _ = np.linalg.svd(self.X, full_matrices=False)
        keep = s > _EPS
        if not np.any(keep):
            # X = 0; equality forces α = mean(y).
            return float(np.mean(y))

        U = U[:, keep]
        s = s[keep]
        ones = np.ones(self.n, dtype=float)
        Uy = U.T @ y
        U1 = U.T @ ones

        a = Uy / s             # Σ^{-1} U^T y on the row space of X
        b = U1 / s             # Σ^{-1} U^T 1 on the row space of X
        denom = float(b @ b)
        if denom > _EPS:
            return float((b @ a) / denom)

        # Nullspace fallback: enforce (I - U U^T)(y - α1) = 0.
        h = ones - U @ U1      # (I - U U^T) 1
        h1 = float(h @ ones)
        if abs(h1) > _EPS:
            return float((h @ y) / h1)

        # Extremely degenerate: return mean(y) as a safe stabilizer.
        return float(np.mean(y))


    def mu_quotient(self, y: np.ndarray) -> float:
        """
        Dual quotient in sample space: α = (1^T K^{+} y)/(1^T K^{+} 1),  K = X X^T.
        Implemented via SPD solve when possible, else symmetric pseudoinverse.
        A tiny ridge is added only if both solves are ill-conditioned.
        """
        y = np.asarray(y, dtype=float)
        if y.shape[0] != self.n:
            raise ValueError("mu_quotient (K): y length must equal n.")

        if self.n == 0:
            return 0.0

        X = self.X
        K = X @ X.T
        ones = np.ones(self.n, dtype=float)

        # Prefer direct solves (fast, stable when K is PD).
        try:
            z_y = np.linalg.solve(K, y)
            z_1 = np.linalg.solve(K, ones)
        except np.linalg.LinAlgError:
            Kp = _safe_pinv_sym(K)
            z_y = Kp @ y
            z_1 = Kp @ ones

        denom = float(ones @ z_1)
        if abs(denom) > _EPS:
            return float((ones @ z_y) / denom)

        # Small ridge (ridgeless limit)
        trK = float(np.trace(K))
        delta = 1e-8 * (trK / max(self.n, 1)) if trK > 0.0 else 1e-8
        z_y = np.linalg.solve(K + delta * np.eye(self.n), y)
        z_1 = np.linalg.solve(K + delta * np.eye(self.n), ones)
        denom = float(ones @ z_1)
        if abs(denom) > _EPS:
            return float((ones @ z_y) / denom)

        # Last resort (consistent when 1∈Null(X^T))
        return float(np.mean(y))


def build_geometry(X: np.ndarray, branch: Literal["M", "K", "auto"] = "auto") -> MGeometry | KGeometry:
    """
    Factory: build an M-branch geometry if n > p, otherwise K-branch.
    If branch is provided explicitly, obey it.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("build_geometry: X must be 2D (n×p).")
    n, p = X.shape
    if branch == "auto":
        branch = "M" if n > p else "K"
    return MGeometry.from_X(X) if branch == "M" else KGeometry.from_X(X)



# =====================================================================
# PART 2. Swap sensitivities
# =====================================================================

## --------------------------------------------------------------------
## 2-A. Definition-faithful sensitivities via model refits
## --------------------------------------------------------------------

def deletion_delta_mu_refit_M(X: np.ndarray, y: np.ndarray, idx: int) -> float:
    """Definition-faithful M-branch deletion delta: mu(X\\{i}, y\\{i}) - mu(X, y), using OLS refits."""
    mu_before = MGeometry.from_X(X).mu_ols(y)
    mu_after  = MGeometry.from_X(np.delete(X, idx, axis=0)).mu_ols(np.delete(y, idx))
    return float(mu_after - mu_before)


def insertion_delta_mu_refit_M(X: np.ndarray, y: np.ndarray, x_new: np.ndarray, y_new: float) -> float:
    """Definition-faithful M-branch insertion delta: mu([X;x], [y;y_new]) - mu(X, y), using OLS refits."""
    mu_before = MGeometry.from_X(X).mu_ols(y)
    X_plus = np.vstack([X, np.asarray(x_new, dtype=float).reshape(1, -1)])
    y_plus = np.concatenate([y, [float(y_new)]])
    mu_after  = MGeometry.from_X(X_plus).mu_ols(y_plus)
    return float(mu_after - mu_before)


def deletion_delta_mu_refit_K(X: np.ndarray, y: np.ndarray, idx: int) -> float:
    """Definition-faithful K-branch deletion delta via ridgeless OLS refits."""
    mu_before = KGeometry.from_X(X).mu_ols(y)
    mu_after  = KGeometry.from_X(np.delete(X, idx, axis=0)).mu_ols(np.delete(y, idx))
    return float(mu_after - mu_before)


def insertion_delta_mu_refit_K(X: np.ndarray, y: np.ndarray, x_new: np.ndarray, y_new: float) -> float:
    """Definition-faithful K-branch insertion delta via ridgeless OLS refits."""
    mu_before = KGeometry.from_X(X).mu_ols(y)
    X_plus = np.vstack([X, np.asarray(x_new, dtype=float).reshape(1, -1)])
    y_plus = np.concatenate([y, [float(y_new)]])
    mu_after  = KGeometry.from_X(X_plus).mu_ols(y_plus)
    return float(mu_after - mu_before)


## --------------------------------------------------------------------
## 2-B. M-branch helpers and cache-based shortcut
## --------------------------------------------------------------------

def _inv_rank1_add(Ainv: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Rank-1 update for (A + u u^T)^{-1} via Sherman–Morrison.
    Returns updated inverse or raises LinAlgError if the Schur denominator is tiny.
    """
    u = u.reshape(-1)
    Au = Ainv @ u
    denom = 1.0 + float(u @ Au)
    if abs(denom) <= _EPS:
        raise np.linalg.LinAlgError("rank-1 add: singular update")
    return Ainv - np.outer(Au, Au) / denom


def _inv_rank1_del(Ainv: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Rank-1 downdate for (A - u u^T)^{-1} via Sherman–Morrison.
    Returns updated inverse or raises LinAlgError if the Schur denominator is tiny.
    """
    u = u.reshape(-1)
    Au = Ainv @ u
    denom = 1.0 - float(u @ Au)
    if abs(denom) <= _EPS:
        raise np.linalg.LinAlgError("rank-1 del: singular update")
    return Ainv + np.outer(Au, Au) / denom


def _mu_M_from_stats(n: int, sumy: float, a: np.ndarray, b: np.ndarray, Ginv: np.ndarray) -> float:
    """
    Manuscript M-branch intercept via FWL quotient:
      mu = (1^T y - a^T Ginv b) / (n - a^T Ginv a),
      with a = X^T 1, b = X^T y, Ginv = (X^T X)^{-1}.
    """
    Ga = Ginv @ a
    Gb = Ginv @ b
    num = float(sumy - a @ Gb)
    den = float(n - a @ Ga)
    if abs(den) <= _EPS:
        # 1 is (near) in col(X); the quotient becomes unstable: caller should refit.
        return np.nan
    return num / den


class MArmCache:
    """
    M-branch no-refit cache (n>p). Stores only manuscript-native stats:
      Ginv = (X^T X)^{-1} (exact; if only pseudoinverse is possible, updates are disabled),
      a    = X^T 1,
      b    = X^T y,
      sumy = 1^T y,
      n    = |arm|,
      and the current X,y to replay updates.

    Methods (exact when updates enabled; otherwise rebuild on demand):
      • mu(): current intercept via FWL quotient (falls back to OLS if quotient unstable).
      • delete(idx): rank-1 downdate (X^T X)^{-1} and update (a,b,sumy,n).
      • insert(x,y): rank-1 update and stats update.
    """
    __slots__ = ("X", "y", "n", "p", "Ginv", "a", "b", "sumy", "_allow_updates")

    def __init__(self, X: np.ndarray, y: np.ndarray, Ginv: np.ndarray, a: np.ndarray, b: np.ndarray, sumy: float, allow_updates: bool):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.n, self.p = self.X.shape
        self.Ginv = np.asarray(Ginv, dtype=float)
        self.a = np.asarray(a, dtype=float)
        self.b = np.asarray(b, dtype=float)
        self.sumy = float(sumy)
        self._allow_updates = bool(allow_updates)

    @classmethod
    def build(cls, X: np.ndarray, y: np.ndarray) -> "MArmCache":
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        if X.shape[0] != y.shape[0]:
            raise ValueError("MArmCache.build: X and y must have the same number of rows.")
        n, p = X.shape
        # Core stats
        G = X.T @ X
        allow = True
        try:
            Ginv = np.linalg.inv(G)
        except np.linalg.LinAlgError:
            Ginv = np.linalg.pinv(G)  # pseudoinverse: valid for mu formula, but NOT for rank-1 inverse updates
            allow = False             # disable updates; fall back to refit when asked to update
        a = X.T @ np.ones(n, dtype=float)
        b = X.T @ y
        sumy = float(np.sum(y))
        return cls(X, y, Ginv, a, b, sumy, allow)

    def mu(self) -> float:
        mu = _mu_M_from_stats(self.n, self.sumy, self.a, self.b, self.Ginv)
        if not np.isnan(mu):
            return float(mu)
        # Fallback to a definition-faithful refit when the quotient is unstable
        return float(MGeometry.from_X(self.X).mu_ols(self.y))

    def delete(self, idx: int) -> "MArmCache":
        """
        Return a NEW cache after removing row idx. Exact rank-1 downdate when allowed; else rebuild (refit path).
        """
        x_i = self.X[idx, :].reshape(-1)
        y_i = float(self.y[idx])

        if self._allow_updates:
            try:
                Ginv_new = _inv_rank1_del(self.Ginv, x_i)
                a_new = self.a - x_i
                b_new = self.b - x_i * y_i
                sumy_new = self.sumy - y_i
                X_new = np.delete(self.X, idx, axis=0)
                y_new = np.delete(self.y, idx, axis=0)
                return MArmCache(X_new, y_new, Ginv_new, a_new, b_new, sumy_new, True)
            except np.linalg.LinAlgError:
                pass  # fall through to rebuild

        # Rebuild from scratch (definition-faithful statistics)
        X_new = np.delete(self.X, idx, axis=0)
        y_new = np.delete(self.y, idx, axis=0)
        return MArmCache.build(X_new, y_new)

    def insert(self, x_new: np.ndarray, y_new: float) -> "MArmCache":
        """
        Return a NEW cache after appending (x_new, y_new). Exact rank-1 update when allowed; else rebuild.
        """
        x_new = np.asarray(x_new, dtype=float).reshape(-1)
        y_new = float(y_new)

        if self._allow_updates:
            try:
                Ginv_new = _inv_rank1_add(self.Ginv, x_new)
                a_new = self.a + x_new
                b_new = self.b + x_new * y_new
                sumy_new = self.sumy + y_new
                X_new = np.vstack([self.X, x_new.reshape(1, -1)])
                y_new_vec = np.concatenate([self.y, [y_new]])
                return MArmCache(X_new, y_new_vec, Ginv_new, a_new, b_new, sumy_new, True)
            except np.linalg.LinAlgError:
                pass  # fall through to rebuild

        X_new = np.vstack([self.X, x_new.reshape(1, -1)])
        y_new_vec = np.concatenate([self.y, [y_new]])
        return MArmCache.build(X_new, y_new_vec)


## --------------------------------------------------------------------
## 2-C. K-branch helpers and cache-based shortcut
## --------------------------------------------------------------------

def _mu_K_from_Kinv_y(Kinv: np.ndarray, y: np.ndarray) -> float:
    """
    K-branch dual intercept: mu = (1^T Kinv y) / (1^T Kinv 1).
    """
    ones = np.ones(y.shape[0], dtype=float)
    Ky = Kinv @ y
    Ko = Kinv @ ones
    den = float(ones @ Ko)
    if abs(den) <= _EPS:
        return float(np.mean(y))  # last-resort stabilizer
    return float((ones @ Ky) / den)


class KArmCache:
    """
    K-branch no-refit cache (n≤p). Stores:
      Kinv = (X X^T)^{-1} (exact; if only pseudoinverse is possible, updates are disabled),
      X, y.

    Methods:
      • mu(): current intercept via dual quotient.
      • delete(idx): principal-minor update (Schur complement on Kinv).
      • insert(x,y): block-inverse update using k = X x_new, κ = ||x_new||^2.
    """
    __slots__ = ("X", "y", "n", "p", "Kinv", "_allow_updates")

    def __init__(self, X: np.ndarray, y: np.ndarray, Kinv: np.ndarray, allow_updates: bool):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.n, self.p = X.shape
        self.Kinv = np.asarray(Kinv, dtype=float)
        self._allow_updates = bool(allow_updates)

    @classmethod
    def build(cls, X: np.ndarray, y: np.ndarray) -> "KArmCache":
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        if X.shape[0] != y.shape[0]:
            raise ValueError("KArmCache.build: X and y must have the same number of rows.")
        K = X @ X.T
        allow = True
        try:
            Kinv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            Kinv = _safe_pinv_sym(K)
            allow = False      # disable updates; fall back to rebuild if we need to modify the arm
        return cls(X, y, Kinv, allow)

    def mu(self) -> float:
        return _mu_K_from_Kinv_y(self.Kinv, self.y)

    def delete(self, idx: int) -> "KArmCache":
        """
        Return a NEW cache after removing row idx. Exact principal-minor update when allowed; else rebuild.
        """
        n = self.n
        if n == 0:
            return KArmCache(self.X, self.y, self.Kinv, self._allow_updates)
        if n == 1:
            # Removing the only observation -> empty arm
            X_new = self.X[:0, :]
            y_new = self.y[:0]
            return KArmCache.build(X_new, y_new)

        if self._allow_updates:
            # Move idx to the last position to form a 2x2 block; avoid forming permutation matrices explicitly.
            order = [i for i in range(n) if i != idx] + [idx]
            Kinv_perm = self.Kinv[np.ix_(order, order)]
            alpha = float(Kinv_perm[-1, -1])
            beta = Kinv_perm[:-1, -1].reshape(-1, 1)
            Gamma = Kinv_perm[:-1, :-1]
            if abs(alpha) > _EPS:
                Kinv_new = Gamma - (beta @ beta.T) / alpha
                X_new = self.X[order[:-1], :]
                y_new = self.y[order[:-1]]
                return KArmCache(X_new, y_new, Kinv_new, True)
            # else fall through to rebuild

        # Rebuild if updates are not allowed or alpha≈0
        X_new = np.delete(self.X, idx, axis=0)
        y_new = np.delete(self.y, idx, axis=0)
        return KArmCache.build(X_new, y_new)

    def insert(self, x_new: np.ndarray, y_new: float) -> "KArmCache":
        """
        Return a NEW cache after appending (x_new, y_new). Exact block inverse when allowed; else rebuild.
        """
        x_new = np.asarray(x_new, dtype=float).reshape(-1)
        y_new = float(y_new)

        if self._allow_updates:
            k = (self.X @ x_new.reshape(-1, 1)).reshape(self.n)  # (n,)
            kappa = float(x_new @ x_new)
            t = self.Kinv @ k
            S = kappa - float(k @ t)
            if S > _EPS:
                invS = 1.0 / S
                Kinv_plus = np.empty((self.n + 1, self.n + 1), dtype=float)
                Kinv_plus[:-1, :-1] = self.Kinv + np.outer(t, t) * invS
                Kinv_plus[:-1, -1] = -t * invS
                Kinv_plus[-1, :-1] = (-t * invS).T
                Kinv_plus[-1, -1] = invS
                X_new = np.vstack([self.X, x_new.reshape(1, -1)])
                y_vec = np.concatenate([self.y, [y_new]])
                return KArmCache(X_new, y_vec, Kinv_plus, True)
            # else fall through to rebuild

        X_new = np.vstack([self.X, x_new.reshape(1, -1)])
        y_vec = np.concatenate([self.y, [y_new]])
        return KArmCache.build(X_new, y_vec)


## --------------------------------------------------------------------
## 2-D. Thin convenience wrappers for no-refit deltas
## --------------------------------------------------------------------

def deletion_delta_mu_norefit_M(X: np.ndarray, y: np.ndarray, idx: int) -> float:
    """M-branch: Δμ (delete idx) via cache-based exact rank‑1 downdate."""
    c = MArmCache.build(X, y)
    return float(c.delete(idx).mu() - c.mu())


def insertion_delta_mu_norefit_M(X: np.ndarray, y: np.ndarray, x_new: np.ndarray, y_new: float) -> float:
    """M-branch: Δμ (insert (x,y)) via cache-based exact rank‑1 update."""
    c = MArmCache.build(X, y)
    return float(c.insert(x_new, y_new).mu() - c.mu())


def deletion_delta_mu_norefit_K(X: np.ndarray, y: np.ndarray, idx: int) -> float:
    """K-branch: Δμ (delete idx) via cache-based principal-minor update."""
    c = KArmCache.build(X, y)
    return float(c.delete(idx).mu() - c.mu())


def insertion_delta_mu_norefit_K(X: np.ndarray, y: np.ndarray, x_new: np.ndarray, y_new: float) -> float:
    """K-branch: Δμ (insert (x,y)) via cache-based block-inverse update."""
    c = KArmCache.build(X, y)
    return float(c.insert(x_new, y_new).mu() - c.mu())


## --------------------------------------------------------------------
## 2-E. Swap sensitivities for RA and DiM
## --------------------------------------------------------------------

def swap_delta_RA(
    X: np.ndarray,
    y1: np.ndarray,
    y0: np.ndarray,
    S1: np.ndarray,
    i_in_S1: int,
    j_in_S0: int,
    branch: Literal["M", "K", "auto"] = "auto",
    mode: Literal["no_refit", "refit"] = "no_refit",
) -> float:
    """
    Δ τ̂_RA from swapping i∈S1 with j∈S0 (global indices).

    - 'refit':      recompute arm-level intercepts before/after the swap using Part 1's mu_ols.
    - 'no_refit':   use exact sequential cache updates (delete then insert) in the chosen branch for each arm.
                    This path is algebraically exact; we fall back to refit if an update would be unstable.
    """
    n = X.shape[0]
    in1 = np.zeros(n, dtype=bool); in1[S1] = True
    S0 = np.where(~in1)[0]

    # Validate membership: i must be in S1; j must be in S0
    if i_in_S1 not in set(map(int, S1)):
        raise ValueError("swap_delta_RA: i_in_S1 must be an element of S1")
    if j_in_S0 in set(map(int, S1)):
        raise ValueError("swap_delta_RA: j_in_S0 must be in the complement of S1")

    pos_i = int(np.where(S1 == i_in_S1)[0][0])   # local position in S1
    pos_j = int(np.where(S0 == j_in_S0)[0][0])   # local position in S0

    X1, X0 = X[S1, :], X[S0, :]
    y1S, y0S = y1[S1], y0[S0]

    # choose branch separately per arm if 'auto'
    br1 = ("M" if X1.shape[0] > X1.shape[1] else "K") if branch == "auto" else branch
    br0 = ("M" if X0.shape[0] > X0.shape[1] else "K") if branch == "auto" else branch

    if mode == "refit":
        # before
        mu1_b = (MGeometry.from_X(X1).mu_ols(y1S) if br1 == "M" else KGeometry.from_X(X1).mu_ols(y1S))
        mu0_b = (MGeometry.from_X(X0).mu_ols(y0S) if br0 == "M" else KGeometry.from_X(X0).mu_ols(y0S))
        # after
        S1_new = S1.copy(); S1_new[pos_i] = j_in_S0
        in1_new = np.zeros(n, dtype=bool)
        in1_new[S1_new] = True
        S0_new = np.where(~in1_new)[0]
        X1a, y1a = X[S1_new, :], y1[S1_new]
        X0a, y0a = X[S0_new, :], y0[S0_new]
        mu1_a = (MGeometry.from_X(X1a).mu_ols(y1a) if br1 == "M" else KGeometry.from_X(X1a).mu_ols(y1a))
        mu0_a = (MGeometry.from_X(X0a).mu_ols(y0a) if br0 == "M" else KGeometry.from_X(X0a).mu_ols(y0a))
        return (mu1_a - mu0_a) - (mu1_b - mu0_b)

    # no_refit path: exact sequential updates via caches
    if br1 == "M":
        cache1 = MArmCache.build(X1, y1S)
        cache1_after = cache1.delete(pos_i).insert(X[j_in_S0, :], float(y1[j_in_S0]))
        d1 = cache1_after.mu() - cache1.mu()
    else:
        cache1 = KArmCache.build(X1, y1S)
        cache1_after = cache1.delete(pos_i).insert(X[j_in_S0, :], float(y1[j_in_S0]))
        d1 = cache1_after.mu() - cache1.mu()

    if br0 == "M":
        cache0 = MArmCache.build(X0, y0S)
        cache0_after = cache0.delete(pos_j).insert(X[i_in_S1, :], float(y0[i_in_S1]))
        d0 = cache0_after.mu() - cache0.mu()
    else:
        cache0 = KArmCache.build(X0, y0S)
        cache0_after = cache0.delete(pos_j).insert(X[i_in_S1, :], float(y0[i_in_S1]))
        d0 = cache0_after.mu() - cache0.mu()

    return float(d1 - d0)


def _swap_delta_RA_on_set(
    X: np.ndarray,
    y1: np.ndarray,
    y0: np.ndarray,
    S1_treat: np.ndarray,   # treated set to evaluate at (arbitrary; e.g., S_{t-1}^{prox}(i,T))
    i_glob: int,
    j_glob: int,
    branch: Literal["M", "K", "auto"] = "auto",
) -> float:
    """
    Δ τ̂_RA from swapping i∈S1_treat with j∈S0_treat, evaluated on the provided set S1_treat.
    Uses exact cache updates (downdate + update) with safe fallbacks.
    """
    n = X.shape[0]
    mask = np.zeros(n, dtype=bool); mask[S1_treat] = True
    S0_treat = np.where(~mask)[0]

    # local positions
    pos_i = int(np.where(S1_treat == i_glob)[0][0])
    pos_j = int(np.where(S0_treat == j_glob)[0][0])

    X1, X0 = X[S1_treat, :], X[S0_treat, :]
    y1S, y0S = y1[S1_treat], y0[S0_treat]

    br1 = ("M" if X1.shape[0] > X1.shape[1] else "K") if branch == "auto" else branch
    br0 = ("M" if X0.shape[0] > X0.shape[1] else "K") if branch == "auto" else branch

    if br1 == "M":
        cache1 = MArmCache.build(X1, y1S)
        d1 = cache1.delete(pos_i).insert(X[j_glob, :], float(y1[j_glob])).mu() - cache1.mu()
    else:
        cache1 = KArmCache.build(X1, y1S)
        d1 = cache1.delete(pos_i).insert(X[j_glob, :], float(y1[j_glob])).mu() - cache1.mu()

    if br0 == "M":
        cache0 = MArmCache.build(X0, y0S)
        d0 = cache0.delete(pos_j).insert(X[i_glob, :], float(y0[i_glob])).mu() - cache0.mu()
    else:
        cache0 = KArmCache.build(X0, y0S)
        d0 = cache0.delete(pos_j).insert(X[i_glob, :], float(y0[i_glob])).mu() - cache0.mu()

    return float(d1 - d0)


def _swap_deltas_RA_over_all_controls_on_set(
    X: np.ndarray,
    y1: np.ndarray,
    y0: np.ndarray,
    S1_treat: np.ndarray,   # e.g., S_{t-1}^{prox}(i,T)
    i_glob: int,
    j_list_all: list[int],
    branch: Literal["M", "K", "auto"] = "auto",
) -> np.ndarray:
    """
    For a fixed proxied treated set S1_treat and fixed i ∈ S1_treat,
    return the array [Δ_{i,j} τ̂_RA]_{j∈S0_treat} evaluated on S1_treat,
    sharing arm caches across all j for efficiency.
    """
    n = X.shape[0]
    mask = np.zeros(n, dtype=bool); mask[S1_treat] = True
    S0_treat = np.where(~mask)[0]

    # restrict to valid controls in the current proxied control set
    S0_set = set(int(u) for u in S0_treat)
    j_list = [int(j) for j in j_list_all if int(j) in S0_set]
    if not j_list:
        return np.asarray([], dtype=float)

    # local positions
    pos_i = int(np.where(S1_treat == i_glob)[0][0])
    pos_map_S0 = {int(g): int(k) for k, g in enumerate(S0_treat)}

    # arm data
    X1, X0 = X[S1_treat, :], X[S0_treat, :]
    y1S, y0S = y1[S1_treat], y0[S0_treat]

    br1 = ("M" if X1.shape[0] > X1.shape[1] else "K") if branch == "auto" else branch
    br0 = ("M" if X0.shape[0] > X0.shape[1] else "K") if branch == "auto" else branch

    # build base caches once
    if br1 == "M":
        cache1 = MArmCache.build(X1, y1S)
    else:
        cache1 = KArmCache.build(X1, y1S)
    mu1_base = float(cache1.mu())
    cache1_del = cache1.delete(pos_i)  # same for all j

    if br0 == "M":
        cache0 = MArmCache.build(X0, y0S)
    else:
        cache0 = KArmCache.build(X0, y0S)
    mu0_base = float(cache0.mu())

    out = np.empty(len(j_list), dtype=float)
    for idx, j_glob in enumerate(j_list):
        # treated arm: delete i, insert j
        mu1_after = cache1_del.insert(X[j_glob, :], float(y1[j_glob])).mu()
        d1 = float(mu1_after - mu1_base)

        # control arm: delete j, insert i
        pos_j = pos_map_S0[j_glob]
        cache0_del = cache0.delete(pos_j)
        mu0_after = cache0_del.insert(X[i_glob, :], float(y0[i_glob])).mu()
        d0 = float(mu0_after - mu0_base)

        out[idx] = d1 - d0
    return out

def swap_deltas_RA_for_pairs(
    X: np.ndarray,
    y1: np.ndarray,
    y0: np.ndarray,
    S1: np.ndarray,
    pairs: list[tuple[int, int]],
    branch: Literal["M", "K", "auto"] = "auto",
) -> np.ndarray:
    """
    Compute RA swap deltas for many pairs (i,j) at a fixed assignment S1.

    This is mathematically identical to repeated calls to swap_delta_RA(...),
    but builds the two base arm caches only once. This is especially useful
    for MCBias, where many (i,j) pairs are sampled for the same assignment.
    """
    n = int(X.shape[0])
    S1 = _validate_S1(S1, n)
    if not pairs:
        return np.asarray([], dtype=float)

    mask = np.zeros(n, dtype=bool)
    mask[S1] = True
    S0 = np.where(~mask)[0]

    pos1 = {int(g): int(k) for k, g in enumerate(S1)}
    pos0 = {int(g): int(k) for k, g in enumerate(S0)}

    X1, X0 = X[S1, :], X[S0, :]
    y1S, y0S = y1[S1], y0[S0]

    br1 = ("M" if X1.shape[0] > X1.shape[1] else "K") if branch == "auto" else branch
    br0 = ("M" if X0.shape[0] > X0.shape[1] else "K") if branch == "auto" else branch

    cache1 = MArmCache.build(X1, y1S) if br1 == "M" else KArmCache.build(X1, y1S)
    cache0 = MArmCache.build(X0, y0S) if br0 == "M" else KArmCache.build(X0, y0S)

    mu1_base = float(cache1.mu())
    mu0_base = float(cache0.mu())

    out = np.empty(len(pairs), dtype=float)
    for k, (i_glob, j_glob) in enumerate(pairs):
        i_glob = int(i_glob)
        j_glob = int(j_glob)

        if i_glob not in pos1 or j_glob not in pos0:
            raise ValueError(
                "swap_deltas_RA_for_pairs: each pair must satisfy i in S1 and j in S0."
            )

        d1 = (
            cache1.delete(pos1[i_glob])
            .insert(X[j_glob, :], float(y1[j_glob]))
            .mu()
            - mu1_base
        )
        d0 = (
            cache0.delete(pos0[j_glob])
            .insert(X[i_glob, :], float(y0[i_glob]))
            .mu()
            - mu0_base
        )
        out[k] = float(d1 - d0)

    return out

def swap_delta_DIM(
    y1: np.ndarray,
    y0: np.ndarray,
    S1: np.ndarray,
    i_in_S1: int,
    j_in_S0: int,
) -> float:
    """
    Exact Δ τ̂_DIM from swapping i∈S1 with j∈S0:
        Δ = (y1[j]-y1[i]) / n1 - (y0[i]-y0[j]) / n0.
    """
    n1 = len(S1)
    n0 = len(y1) - n1
    return float((y1[j_in_S0] - y1[i_in_S1]) / n1 - (y0[i_in_S1] - y0[j_in_S0]) / n0)


# =====================================================================
# PART 3. Monte-Carlo estimates of B, R, V
# =====================================================================

## --------------------------------------------------------------------
## 3-A. Helpers and concentration parameter estimates
## --------------------------------------------------------------------

def tau_hat_RA(
    S1: np.ndarray,
    X: np.ndarray,
    y1: np.ndarray,
    y0: np.ndarray,
    branch: Literal["M", "K", "auto"] = "auto",
) -> float:
    """
    RA estimator: armwise OLS intercepts using the manuscript-consistent
    convention: unpenalized intercept and minimum-norm slope among LS minimizers.

    Uses the fast Q-ratio/cache implementation of Proposition 1. This is
    mathematically equivalent to the definition-faithful OLS intercept, but
    avoids repeated SVDs in high-dimensional/interpolating regimes.
    """
    n = int(X.shape[0])
    S1 = _validate_S1(S1, n)
    S0 = _indices_complement(S1, n)

    def _arm_mu_fast(Xa: np.ndarray, ya: np.ndarray) -> float:
        br = ("M" if Xa.shape[0] > Xa.shape[1] else "K") if branch == "auto" else branch
        if br == "M":
            return float(MArmCache.build(Xa, ya).mu())
        return float(KArmCache.build(Xa, ya).mu())

    mu1 = _arm_mu_fast(X[S1, :], y1[S1])
    mu0 = _arm_mu_fast(X[S0, :], y0[S0])
    return float(mu1 - mu0)


def tau_hat_DIM(
    S1: np.ndarray,
    y1: np.ndarray,
    y0: np.ndarray,
) -> float:
    """
    Unadjusted difference-in-means via ols_primitives.difference_in_means.
    """
    n = y1.shape[0]
    S1 = _validate_S1(S1, n)
    # Prefer the primitive when available
    if (_ols_mod is not None) and hasattr(_ols_mod, "difference_in_means"):
        T = np.zeros(n, dtype=np.int8); T[S1] = 1
        y_obs = T * y1 + (1 - T) * y0
        return float(_ols_mod.difference_in_means(y_obs, T))  # type: ignore[attr-defined]
    # Fallback: manual
    S0 = _indices_complement(S1, n)
    return float(np.mean(y1[S1]) - np.mean(y0[S0]))


#---------------------------------------------------------

def ra_vs_dim_penalty(S1: np.ndarray, X: np.ndarray, y1: np.ndarray, y0: np.ndarray) -> float:
    """
    η(S) = | τ̂_RA(S) - τ̂_DIM(S) | computed exactly for the realized assignment S1.
    """
    n = int(X.shape[0])
    S1 = _validate_S1(S1, n)
    tau_dim = tau_hat_DIM(S1, y1, y0)
    tau_ra = tau_hat_RA(S1, X, y1, y0, branch="auto")
    return float(abs(tau_ra - tau_dim))


def estimate_VR_DIM_exact(
    S1: np.ndarray,
    X: np.ndarray,
    y1: np.ndarray,
    y0: np.ndarray,
    Pi: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float]:
    """
    Exact oracle (V*, R*) for DiM under the remaining-pool reveal law.

    For DiM, Delta_{ij} = a_j - a_i with
        a_i = y1_i / n1 + y0_i / n0.
    Conditional on the revealed past, i ranges over the remaining pool R,
    and J is uniform on R \\ {i}. Hence
        zeta_t(i) = E_J[a_J - a_i]
                  = (sum_{u in R} a_u - |R| a_i) / (|R|-1).
    """
    n = int(X.shape[0])
    S1 = _validate_S1(S1, n)
    n1 = int(len(S1))
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

        alpha_t = _alpha_t(n, n1, t + 1)
        Vsum += (alpha_t ** 2) * float(np.var(zeta, ddof=0))
        Rmax = max(Rmax, alpha_t * float(np.max(np.abs(zeta))))

        available[int(S1[int(pos)])] = False

    return float(Vsum), float(Rmax)


def hybrid_ra_radius(
    S1: np.ndarray,
    X: np.ndarray,
    y1: np.ndarray,
    y0: np.ndarray,
    Pi: np.ndarray,
    delta: float,
    Bstar_RA: float,
    rng: np.random.Generator,
) -> float:
    """
    r_hybrid = Freedman_DIM(δ) + η(S) + B*_RA.
    """
    V_dim, R_dim = estimate_VR_DIM_exact(S1, X, y1, y0, Pi, rng)
    L = math.log(2.0 / float(delta))
    rad_dim = math.sqrt(2.0 * V_dim * L) + (R_dim / 3.0) * L
    eta = ra_vs_dim_penalty(S1, X, y1, y0)
    return float(rad_dim + eta + Bstar_RA)

    
#---------------------------------------------------------


def true_tau(y1: np.ndarray, y0: np.ndarray) -> float:
    """Finite-population ATE."""
    if _true_tau_population is not None:
        return float(_true_tau_population(y1, y0))
    return float(np.mean(y1) - np.mean(y0))


def make_random_reveal_order(n1: int, rng: np.random.Generator) -> np.ndarray:
    """Random reveal order Π over positions [0..n1-1]."""
    return rng.permutation(n1).astype(int)


def _alpha_t(n: int, n1: int, t: int) -> float:
    """Default α_t = n0 / (n - t + 1) used in the paper's progressive reveal."""
    n0 = n - n1
    return float(n0 / (n - t + 1))


def estimate_VR_for_assignment(
    S1: np.ndarray,
    X: np.ndarray,
    y1: np.ndarray,
    y0: np.ndarray,
    Pi: np.ndarray,
    Bi: int,
    Bcond: int,
    rng: np.random.Generator,
    method: Literal["RA", "DIM"] = "RA",
    branch: Literal["M", "K", "auto"] = "auto",
    eta: Optional[float] = None,
    delta_max: Optional[float] = None,   # reserved for future envelope-based refinements
    apply_ucb: bool = False,
    Bj: int = 0,                         # 0 enumerates all admissible controls J
    return_rswap: bool = False,
) -> Tuple[float, ...]:
    r"""
    Monte-Carlo approximation of (V*, R*) for a *fixed* realized assignment S1 and reveal order Pi.

    Tightened implementation that matches Algorithm MCVarRange in Appx04:
      - At step t (0-based here; 1-based in the paper), candidate i's are sampled
        from the remaining pool R_{t-1} = [n] \\ S_past, matching Proposition 2.
      - For each candidate i, draw Bcond proxied completions T_b uniformly from
        subsets of R_{t-1}\\{i} of size (n1 - t - 1).
      - For each T_b, either average over all admissible controls J
        if Bj <= 0, or sample Bj admissible controls uniformly without replacement.
      - Accumulate v_t^* using the across-i variance of \hat{\zeta}_t(i),
        with the average within-i Monte Carlo noise correction subtracted and truncated at zero.
      - Accumulate r_t^* \approx \alpha_t \max_i |\hat{\zeta}_t(i)|.

    Notes:
      - eta controls the one-sided normal UCB used when apply_ucb=True.
      - delta_max is currently reserved for future envelope-based refinements
        and is not used below.
    """

    n = X.shape[0]
    n1 = len(S1)
    mask = np.zeros(n, dtype=bool); mask[S1] = True
    S0 = np.where(~mask)[0]

    Vsum = 0.0
    Rmax = 0.0
    V_pqv_naive_accum = 0.0
    Rswap_max = 0.0

    I_total = 0  # Σ_t |I_t| for the union bound in the UCB
    R_chunks: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []

    for t, pos in enumerate(Pi):
        a = _alpha_t(n, n1, t + 1)

        S_past = S1[Pi[:t]]

        # Candidate I is distributed over the full remaining pool R_{t-1},
        # not only the realized future-treated suffix.
        R_pool = np.setdiff1d(np.arange(n, dtype=int), S_past, assume_unique=False)

        Bi_eff = min(R_pool.size, max(int(Bi), 2))
        if Bi_eff < R_pool.size:
            i_list = rng.choice(R_pool, size=Bi_eff, replace=False).astype(int).tolist()
        else:
            i_list = list(map(int, R_pool))

        I_total += len(i_list)

        # --- compute per-i Monte Carlo means and within-i variances, then de-noise var across i ---
        m_list: list[float] = []   # per-i Monte Carlo means \hat zeta_t(i)
        s2_list: list[float] = []  # per-i within-i sample variances of xi's
        B_list: list[float] = []   # per-i effective B (B=inf under exact DIM enumeration ⇒ no correction)

        raw_abs_t = 0.0
        if method == "DIM":
            # For DiM, Delta_{iJ} does not depend on T, and the marginal law of
            # J given (F_{t-1}, i) is uniform on R_{t-1}\\{i}.
            for i_glob in i_list:
                pool_wo_i = R_pool[R_pool != int(i_glob)]

                if int(Bj) <= 0 or int(Bj) >= pool_wo_i.size:
                    j_list = list(map(int, pool_wo_i))
                    xis = [
                        float(swap_delta_DIM(y1, y0, S1, int(i_glob), int(j_glob)))
                        for j_glob in j_list
                    ]
                    if xis:
                        raw_abs_t = max(raw_abs_t, float(np.max(np.abs(xis))))

                    m_i = float(np.mean(xis)) if xis else 0.0
                    s2_list.append(0.0)
                    B_list.append(float("inf"))
                else:
                    j_samp = rng.choice(pool_wo_i, size=int(Bj), replace=False)
                    xis = [
                        float(swap_delta_DIM(y1, y0, S1, int(i_glob), int(j_glob)))
                        for j_glob in j_samp
                    ]
                    if xis:
                        raw_abs_t = max(raw_abs_t, float(np.max(np.abs(xis))))
                        
                    m_i = float(np.mean(xis)) if xis else 0.0
                    s2_list.append(float(np.var(xis, ddof=1)) if len(xis) > 1 else 0.0)
                    B_list.append(float(len(xis)))

                m_list.append(m_i)
        else:
            # RA branch: approximate zeta_t(i)=E_{T,J}[Delta_{iJ} f(S_past ∪ {i} ∪ T)].
            T_size = max(0, n1 - t - 1)
            Bnow = max(int(Bcond), 1)

            for i_glob in i_list:
                i_glob = int(i_glob)
                pool_wo_i = R_pool[R_pool != i_glob]

                xi_bar_list = []
                for _ in range(Bnow):
                    if T_size > 0:
                        T = rng.choice(pool_wo_i, size=int(T_size), replace=False).astype(int)
                    else:
                        T = np.array([], dtype=int)

                    S1_prox = np.concatenate(
                        [S_past, np.array([i_glob], dtype=int), T]
                    ).astype(int)

                    admissible_J = np.setdiff1d(pool_wo_i, T, assume_unique=False)
                    if int(Bj) > 0 and int(Bj) < admissible_J.size:
                        j_list = rng.choice(admissible_J, size=int(Bj), replace=False).astype(int).tolist()
                    else:
                        j_list = list(map(int, admissible_J))

                    vals = _swap_deltas_RA_over_all_controls_on_set(
                        X, y1, y0, S1_prox, i_glob, j_list, branch
                    )
                    if len(vals) > 0:
                        raw_abs_t = max(raw_abs_t, float(np.max(np.abs(vals))))
                        xi_bar_list.append(float(np.mean(vals)))
                    else:
                        xi_bar_list.append(0.0)

                m_i = float(np.mean(xi_bar_list))
                s2_i = float(np.var(xi_bar_list, ddof=1)) if Bnow > 1 else 0.0
                m_list.append(m_i)
                s2_list.append(s2_i)
                B_list.append(Bnow)

        # m_arr is the array of per-i means already built above
        m_arr = np.asarray(m_list, dtype=float)
        s2_arr = np.asarray(s2_list, dtype=float)
        B_arr  = np.asarray(B_list, dtype=float)
   
        # Stash for UCB aggregation over all (t,i)
        # keep per-i standard errors across completions so we can form a proper UCB
        R_chunks.append((a, np.abs(m_arr), s2_arr, B_arr))

        # detect full enumeration of i's over the current remaining pool R_{t-1}
        R_pool_arr = np.asarray(R_pool, dtype=int)
        i_arr      = np.asarray(i_list, dtype=int)
        enum_i = (i_arr.size == R_pool_arr.size) and np.array_equal(np.sort(i_arr), np.sort(R_pool_arr))
        
        # pick ddof: population variance when enumerating all i; unbiased sample variance otherwise
        ddof_i = 0 if enum_i else 1

        # variance across i of the per-i means, and the (uncentered) max for R
        if m_arr.size <= 1:
            var_across_i = 0.0
            r_core = float(abs(m_arr[0])) if m_arr.size == 1 else 0.0
        else:
            var_across_i = float(np.var(m_arr, ddof=ddof_i))
            r_core = float(np.max(np.abs(m_arr)))

        # unbiased de-noising: subtract average within-i variance divided by B
        # entries with 'enumeration' were marked with huge B ⇒ contribute ~0
        denom = np.clip(B_arr, 1.0, np.inf)
        noise_correction = float(np.mean(s2_arr / denom))
        var_t = max(var_across_i - noise_correction, 0.0)
        r_t   = r_core

        Vsum += (a * a) * var_t
        Rmax = max(Rmax, a * r_t)
        Rswap_max = max(Rswap_max, a * raw_abs_t)

        s2_bar = float(m_arr.var(ddof=1)) if m_arr.size > 1 else 0.0
        V_pqv_naive_accum += (a * a) * s2_bar

    # Optional UCB on the RB increment using per-i SE across completions
    if apply_ucb and (I_total > 0):
        if eta is None:
            eta = 0.01
        if not (0.0 < float(eta) < 1.0):
            raise ValueError(f"eta must lie in (0,1); got {eta}.")
        z = float(_STD_NORMAL.inv_cdf(1.0 - float(eta)))

        Rmax_ucb = 0.0
        for a_t, abs_m, s2, B in R_chunks:
            se = np.sqrt(np.maximum(s2, 0.0) / np.clip(B, 1.0, np.inf))
            Rmax_ucb = max(Rmax_ucb, float(a_t * np.max(abs_m + z * se)))
        if return_rswap:
            return Vsum, Rmax_ucb, V_pqv_naive_accum, Rswap_max
        return Vsum, Rmax_ucb, V_pqv_naive_accum
    else:
        if return_rswap:
            return Vsum, Rmax, V_pqv_naive_accum, Rswap_max
        return Vsum, Rmax, V_pqv_naive_accum


def compute_R_emp_RA(
    S1: np.ndarray,
    X: np.ndarray,
    y1: np.ndarray,
    y0: np.ndarray,
    Pi: np.ndarray,
    Bcond: int,
    rng: np.random.Generator,
    branch: Literal["M", "K", "auto"] = "auto",
    Bi_emp: int = 0,   # 0 enumerates candidates i
    Bj: int = 0,       # 0 enumerates admissible controls J
) -> float:
    r"""
    Sampled raw-swap range diagnostic for RA:
        R_{\mathrm{swap}}(S_1,\Pi)
        = \max_t \alpha_t \max_i
          \mathbb{E}_T\!\left[\max_{J \in \mathcal{J}(T,i)}
          |\Delta_{iJ}(S_{t-1}^{\mathrm{prox}}(i,T))|\right],
    approximated by Bcond random completions per i and, when Bj > 0,
    by a uniform subsample of admissible controls J per completion.

    This is a diagnostic built from the same sampled objects as
    estimate_VR_for_assignment. It is not the oracle range R^*, and when
    Bj > 0 it is not an exact all-control envelope.
    """
    n  = X.shape[0]
    n1 = len(S1)

    Rmax_emp = 0.0
    for t, _ in enumerate(Pi):
        a = _alpha_t(n, n1, t + 1)
        S_past = S1[Pi[:t]]
        R_pool = np.setdiff1d(np.arange(n, dtype=int), S_past, assume_unique=False)
        if R_pool.size == 0:
            continue

        T_size = max(n1 - (t + 1), 0)

        if int(Bi_emp) > 0 and int(Bi_emp) < R_pool.size:
            i_candidates = rng.choice(R_pool, size=int(Bi_emp), replace=False).astype(int)
        else:
            i_candidates = R_pool

        for i_glob in map(int, i_candidates):
            pool_wo_i = R_pool[R_pool != i_glob]
            Bnow = max(int(Bcond), 1) if T_size > 0 else 1

            maxabs_list: list[float] = []
            for _ in range(Bnow):
                if T_size > 0:
                    T = rng.choice(pool_wo_i, size=T_size, replace=False).astype(int)
                else:
                    T = np.array([], dtype=int)

                S1_prox = np.concatenate(
                    [S_past, np.array([i_glob], dtype=int), T]
                ).astype(int)

                admissible_J = np.setdiff1d(pool_wo_i, T, assume_unique=False)
                if int(Bj) > 0 and int(Bj) < admissible_J.size:
                    j_list = rng.choice(admissible_J, size=int(Bj), replace=False).astype(int).tolist()
                else:
                    j_list = list(map(int, admissible_J))

                vals = _swap_deltas_RA_over_all_controls_on_set(
                    X, y1, y0, S1_prox, i_glob, j_list, branch
                )
                maxabs_list.append(float(np.max(np.abs(vals))) if len(vals) > 0 else 0.0)

            m_i = float(np.mean(maxabs_list)) if maxabs_list else 0.0
            Rmax_emp = max(Rmax_emp, float(a * m_i))

    return Rmax_emp


## --------------------------------------------------------------------
## 3-B. Helpers and bias parameter estimates
## --------------------------------------------------------------------

def _lambda_hat(n: int, n1: int, EGamma: float, Varf: float, mode: Literal["gap", "ratio", "max"]) -> float:
    """λ̂ choices: 'gap'=n/(n1 n0), 'ratio'=EGamma/Varf (guarded), 'max'=max of both."""
    n0 = n - n1
    gap = float(n) / float(n1 * n0) if (n1 > 0 and n0 > 0) else 0.0
    if mode == "gap":
        return gap
    if Varf <= _EPS:
        return gap
    ratio = EGamma / Varf
    return ratio if mode == "ratio" else max(gap, ratio)


def tau_hat_for_method(
    S1: np.ndarray,
    X: np.ndarray,
    y1: np.ndarray,
    y0: np.ndarray,
    method: Literal["RA", "DIM"],
    branch: Literal["M", "K", "auto"] = "auto",
) -> float:
    if method == "DIM":
        return tau_hat_DIM(S1, y1, y0)
    elif method == "RA":
        return tau_hat_RA(S1, X, y1, y0, branch)
    else:
        raise ValueError("method must be 'RA' or 'DIM'")


def pairs_for_assignment(
    S1: np.ndarray,
    n: int,
    Bpair: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    S1 = _validate_S1(S1, n)
    if Bpair <= 0:
        return []
    S0 = _indices_complement(S1, n)
    S1s = np.sort(S1)
    S0s = np.sort(S0)
    n1, n0 = int(S1s.size), int(S0s.size)
    total  = n1 * n0
    if Bpair >= total:
        # enumerate all pairs
        return [(int(i), int(j)) for i in S1s for j in S0s]
    # sample linear indices without replacement, then map to (i,j)
    idx = np.asarray(rng.choice(total, size=int(Bpair), replace=False), dtype=int)
    i_idx = (idx // n0).astype(int)
    j_idx = (idx %  n0).astype(int)
    return [(int(S1s[i]), int(S0s[j])) for i, j in zip(i_idx, j_idx)]


def centered_tau_hat(S1: np.ndarray,
                     X: np.ndarray, y1: np.ndarray, y0: np.ndarray,
                     method: Literal["RA","DIM"], branch: Literal["M","K","auto"],
                     tau_true: float) -> float:
    """Return tau_hat(S1) - tau_true for the chosen method/branch."""
    return tau_hat_for_method(S1, X, y1, y0, method, branch) - tau_true


def estimate_Bstar(
    S1_size: int,
    X: np.ndarray,
    y1: np.ndarray,
    y0: np.ndarray,
    BS: int,
    Bpair: int,
    rng: np.random.Generator,
    method: Literal["RA", "DIM"] = "RA",
    branch: Literal["M", "K", "auto"] = "auto",
    lambda_mode: Literal["gap", "ratio", "max"] = "max",
) -> Tuple[float, float, float]:
    """
    Monte-Carlo estimate of the corrected oracle bias parameter B*.

    The corrected manuscript definition is
        B* = sqrt( E[(Lf + lambda* f)^2] ) / lambda*,
        lambda* = E[Gamma(f)] / Var(f),
    where f(S)=tau_hat(S)-tau and Lf(S) is the Johnson-walk average swap
    drift E[f(S')-f(S) | S].

    This routine returns (B_hat, EGamma_hat, Varf_hat) to preserve the public
    interface used by run_experiments_finite.py. Internally, it estimates both
    Gamma(f)(S) and Lf(S) from the same sampled or enumerated swap pairs.
    """
    n = int(X.shape[0])
    if not (0 < S1_size < n):
        raise ValueError("S1_size must satisfy 0 < S1_size < n.")
    if Bpair <= 0:
        raise ValueError("Bpair must be positive to estimate Gamma(f) and Lf.")

    tau_true = (
        float(np.mean(y1) - np.mean(y0))
        if (_true_tau_population is None)
        else float(_true_tau_population(y1, y0))
    )

    total_assign = math.comb(n, S1_size)
    enumerate_assignments = BS >= total_assign

    Gamma_vals: list[float] = []
    Lf_vals: list[float] = []
    f_vals: list[float] = []

    if enumerate_assignments:
        assignment_iter = (
            np.array(S1_tuple, dtype=int)
            for S1_tuple in itertools.combinations(range(n), S1_size)
        )
    else:
        assignment_iter = (
            rng.choice(n, size=S1_size, replace=False).astype(int)
            for _ in range(BS)
        )

    for S1 in assignment_iter:
        f_vals.append(centered_tau_hat(S1, X, y1, y0, method, branch, tau_true))
        pairs = pairs_for_assignment(S1, n, Bpair, rng)

        if method == "DIM":
            deltas = np.asarray(
                [swap_delta_DIM(y1, y0, S1, i_glob, j_glob)
                 for i_glob, j_glob in pairs],
                dtype=float,
            )
        else:
            deltas = swap_deltas_RA_for_pairs(X, y1, y0, S1, pairs, branch=branch)

        dsum = float(np.sum(deltas))
        dsq = float(deltas @ deltas)

        denom = float(len(pairs))
        Gamma_vals.append(0.5 * dsq / denom)
        Lf_vals.append(dsum / denom)

    f_arr = np.asarray(f_vals, dtype=float)
    Gamma_arr = np.asarray(Gamma_vals, dtype=float)
    Lf_arr = np.asarray(Lf_vals, dtype=float)

    # If all assignments are enumerated, use the exact population variance.
    # Otherwise use the usual unbiased sample variance as a Monte Carlo estimate.
    ddof = 0 if enumerate_assignments else 1
    Varf_hat = float(np.var(f_arr, ddof=ddof)) if f_arr.size > ddof else 0.0
    EGamma_hat = float(np.mean(Gamma_arr)) if Gamma_arr.size > 0 else 0.0

    lam = _lambda_hat(n, S1_size, EGamma_hat, Varf_hat, lambda_mode)

    if lam <= 0.0:
        B_hat = float(abs(np.mean(f_arr))) if f_arr.size > 0 else 0.0
    else:
        stein_resid = Lf_arr + lam * f_arr
        B_hat = math.sqrt(max(float(np.mean(stein_resid * stein_resid)), 0.0)) / max(lam, _EPS)

    return B_hat, EGamma_hat, Varf_hat



# =====================================================================
# PART 4. Self-tests (Sanity checks)
# =====================================================================

## --------------------------------------------------------------------
## 4-A. Test functions
## --------------------------------------------------------------------

def _make_problem(n: int, p: int, n1: int, rng: np.random.Generator):
    X = rng.normal(size=(n, p))
    beta = rng.normal(size=p)
    tau = 0.5
    y0 = X @ beta + rng.normal(scale=0.7, size=n)
    y1 = y0 + tau
    S1 = rng.choice(n, size=n1, replace=False)
    return X, y1, y0, S1


def _timeit(fn, repeats: int = 5) -> float:
    tmin = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        tmin = min(tmin, t1 - t0)
    return tmin


def _check_close(a: float, b: float, name: str, rtol=1e-6, atol=1e-8):
    if not np.isclose(a, b, rtol=rtol, atol=atol):
        raise AssertionError(f"{name}: mismatch: {a} vs {b} (rtol={rtol}, atol={atol})")


def _test_mu_section(verbose: bool = True) -> None:
    rng = np.random.default_rng(7)

    # ----- M-branch -----
    n, p, n1 = 800, 40, 400
    X, y1, y0, S1 = _make_problem(n, p, n1, rng)
    X1 = X[S1, :]; y1S = y1[S1]
    mg = MGeometry.from_X(X1)
    mu_q = mg.mu_quotient(y1S); mu_o = mg.mu_ols(y1S)
    _check_close(mu_q, mu_o, "M mu_quotient==mu_ols")
    t_q = _timeit(lambda: mg.mu_quotient(y1S))
    t_o = _timeit(lambda: mg.mu_ols(y1S))
    if verbose:
        print(f"[M] mu_quotient vs mu_ols: {t_q:.4f}s vs {t_o:.4f}s (speedup ×{t_o/max(t_q,1e-9):.2f})")

    # ----- K-branch -----
    n, p, n1 = 300, 1000, 150
    X, y1, y0, S1 = _make_problem(n, p, n1, rng)
    X1 = X[S1, :]; y1S = y1[S1]
    kg = KGeometry.from_X(X1)
    mu_q = kg.mu_quotient(y1S); mu_o = kg.mu_ols(y1S)
    _check_close(mu_q, mu_o, "K mu_quotient==mu_ols")
    t_q = _timeit(lambda: kg.mu_quotient(y1S))
    t_o = _timeit(lambda: kg.mu_ols(y1S))
    if verbose:
        print(f"[K] mu_quotient vs mu_ols: {t_q:.4f}s vs {t_o:.4f}s (speedup ×{t_o/max(t_q,1e-9):.2f})")

    # ----- Corner: 1 ∈ col(X) (forces fallback) -----
    n, p, n1 = 200, 5, 100
    X, y1, y0, S1 = _make_problem(n, p, n1, rng)
    X1 = np.column_stack([np.ones(len(S1)), X[S1, :]])
    mg = MGeometry.from_X(X1)
    mu_q = mg.mu_quotient(y1[S1])
    mu_o = mg.mu_ols(y1[S1])
    _check_close(mu_q, mu_o, "M corner: mu_quotient fallback==mu_ols", rtol=1e-5, atol=1e-7)


def _test_deltas_section(verbose: bool = True) -> None:
    rng = np.random.default_rng(7)

    # ----- M-branch delete/insert -----
    n, p, n1 = 1200, 40, 600
    X, y1, y0, S1 = _make_problem(n, p, n1, rng)
    X1 = X[S1, :]; y1S = y1[S1]
    pos = int(rng.integers(0, len(S1)))
    x_new = X[~np.isin(np.arange(len(y1)), S1)][0, :]
    y_new = float(y1[~np.isin(np.arange(len(y1)), S1)][0])
    d_ref = deletion_delta_mu_refit_M(X1, y1S, pos)
    d_fast = deletion_delta_mu_norefit_M(X1, y1S, pos)
    _check_close(d_ref, d_fast, "M deletion Δμ no-refit==refit")
    i_ref = insertion_delta_mu_refit_M(X1, y1S, x_new, y_new)
    i_fast = insertion_delta_mu_norefit_M(X1, y1S, x_new, y_new)
    _check_close(i_ref, i_fast, "M insertion Δμ no-refit==refit")
    t_del = _timeit(lambda: (MArmCache.build(X1, y1S).delete(int(rng.integers(0, len(S1)))).mu()))
    t_ins = _timeit(lambda: (MArmCache.build(X1, y1S).insert(x_new, y_new).mu()))
    if verbose:
        print(f"[M] Δμ delete/insert (no-refit): {t_del:.5f}s / {t_ins:.5f}s (per op)")

    # ----- K-branch delete/insert -----
    n, p, n1 = 300, 1000, 150
    X, y1, y0, S1 = _make_problem(n, p, n1, rng)
    X1 = X[S1, :]; y1S = y1[S1]
    pos = int(rng.integers(0, len(S1)))
    x_new = X[~np.isin(np.arange(len(y1)), S1)][0, :]
    y_new = float(y1[~np.isin(np.arange(len(y1)), S1)][0])
    d_ref = deletion_delta_mu_refit_K(X1, y1S, pos)
    d_fast = deletion_delta_mu_norefit_K(X1, y1S, pos)
    _check_close(d_ref, d_fast, "K deletion Δμ no-refit==refit", rtol=1e-6, atol=1e-8)
    i_ref = insertion_delta_mu_refit_K(X1, y1S, x_new, y_new)
    i_fast = insertion_delta_mu_norefit_K(X1, y1S, x_new, y_new)
    _check_close(i_ref, i_fast, "K insertion Δμ no-refit==refit", rtol=1e-6, atol=1e-8)
    t_del = _timeit(lambda: (KArmCache.build(X1, y1S).delete(int(rng.integers(0, len(S1)))).mu()))
    t_ins = _timeit(lambda: (KArmCache.build(X1, y1S).insert(x_new, y_new).mu()))
    if verbose:
        print(f"[K] Δμ delete/insert (no-refit): {t_del:.5f}s / {t_ins:.5f}s (per op)")


def _test_stress_section(verbose: bool = True) -> None:
    rng = np.random.default_rng(7)

    # Stress M
    n, p, n1 = 1000, 60, 250
    X, y1, y0, S1 = _make_problem(n, p, n1, rng)
    X1 = X[S1, :]; y1S = y1[S1]
    j = int(rng.integers(0, len(S1)))
    _check_close(deletion_delta_mu_refit_M(X1, y1S, j),
                 deletion_delta_mu_norefit_M(X1, y1S, j),
                 "M stress deletion Δμ")

    # Stress K
    n, p, n1 = 80, 1000, 40
    X, y1, y0, S1 = _make_problem(n, p, n1, rng)
    X1 = X[S1, :]; y1S = y1[S1]
    j = int(rng.integers(0, len(S1)))
    _check_close(deletion_delta_mu_refit_K(X1, y1S, j),
                 deletion_delta_mu_norefit_K(X1, y1S, j),
                 "K stress deletion Δμ")


def _test_tau_hat_RA_matches_ols_primitives():
    rng = np.random.default_rng(123)
    n, p, n1 = 60, 7, 25
    X = rng.normal(size=(n, p))
    beta = rng.normal(size=p)
    y0 = X @ beta + rng.normal(scale=0.5, size=n)
    y1 = y0 + 0.3
    S1 = rng.choice(n, size=n1, replace=False)
    # ours
    tau1 = tau_hat_RA(S1, X, y1, y0, branch="auto")
    # direct ols_primitives
    _mod = _load_ols_primitives_module()
    if _mod is None:
        # Skip test if module not available; RA estimator correctness is covered elsewhere.
        return
    _ols = getattr(_mod, "_ols_with_intercept")

    mask = np.zeros(n, dtype=bool); mask[S1] = True
    S0 = np.where(~mask)[0]
    fit1 = _ols(y1[S1], X[S1, :]); fit0 = _ols(y0[S0], X[S0, :])
    tau2 = float(fit1.mu_hat - fit0.mu_hat)
    _check_close(tau1, tau2, "tau_hat_RA matches ols_primitives")

def _test_tau_hat_RA_highdim_uses_unpenalized_intercept():
    """
    In high-dimensional arms, np.linalg.lstsq([1, X], y) minimum-norms the
    intercept as well as the slopes. The manuscript estimator leaves the
    intercept unpenalized. This test ensures tau_hat_RA uses the latter.
    """
    rng = np.random.default_rng(20260420)
    n, p, n1 = 50, 133, 15
    X = rng.normal(size=(n, p))
    y0 = rng.normal(size=n)
    y1 = rng.normal(size=n)
    S1 = rng.choice(n, size=n1, replace=False)
    S0 = _indices_complement(S1, n)

    tau = tau_hat_RA(S1, X, y1, y0, branch="auto")
    tau_q = (
        build_geometry(X[S1, :], "auto").mu_ols(y1[S1])
        -
        build_geometry(X[S0, :], "auto").mu_ols(y0[S0])
    )
    _check_close(tau, tau_q, "tau_hat_RA high-dimensional unpenalized intercept", rtol=1e-12, atol=1e-12)

def _test_swap_delta_RA_refit_vs_norefit():
    rng = np.random.default_rng(321)
    for n, p in [(80, 10), (40, 80)]:  # M-branch-like and K-branch-like
        n1 = n // 3
        X = rng.normal(size=(n, p))
        beta = rng.normal(size=p)
        y0 = X @ beta + rng.normal(scale=0.7, size=n)
        y1 = y0 + 0.5
        S1 = rng.choice(n, size=n1, replace=False)
        mask = np.zeros(n, dtype=bool); mask[S1] = True
        S0 = np.where(~mask)[0]
        i = int(rng.choice(S1)); j = int(rng.choice(S0))
        d_refit = swap_delta_RA(X, y1, y0, S1, i, j, branch="auto", mode="refit")
        d_nrf  = swap_delta_RA(X, y1, y0, S1, i, j, branch="auto", mode="no_refit")
        _check_close(d_refit, d_nrf, "swap_delta_RA refit==no_refit", rtol=1e-8, atol=1e-10)


def _test_MCVarRange_DIM_enumeration():
    rng = np.random.default_rng(7)
    n, p, n1 = 22, 4, 8
    X = rng.normal(size=(n, p))
    beta = rng.normal(size=p)
    y0 = X @ beta + rng.normal(size=n)
    y1 = y0 + 0.4
    S1 = rng.choice(n, size=n1, replace=False)
    Pi = make_random_reveal_order(n1, rng)

    # Exact remaining-pool oracle quantities.
    Vsum_ex, Rmax_ex = estimate_VR_DIM_exact(S1, X, y1, y0, Pi, rng=None)

    # Full enumeration in the generic MCVarRange routine should match the exact formula.
    rng_mc = np.random.default_rng(7)
    Vsum_mc, Rmax_mc, _ = estimate_VR_for_assignment(
        S1, X, y1, y0, Pi,
        Bi=n, Bcond=1, Bj=0, rng=rng_mc,
        method="DIM", branch="auto", apply_ucb=False
    )

    _check_close(Vsum_mc, Vsum_ex, "MCVarRange DIM Vsum exact", rtol=1e-12, atol=1e-12)
    _check_close(Rmax_mc, Rmax_ex, "MCVarRange DIM Rmax exact", rtol=1e-12, atol=1e-12)
    print("✓ MCVarRange (DIM) matches the exact remaining-pool formula")
    

def _test_MCBias_DIM_small_exact():
    rng = np.random.default_rng(11)
    n, p, n1 = 6, 3, 3
    X = rng.normal(size=(n, p))
    beta = rng.normal(size=p)
    y0 = X @ beta + rng.normal(scale=0.3, size=n)
    y1 = y0 + 0.2

    # Exact E[Gamma], Var(f), and corrected B* under full enumeration.
    from itertools import combinations
    EG_list = []
    f_list = []
    tau_true = true_tau(y1, y0)
    for S1_tuple in combinations(range(n), n1):
        S1 = np.array(S1_tuple, dtype=int)
        mask = np.zeros(n, dtype=bool); mask[S1] = True
        S0 = np.where(~mask)[0]
        f_list.append(tau_hat_DIM(S1, y1, y0) - tau_true)

        dsq_sum = 0.0
        cnt = 0
        for i in S1:
            for j in S0:
                d = swap_delta_DIM(y1, y0, S1, int(i), int(j))
                dsq_sum += d * d
                cnt += 1
        EG_list.append(0.5 * dsq_sum / cnt)

    EG_ex = float(np.mean(EG_list))
    Varf_ex = float(np.var(np.asarray(f_list, dtype=float), ddof=0)) if len(f_list) > 0 else 0.0

    # MC with full enumeration
    BS = math.comb(n, n1)
    Bpair = n1 * (n - n1)
    rng2 = np.random.default_rng(11)
    Bhat, EG_mc, Varf_mc = estimate_Bstar(
        n1, X, y1, y0, BS, Bpair, rng2, method="DIM", lambda_mode="max"
    )

    _check_close(EG_mc, EG_ex, "MCBias DIM E[Gamma] exact", rtol=1e-12, atol=1e-12)
    _check_close(Varf_mc, Varf_ex, "MCBias DIM Var(f) exact", rtol=1e-12, atol=1e-12)
    _check_close(Bhat, 0.0, "MCBias DIM corrected B* is zero", rtol=1e-12, atol=1e-12)


def _test_MCBias_RA_small_exact():
    """MCBias for RA on a tiny instance: exact enumeration equals full-coverage computation."""
    rng = np.random.default_rng(5)
    n, p, n1 = 8, 3, 3
    X = rng.normal(size=(n, p))
    beta = rng.normal(size=p)
    y0 = X @ beta + rng.normal(scale=0.3, size=n)
    y1 = y0 + 0.2

    # Exact enumeration
    from itertools import combinations
    tau_true = true_tau(y1, y0)
    EG_list, f_list, Lf_list = [], [], []
    for S1_tuple in combinations(range(n), n1):
        S1 = np.array(S1_tuple, dtype=int)
        f_list.append(tau_hat_RA(S1, X, y1, y0, branch="auto") - tau_true)

        dsum = 0.0
        dsq_sum = 0.0
        cnt = 0
        S0 = _indices_complement(S1, n)
        for i in S1:
            for j in S0:
                d = swap_delta_RA(
                    X, y1, y0, S1, int(i), int(j),
                    branch="auto", mode="no_refit"
                )
                d = float(d)
                dsum += d
                dsq_sum += d * d
                cnt += 1

        EG_list.append(0.5 * dsq_sum / cnt)
        Lf_list.append(dsum / cnt)

    f_arr = np.asarray(f_list, dtype=float)
    Lf_arr = np.asarray(Lf_list, dtype=float)
    EG_ex = float(np.mean(EG_list))
    Varf_ex = float(np.var(f_arr, ddof=0)) if f_arr.size > 0 else 0.0
    lam_ex = _lambda_hat(n, n1, EG_ex, Varf_ex, "max")
    B_ex = math.sqrt(float(np.mean((Lf_arr + lam_ex * f_arr) ** 2))) / max(lam_ex, _EPS)

    # MC with full enumeration
    BS = math.comb(n, n1)
    Bpair = n1 * (n - n1)
    rng2 = np.random.default_rng(5)
    Bhat, EG_mc, Varf_mc = estimate_Bstar(
        n1, X, y1, y0, BS, Bpair, rng2, method="RA", branch="auto"
    )

    _check_close(EG_mc, EG_ex, "MCBias RA E[Gamma] exact", rtol=1e-12, atol=1e-12)
    _check_close(Varf_mc, Varf_ex, "MCBias RA Var(f) exact", rtol=1e-12, atol=1e-12)
    _check_close(Bhat, B_ex, "MCBias RA corrected B* exact", rtol=1e-12, atol=1e-12)


def _test_MCBias_sampling_reproducible():
    """
    With fixed helpers and RNG separation, repeated runs with the same seed match exactly.
    """
    rng_data = np.random.default_rng(42)   # data RNG
    n, p, n1 = 30, 6, 10
    X = rng_data.normal(size=(n, p))
    beta = rng_data.normal(size=p)
    y0 = X @ beta + rng_data.normal(scale=0.4, size=n)
    y1 = y0 + 0.25

    BS, Bpair = 9, 7  # sampling regime (non-enumeration)
    rng_a = np.random.default_rng(777)     # algorithm RNG A
    rng_b = np.random.default_rng(777)     # algorithm RNG B (same seed, fresh state)

    out1 = estimate_Bstar(n1, X, y1, y0, BS, Bpair, rng_a, method="DIM")
    out2 = estimate_Bstar(n1, X, y1, y0, BS, Bpair, rng_b, method="DIM")
    for a, b in zip(out1, out2):
        _check_close(a, b, "MCBias sampling reproducible", rtol=0.0, atol=0.0)

    print("✓ MCBias (DIM) reproducibility: identical outputs with same RNG seed")


def _test_MCBias_sampling_reproducible_RA():
    rng_data = np.random.default_rng(12)
    n, p, n1 = 25, 7, 9
    X = rng_data.normal(size=(n, p))
    beta = rng_data.normal(size=p)
    y0 = X @ beta + rng_data.normal(scale=0.5, size=n)
    y1 = y0 + 0.2

    BS, Bpair = 11, 8
    rng_a = np.random.default_rng(888)
    rng_b = np.random.default_rng(888)

    out1 = estimate_Bstar(n1, X, y1, y0, BS, Bpair, rng_a, method="RA", branch="auto")
    out2 = estimate_Bstar(n1, X, y1, y0, BS, Bpair, rng_b, method="RA", branch="auto")
    for a, b in zip(out1, out2):
        _check_close(a, b, "MCBias (RA) sampling reproducible", rtol=0.0, atol=0.0)

    print("✓ MCBias (RA) reproducibility: identical outputs with same RNG seed")


def _test_pairs_for_determinism_and_corners():
    rng = np.random.default_rng(99)
    n, p, n1 = 40, 5, 15
    X = rng.normal(size=(n, p))
    y0 = rng.normal(size=n); y1 = y0 + 0.1
    S1 = rng.choice(n, size=n1, replace=False)

    rngA = np.random.default_rng(2024)
    rngB = np.random.default_rng(2024)
    P1 = pairs_for_assignment(S1, n, Bpair=13, rng=rngA)
    P2 = pairs_for_assignment(S1, n, Bpair=13, rng=rngB)
    assert P1 == P2, "pairs_for_assignment reproducibility failed"

    P0 = pairs_for_assignment(S1, n, Bpair=0, rng=rngA)
    assert P0 == [], "pairs_for_assignment(Bpair=0) should return an empty list"

    print("✓ pairs_for_assignment: deterministic sampling and Bpair=0 corner handled")


def _test_MCVarRange_RA_basic_invariants():
    rng = np.random.default_rng(123)
    n, p, n1 = 10, 4, 4
    X = rng.normal(size=(n, p))
    beta = rng.normal(size=p)
    y0 = X @ beta + rng.normal(scale=0.5, size=n)
    y1 = y0 + 0.3

    S1 = rng.choice(n, size=n1, replace=False)
    Pi = make_random_reveal_order(n1, rng)

    Vsum, Rmax, Vpqv, Rswap = estimate_VR_for_assignment(
        S1, X, y1, y0, Pi,
        Bi=n, Bcond=3, Bj=0, rng=np.random.default_rng(999),
        method="RA", branch="auto", apply_ucb=False, return_rswap=True
    )

    assert np.isfinite(Vsum) and np.isfinite(Rmax)
    assert np.isfinite(Vpqv) and np.isfinite(Rswap)
    assert Vsum >= 0.0 and Vpqv >= 0.0
    assert Rswap >= Rmax - 1e-12
    print("✓ MCVarRange (RA) returns finite outputs and Rswap ≥ Rhat")


def _test_RA_RB_reduces_within_i_variance():
    rng = np.random.default_rng(123)
    n, p, n1 = 60, 6, 30
    X = rng.normal(size=(n, p))
    beta = rng.normal(size=p)
    y0 = X @ beta + rng.normal(scale=0.5, size=n)
    y1 = y0 + 0.3
    S1 = rng.choice(n, size=n1, replace=False)
    Pi = rng.permutation(n1).astype(int)

    # Fix step t=0 and a representative candidate i
    S_past = S1[Pi[:0]]
    S_future = S1[Pi[0:]]
    i_glob = int(S_future[0])
    S0 = _indices_complement(S1, n)
    future_wo_i = [int(u) for u in S_future if int(u) != int(i_glob)]
    T_size = max(0, n1 - 0 - 1)
    Bcond = 40

    # Precompute identical T_b draws
    T_draws = []
    for _ in range(Bcond):
        if T_size > 0:
            T = (np.asarray(future_wo_i, dtype=int)
                 if len(future_wo_i) <= T_size
                 else rng.choice(future_wo_i, size=T_size, replace=False).astype(int))
        else:
            T = np.array([], dtype=int)
        T_draws.append(T)

    # RB variance across b
    xi_RB = []
    for T in T_draws:
        S1_prox = np.concatenate([S_past, np.array([i_glob], dtype=int), T]).astype(int)
        vals = _swap_deltas_RA_over_all_controls_on_set(
            X, y1, y0, S1_prox, int(i_glob), list(map(int, S0)), branch="auto"
        )
        xi_RB.append(float(np.mean(vals)) if vals.size > 0 else 0.0)
    s2_RB = float(np.var(np.asarray(xi_RB), ddof=1))

    # Naive variance across b: same T_b, but one random control j per draw
    xi_naive = []
    for T in T_draws:
        S1_prox = np.concatenate([S_past, np.array([i_glob], dtype=int), T]).astype(int)
        j_b = int(rng.choice(S0))
        xi_naive.append(float(_swap_delta_RA_on_set(X, y1, y0, S1_prox, int(i_glob), j_b, branch="auto")))
    s2_naive = float(np.var(np.asarray(xi_naive), ddof=1))

    assert s2_RB <= s2_naive + 1e-12, "RB variance should not exceed naive variance"
    print("✓ RB variance ≤ naive variance (fixed T_b)")


def _test_VR_UCB_monotone():
    rng = np.random.default_rng(77)
    n, p, n1 = 24, 6, 8
    X = rng.normal(size=(n, p))
    beta = rng.normal(size=p)
    y0 = X @ beta + rng.normal(scale=0.6, size=n)
    y1 = y0 + 0.4
    S1 = rng.choice(n, size=n1, replace=False)
    Pi = make_random_reveal_order(n1, rng)

    # Same randomness for both calls
    rng_plain = np.random.default_rng(1001)
    rng_ucb   = np.random.default_rng(1001)

    V_plain, R_plain, _ = estimate_VR_for_assignment(
        S1, X, y1, y0, Pi, Bi=min(5, len(S1)), Bcond=5, rng=rng_plain, method="RA",
        branch="auto", apply_ucb=False
    )
    V_ucb, R_ucb, _ = estimate_VR_for_assignment(
        S1, X, y1, y0, Pi, Bi=min(5, len(S1)), Bcond=5, rng=rng_ucb, method="RA",
        branch="auto", apply_ucb=True, eta=0.05, delta_max=5.0
    )
    _check_close(V_plain, V_ucb, "V unaffected by UCB switch", rtol=0.0, atol=0.0)
    assert R_ucb >= R_plain - 1e-15, "UCB radius should be ≥ plain radius"
    print("✓ UCB monotonicity: R_ucb ≥ R_plain and V unchanged")


def _test_RA_equals_DIM_when_X_empty_pointwise():
    rng = np.random.default_rng(303)
    n, p, n1 = 14, 0, 5   # empty design ⇒ RA degenerates to DiM
    X = np.empty((n, 0))
    y0 = rng.normal(size=n)
    y1 = y0 + 0.25
    S1 = rng.choice(n, size=n1, replace=False)
    S0 = _indices_complement(S1, n)

    tau_ra = tau_hat_RA(S1, X, y1, y0, branch="auto")
    tau_dim = tau_hat_DIM(S1, y1, y0)
    _check_close(tau_ra, tau_dim, "RA equals DIM when X is empty: tau", rtol=0.0, atol=1e-12)

    i = int(rng.choice(S1))
    j = int(rng.choice(S0))
    d_ra = _swap_delta_RA_on_set(X, y1, y0, S1, i, j, branch="auto")
    d_dim = swap_delta_DIM(y1, y0, S1, i, j)
    _check_close(d_ra, d_dim, "RA equals DIM when X is empty: swap delta", rtol=0.0, atol=1e-12)

    print("✓ RA equals DIM when X is empty (pointwise sanity check)")



## --------------------------------------------------------------------
## 4-B. Run tests
## --------------------------------------------------------------------

def run_tests(verbose: bool = True) -> None:
    registry = [
        ("mu-section", lambda: _test_mu_section(verbose)),
        ("deltas-section", lambda: _test_deltas_section(verbose)),
        ("stress-section", lambda: _test_stress_section(verbose)),
        ("tau_RA_vs_primitives", _test_tau_hat_RA_matches_ols_primitives),
        ("tau_RA_highdim_unpenalized_intercept", _test_tau_hat_RA_highdim_uses_unpenalized_intercept),
        ("swap_RA_refit_vs_norefit", _test_swap_delta_RA_refit_vs_norefit),
        ("VR_DIM_enumeration", _test_MCVarRange_DIM_enumeration),
        ("MCBias_DIM_exact", _test_MCBias_DIM_small_exact),
        ("MCBias_RA_exact", _test_MCBias_RA_small_exact),
        ("MCBias_sampling_repro_DIM", _test_MCBias_sampling_reproducible),
        ("MCBias_sampling_repro_RA", _test_MCBias_sampling_reproducible_RA),
        ("pairs_sampler_determinism", _test_pairs_for_determinism_and_corners),
        ("VR_RA_basic_invariants", _test_MCVarRange_RA_basic_invariants),
        ("RA_RB_variance_reduction", _test_RA_RB_reduces_within_i_variance),
        ("VR_UCB_monotone", _test_VR_UCB_monotone),
        ("RA_equals_DIM_when_X_empty_pointwise", _test_RA_equals_DIM_when_X_empty_pointwise),
    ]
    for name, test in registry:
        t0 = time.perf_counter()
        try:
            test()
            dt = time.perf_counter() - t0
            print(f"✓ {name} — passed in {dt:.3f}s")
        except Exception as e:
            dt = time.perf_counter() - t0
            print(f"✗ {name} — FAILED in {dt:.3f}s: {e}")
            raise
    if verbose:
        print("All tests passed.")


if __name__ == "__main__":  # pragma: no cover
    run_tests(verbose=True)
