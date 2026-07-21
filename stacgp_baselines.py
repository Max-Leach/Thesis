"""
stacgp_baselines.py
====================
Section 5.2 baseline comparison for STAC-GP, built on top of stacgp_v2.py.

Design notes (confirmed / documented choices)
----------------------------------------------
Most of the named baselines are points on the same 2x2x2 ablation grid that
section 5.4 describes (GP on/off x localiser on/off x adaptive on/off),
differing only in the calibration LAYER wrapped around the *same* frozen GP
width w_GP:

    STAC-GP        = GP on, localiser on,  adaptive on   (Algorithm 1, as before)
    ACI             = GP on, localiser off, adaptive on   (global recency-weighted
                                                           quantile + eq. 13 feedback)
    LCP             = GP on, localiser on,  adaptive off  (localised quantile at a
                                                           fixed nominal alpha)
    Split conformal = GP on, no rolling recompute at all -- a SINGLE global
                      threshold fit once on a calibration slice and then frozen
                      for the rest of the evaluation stream (this is the one
                      baseline that genuinely isn't a point on the toggle grid,
                      since ACI/LCP/STAC-GP all recompute their quantile every
                      step; split conformal by definition does not).

Two more are NOT conformal at all ("no CP"):

    Start-of-day prior = g=1 (no GP), no data-driven calibration whatsoever --
                          a purely parametric band using a Gaussian assumption
                          baked into eq. (4)'s Brownian scaling: since S_t =
                          |Y_t|/w_prior_t ~ half-normal(1) under that assumption,
                          the (1-alpha) quantile is q = Phi^-1(1 - alpha/2)
                          (same as a textbook two-sided z-interval), applied as
                          a constant multiplier throughout, never recalibrated.
    GP credible band    = uses the frozen GP's own posterior predictive
                          mean/std of z = log(|Y|/w_prior) (the WhiteKernel
                          term in the kernel means predict(..., return_std=True)
                          already reflects total predictive uncertainty, not
                          just epistemic function uncertainty) to build a
                          credible interval q_t = exp(mean_z(x_t) + z*std_z(x_t)),
                          again with NO empirical/conformal correction -- the
                          whole point of this baseline is to show what the raw
                          GP's nominal coverage claim looks like empirically.

CQR is a genuinely different architecture (gradient-boosted quantile
regression instead of a GP), conformalised with a single global additive
correction computed once on a calibration slice (standard split-CQR, Romano,
Patterson & Candes 2019) -- the regression prediction is per-x ("adaptive
quantile score"), the correction constant is not ("global threshold").

"Third-party bid-offer" is skipped: there is no bid/ask or spread data
anywhere in Data/, only OHLCV bars, so an external reference width cannot be
constructed without fabricating a proxy. The comparison table reports this
metric as not available rather than approximating it.

Fair-comparison alignment
--------------------------
All 7 implemented methods share: the same frozen GP (fit once on the training
sessions), the same test-session pool, and the same *evaluation window* --
the first `cal_bars` bars of the test stream are reserved as a calibration
slice (used by split conformal / CQR to fit their single global correction;
simply "warmed up over" by the rolling/online methods so their windows are
full), and every method's reported metrics are computed on the identical
remaining bars.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from stacgp_v2 import (StaticGPWidth, weighted_quantile,
                        GP_FEATURES_ALL, LOC_FEATURES)


# ----------------------------------------------------------------------
# Shared setup: fit the frozen GP once, split test stream into cal/eval
# ----------------------------------------------------------------------
@dataclass
class SharedSetup:
    panel: pd.DataFrame
    gp: StaticGPWidth
    gp_features: list
    loc_features: list
    w_prior: np.ndarray
    w_GP: np.ndarray
    S_gp: np.ndarray          # |Y|/w_GP  (coupled score, GP-based)
    S_prior: np.ndarray       # |Y|/w_prior (score against the raw Brownian prior)
    is_test: np.ndarray
    train_mask: np.ndarray
    cal_idx: np.ndarray       # panel row positions: calibration-only bars
    eval_idx: np.ndarray      # panel row positions: the shared evaluation window
    R: int
    rho: float
    h: float
    gamma: float
    alpha: float
    bars_per_session: int


def build_shared_setup(panel: pd.DataFrame, alpha: float = 0.10,
                        train_frac: float = 0.40, R_sessions: int = 10,
                        halflife_sessions: int = 5, cal_sessions: Optional[int] = None,
                        seed: int = 0) -> SharedSetup:
    gp_features = [c for c in GP_FEATURES_ALL if c in panel.columns]
    loc_features = [c for c in LOC_FEATURES if c in panel.columns]
    need = ["Y", "w_prior"] + gp_features
    valid = np.isfinite(panel[need].to_numpy(float)).all(1) & (panel["w_prior"] > 0)
    panel = panel.loc[valid].reset_index(drop=True)

    sessions = pd.unique(panel["session"])
    n_train_sessions = max(1, int(round(len(sessions) * train_frac)))
    train_sessions = set(sessions[:n_train_sessions])
    is_test = ~panel["session"].isin(train_sessions).to_numpy()
    train_mask = ~is_test

    bars_per_session = int(np.median(panel["session"].value_counts().to_numpy()))
    R = R_sessions * bars_per_session
    halflife_bars = halflife_sessions * bars_per_session
    rho = 0.5 ** (1.0 / halflife_bars)
    cal_bars = (cal_sessions or R_sessions) * bars_per_session

    z = np.log(np.clip(np.abs(panel["Y"].to_numpy()), 1e-12, None) / panel["w_prior"].to_numpy())
    Xgp = panel[gp_features].to_numpy(float)
    gp = StaticGPWidth(n_gp_max=600, n_restarts=1, seed=seed).fit(Xgp[train_mask], z[train_mask])

    w_prior = panel["w_prior"].to_numpy()
    w_GP = np.exp(gp.log_factor(Xgp)) * w_prior
    S_gp = np.abs(panel["Y"].to_numpy()) / w_GP
    S_prior = np.abs(panel["Y"].to_numpy()) / w_prior

    test_positions = np.where(is_test)[0]
    cal_idx = test_positions[:cal_bars]
    eval_idx = test_positions[cal_bars:]

    Xloc_train = panel.loc[train_mask, loc_features].to_numpy(float)
    h = _median_heuristic_h(Xloc_train)

    T_eval = len(eval_idx)
    gamma = 1.0 / np.sqrt(max(T_eval, 1))

    return SharedSetup(panel=panel, gp=gp, gp_features=gp_features, loc_features=loc_features,
                        w_prior=w_prior, w_GP=w_GP, S_gp=S_gp, S_prior=S_prior,
                        is_test=is_test, train_mask=train_mask, cal_idx=cal_idx, eval_idx=eval_idx,
                        R=R, rho=rho, h=h, gamma=gamma, alpha=alpha,
                        bars_per_session=bars_per_session)


def _median_heuristic_h(X: np.ndarray) -> float:
    mu = X.mean(0); sd = X.std(0) + 1e-12
    Z = (X - mu) / sd
    m = min(len(Z), 300)
    idx = np.linspace(0, len(Z) - 1, m).astype(int)
    Zs = Z[idx]
    G = ((Zs[:, None, :] - Zs[None, :, :]) ** 2).sum(-1)
    med = np.median(G[np.triu_indices(m, k=1)]) if m > 1 else 1.0
    return float(max(med, 1e-6))


def _frame_from_bands(setup: SharedSetup, half_width: np.ndarray, idx: np.ndarray,
                       extra: Optional[dict] = None) -> pd.DataFrame:
    panel = setup.panel
    mid = panel["mid"].to_numpy()[idx]
    Y = panel["Y"].to_numpy()[idx]
    half = half_width[idx]
    f = panel.loc[idx, ["ts", "session", "mid", "Y"]].copy()
    f["lower"] = mid - half
    f["upper"] = mid + half
    f["width"] = 2.0 * half
    f["covered"] = np.abs(Y) <= half
    if extra:
        for k, v in extra.items():
            f[k] = np.asarray(v)[idx] if np.ndim(v) else v
    return f.reset_index(drop=True)


# ----------------------------------------------------------------------
# 1. STAC-GP / ACI / LCP: one generalised online loop, three toggle settings
# ----------------------------------------------------------------------
def generalized_online_calibration(setup: SharedSetup, use_localiser: bool, use_adaptive: bool,
                                    alpha_floor: float = 0.01, alpha_ceil: float = 0.50) -> pd.DataFrame:
    panel = setup.panel
    Xloc_all = panel[setup.loc_features].to_numpy(float)
    Y = panel["Y"].to_numpy(float)
    S = setup.S_gp
    w_used = setup.w_GP
    n = len(panel)
    R, rho, h, gamma, alpha = setup.R, setup.rho, setup.h, setup.gamma, setup.alpha

    run_from = setup.is_test.nonzero()[0][0] if setup.is_test.any() else 0
    run_to = n

    alpha_t = alpha
    half = np.full(n, np.nan)
    q_arr = np.full(n, np.nan)
    alpha_path = np.full(n, np.nan)

    for t in range(run_from, run_to):
        lo_i = max(0, t - R)
        cal_idx = np.arange(lo_i, t)
        if len(cal_idx) == 0:
            continue

        if use_localiser:
            Xw = Xloc_all[cal_idx]
            xt = Xloc_all[t]
            mu = Xw.mean(0); sd = Xw.std(0) + 1e-12
            Zw = (Xw - mu) / sd
            zt = (xt - mu) / sd
            d2 = np.einsum("ij,ij->i", Zw - zt, Zw - zt)
            H_loc = np.exp(-d2 / max(h, 1e-12))
        else:
            H_loc = np.ones(len(cal_idx))     # Hh = 1: no covariate localisation

        recency = rho ** (t - cal_idx)
        w = H_loc * recency

        q = weighted_quantile(S[cal_idx], w, 1.0 - alpha_t)
        q_arr[t] = q
        alpha_path[t] = alpha_t

        half[t] = q * w_used[t] if np.isfinite(q) else np.inf

        if use_adaptive:
            covered = abs(Y[t]) <= half[t]
            err = 0.0 if covered else 1.0
            alpha_t = float(np.clip(alpha_t + gamma * (alpha - err), alpha_floor, alpha_ceil))
        # else: alpha_t stays fixed at the nominal alpha throughout (LCP)

    f = _frame_from_bands(setup, half, setup.eval_idx, extra={"q": q_arr, "alpha_t": alpha_path})
    return f


# ----------------------------------------------------------------------
# 2. Split conformal: one static global threshold, frozen after calibration
# ----------------------------------------------------------------------
def split_conformal_baseline(setup: SharedSetup) -> pd.DataFrame:
    S = setup.S_gp
    q_global = weighted_quantile(S[setup.cal_idx], np.ones(len(setup.cal_idx)), 1.0 - setup.alpha)
    half = np.full(len(setup.panel), np.nan)
    half[setup.eval_idx] = q_global * setup.w_GP[setup.eval_idx]
    f = _frame_from_bands(setup, half, setup.eval_idx, extra={"q": q_global})
    return f


# ----------------------------------------------------------------------
# 3. Parametric "no-CP" baselines
# ----------------------------------------------------------------------
def start_of_day_prior_baseline(setup: SharedSetup) -> pd.DataFrame:
    """g=1 (no GP), no conformal calibration: q = Phi^-1(1-alpha/2), the
    textbook two-sided normal critical value, applied as a constant."""
    z = norm.ppf(1.0 - setup.alpha / 2.0)
    half = np.full(len(setup.panel), np.nan)
    half[setup.eval_idx] = z * setup.w_prior[setup.eval_idx]
    return _frame_from_bands(setup, half, setup.eval_idx, extra={"q": z})


def gp_credible_band_baseline(setup: SharedSetup) -> pd.DataFrame:
    """Frozen GP's own posterior predictive credible interval for
    S = |Y|/w_GP, no conformal correction: q_t = exp(z_{1-alpha} * std_z(x_t))."""
    Xgp = setup.panel[setup.gp_features].to_numpy(float)
    mean_z, std_z = setup.gp.log_factor_mean_std(Xgp)
    z = norm.ppf(1.0 - setup.alpha)
    q_t = np.exp(z * std_z)
    half = np.full(len(setup.panel), np.nan)
    half[setup.eval_idx] = q_t[setup.eval_idx] * setup.w_GP[setup.eval_idx]
    return _frame_from_bands(setup, half, setup.eval_idx, extra={"q": q_t, "std_z": std_z})


# ----------------------------------------------------------------------
# 4. CQR: gradient-boosted quantile regression + one global conformal constant
# ----------------------------------------------------------------------
def cqr_baseline(setup: SharedSetup, seed: int = 0) -> pd.DataFrame:
    from sklearn.ensemble import GradientBoostingRegressor

    panel = setup.panel
    X = panel[setup.gp_features].to_numpy(float)
    y = setup.S_prior   # |Y| / w_prior: CQR replaces the GP entirely, so it is
                         # scored against the raw Brownian prior, not w_GP.

    gbr = GradientBoostingRegressor(loss="quantile", alpha=1.0 - setup.alpha,
                                     n_estimators=200, max_depth=3,
                                     learning_rate=0.05, random_state=seed)
    gbr.fit(X[setup.train_mask], y[setup.train_mask])
    qhat = gbr.predict(X)

    resid = y[setup.cal_idx] - qhat[setup.cal_idx]
    Qcorr = weighted_quantile(resid, np.ones(len(setup.cal_idx)), 1.0 - setup.alpha)
    # weighted_quantile assumes non-negative scores; resid can be signed, so
    # compute its corrected quantile directly instead of routing through it.
    resid_sorted = np.sort(resid)
    n_cal = len(resid_sorted)
    lvl = min(1.0, (1.0 - setup.alpha) * (1.0 + 1.0 / n_cal))
    k = int(np.ceil(lvl * n_cal)) - 1
    k = min(max(k, 0), n_cal - 1)
    Qcorr = float(resid_sorted[k])

    q_final = np.clip(qhat + Qcorr, 1e-6, None)
    half = np.full(len(panel), np.nan)
    half[setup.eval_idx] = q_final[setup.eval_idx] * setup.w_prior[setup.eval_idx]
    return _frame_from_bands(setup, half, setup.eval_idx, extra={"q": q_final})


# ----------------------------------------------------------------------
# 5. Run everything
# ----------------------------------------------------------------------
def run_all_methods(setup: SharedSetup) -> dict:
    methods = {
        "Start-of-day prior": start_of_day_prior_baseline(setup),
        "GP credible band": gp_credible_band_baseline(setup),
        "Split conformal": split_conformal_baseline(setup),
        "CQR": cqr_baseline(setup),
        "ACI": generalized_online_calibration(setup, use_localiser=False, use_adaptive=True),
        "LCP": generalized_online_calibration(setup, use_localiser=True, use_adaptive=False),
        "STAC-GP (ours)": generalized_online_calibration(setup, use_localiser=True, use_adaptive=True),
    }
    return methods


__all__ = ["SharedSetup", "build_shared_setup", "generalized_online_calibration",
           "split_conformal_baseline", "start_of_day_prior_baseline",
           "gp_credible_band_baseline", "cqr_baseline", "run_all_methods"]
