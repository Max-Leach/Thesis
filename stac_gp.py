"""
stac_gp.py
==========
STAC-GP: a Static Gaussian-Process width model with an Online Localised
Conformal calibration layer, specialised to short-horizon (e.g. 5-minute,
30-minute) equity index futures.

This is a direct implementation of Algorithm 1 of

    Leach, Koukorinis & Tobar,
    "Calibrated uncertainty in Gaussian Process models using
     Online Localised Conformal Prediction" (STAC-GP).

Design (thesis Section 4)
-------------------------
The point forecast of the short-horizon move is the martingale Yhat = 0, so all
modelling lives in the *half-width* of the band:

    w_prior_t = sigma0_session * sqrt(H_bars)      # start-of-day realised-vol prior
                                                   # (constant within a session)
    w_GP_t    = exp(f(x_t)) * w_prior_t            # a STATIC, frozen GP rescales it
                                                   #   intraday (log-vol adjustment)
    S_t       = |Y_t| / w_GP_t                     # scale-free non-conformity score

Online localised conformal calibration (both axes of non-exchangeability):

    weights   w_{t,i} = H_h(x_t, x_i) * rho^{t-i}      # covariate localiser x recency (eq. 10)
              H_h(x, x') = exp( -||x_std - x'_std||^2 / h )
    q_t(x_t)  = Q(1 - alpha_t ; sum_i w_{t,i} delta_{S_i})   # localised weighted quantile
    band      C_t = [ mid_t - q_t w_GP_t , mid_t + q_t w_GP_t ]
    err_t     = 1{ |Y_t| > q_t w_GP_t }
    alpha_{t+1} = clip( alpha_t + gamma (alpha - err_t) , 0, 1 )   # online level feedback

The band family is monotone in alpha, so the projected-update long-run coverage
guarantee (Lai & Raskutti) transfers. The localisation buys *efficiency*
(correctly sized bands by regime), not conditional coverage.

Expected input schema
---------------------
A single pandas DataFrame of intraday bars in chronological order. Column names
are mapped through `ColumnMap` (defaults in parentheses):

    timestamp (`timestamp`)  : parseable bar-close datetime
    close     (`close`)      : mid-price proxy (or set close=(bid+ask)/2 upstream)
    high      (`high`)       : bar high        [optional -> feature]
    low       (`low`)        : bar low         [optional -> feature]
    volume    (`volume`)     : bar volume      [optional -> feature]
    ofi       (`ofi`)        : order-flow imbalance   [optional -> feature]
    queue     (`queue`)      : queue depth / liquidity[optional -> feature]
    event     (`event`)      : scheduled-event flag {0,1} [optional -> GP + localiser]

Bars are assumed to be a single contract at a fixed `bar_minutes` frequency,
already restricted to the trading session(s) you want to evaluate. Sessions are
inferred from the calendar date of `timestamp` (override via `session_key`).

Typical use
-----------
    from stac_gp import STACGP, ColumnMap

    model = STACGP(horizon_minutes=5, bar_minutes=1, alpha=0.10,
                   gamma=0.05, R=300, rho=0.99)
    res = model.fit_run(bars, train_frac=0.5)     # earlier sessions fit+freeze GP
    print(res.summary())                          # coverage / width / width-vol
    df = res.frame                                # per-bar bands & diagnostics

Only realised-volatility is used for sigma0 (no implied-vol dependency).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Column mapping / configuration
# ----------------------------------------------------------------------
@dataclass
class ColumnMap:
    timestamp: str = "timestamp"
    close: str = "close"
    high: str = "high"
    low: str = "low"
    volume: str = "volume"
    ofi: str = "ofi"          # order-flow imbalance (optional)
    queue: str = "queue"      # queue depth / liquidity proxy (optional)
    event: str = "event"      # scheduled-event flag 0/1 (optional)


# ----------------------------------------------------------------------
# Weighted conformal quantile (finite-sample corrected, with test self-mass)
# ----------------------------------------------------------------------
def weighted_conformal_quantile(scores: np.ndarray,
                                weights: np.ndarray,
                                level: float) -> float:
    """
    Localised weighted `level`-quantile of calibration `scores` with a
    finite-sample correction based on the Kish effective sample size.

    Weights are normalised over the calibration set (they sum to 1); the target
    level is inflated by the standard (1 + 1/n_eff) split-conformal factor, where
    n_eff = (sum w)^2 / sum(w^2) is the effective number of local points. This
    mirrors the ceil((n+1)(1-alpha)) correction in the equal-weight case while
    staying finite (when the inflated level exceeds all mass, the largest local
    score is returned). The online alpha_t feedback then secures long-run
    coverage under drift, so we do not force +inf bands on thin local samples.

        p_i    = w_i / sum_j w_j
        n_eff  = (sum w)^2 / sum(w^2)
        lvl    = min(1, level * (1 + 1/n_eff))
        q      = smallest S_(k) with sum_{i: S_i <= S_(k)} p_i >= lvl
                 (falls back to max local score if lvl exceeds cumulative mass)
    """
    scores = np.asarray(scores, float)
    weights = np.asarray(weights, float)
    if scores.size == 0:
        return np.inf
    sw = weights.sum()
    if sw <= 0:                                  # no local mass -> unweighted
        weights = np.ones_like(scores)
        sw = weights.sum()
    n_eff = (sw * sw) / float(np.sum(weights ** 2))
    lvl = min(1.0, float(level) * (1.0 + 1.0 / max(n_eff, 1.0)))
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    cum = np.cumsum(weights[order]) / sw
    idx = int(np.searchsorted(cum, lvl, side="left"))
    if idx >= len(s):
        return float(s[-1])                      # largest local score (finite)
    return float(s[idx])


# ----------------------------------------------------------------------
# Static GP width model (fit once, frozen)
# ----------------------------------------------------------------------
class StaticGPWidth:
    """
    Fits f ~ GP(0, k) to the log realised-volatility residual
        z_t = log( |Y_t| / w_prior_t )
    on historical bars, then freezes the posterior mean. The width factor is
    g(x) = exp(f(x)); the GP-driven half-width is w_GP = g(x) * w_prior.

    Exact GP inference is O(n^3); for intraday data we cap the training set at
    `n_gp_max` points by uniform subsampling (a standard, faithful compromise
    for a *static* fit; swap in sparse-spectrum/inducing points if desired).
    """

    def __init__(self, kernel: str = "rbf", n_gp_max: int = 1500,
                 n_restarts: int = 2, seed: int = 0):
        self.kernel_name = kernel
        self.n_gp_max = int(n_gp_max)
        self.n_restarts = int(n_restarts)
        self.seed = int(seed)
        self.gpr = None
        self.mu_ = None
        self.sd_ = None

    def _make_kernel(self, d: int):
        from sklearn.gaussian_process.kernels import (
            RBF, Matern, ConstantKernel, WhiteKernel)
        base = (Matern(np.ones(d), (1e-1, 1e2), nu=1.5)
                if self.kernel_name == "matern"
                else RBF(np.ones(d), (1e-1, 1e2)))
        return ConstantKernel(1.0, (1e-2, 1e2)) * base + WhiteKernel(0.5, (1e-3, 1e1))

    def fit(self, X: np.ndarray, z: np.ndarray) -> "StaticGPWidth":
        from sklearn.gaussian_process import GaussianProcessRegressor
        X = np.asarray(X, float)
        z = np.asarray(z, float)
        # causal standardisation using the training window only
        self.mu_ = X.mean(0)
        self.sd_ = X.std(0) + 1e-12
        Xs = (X - self.mu_) / self.sd_
        # subsample for tractable exact inference
        rng = np.random.RandomState(self.seed)
        n = len(Xs)
        if n > self.n_gp_max:
            sel = rng.choice(n, self.n_gp_max, replace=False)
            Xs, z = Xs[sel], z[sel]
        self.gpr = GaussianProcessRegressor(
            kernel=self._make_kernel(Xs.shape[1]), alpha=0.0,
            n_restarts_optimizer=self.n_restarts, normalize_y=True)
        self.gpr.fit(Xs, z)
        return self

    def log_factor(self, X: np.ndarray, clip: float = 3.0) -> np.ndarray:
        """Return f(x) (log width factor), clipped for numerical safety."""
        Xs = (np.asarray(X, float) - self.mu_) / self.sd_
        return np.clip(self.gpr.predict(Xs), -clip, clip)

    @property
    def kernel_(self):
        return None if self.gpr is None else self.gpr.kernel_


# ----------------------------------------------------------------------
# Feature / target construction (all causal, session-aware)
# ----------------------------------------------------------------------
def _session_key(ts: pd.Series, session_key: Optional[str]) -> np.ndarray:
    if session_key is not None:
        return ts.dt.floor(session_key).to_numpy()
    return ts.dt.normalize().to_numpy()   # calendar-date sessions


def build_panel(df: pd.DataFrame,
                horizon_minutes: int,
                bar_minutes: int,
                cols: ColumnMap,
                sigma0_lookback_sessions: int = 5,
                ewma_halflife_bars: int = 60,
                session_key: Optional[str] = None) -> pd.DataFrame:
    """
    Turn raw bars into a per-bar panel with the response, the start-of-day
    volatility prior, and the causal feature inputs. Returns a DataFrame with:

        ts, session, mid, Y, w_prior, tod_frac,
        ret1, rv, rng, vol_z, [ofi, queue, event]   (whichever are present)

    `Y_t` is the forward mid change over the horizon, computed *within* a session
    (never across the overnight gap). `w_prior_t = sigma0 * sqrt(H_bars)` with
    sigma0 the per-bar realised vol fixed at the session open (start-of-day).
    """
    d = df.copy()
    ts = pd.to_datetime(d[cols.timestamp])
    d = d.assign(_ts=ts).sort_values("_ts").reset_index(drop=True)
    ts = d["_ts"]
    mid = d[cols.close].astype(float).to_numpy()
    session = _session_key(ts, session_key)

    H_bars = max(1, int(round(horizon_minutes / bar_minutes)))

    # ---- forward horizon change within a session (Yhat = 0) ----
    s = pd.Series(mid)
    fwd = s.shift(-H_bars) - s
    same_session = pd.Series(session).shift(-H_bars).to_numpy() == session
    Y = np.where(same_session, fwd.to_numpy(), np.nan)

    # ---- causal 1-bar return & EWMA realised vol ----
    ret1 = np.diff(mid, prepend=mid[0])
    # zero the return across session boundaries
    new_sess = np.concatenate([[True], session[1:] != session[:-1]])
    ret1 = np.where(new_sess, 0.0, ret1)
    ewma_var = (pd.Series(ret1 ** 2)
                .ewm(halflife=ewma_halflife_bars, min_periods=ewma_halflife_bars)
                .mean().shift(1).to_numpy())          # causal per-bar variance
    sigma_bar = np.sqrt(ewma_var)

    # ---- start-of-day sigma0: value at the first bar of each session,
    #      blended over the last `sigma0_lookback_sessions` session-opens ----
    panel = pd.DataFrame({"ts": ts.to_numpy(), "session": session,
                          "mid": mid, "sigma_bar": sigma_bar})
    first_idx = panel.groupby("session", sort=False).head(1).index
    open_sigma = pd.Series(np.nan, index=panel.index)
    open_sigma.loc[first_idx] = panel["sigma_bar"].loc[first_idx].to_numpy()
    # per-session open sigma, then trailing mean over prior sessions (causal)
    sess_open = (panel.assign(open_sigma=open_sigma)
                 .groupby("session", sort=False)["open_sigma"].first())
    sigma0_sess = sess_open.rolling(sigma0_lookback_sessions, min_periods=1).mean().shift(0)
    # note: sess_open at a session uses that session's OPEN bar sigma_bar, which is
    # itself shifted(1) => only prior-bar info; the rolling mean smooths across days.
    sigma0 = panel["session"].map(sigma0_sess).to_numpy()
    w_prior = sigma0 * np.sqrt(H_bars)

    # ---- causal features ----
    # time-of-day fraction within the session (0 at open, 1 at close)
    tod = panel.groupby("session", sort=False).cumcount().to_numpy().astype(float)
    tod_n = panel.groupby("session", sort=False)["session"].transform("size").to_numpy()
    tod_frac = np.where(tod_n > 1, tod / np.maximum(tod_n - 1, 1), 0.0)

    absret = np.abs(ret1)
    rv = pd.Series(absret).shift(1).rolling(H_bars * 3, min_periods=1).mean().to_numpy()
    out = {
        "ts": ts.to_numpy(), "session": session, "mid": mid,
        "Y": Y, "w_prior": w_prior, "tod_frac": tod_frac,
        "ret_recent": pd.Series(ret1).shift(1).rolling(H_bars, min_periods=1).sum().to_numpy(),
        "rv": rv,
        "sigma_bar": sigma_bar,
    }
    if cols.high in d and cols.low in d:
        rng = (d[cols.high].astype(float).to_numpy() - d[cols.low].astype(float).to_numpy())
        out["rng"] = pd.Series(rng).shift(1).rolling(H_bars, min_periods=1).mean().to_numpy()
    if cols.volume in d:
        lv = np.log(np.clip(d[cols.volume].astype(float).to_numpy(), 1.0, None))
        out["vol_z"] = pd.Series(lv).shift(1).rolling(H_bars * 5, min_periods=1).mean().to_numpy()
    if cols.ofi in d:
        out["ofi"] = pd.Series(d[cols.ofi].astype(float).to_numpy()).shift(1).to_numpy()
    if cols.queue in d:
        out["queue"] = pd.Series(d[cols.queue].astype(float).to_numpy()).shift(1).to_numpy()
    if cols.event in d:
        out["event"] = d[cols.event].astype(float).to_numpy()   # known ahead of time

    return pd.DataFrame(out)


# feature groups: everything drives the GP; vol-state features drive the localiser
GP_FEATURES_ALL = ["tod_frac", "ret_recent", "rv", "rng", "vol_z", "ofi", "queue", "event"]
LOC_FEATURES_VOLSTATE = ["rv", "rng", "ret_recent", "event"]


def _present(panel: pd.DataFrame, names: Sequence[str]) -> list:
    return [c for c in names if c in panel.columns]


# ----------------------------------------------------------------------
# Results container
# ----------------------------------------------------------------------
@dataclass
class STACGPResult:
    frame: pd.DataFrame
    alpha: float
    meta: dict = field(default_factory=dict)

    def summary(self, by_vol_quartile: bool = True) -> pd.DataFrame:
        f = self.frame
        fin = f[np.isfinite(f["width"])]
        rows = [{
            "scope": "all",
            "n": int(len(f)),
            "coverage": float(f["covered"].mean()),
            "target": 1 - self.alpha,
            "avg_width": float(fin["width"].mean()),
            "median_width": float(fin["width"].median()),
            "width_vol": float(fin["width"].diff().abs().mean()),
            "inf_rate": float((~np.isfinite(f["width"])).mean()),
        }]
        if by_vol_quartile and "w_GP" in f and len(f) >= 8:
            q = pd.qcut(f["w_GP"], 4, labels=["Q1 calm", "Q2", "Q3", "Q4 stressed"],
                        duplicates="drop")
            for lab, g in f.groupby(q, observed=True):
                gfin = g[np.isfinite(g["width"])]
                rows.append({
                    "scope": f"vol {lab}", "n": int(len(g)),
                    "coverage": float(g["covered"].mean()), "target": 1 - self.alpha,
                    "avg_width": float(gfin["width"].mean()),
                    "median_width": float(gfin["width"].median()),
                    "width_vol": float(gfin["width"].diff().abs().mean()),
                    "inf_rate": float((~np.isfinite(g["width"])).mean()),
                })
        return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# STAC-GP: Algorithm 1
# ----------------------------------------------------------------------
class STACGP:
    def __init__(self,
                 horizon_minutes: int = 5,
                 bar_minutes: int = 1,
                 alpha: float = 0.10,
                 gamma: Optional[float] = 0.05,
                 alpha_floor: float = 0.01,
                 alpha_ceil: float = 0.50,
                 R: int = 300,
                 rho: float = 0.99,
                 h: Optional[float] = None,
                 kernel: str = "rbf",
                 n_gp_max: int = 1500,
                 sigma0_lookback_sessions: int = 5,
                 ewma_halflife_bars: int = 60,
                 gp_features: Optional[Sequence[str]] = None,
                 loc_features: Optional[Sequence[str]] = None,
                 cols: Optional[ColumnMap] = None,
                 session_key: Optional[str] = None,
                 seed: int = 0):
        self.horizon_minutes = horizon_minutes
        self.bar_minutes = bar_minutes
        self.alpha = float(alpha)
        self.gamma = gamma
        # bound the adaptive nominal level away from {0,1}: at alpha_t=0 the target
        # quantile level is 1 and q_t collapses to the max window score (huge bands);
        # a small floor caps q_t at a high-but-finite quantile instead.
        self.alpha_floor = float(alpha_floor)
        self.alpha_ceil = float(alpha_ceil)
        self.R = int(R)
        self.rho = float(rho)
        self.h = h
        self.kernel = kernel
        self.n_gp_max = n_gp_max
        self.sigma0_lookback_sessions = sigma0_lookback_sessions
        self.ewma_halflife_bars = ewma_halflife_bars
        self.gp_features = gp_features
        self.loc_features = loc_features
        self.cols = cols or ColumnMap()
        self.session_key = session_key
        self.seed = seed
        self.gp_: Optional[StaticGPWidth] = None
        self.gp_feat_: Optional[list] = None
        self.loc_feat_: Optional[list] = None

    # ---- offline: build panel and freeze the GP on the training slice ----
    def fit_run(self, bars: pd.DataFrame, train_frac: float = 0.5,
                n_train_sessions: Optional[int] = None) -> STACGPResult:
        panel = build_panel(bars, self.horizon_minutes, self.bar_minutes, self.cols,
                            self.sigma0_lookback_sessions, self.ewma_halflife_bars,
                            self.session_key)

        self.gp_feat_ = _present(panel, self.gp_features or GP_FEATURES_ALL)
        self.loc_feat_ = _present(panel, self.loc_features or LOC_FEATURES_VOLSTATE)

        # valid rows: finite target, prior, and all needed features
        need = list(dict.fromkeys(["Y", "w_prior"] + self.gp_feat_ + self.loc_feat_))
        valid = np.isfinite(panel[need].to_numpy().astype(float)).all(1) & (panel["w_prior"] > 0)
        panel = panel.loc[valid].reset_index(drop=True)

        # train/test split by session (fit on earlier sessions, stream the rest)
        sessions = pd.unique(panel["session"])
        if n_train_sessions is None:
            n_train_sessions = max(1, int(round(len(sessions) * train_frac)))
        train_sessions = set(sessions[:n_train_sessions])
        is_test = ~panel["session"].isin(train_sessions).to_numpy()
        train_mask = ~is_test

        # ---- static GP fit on the training slice, then freeze ----
        z = np.log(np.clip(np.abs(panel["Y"].to_numpy()), 1e-12, None)
                   / panel["w_prior"].to_numpy())
        Xgp = panel[self.gp_feat_].to_numpy(float)
        self.gp_ = StaticGPWidth(kernel=self.kernel, n_gp_max=self.n_gp_max,
                                 seed=self.seed).fit(Xgp[train_mask], z[train_mask])

        # frozen width for the whole panel
        w_GP = np.exp(self.gp_.log_factor(Xgp)) * panel["w_prior"].to_numpy()
        S = np.abs(panel["Y"].to_numpy()) / w_GP

        frame = self._online_localised_conformal(panel, S, w_GP, is_test)
        meta = dict(horizon_minutes=self.horizon_minutes, bar_minutes=self.bar_minutes,
                    H_bars=max(1, int(round(self.horizon_minutes / self.bar_minutes))),
                    gp_features=self.gp_feat_, loc_features=self.loc_feat_,
                    n_train_sessions=int(n_train_sessions), R=self.R, rho=self.rho,
                    gamma=self._gamma(is_test.sum()), h=self._h_used,
                    alpha_floor=self.alpha_floor, alpha_ceil=self.alpha_ceil,
                    kernel=str(self.gp_.kernel_))
        return STACGPResult(frame=frame, alpha=self.alpha, meta=meta)

    def _gamma(self, T_test: int) -> float:
        return (0.5 / np.sqrt(max(T_test, 1))) if self.gamma is None else float(self.gamma)

    # ---- online loop (Algorithm 1) ----
    def _online_localised_conformal(self, panel, S, w_GP, is_test) -> pd.DataFrame:
        Xloc_all = panel[self.loc_feat_].to_numpy(float)
        Y = panel["Y"].to_numpy(float)
        mid = panel["mid"].to_numpy(float)
        n = len(panel)
        gamma = self._gamma(int(is_test.sum()))

        # bandwidth: median heuristic on squared distances in the first full window
        self._h_used = self._resolve_h(Xloc_all, is_test)

        alpha_t = self.alpha
        R, rho, h = self.R, self.rho, self._h_used

        lo = np.full(n, np.nan); hi = np.full(n, np.nan)
        width = np.full(n, np.nan); q_arr = np.full(n, np.nan)
        covered = np.zeros(n, bool); alpha_path = np.full(n, np.nan)
        out_rows = np.zeros(n, bool)

        # calibration memory: indices of past points that had a revealed score
        for t in range(n):
            if not is_test[t]:
                continue
            lo_i = max(0, t - R)
            cal_idx = np.arange(lo_i, t)
            if len(cal_idx) == 0:
                continue

            # localiser (standardise within the window) x recency decay (eq. 10)
            Xw = Xloc_all[cal_idx]
            xt = Xloc_all[t]
            mu = Xw.mean(0); sd = Xw.std(0) + 1e-12
            Zw = (Xw - mu) / sd
            zt = (xt - mu) / sd
            d2 = np.einsum("ij,ij->i", Zw - zt, Zw - zt)      # squared distances
            H_loc = np.exp(-d2 / max(h, 1e-12))
            recency = rho ** (t - cal_idx)                     # rho^{t-i}
            w = H_loc * recency

            q = weighted_conformal_quantile(S[cal_idx], w, 1.0 - alpha_t)
            q_arr[t] = q
            alpha_path[t] = alpha_t
            out_rows[t] = True

            if np.isfinite(q):
                half = q * w_GP[t]
                lo[t], hi[t] = mid[t] - half, mid[t] + half
                width[t] = 2.0 * half
                covered[t] = abs(Y[t]) <= half
            else:
                lo[t], hi[t] = -np.inf, np.inf
                width[t] = np.inf
                covered[t] = True

            err = 0.0 if covered[t] else 1.0
            alpha_t = float(np.clip(alpha_t + gamma * (self.alpha - err),
                                    self.alpha_floor, self.alpha_ceil))

        f = panel.loc[out_rows, ["ts", "session", "mid", "Y", "w_prior"]].copy()
        idx = np.where(out_rows)[0]
        f["w_GP"] = w_GP[idx]
        f["S"] = S[idx]
        f["q"] = q_arr[idx]
        f["alpha_t"] = alpha_path[idx]
        f["lower"] = lo[idx]
        f["upper"] = hi[idx]
        f["width"] = width[idx]
        f["covered"] = covered[idx]
        return f.reset_index(drop=True)

    def _resolve_h(self, Xloc_all, is_test) -> float:
        if self.h is not None:
            return float(self.h)
        first_test = np.argmax(is_test)
        lo_i = max(0, first_test)
        win = Xloc_all[lo_i: lo_i + self.R]
        if len(win) < 5:
            win = Xloc_all[:max(5, self.R)]
        mu = win.mean(0); sd = win.std(0) + 1e-12
        Z = (win - mu) / sd
        # median pairwise squared distance (subsample for speed)
        m = min(len(Z), 200)
        Zs = Z[np.linspace(0, len(Z) - 1, m).astype(int)]
        G = ((Zs[:, None, :] - Zs[None, :, :]) ** 2).sum(-1)
        med = np.median(G[np.triu_indices(m, k=1)]) if m > 1 else 1.0
        return float(max(med, 1e-6))


__all__ = ["STACGP", "STACGPResult", "StaticGPWidth", "ColumnMap",
           "build_panel", "weighted_conformal_quantile",
           "GP_FEATURES_ALL", "LOC_FEATURES_VOLSTATE"]
