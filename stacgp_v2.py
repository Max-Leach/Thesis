"""
stacgp_v2.py
============
A from-scratch implementation of STAC-GP (Algorithm 1 of the thesis paper
"Calibrated uncertainty in Gaussian Process models using Online Localised
Conformal Prediction"), built directly against equations (4)-(13) and confirmed,
explicit modelling choices for the NQU6 30-minute futures data in Data/.

This module is intentionally independent of the pre-existing stac_gp.py in this
repo (kept only as reference / prior work). Every non-obvious modelling choice
below was confirmed with the thesis author rather than assumed; see the
"CONFIRMED CHOICES" block.

CONFIRMED CHOICES
-----------------
- Instrument / bars   : NQU6 Index, 30-minute bars, restricted to "dense"
                        sessions (>= 30 of the ~46 possible bars in a session).
- sigma0 (eq. 4)      : average of HistVol10 and ImpliedVol from the daily
                        NQU6_HistoricalVolatility.csv file, read from the most
                        recent PRIOR trading day (no lookahead), converted from
                        an annualised percentage to a per-bar price-scale sigma
                        via sigma0 = P_open * (blend/100) * sqrt(1/(252*B)),
                        B = median bars per dense session.
- Horizon H           : 1 bar (30 minutes).
- Features xt         : restricted to what the data actually supports --
                        time-of-day fraction, recent return, recent realised
                        range, recent realised |return| (volatility proxy),
                        and a log-volume liquidity proxy. No OFI / queue-depth
                        / scheduled-event features (not present in the data).
- Localiser features  : the volatility-state subset (recent return, recent
                        realised range, recent realised |return|).
- Kernel              : RBF with ARD (one lengthscale per feature).
- Bandwidth h         : fixed once via the median-heuristic on the training
                        window (no online grid/Hedge selection over h).
- alpha (target)      : 0.10 (90% nominal coverage).
- R (calibration window): 10 sessions worth of bars (~460 bars).
- rho (recency decay) : half-life of 5 sessions worth of bars (~230 bars).
- gamma (step size)   : 1/sqrt(T_test), the standard ACI-style schedule.
- Train/test split    : chronological ~40/60 split of the dense-session pool
                        (earliest ~40% of sessions freeze the GP; the rest
                        stream through the online localised conformal layer).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 1. Data panel: dense sessions, sigma0 blend, causal features
# ----------------------------------------------------------------------
TRADING_DAYS_PER_YEAR = 252


def _dense_sessions(bars: pd.DataFrame, min_bars: int) -> pd.Series:
    counts = bars.groupby(bars["Date"].dt.date).size()
    return counts[counts >= min_bars].index


def load_bars(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def load_hist_vol(csv_path: str) -> pd.DataFrame:
    hv = pd.read_csv(csv_path)
    hv["Date"] = pd.to_datetime(hv["Date"])
    return hv.sort_values("Date").reset_index(drop=True)


def build_panel(bars: pd.DataFrame,
                 hv: pd.DataFrame,
                 min_bars_dense: int = 30,
                 H_bars: int = 1,
                 ret_lookback: int = 4,
                 rng_lookback: int = 4,
                 vol_lookback: int = 12) -> pd.DataFrame:
    """
    Build the causal per-bar panel: response Y, start-of-day prior w_prior,
    and GP/localiser feature columns. Restricted to dense sessions only
    (sparse-day snapshots are dropped -- they are not real intraday bars and
    would otherwise create spurious multi-hour "returns").
    """
    dense_days = set(_dense_sessions(bars, min_bars_dense))
    d = bars.copy()
    d["session"] = d["Date"].dt.date
    d = d[d["session"].isin(dense_days)].reset_index(drop=True)

    mid = d["Close"].astype(float).to_numpy()
    session = d["session"].to_numpy()
    ts = d["Date"]

    # ---- forward H-bar response, never crossing a session boundary ----
    s = pd.Series(mid)
    fwd = s.shift(-H_bars) - s
    same_session = pd.Series(session).shift(-H_bars).to_numpy() == session
    Y = np.where(same_session, fwd.to_numpy(), np.nan)

    # ---- bars per session (median, for the annualisation convention) ----
    bars_per_session = pd.Series(session).groupby(pd.Series(session)).transform("size")
    B = int(np.median(pd.Series(session).value_counts().to_numpy()))

    # ---- session-open price (reference price level for the sigma0 blend) ----
    first_idx = d.groupby("session", sort=False).head(1).index
    open_price = pd.Series(np.nan, index=d.index)
    open_price.loc[first_idx] = d["Open"].loc[first_idx].astype(float)
    open_price = open_price.groupby(d["session"]).transform("first")

    # ---- sigma0 blend: HistVol10 & ImpliedVol from the most recent PRIOR
    #      trading day (merge_asof, backward, strictly before the session date)
    sessions_df = pd.DataFrame({"session_date": pd.to_datetime(pd.Series(session).unique())})
    sessions_df = sessions_df.sort_values("session_date").reset_index(drop=True)
    merged = pd.merge_asof(sessions_df, hv, left_on="session_date", right_on="Date",
                            direction="backward", allow_exact_matches=False)
    merged["sigma_blend_pct"] = 0.5 * (merged["HistVol10"] + merged["ImpliedVol"])
    sigma_map = merged.set_index(merged["session_date"].dt.date)["sigma_blend_pct"]

    sigma_blend_pct = pd.Series(session).map(sigma_map).to_numpy(dtype=float)
    sigma_annual = sigma_blend_pct / 100.0
    period_years = 1.0 / (TRADING_DAYS_PER_YEAR * B)
    sigma0_price = open_price.to_numpy() * sigma_annual * np.sqrt(period_years)
    w_prior = sigma0_price * np.sqrt(H_bars)

    # ---- causal features ----
    tod = d.groupby("session", sort=False).cumcount().to_numpy().astype(float)
    tod_n = bars_per_session.to_numpy().astype(float)
    tod_frac = np.where(tod_n > 1, tod / np.maximum(tod_n - 1, 1), 0.0)

    ret1 = np.diff(mid, prepend=mid[0])
    new_sess = np.concatenate([[True], session[1:] != session[:-1]])
    ret1 = np.where(new_sess, 0.0, ret1)   # no overnight-gap "return"

    ret_recent = pd.Series(ret1).shift(1).rolling(ret_lookback, min_periods=1).sum().to_numpy()
    absret_recent = pd.Series(np.abs(ret1)).shift(1).rolling(vol_lookback, min_periods=1).mean().to_numpy()
    hi = d["High"].astype(float).to_numpy()
    lo = d["Low"].astype(float).to_numpy()
    rng_recent = pd.Series(hi - lo).shift(1).rolling(rng_lookback, min_periods=1).mean().to_numpy()

    panel = pd.DataFrame({
        "ts": ts.to_numpy(), "session": session, "mid": mid,
        "Y": Y, "w_prior": w_prior, "sigma0_price": sigma0_price,
        "tod_frac": tod_frac, "ret_recent": ret_recent,
        "absret_recent": absret_recent, "rng_recent": rng_recent,
    })
    # Volume is present for ES (ESZ6) bars but not for NQ (NQU6) bars in this
    # data set -- only add the liquidity proxy feature when it exists, rather
    # than assuming it's there.
    if "Volume" in d.columns:
        logvol = np.log(np.clip(d["Volume"].astype(float).to_numpy(), 1.0, None))
        panel["vol_recent"] = pd.Series(logvol).shift(1).rolling(vol_lookback, min_periods=1).mean().to_numpy()
    return panel


GP_FEATURES_ALL = ["tod_frac", "ret_recent", "absret_recent", "rng_recent", "vol_recent"]
LOC_FEATURES = ["ret_recent", "absret_recent", "rng_recent"]


# ----------------------------------------------------------------------
# 2. Static GP width model (eq. 5-7): fit once on the training slice, freeze
# ----------------------------------------------------------------------
class StaticGPWidth:
    """f ~ GP(0, k) fit on z_t = log(|Y_t| / w_prior_t) over the training
    slice, then frozen. g(x) = exp(f(x)) is the multiplicative width factor."""

    def __init__(self, n_gp_max: int = 600, n_restarts: int = 1, seed: int = 0):
        self.n_gp_max = int(n_gp_max)
        self.n_restarts = int(n_restarts)
        self.seed = int(seed)
        self.gpr = None
        self.mu_ = None
        self.sd_ = None

    def _kernel(self, d: int):
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
        return (ConstantKernel(1.0, (1e-2, 1e2)) * RBF(np.ones(d), (1e-1, 1e2))
                + WhiteKernel(0.5, (1e-3, 1e1)))

    def fit(self, X: np.ndarray, z: np.ndarray) -> "StaticGPWidth":
        from sklearn.gaussian_process import GaussianProcessRegressor
        X = np.asarray(X, float)
        z = np.asarray(z, float)
        self.mu_ = X.mean(0)
        self.sd_ = X.std(0) + 1e-12
        Xs = (X - self.mu_) / self.sd_
        rng = np.random.RandomState(self.seed)
        n = len(Xs)
        if n > self.n_gp_max:
            sel = rng.choice(n, self.n_gp_max, replace=False)
            Xs, z = Xs[sel], z[sel]
        self.gpr = GaussianProcessRegressor(
            kernel=self._kernel(Xs.shape[1]), alpha=0.0,
            n_restarts_optimizer=self.n_restarts, normalize_y=True)
        self.gpr.fit(Xs, z)
        return self

    def log_factor(self, X: np.ndarray, clip: float = 3.0) -> np.ndarray:
        Xs = (np.asarray(X, float) - self.mu_) / self.sd_
        return np.clip(self.gpr.predict(Xs), -clip, clip)

    def log_factor_mean_std(self, X: np.ndarray, clip: float = 3.0):
        """Posterior predictive mean and std of z=log(|Y|/w_prior) (includes the
        WhiteKernel noise term, since it is part of the fitted kernel sum) --
        used for the GP-credible-band baseline, which needs the GP's own
        uncertainty, not just its posterior mean."""
        Xs = (np.asarray(X, float) - self.mu_) / self.sd_
        mean, std = self.gpr.predict(Xs, return_std=True)
        return np.clip(mean, -clip, clip), std

    @property
    def kernel_(self):
        return None if self.gpr is None else self.gpr.kernel_


# ----------------------------------------------------------------------
# 3. Weighted, finite-sample-corrected conformal quantile (eq. 11-12)
# ----------------------------------------------------------------------
def weighted_quantile(scores: np.ndarray, weights: np.ndarray, level: float) -> float:
    """Localised weighted `level`-quantile with a finite-sample correction
    based on the Kish effective sample size n_eff = (sum w)^2 / sum(w^2),
    mirroring the ceil((n+1)(1-alpha)) correction of the exchangeable case."""
    scores = np.asarray(scores, float)
    weights = np.asarray(weights, float)
    if scores.size == 0:
        return np.inf
    sw = weights.sum()
    if sw <= 0:
        weights = np.ones_like(scores)
        sw = weights.sum()
    n_eff = (sw * sw) / float(np.sum(weights ** 2))
    lvl = min(1.0, float(level) * (1.0 + 1.0 / max(n_eff, 1.0)))
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    cum = np.cumsum(weights[order]) / sw
    idx = int(np.searchsorted(cum, lvl, side="left"))
    if idx >= len(s):
        return float(s[-1])
    return float(s[idx])


# ----------------------------------------------------------------------
# 4. Results container
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
            "scope": "all", "n": int(len(f)),
            "coverage": float(f["covered"].mean()), "target": 1 - self.alpha,
            "avg_width": float(fin["width"].mean()),
            "median_width": float(fin["width"].median()),
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
                    "inf_rate": float((~np.isfinite(g["width"])).mean()),
                })
        return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 5. STAC-GP: Algorithm 1 end to end
# ----------------------------------------------------------------------
class STACGP:
    def __init__(self,
                 alpha: float = 0.10,
                 R_sessions: int = 10,
                 halflife_sessions: int = 5,
                 gamma: Optional[float] = None,
                 alpha_floor: float = 0.01,
                 alpha_ceil: float = 0.50,
                 seed: int = 0):
        self.alpha = float(alpha)
        self.R_sessions = int(R_sessions)
        self.halflife_sessions = int(halflife_sessions)
        self.gamma = gamma
        self.alpha_floor = float(alpha_floor)
        self.alpha_ceil = float(alpha_ceil)
        self.seed = seed
        self.gp_: Optional[StaticGPWidth] = None

    def fit_run(self, panel: pd.DataFrame, train_frac: float = 0.40) -> STACGPResult:
        gp_features = [c for c in GP_FEATURES_ALL if c in panel.columns]
        need = ["Y", "w_prior"] + gp_features
        valid = np.isfinite(panel[need].to_numpy(float)).all(1) & (panel["w_prior"] > 0)
        panel = panel.loc[valid].reset_index(drop=True)

        sessions = pd.unique(panel["session"])
        n_train_sessions = max(1, int(round(len(sessions) * train_frac)))
        train_sessions = set(sessions[:n_train_sessions])
        is_test = ~panel["session"].isin(train_sessions).to_numpy()
        train_mask = ~is_test

        bars_per_session = int(np.median(panel["session"].value_counts().to_numpy()))
        R = self.R_sessions * bars_per_session
        halflife_bars = self.halflife_sessions * bars_per_session
        rho = 0.5 ** (1.0 / halflife_bars)

        # ---- static GP fit on the training slice, then freeze (eq. 5-7) ----
        z = np.log(np.clip(np.abs(panel["Y"].to_numpy()), 1e-12, None)
                   / panel["w_prior"].to_numpy())
        Xgp = panel[gp_features].to_numpy(float)
        self.gp_ = StaticGPWidth(seed=self.seed).fit(Xgp[train_mask], z[train_mask])
        self.gp_features_ = gp_features

        w_GP = np.exp(self.gp_.log_factor(Xgp)) * panel["w_prior"].to_numpy()
        S = np.abs(panel["Y"].to_numpy()) / w_GP

        h = self._median_heuristic_h(panel.loc[train_mask, LOC_FEATURES].to_numpy(float))
        T_test = int(is_test.sum())
        gamma = self.gamma if self.gamma is not None else 1.0 / np.sqrt(max(T_test, 1))

        frame = self._online_localised_conformal(panel, S, w_GP, is_test, R, rho, h, gamma)
        meta = dict(alpha=self.alpha, R=R, R_sessions=self.R_sessions, rho=rho,
                    halflife_sessions=self.halflife_sessions, h=h, gamma=gamma,
                    n_train_sessions=int(n_train_sessions), gp_features=gp_features,
                    bars_per_session=bars_per_session, kernel=str(self.gp_.kernel_))
        return STACGPResult(frame=frame, alpha=self.alpha, meta=meta)

    @staticmethod
    def _median_heuristic_h(X: np.ndarray) -> float:
        mu = X.mean(0); sd = X.std(0) + 1e-12
        Z = (X - mu) / sd
        m = min(len(Z), 300)
        idx = np.linspace(0, len(Z) - 1, m).astype(int)
        Zs = Z[idx]
        G = ((Zs[:, None, :] - Zs[None, :, :]) ** 2).sum(-1)
        med = np.median(G[np.triu_indices(m, k=1)]) if m > 1 else 1.0
        return float(max(med, 1e-6))

    def _online_localised_conformal(self, panel, S, w_GP, is_test, R, rho, h, gamma) -> pd.DataFrame:
        Xloc_all = panel[LOC_FEATURES].to_numpy(float)
        Y = panel["Y"].to_numpy(float)
        mid = panel["mid"].to_numpy(float)
        n = len(panel)

        alpha_t = self.alpha
        lo = np.full(n, np.nan); hi = np.full(n, np.nan)
        width = np.full(n, np.nan); q_arr = np.full(n, np.nan)
        covered = np.zeros(n, bool); alpha_path = np.full(n, np.nan)
        out_rows = np.zeros(n, bool)

        for t in range(n):
            if not is_test[t]:
                continue
            lo_i = max(0, t - R)
            cal_idx = np.arange(lo_i, t)
            if len(cal_idx) == 0:
                continue

            Xw = Xloc_all[cal_idx]
            xt = Xloc_all[t]
            mu = Xw.mean(0); sd = Xw.std(0) + 1e-12
            Zw = (Xw - mu) / sd
            zt = (xt - mu) / sd
            d2 = np.einsum("ij,ij->i", Zw - zt, Zw - zt)
            H_loc = np.exp(-d2 / max(h, 1e-12))
            recency = rho ** (t - cal_idx)
            w = H_loc * recency

            q = weighted_quantile(S[cal_idx], w, 1.0 - alpha_t)
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


__all__ = ["STACGP", "STACGPResult", "StaticGPWidth", "load_bars", "load_hist_vol",
           "build_panel", "weighted_quantile", "GP_FEATURES_ALL", "LOC_FEATURES"]
