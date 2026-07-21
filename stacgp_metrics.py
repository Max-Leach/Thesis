"""
stacgp_metrics.py
==================
Section 5.3 metrics for the STAC-GP baseline comparison.

    (i)   empirical coverage vs target 1-alpha
    (ii)  relative width vs third-party bid-offer     -- NOT AVAILABLE: no
          bid/ask or spread data exists anywhere in Data/, so this is
          reported as N/A rather than approximated by a proxy.
    (iii) stress-response time                        -- how quickly the band
          widens after a shock. No single textbook definition exists, so we
          define one explicitly here rather than assuming: a "shock" bar is
          one where the realised |return| exceeds `shock_mult` times its own
          trailing median (a regime-independent, data-driven shock detector,
          computed once from the shared panel so every method faces the same
          shock set). For each shock at bar t0, we measure the number of bars
          until the method's OWN band width first reaches |Y_t0| again (i.e.
          would have covered a repeat of that shock) -- shorter is faster to
          react. Methods that never catch up within `lookahead` bars are
          scored at `lookahead` (a censored, conservative penalty).
    (iv)  width volatility                             -- mean absolute
          bar-to-bar change in band width ("jumpiness").

All four are reported both as a single overall number and stratified by a
volatility regime quartile. The regime label is computed once from a
method-independent, purely data-driven proxy (trailing realised |return|,
`absret_recent`) so that every method is judged against the same regime
definition rather than its own internal notion of "calm" vs "stressed".
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def regime_labels(panel: pd.DataFrame, idx: np.ndarray, n_quantiles: int = 4) -> pd.Series:
    proxy = panel.loc[idx, "absret_recent"]
    labels = [f"Q{i+1}" for i in range(n_quantiles)]
    labels[0] += " calm"; labels[-1] += " stressed"
    return pd.qcut(proxy, n_quantiles, labels=labels, duplicates="drop")


def _finite(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[np.isfinite(frame["width"])]


def basic_metrics(frame: pd.DataFrame, alpha: float) -> dict:
    fin = _finite(frame)
    return {
        "n": int(len(frame)),
        "coverage": float(frame["covered"].mean()),
        "target": 1 - alpha,
        "avg_width": float(fin["width"].mean()) if len(fin) else np.nan,
        "median_width": float(fin["width"].median()) if len(fin) else np.nan,
        "width_volatility": float(fin["width"].diff().abs().mean()) if len(fin) > 1 else np.nan,
        "inf_rate": float((~np.isfinite(frame["width"])).mean()),
    }


def stress_response_time(frame: pd.DataFrame, panel: pd.DataFrame,
                          shock_mult: float = 3.0, trailing_window: int = 40,
                          lookahead: int = 40) -> float:
    """Bars from a shock to the method's own band first re-covering a move of
    that size. `frame` must be aligned (same row order/timestamps) to a
    contiguous slice of `panel` -- true for every method here since all are
    built on the shared evaluation window."""
    # frame carries its own Y/width columns already aligned to the eval window
    Y = frame["Y"].abs().to_numpy()
    width = frame["width"].to_numpy()
    n = len(frame)
    if n < trailing_window + 2:
        return np.nan

    trailing_med = pd.Series(Y).rolling(trailing_window, min_periods=trailing_window).median().to_numpy()
    is_shock = Y > shock_mult * trailing_med
    is_shock[:trailing_window] = False

    shock_positions = np.where(is_shock)[0]
    if len(shock_positions) == 0:
        return np.nan

    response_times = []
    for t0 in shock_positions:
        target = Y[t0]
        found = lookahead
        for k in range(0, min(lookahead, n - t0)):
            w = width[t0 + k]
            if np.isfinite(w) and (w / 2.0) >= target:
                found = k
                break
        response_times.append(found)
    return float(np.mean(response_times))


def metrics_table(methods: dict, alpha: float, panel: pd.DataFrame,
                   shock_mult: float = 3.0, trailing_window: int = 40,
                   lookahead: int = 40) -> pd.DataFrame:
    rows = []
    for name, frame in methods.items():
        m = basic_metrics(frame, alpha)
        m["method"] = name
        m["relative_width_vs_3rd_party_bid_offer"] = "N/A (no bid/offer data)"
        m["stress_response_time_bars"] = stress_response_time(
            frame, panel, shock_mult, trailing_window, lookahead)
        rows.append(m)
    cols = ["method", "n", "coverage", "target", "avg_width", "median_width",
            "width_volatility", "stress_response_time_bars", "inf_rate",
            "relative_width_vs_3rd_party_bid_offer"]
    return pd.DataFrame(rows)[cols]


def metrics_by_regime(methods: dict, alpha: float, panel: pd.DataFrame,
                       n_quantiles: int = 4) -> pd.DataFrame:
    # Frames returned by the baseline module are reset_index(drop=True) but
    # keep 'ts', so the regime label (built from the panel) is joined back by
    # timestamp rather than row position.
    rows = []
    for name, frame in methods.items():
        rows.extend(_regime_rows(name, frame, alpha, panel, n_quantiles))
    return pd.DataFrame(rows)


def _regime_rows(name: str, frame: pd.DataFrame, alpha: float, panel: pd.DataFrame,
                  n_quantiles: int) -> list:
    # Build the regime label on the panel rows the frame actually covers,
    # matched by timestamp (robust to any row filtering upstream).
    sub = panel.set_index("ts").loc[frame["ts"].to_numpy(), "absret_recent"]
    labels = [f"Q{i+1}" for i in range(n_quantiles)]
    labels[0] += " calm"; labels[-1] += " stressed"
    regime = pd.qcut(sub.to_numpy(), n_quantiles, labels=labels, duplicates="drop")
    out = []
    f = frame.copy()
    f["regime"] = np.asarray(regime)
    for lab, g in f.groupby("regime", observed=True):
        m = basic_metrics(g, alpha)
        m["method"] = name
        m["regime"] = lab
        out.append(m)
    return out


def rolling(a, w: int = 250):
    return pd.Series(a).rolling(w, min_periods=max(1, w // 4)).mean()


__all__ = ["regime_labels", "basic_metrics", "stress_response_time",
           "metrics_table", "metrics_by_regime", "rolling"]
