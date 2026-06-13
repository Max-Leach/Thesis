
from tqdm.auto import tqdm
def iter_data(data, show_pbar=True, desc=""):
    return tqdm(data, desc=desc, leave=False) if show_pbar else data


def run_plain_cp(calib_scores, alpha, plus_one=True):
    calib_scores = np.asarray(calib_scores, float).ravel()
    r = len(calib_scores)
    if r == 0:
        return np.inf
    if plus_one:
        k = int(np.ceil((1 - alpha) * (r + 1)))
    else:
        k = int(np.ceil((1 - alpha) * r))
    k = min(max(k, 1), r)
    return float(np.sort(calib_scores)[k - 1])
import numpy as np

def _weighted_quantile_sorted(v_sorted, w_sorted, quantile):
    # v_sorted ascending
    cw = np.cumsum(w_sorted)
    tot = cw[-1]
    if tot < 1e-12:
        w_sorted = np.ones_like(w_sorted, dtype=float)
        cw = np.cumsum(w_sorted)
        tot = cw[-1]
    target = quantile * tot
    j = int(np.searchsorted(cw, target, side="left"))
    j = min(max(j, 0), len(v_sorted) - 1)
    return float(v_sorted[j])

def lcp_q_fast(calib_scores, calib_X, x_new, alpha, h):
    """
    Localized CP:
      - sort scores once
      - compute weights then reorder weights by score-order
      - compute weighted quantile by cumulative weights
    """
    V = np.asarray(calib_scores, float).ravel()
    Xc = np.asarray(calib_X, float)
    x = np.asarray(x_new, float).ravel()

    order = np.argsort(V)
    V_sorted = V[order]

    mu = Xc.mean(axis=0)
    sd = Xc.std(axis=0, ddof=0)
    sd = np.where(np.isfinite(sd) & (sd >= 1e-12), sd, 1.0)

    Z = (Xc - mu) / sd
    z = (x - mu) / sd

    # distances in z-space
    diff = Z - z[None, :]
    d = np.sqrt(np.einsum("ij,ij->i", diff, diff))  # fast norm

    w = np.exp(-d / max(float(h), 1e-12))
    w_sorted = w[order]

    return _weighted_quantile_sorted(V_sorted, w_sorted, 1 - float(alpha))


def lcp_q_grid_fast(calib_scores, calib_X, x_new, alpha_vec, H_grid):
    """
    Compute q for multiple bandwidths (and potentially different alphas) efficiently.
    Returns q_list aligned with H_grid.
    alpha_vec can be scalar or length len(H_grid).
    """
    V = np.asarray(calib_scores, float).ravel()
    Xc = np.asarray(calib_X, float)
    x = np.asarray(x_new, float).ravel()

    order = np.argsort(V)
    V_sorted = V[order]

    mu = Xc.mean(axis=0)
    sd = Xc.std(axis=0, ddof=0)
    sd = np.where(np.isfinite(sd) & (sd >= 1e-12), sd, 1.0)

    Z = (Xc - mu) / sd
    z = (x - mu) / sd

    diff = Z - z[None, :]
    d = np.sqrt(np.einsum("ij,ij->i", diff, diff))  # (R,)

    H = np.asarray(H_grid, float).ravel()
    M = len(H)

    # broadcast weights for all bandwidths at once: (R, M)
    W = np.exp(-d[:, None] / np.maximum(H[None, :], 1e-12))
    W_sorted = W[order, :]  # (R, M) sorted by score-order

    # normalize alpha vector
    if np.isscalar(alpha_vec):
        alpha_vec = np.full(M, float(alpha_vec), dtype=float)
    else:
        alpha_vec = np.asarray(alpha_vec, float).ravel()
        assert len(alpha_vec) == M

    qs = np.empty(M, dtype=float)
    for m in range(M):
        qs[m] = _weighted_quantile_sorted(V_sorted, W_sorted[:, m], 1 - alpha_vec[m])
    return qs

def default_bandwidth_h0(X_ref, R, max_points=300):
    X_ref = np.asarray(X_ref, float)
    d = X_ref.shape[1]
    return float((4.0 / (d + 2.0)) ** (1.0 / (d + 4.0)) * (R ** (-1.0 / (d + 4.0))) * np.sqrt(d))

import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# -------------------------------------------------
# helpers
# -------------------------------------------------
def _past_test_idx(test_idx, j, R):
    """Indices of the previous at most R test points before test position j."""
    return test_idx[max(0, j - R):j]

def _make_interval(pred_t, q, normalized_score=False):
    """
    normalized_score=False -> additive interval: pred ± q
    normalized_score=True  -> multiplicative interval: pred * (1 ± q)
    """
    if normalized_score:
        return pred_t * (1.0 - q), pred_t * (1.0 + q)
    else:
        return pred_t - q, pred_t + q

def _pinball_loss(beta, theta, alpha_star):
    z = beta - theta
    return alpha_star * z - min(0.0, z)


# -------------------------------------------------
# CP
# -------------------------------------------------
def eval_cp(data, alpha=0.1, R=100, exclude_const_test=False, show_pbar=True,
            normalized_score=False):
    t0 = time.perf_counter()
    rows = []

    for d in iter_data(data, show_pbar=show_pbar, desc="CP"):
        if exclude_const_test and d["const_test"]:
            continue

        y = np.asarray(d["y"], dtype=float)
        pred = np.asarray(d["pred"], dtype=float)
        score = np.asarray(d["score"], dtype=float)
        is_test = np.asarray(d["is_test"], dtype=bool)

        test_idx = np.where(is_test)[0]
        if len(test_idx) == 0:
            continue

        for j, t in enumerate(test_idx):
            calib_idx = _past_test_idx(test_idx, j, R)
            if len(calib_idx) == 0:
                continue

            calib_scores = score[calib_idx]
            q = run_plain_cp(calib_scores, alpha, plus_one=True)
            lo, hi = _make_interval(pred[t], q, normalized_score=normalized_score)

            rows.append((d["series_id"], t, y[t], pred[t], lo, hi))

    df = pd.DataFrame(rows, columns=["series_id", "t", "y", "pred", "lower", "upper"])
    df["covered"] = (df["y"] >= df["lower"]) & (df["y"] <= df["upper"])
    df["width"] = df["upper"] - df["lower"]

    dt = time.perf_counter() - t0
    print(f"CP done in {dt:.2f}s | rows={len(df)}")
    return df, dt


# -------------------------------------------------
# LCP
# -------------------------------------------------
def eval_lcp(data, alpha=0.1, R=100, exclude_const_test=False, show_pbar=True,
             normalized_score=False):
    t0 = time.perf_counter()
    rows = []

    for d in iter_data(data, show_pbar=show_pbar, desc="LCP"):
        if exclude_const_test and d["const_test"]:
            continue

        y = np.asarray(d["y"], dtype=float)
        pred = np.asarray(d["pred"], dtype=float)
        score = np.asarray(d["score"], dtype=float)
        X = np.asarray(d["X"], dtype=float)
        is_test = np.asarray(d["is_test"], dtype=bool)

        test_idx = np.where(is_test)[0]
        if len(test_idx) == 0:
            continue

        # bandwidth from validation only
        pretest_idx = np.where(~is_test)[0]
        if len(pretest_idx) == 0:
            continue

        X_ref = X[pretest_idx[-min(300, len(pretest_idx))]:]
        h0 = default_bandwidth_h0(X_ref, R)

        for j, t in enumerate(test_idx):
            calib_idx = _past_test_idx(test_idx, j, R)
            if len(calib_idx) == 0:
                continue

            calib_scores = score[calib_idx]
            calib_X = X[calib_idx]
            q = lcp_q_fast(calib_scores, calib_X, X[t], alpha, h0)
            # q, info = lcp_q_fast_debug(calib_scores, calib_X, X[t], alpha, h0, verbose=False)
            # if not np.isfinite(q):
            #     print(f"[LCP bad] series={d['series_id']} t={t}")
            #     print(info)
            lo, hi = _make_interval(pred[t], q, normalized_score=normalized_score)

            rows.append((d["series_id"], t, y[t], pred[t], lo, hi, h0))

    df = pd.DataFrame(rows, columns=["series_id", "t", "y", "pred", "lower", "upper", "h0"])
    df["covered"] = (df["y"] >= df["lower"]) & (df["y"] <= df["upper"])
    df["width"] = df["upper"] - df["lower"]

    dt = time.perf_counter() - t0
    print(f"LCP done in {dt:.2f}s | rows={len(df)}")
    return df, dt


# -------------------------------------------------
# ACI
# -------------------------------------------------
def eval_aci(data, alpha=0.1, R=100, gamma=None, exclude_const_test=False,
             show_pbar=True, normalized_score=False):
    t0 = time.perf_counter()
    rows = []

    for d in iter_data(data, show_pbar=show_pbar, desc="ACI"):
        if exclude_const_test and d["const_test"]:
            continue

        y = np.asarray(d["y"], dtype=float)
        pred = np.asarray(d["pred"], dtype=float)
        score = np.asarray(d["score"], dtype=float)
        is_test = np.asarray(d["is_test"], dtype=bool)

        test_idx = np.where(is_test)[0]
        if len(test_idx) == 0:
            continue

        T = len(test_idx)
        g = (0.5 / np.sqrt(T)) if gamma is None else float(gamma)

        alpha_t = float(alpha)
        for j, t in enumerate(test_idx):
            calib_idx = _past_test_idx(test_idx, j, R)
            if len(calib_idx) == 0:
                continue

            calib_scores = score[calib_idx]
            q = run_plain_cp(calib_scores, alpha_t, plus_one=False)
            lo, hi = _make_interval(pred[t], q, normalized_score=normalized_score)

            err = 0.0 if (lo <= y[t] <= hi) else 1.0
            alpha_t = float(np.clip(alpha_t + g * (alpha - err), 0.0, 1.0))

            rows.append((d["series_id"], t, y[t], pred[t], lo, hi, alpha_t))

    df = pd.DataFrame(rows, columns=["series_id", "t", "y", "pred", "lower", "upper", "alpha_t"])
    df["covered"] = (df["y"] >= df["lower"]) & (df["y"] <= df["upper"])
    df["width"] = df["upper"] - df["lower"]

    dt = time.perf_counter() - t0
    print(f"ACI done in {dt:.2f}s | rows={len(df)}")
    return df, dt


# -------------------------------------------------
# DtACI
# -------------------------------------------------
def eval_dtaci(data, alpha=0.1, R=100, gamma_base=None, exclude_const_test=False,
               show_pbar=True, normalized_score=False):
    t0 = time.perf_counter()
    rows = []

    for d in iter_data(data, show_pbar=show_pbar, desc="DtACI"):
        if exclude_const_test and d["const_test"]:
            continue

        y = np.asarray(d["y"], dtype=float)
        pred = np.asarray(d["pred"], dtype=float)
        score = np.asarray(d["score"], dtype=float)
        is_test = np.asarray(d["is_test"], dtype=bool)

        test_idx = np.where(is_test)[0]
        if len(test_idx) == 0:
            continue

        T = len(test_idx)
        g0 = (0.5 / np.sqrt(T)) if gamma_base is None else float(gamma_base)

        GAMMA_GRID = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5], dtype=float) * g0
        k = len(GAMMA_GRID)

        I_size = 500
        sigma_dt = 1.0 / (2.0 * I_size)
        denom = ((1.0 - alpha) ** 2 * alpha ** 3 + alpha ** 2 * (1.0 - alpha) ** 3) / 3.0
        eta_dt = np.sqrt(3.0 / I_size) * np.sqrt((np.log(I_size * k) + 2.0) / denom)

        alpha_i = np.full(k, alpha, dtype=float)
        w_i = np.ones(k, dtype=float)

        for j, t in enumerate(test_idx):
            calib_idx = _past_test_idx(test_idx, j, R)
            if len(calib_idx) == 0:
                continue

            calib_scores = score[calib_idx]
            r_t = score[t]
            beta_t = float(np.mean(calib_scores >= r_t))

            p_i = w_i / w_i.sum()
            alpha_dt = float(np.dot(p_i, alpha_i))

            q = run_plain_cp(calib_scores, alpha_dt, plus_one=False)
            lo, hi = _make_interval(pred[t], q, normalized_score=normalized_score)

            losses_i = np.array([_pinball_loss(beta_t, a_i, alpha) for a_i in alpha_i], dtype=float)
            w_bar = w_i * np.exp(-eta_dt * losses_i)
            W_bar = w_bar.sum()
            w_i = (1.0 - sigma_dt) * w_bar + (sigma_dt * W_bar / k)

            err_i = (alpha_i >= beta_t).astype(float)
            alpha_i = np.clip(alpha_i + GAMMA_GRID * (alpha - err_i), 0.0, 1.0)

            rows.append((d["series_id"], t, y[t], pred[t], lo, hi, alpha_dt))

    df = pd.DataFrame(rows, columns=["series_id", "t", "y", "pred", "lower", "upper", "alpha_dt"])
    df["covered"] = (df["y"] >= df["lower"]) & (df["y"] <= df["upper"])
    df["width"] = df["upper"] - df["lower"]

    dt = time.perf_counter() - t0
    print(f"DtACI done in {dt:.2f}s | rows={len(df)}")
    return df, dt


# -------------------------------------------------
# OLCP
# -------------------------------------------------
def eval_olcp(data, alpha=0.1, R=100, gamma=None, exclude_const_test=False,
              show_pbar=True, normalized_score=False):
    t0 = time.perf_counter()
    rows = []

    for d in iter_data(data, show_pbar=show_pbar, desc="OLCP"):
        if exclude_const_test and d["const_test"]:
            continue

        y = np.asarray(d["y"], dtype=float)
        pred = np.asarray(d["pred"], dtype=float)
        score = np.asarray(d["score"], dtype=float)
        X = np.asarray(d["X"], dtype=float)
        is_test = np.asarray(d["is_test"], dtype=bool)

        test_idx = np.where(is_test)[0]
        if len(test_idx) == 0:
            continue

        pretest_idx = np.where(~is_test)[0]
        if len(pretest_idx) == 0:
            continue

        X_ref = X[pretest_idx[-min(300, len(pretest_idx))]:]
        h0 = default_bandwidth_h0(X_ref, R)

        T = len(test_idx)
        g = (0.5 / np.sqrt(T)) if gamma is None else float(gamma)

        alpha_t = float(alpha)
        for j, t in enumerate(test_idx):
            calib_idx = _past_test_idx(test_idx, j, R)
            if len(calib_idx) == 0:
                continue

            calib_scores = score[calib_idx]
            calib_X = X[calib_idx]
            q = lcp_q_fast(calib_scores, calib_X, X[t], alpha_t, h0)
            lo, hi = _make_interval(pred[t], q, normalized_score=normalized_score)

            err = 0.0 if (lo <= y[t] <= hi) else 1.0
            alpha_t = float(np.clip(alpha_t + g * (alpha - err), 0.0, 1.0))

            rows.append((d["series_id"], t, y[t], pred[t], lo, hi, alpha_t, h0))

    df = pd.DataFrame(rows, columns=["series_id", "t", "y", "pred", "lower", "upper", "alpha_t", "h0"])
    df["covered"] = (df["y"] >= df["lower"]) & (df["y"] <= df["upper"])
    df["width"] = df["upper"] - df["lower"]

    dt = time.perf_counter() - t0
    print(f"OLCP done in {dt:.2f}s | rows={len(df)}")
    return df, dt


# -------------------------------------------------
# OLCPH
# -------------------------------------------------
def _softmax_theta_over_simplex(theta, lam, eps=1e-12):
    """
    Compute p_j ∝ exp(theta_j / lam).
    """
    theta = np.asarray(theta, dtype=float).ravel()

    if (not np.isfinite(lam)) or lam <= eps:
        return np.full(len(theta), 1.0 / len(theta), dtype=float)

    z = theta / float(lam)
    z = z - np.nanmax(z)
    ez = np.exp(np.clip(z, -745.0, 0.0))
    s = ez.sum()

    if s <= eps or not np.isfinite(s):
        return np.full(len(theta), 1.0 / len(theta), dtype=float)

    return ez / s

def _plain_update_simplex(p, theta, lam_ah, xi, alpha_ah, round_index=None, eps=1e-12):
    """
    Plain multiplicative-weights update on the probability simplex.

    Parameters
    ----------
    p : array-like, shape (K,)
        Current expert distribution.
    theta : array-like, shape (K,)
        Unused by the update, but updated as theta - xi for bookkeeping.
    lam_ah : float
        Unused by the update; returned unchanged for API compatibility.
    xi : array-like, shape (K,)
        Linearized surrogate loss / gradient vector.
    alpha_ah : float
        Here interpreted as the fixed MW learning rate eta.
    round_index : int
        Unused; included for API compatibility.
    eps : float
        Numerical tolerance.

    Returns
    -------
    p_new, theta_new, lam_ah, delta
        Same return format as _adah_update_simplex.
        delta is set to the current mixed loss <xi, p> for diagnostics.
    """
    p = np.asarray(p, dtype=float).ravel()
    theta = np.asarray(theta, dtype=float).ravel()
    xi = np.asarray(xi, dtype=float).ravel()

    K = len(p)
    if len(theta) != K or len(xi) != K:
        raise ValueError("p, theta, xi must have the same length.")

    # Sanitize p
    if (not np.isfinite(p).all()) or p.sum() <= eps:
        p = np.full(K, 1.0 / K, dtype=float)
    else:
        p = np.maximum(p, 0.0)
        s = p.sum()
        p = p / s if s > eps else np.full(K, 1.0 / K, dtype=float)

    # Sanitize xi
    xi = np.where(np.isfinite(xi), xi, 0.0)

    eta = float(alpha_ah)
    if (not np.isfinite(eta)) or eta < 0:
        raise ValueError("For _plain_update_simplex, alpha_ah is interpreted as eta and must be nonnegative.")

    loss_mix = float(np.dot(p, xi))

    # Stable multiplicative weights update:
    # p_new_i ∝ p_i exp(-eta * xi_i)
    if eta <= eps:
        p_new = p.copy()
    else:
        active = p > eps
        if active.sum() == 0:
            p_new = np.full(K, 1.0 / K, dtype=float)
        else:
            z = -eta * xi
            zmax = np.nanmax(z[active])

            w_new = np.zeros(K, dtype=float)
            w_new[active] = p[active] * np.exp(np.clip(z[active] - zmax, -745.0, 0.0))

            s = w_new.sum()
            if s <= eps or not np.isfinite(s):
                p_new = np.full(K, 1.0 / K, dtype=float)
            else:
                p_new = w_new / s

    # Bookkeeping only; plain MW does not use theta/lam_ah.
    theta_new = theta - xi
    delta = loss_mix

    return p_new, theta_new, lam_ah, delta

def _adah_update_simplex(p, theta, lam_ah, xi, alpha_ah, round_index=None, eps=1e-12, eta_ref=0.01):
    """
    One AdaHedge/FTRL update on the probability simplex.

      theta_{t+1} = theta_t - xi_t

      if lambda_t = 0:
          delta_t = -max_j theta_{t,j}
                    + max_j theta_{t+1,j}
                    + <xi_t, p_t>
      else:
          delta_t = lambda_t log(sum_j p_{t,j} exp(-xi_{t,j}/lambda_t))
                    + <xi_t, p_t>

      lambda_{t+1} = lambda_t + delta_t / alpha_AH^2
      p_{t+1,j} ∝ exp(theta_{t+1,j} / lambda_{t+1})
    """
    p = np.asarray(p, dtype=float).ravel()
    theta = np.asarray(theta, dtype=float).ravel()
    xi = np.asarray(xi, dtype=float).ravel()

    K = len(p)
    if len(theta) != K or len(xi) != K:
        raise ValueError("p, theta, xi must have the same length.")

    # sanitize p
    if (not np.isfinite(p).all()) or p.sum() <= eps:
        p = np.full(K, 1.0 / K, dtype=float)
    else:
        p = np.maximum(p, 0.0)
        p = p / p.sum()

    loss_mix = float(np.dot(p, xi))
    theta_new = theta - xi

    # mixability gap
    if (not np.isfinite(lam_ah)) or lam_ah <= eps:
        delta = -float(np.nanmax(theta)) + float(np.nanmax(theta_new)) + loss_mix
    else:
        z = -xi / float(lam_ah)
        zmax = np.nanmax(z)
        log_sum = zmax + np.log(np.sum(p * np.exp(np.clip(z - zmax, -745.0, 0.0))))
        delta = float(lam_ah * log_sum + loss_mix)

    # numerical guard: AdaHedge gap should be nonnegative
    if (not np.isfinite(delta)) or delta < 0.0:
        delta = 0.0

    denom = max(float(alpha_ah) ** 2, eps)
    lam_new = float(lam_ah) + delta / denom
    lam_new = float(np.clip(lam_new, 1 / (1.1 * eta_ref), 1 / (0.9 * eta_ref)))

    lam_was_zero = (not np.isfinite(lam_ah)) or lam_ah <= eps
    if lam_was_zero:
        p_new = np.full(K, 1.0 / K, dtype=float)
    else:
        p_new = _softmax_theta_over_simplex(theta_new, lam_new, eps=eps)

    return p_new, theta_new, lam_new, delta


def eval_olcp_adahedge(
    data,
    alpha=0.1,
    R=100,
    gamma=None,
    exclude_const_test=False,
    show_pbar=True,
    normalized_score=False,
    h_multipliers=None,
    # theorem / Algorithm 2 parameters
    V=1.0,
    G=1.0,
    kappa=None,
    lambda_coco=None,
    alpha_ah=None,
    return_diagnostics=False,
    subroutine="adah",  # or "plain"
):
    """
    OLCP-Hedge implemented with the AdaHedge/FTRL subroutine from the paper.
    """
    t0 = time.perf_counter()
    rows = []
    diag_rows = []

    if h_multipliers is None:
        h_multipliers = np.array([0.5, 0.75, 1.0, 1.25, 1.5], dtype=float)
    else:
        h_multipliers = np.asarray(h_multipliers, dtype=float).ravel()

    def _norm01(a, fill=1.0):
        a = np.asarray(a, float)
        fin = np.isfinite(a)
        if fin.sum() == 0:
            return np.full_like(a, fill, dtype=float)
        mn = a[fin].min()
        mx = a[fin].max()
        den = max(mx - mn, 1e-8)
        out = (a - mn) / den
        out = np.clip(out, 0.0, 1.0)
        out[~fin] = fill
        return out

    for d in iter_data(data, show_pbar=show_pbar, desc="OLCP-AdaHedge"):
        if exclude_const_test and d["const_test"]:
            continue

        y = np.asarray(d["y"], dtype=float)
        pred = np.asarray(d["pred"], dtype=float)
        score = np.asarray(d["score"], dtype=float)
        X = np.asarray(d["X"], dtype=float)
        is_test = np.asarray(d["is_test"], dtype=bool)

        test_idx = np.where(is_test)[0]
        if len(test_idx) == 0:
            continue

        pretest_idx = np.where(~is_test)[0]
        if len(pretest_idx) == 0:
            continue

        X_ref = X[pretest_idx[-min(300, len(pretest_idx))]:]
        h0 = default_bandwidth_h0(X_ref, R)
        H_GRID = np.array([0.5, 0.75, 1.0, 1.25, 1.5]) * h0
        M = len(H_GRID)

        T = len(test_idx)
        g_alpha = (0.5 / np.sqrt(T)) if gamma is None else float(gamma)

        # Theorem parameters
        CAH = 2.0 * np.sqrt(4.0 + np.log(M))

        kappa_eff = (
            1.0 / (np.sqrt(2.0) * CAH * float(G))
            if kappa is None
            else float(kappa)
        )

        lambda_eff = (
            1.0 / (2.0 * np.sqrt(T))
            if lambda_coco is None
            else float(lambda_coco)
        )

        alpha_ah_eff = (
            np.sqrt(max(np.log(M), 1e-12))
            if alpha_ah is None
            else float(alpha_ah)
        )

        if subroutine == "plain":
            update_param = np.sqrt(np.log(M) / T)   # old eta_h
        else:
            update_param = alpha_ah_eff             # AdaHedge alpha_AH

        # AdaHedge state
        p = np.full(M, 1.0 / M, dtype=float)
        theta = np.zeros(M, dtype=float)
        lam_ah = 0.0

        # COCO virtual queue
        Q = 0.0

        # Each bandwidth expert has its own OLCP adaptive alpha
        alpha_e = np.full(M, float(alpha), dtype=float)

        # Number of actual AdaHedge updates for this series
        ah_round = 0

        for j, t in enumerate(test_idx):
            calib_idx = _past_test_idx(test_idx, j, R)
            if len(calib_idx) == 0:
                continue

            calib_scores = score[calib_idx]
            calib_X = X[calib_idx]

            # Full-information expert intervals at current alpha_e
            qs = lcp_q_grid_fast(calib_scores, calib_X, X[t], alpha_e, H_GRID)

            lowers = np.empty(M, dtype=float)
            uppers = np.empty(M, dtype=float)
            raw_widths = np.empty(M, dtype=float)
            err_vec = np.empty(M, dtype=float)

            for e, q in enumerate(qs):
                lo, hi = _make_interval(pred[t], q, normalized_score=normalized_score)

                lowers[e] = lo
                uppers[e] = hi
                raw_widths[e] = hi - lo
                err_vec[e] = 0.0 if (lo <= y[t] <= hi) else 1.0
            
            # Update each OLCP expert's own adaptive alpha level.
            alpha_e = np.clip(
                alpha_e + g_alpha * (float(alpha) - err_vec),
                0.0,
                1.0,
            )

            # Bounded size loss for the OCO/COCO update.
            # Reported sizes remain raw_widths through the sampled interval.
            cost_vec = _norm01(raw_widths, fill=1.0)

            # Current distribution p_t, before observing feedback update.
            if (not np.isfinite(p).all()) or p.sum() <= 1e-12:
                p = np.full(M, 1.0 / M, dtype=float)
            else:
                p = np.maximum(p, 0.0)
                p = p / p.sum()

            # Sample I_t ~ p_t and output sampled expert interval.
            sampled_e = np.random.choice(M, p=p)
            lo_out = float(lowers[sampled_e])
            hi_out = float(uppers[sampled_e])

            g_mix = float(np.dot(p, err_vec) - float(alpha))
            Q = float(Q + kappa_eff * max(0.0, g_mix))
            phi_prime = float(lambda_eff * np.exp(lambda_eff * Q))

            # Linearized surrogate loss xi_t.
            # fhat_t(p) = V*kappa*<cost,p> + Phi'(Q)*kappa*(<err,p>-alpha)_+
            #
            # If g_mix > 0, one subgradient of the positive part is err_vec - alpha.
            # If g_mix <= 0, choose the zero subgradient for the constraint term.
            if g_mix > 0.0:
                xi = V * kappa_eff * cost_vec + phi_prime * kappa_eff * (err_vec - float(alpha))
            else:
                xi = V * kappa_eff * cost_vec

            xi = np.asarray(xi, dtype=float)
            xi = np.where(np.isfinite(xi), xi, 0.0)

            if subroutine == "plain":
                p_next, theta, lam_ah, delta_ah = _plain_update_simplex(
                    p=p,
                    theta=theta,
                    lam_ah=lam_ah,
                    xi=xi,
                    alpha_ah=update_param,   # eta_h for plain MW
                    round_index=ah_round,
                )
            elif subroutine == "adah":
                eta_plain = np.sqrt(np.log(M) / T)
                p_next, theta, lam_ah, delta_ah = _adah_update_simplex(
                    p=p,
                    theta=theta,
                    lam_ah=lam_ah,
                    xi=xi,
                    alpha_ah=update_param,   # alpha_AH for AdaHedge
                    round_index=ah_round,
                    eta_ref=eta_plain,
                )
            else:
                raise ValueError("subroutine must be one of: 'plain', 'adah'")

            

            rows.append((
                d["series_id"],
                t,
                y[t],
                pred[t],
                lo_out,
                hi_out,
                h0,
            ))

            if return_diagnostics:
                diag_rows.append({
                    "series_id": d["series_id"],
                    "t": t,
                    "h0": h0,
                    "sampled_expert": sampled_e,
                    "sampled_h": H_GRID[sampled_e],
                    "sampled_prob": p[sampled_e],
                    "g_mix": g_mix,
                    "Q": Q,
                    "phi_prime": phi_prime,
                    "lambda_ah": lam_ah,
                    "delta_ah": delta_ah,
                    "p_entropy": -float(np.sum(p * np.log(np.maximum(p, 1e-300)))),
                    "min_raw_width": float(np.nanmin(raw_widths)),
                    "max_raw_width": float(np.nanmax(raw_widths)),
                    "sampled_raw_width": float(raw_widths[sampled_e]),
                    "mean_err_expert": float(np.mean(err_vec)),
                    "mix_err": float(np.dot(p, err_vec)),
                })

            p = p_next
            ah_round += 1

    df = pd.DataFrame(
        rows,
        columns=["series_id", "t", "y", "pred", "lower", "upper", "h0"],
    )

    if len(df) > 0:
        df["covered"] = (df["y"] >= df["lower"]) & (df["y"] <= df["upper"])
        df["width"] = df["upper"] - df["lower"]
    else:
        df["covered"] = []
        df["width"] = []

    dt = time.perf_counter() - t0
    print(f"OLCP-AdaHedge done in {dt:.2f}s | rows={len(df)}")

    if return_diagnostics:
        diag_df = pd.DataFrame(diag_rows)
        return df, dt, diag_df

    return df, dt


from quantile_forest import RandomForestQuantileRegressor
import numpy as np

def _fit_qrf_and_predict_quantiles(Xtr, ytr, x_new, qs, rf_kwargs=None):
    if rf_kwargs is None:
        rf_kwargs = dict(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )

    qrf = RandomForestQuantileRegressor(**rf_kwargs)
    qrf.fit(Xtr, ytr)

    qs_list = np.asarray(qs, dtype=float).ravel().tolist()

    qvals = qrf.predict(
        x_new.reshape(1, -1),
        quantiles=qs_list,
    )[0]

    return qrf, np.asarray(qvals, dtype=float)

def _choose_beta_minwidth(q_low_grid, q_high_grid, beta_grid):
    widths = q_high_grid - q_low_grid
    j = int(np.nanargmin(widths))
    return float(beta_grid[j]), float(q_low_grid[j]), float(q_high_grid[j])
def eval_spci(
    data,
    alpha=0.1,
    R=168,
    w_lag=24,
    T_train=None,
    refit_every=24,
    beta_grid_size=21,
    exclude_const_test=False,
    rf_kwargs=None,
    show_pbar=True,
):
    t0 = time.perf_counter()
    rows = []

    if T_train is None:
        T_train = int(R)
    T_train = int(T_train)
    w_lag = int(w_lag)

    if rf_kwargs is None:
        rf_kwargs = dict(
            n_estimators=80,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )

    beta_grid = np.linspace(0.0, float(alpha), beta_grid_size)
    qs_low = beta_grid
    qs_high = 1.0 - float(alpha) + beta_grid
    qs_both = np.clip(np.concatenate([qs_low, qs_high], axis=0), 0.0, 1.0)
    qs_both_list = qs_both.tolist()

    for d in iter_data(data, show_pbar=show_pbar, desc="SPCI"):
        if exclude_const_test and d["const_test"]:
            continue

        y = np.asarray(d["y"], dtype=float)
        pred = np.asarray(d["pred"], dtype=float)
        is_test = np.asarray(d["is_test"], dtype=bool)

        test_idx = np.where(is_test)[0]
        if len(test_idx) == 0:
            continue

        y_test = y[test_idx]
        pred_test = pred[test_idx]
        eps_test = y_test - pred_test

        N = len(eps_test)
        if N <= w_lag + 1:
            continue

        LagX = np.stack([eps_test[k:k+w_lag] for k in range(N - w_lag)], axis=0)
        LagY = eps_test[w_lag:]

        qrf_model = None

        for j, t in enumerate(test_idx):
            if j < w_lag:
                continue

            k1 = j - w_lag
            k0 = max(0, k1 - T_train)
            if k1 <= k0:
                continue

            x_new = eps_test[j - w_lag:j].astype(np.float32)

            if (qrf_model is None) or ((j % refit_every) == 0):
                Xtr = LagX[k0:k1].astype(np.float32)
                ytr = LagY[k0:k1].astype(np.float32)
                if len(Xtr) == 0:
                    continue

                qrf_model, qvals = _fit_qrf_and_predict_quantiles(
                    Xtr, ytr, x_new, qs_both_list, rf_kwargs
                )
            else:
                qvals = qrf_model.predict(
                    x_new.reshape(1, -1),
                    quantiles=qs_both_list,
                )[0]
                qvals = np.asarray(qvals, dtype=float)

            q_low_grid = qvals[:beta_grid_size]
            q_high_grid = qvals[beta_grid_size:]
            beta_hat, q_lo, q_hi = _choose_beta_minwidth(q_low_grid, q_high_grid, beta_grid)

            lo = pred[t] + q_lo
            hi = pred[t] + q_hi
            rows.append((d["series_id"], t, y[t], pred[t], lo, hi, beta_hat))

    df = pd.DataFrame(rows, columns=["series_id", "t", "y", "pred", "lower", "upper", "beta_hat"])
    df["covered"] = (df["y"] >= df["lower"]) & (df["y"] <= df["upper"])
    df["width"] = df["upper"] - df["lower"]

    dt = time.perf_counter() - t0
    print(f"SPCI done in {dt:.2f}s | rows={len(df)} | refit_every={refit_every}")
    return df, dt