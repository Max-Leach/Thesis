"""
(Xu, Lu & Giannakis, 2024; arXiv:2410.05444)

Online Gaussian Process regression with Random Fourier Features (RFF),
ensembled over four kernels: RBF, Laplacian, Matern-3/2, Rational Quadratic.
With online Conformal Prediction with fixed, decaying and varying learning rate.

RFF (Rahimi & Recht, 2007):
    k(x, x') ~= z(x)^T z(x'),
    z(x) = sqrt(1/D) * [ sin(V^T x), cos(V^T x) ]        # 2D features
where the D columns of V are i.i.d. draws from the kernel's spectral density.
This approximates a unit-variance, shift-invariant kernel; the signal variance
sigma_f^2 is handled separately as the weight-space prior in gp_predict().

All samplers below validated by comparing the RFF Gram matrix against the
exact kernel (max abs error ~0.01 at D=2e4, 1-D inputs, l=1).
"""

import numpy as np
from scipy.stats import cauchy, gamma, chi2
from scipy.stats import norm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

# ----------------------------------------------------------------------
# 1. Random Fourier Feature map
# ----------------------------------------------------------------------
def rff_features(X, V):
    """X:(N, d), V:(d, D)  ->  Z:(N, 2D)."""
    D = V.shape[1]
    VX = X @ V
    return np.sqrt(1.0 / D) * np.concatenate([np.sin(VX), np.cos(VX)], axis=1)


# ----------------------------------------------------------------------
# 2. Spectral-density samplers. Each returns V of shape (d, D);
#    every COLUMN of V is one independent frequency omega ~ p(omega).
# ----------------------------------------------------------------------
def sample_rbf(d, D, length_scale, rng):
    # RBF: omega ~ N(0, 1/l^2)
    return rng.normal(0.0, 1.0 / length_scale, size=(d, D))


def sample_laplacian(d, D, length_scale, rng):
    # exp(-|r|/l): omega ~ Cauchy(0, scale = 1/l)
    return cauchy.rvs(loc=0.0, scale=1.0 / length_scale,
                      size=(d, D), random_state=rng)


def sample_matern(d, D, length_scale, nu, rng):
    # Matern-nu: omega ~ Student-t with dof = 2*nu, scale = 1/l.
    #   omega = z / sqrt(g),  z ~ N(0, 1/l^2),  g ~ Gamma(nu, scale=1/nu)
    # NOTE: g is drawn per-frequency (size=D), NOT shared across frequencies.
    g = gamma.rvs(a=nu, scale=1.0 / nu, size=D, random_state=rng)
    z = rng.normal(0.0, 1.0 / length_scale, size=(d, D))
    return z / np.sqrt(g)[None, :]


def sample_rq(d, D, length_scale, alpha, rng):
    # Rational quadratic = scale-mixture of RBFs:
    #   omega = z * sqrt(u),  z ~ N(0, 1),  u ~ Gamma(alpha, scale=1/(alpha*l^2))
    u = gamma.rvs(a=alpha, scale=1.0 / (alpha * length_scale ** 2),
                  size=D, random_state=rng)
    z = rng.normal(0.0, 1.0, size=(d, D))
    return z * np.sqrt(u)[None, :]


# ----------------------------------------------------------------------
# 3. Weight-space GP posterior (mean + diagonal predictive variance)
#    A = Z'Z + (sigma_n^2 / sigma_f^2) I
#    mu_*   = Z_* A^{-1} Z' y
#    var_*  = sigma_n^2 * diag(Z_* A^{-1} Z_*') + sigma_n^2
# ----------------------------------------------------------------------
def gp_predict(Z_train, y_train, Z_test, sigma_n, sigma_f):
    M = Z_train.shape[1]
    A = Z_train.T @ Z_train + (sigma_n ** 2 / sigma_f ** 2) * np.eye(M)
    w = np.linalg.solve(A, Z_train.T @ y_train)          # posterior mean weights
    mu = Z_test @ w
    AinvZt = np.linalg.solve(A, Z_test.T)                # (M, N_test)
    var = sigma_n ** 2 * np.einsum('ij,ji->i', Z_test, AinvZt) + sigma_n ** 2
    return mu, var


# ----------------------------------------------------------------------
# 4. Online learning loop (mini-batch). Returns predictions/truth/variance
#    for every test point, in arrival order.
#    
#    theta_hat_{t+1} <- theta_hat_t + s^-2 Sigma phi (y - yhat)
#    Sigma_{t+1}     <- Sigma_t     - s^-2 Sigma phi phi^T Sigma
# ----------------------------------------------------------------------
def run_online_gp(X_all, y_all, V, sigma_n, sigma_f, n_init=50, batch=50):
    X_tr, y_tr = X_all[:n_init].copy(), y_all[:n_init].copy()
    Z_tr = rff_features(X_tr, V)

    preds, truths, varis = [], [], []
    for i in range(n_init, len(X_all), batch):
        X_new, y_new = X_all[i:i + batch], y_all[i:i + batch]
        Z_new = rff_features(X_new, V)

        mu, var = gp_predict(Z_tr, y_tr, Z_new, sigma_n, sigma_f)
        preds.extend(mu)
        truths.extend(y_new)
        varis.extend(var)

        # absorb the new data into the training set
        X_tr = np.vstack([X_tr, X_new])
        y_tr = np.append(y_tr, y_new)
        Z_tr = np.vstack([Z_tr, Z_new])

    return np.array(preds), np.array(truths), np.array(varis)


def run_online_gp_recursive(X_all, y_all, V, sigma_n, sigma_f, n_init=100):
    # Memoryless Bayesian update of the RFF weight posterior (paper eqs. 10-11).
    M = 2 * V.shape[1]
    theta = np.zeros(M)
    Sigma = (sigma_f ** 2) * np.eye(M) # prior  theta ~ N(0, sigma_f^2 I)
    Z = rff_features(X_all, V)
    sn2 = sigma_n ** 2
    preds, truths, varis = [], [], []
    for t in range(len(X_all)):
        phi = Z[t]
        mu = phi @ theta
        Sphi = Sigma @ phi
        s2 = float(phi @ Sphi + sn2) # predictive variance sigma^2_{t+1|t}
        if t >= n_init:
            preds.append(mu)
            truths.append(y_all[t])
            varis.append(s2)
        gain = Sphi / s2 # recursive update
        theta = theta + gain * (y_all[t] - mu)
        Sigma = Sigma - np.outer(Sphi, Sphi) / s2
        Sigma = 0.5 * (Sigma + Sigma.T) # symmetric
    return np.array(preds), np.array(truths), np.array(varis)

    
# ----------------------------------------------------------------------
# 5. Per-point Gaussian log-likelihood (var = variance, not std)
# ----------------------------------------------------------------------
def gaussian_loglik(y, mu, var):
    var = np.maximum(var, 1e-12)
    return -0.5 * (np.log(2 * np.pi * var) + (y - mu) ** 2 / var)


# ----------------------------------------------------------------------
# 6. Causal multiplicative-weights ensemble (Bayesian model averaging).
#    w_i  <-  w_i * p_i(y_t),  done in log-space for stability.
#    The prediction for point t uses weights from points < t (no leakage).
#    Also returns the mixture predictive mean & variance (moment-matched
#    Gaussian), which the conformal layer uses for the NLL score.
# ----------------------------------------------------------------------
def ensemble(preds, varis, logliks, init_weights=None):
    n_models, T = preds.shape
    w = (np.full(n_models, 1.0 / n_models) if init_weights is None
         else np.asarray(init_weights, float))
    ens_mu = np.zeros(T)
    ens_var = np.zeros(T)
    weight_hist = np.zeros((T, n_models))
    for t in range(T):
        weight_hist[t] = w
        mu_t = w @ preds[:, t]                              # mixture mean
        # variance of a mixture of Gaussians:
        #   Var = sum_i w_i (var_i + mu_i^2) - (sum_i w_i mu_i)^2
        second = w @ (varis[:, t] + preds[:, t] ** 2)
        ens_mu[t] = mu_t
        ens_var[t] = second - mu_t ** 2
        logw = np.log(w + 1e-300) + logliks[:, t]
        logw -= logw.max()
        w = np.exp(logw)
        w /= w.sum()
    return ens_mu, ens_var, weight_hist


# ----------------------------------------------------------------------
# 7. Bayes credible set and standard conformal prediction
#    (Angelopoulos, Barber & Bates, 2022).
#
#    Score (Gaussian NLL):
#        s_t(y) = 0.5*log(2*pi*sigma_t^2) + (y - mu_t)^2 / (2*sigma_t^2)
#    Prediction set  C_t = {y : s_t(y) <= q_t}  is the closed interval
#        mu_t +/- sqrt( 2*sigma_t^2 * (q_t - 0.5*log(2*pi*sigma_t^2)) ),
#    or EMPTY when the radicand is < 0. (Unlike a residual score, the NLL
#    diverges as |y|->inf, so the set is never infinite.)
# ----------------------------------------------------------------------
def bayes_credible(y, mu, var, alpha=0.1):
    c = norm.ppf(1 - alpha / 2)                       # fixed multiplier (1.645 @ alpha=0.1)
    sd = np.sqrt(np.maximum(var, 1e-12))
    return dict(lo=mu - c*sd, hi=mu + c*sd,
                covered=np.abs(y - mu) <= c*sd, size=2*c*sd)

def standard_cp(y, mu, var, alpha=0.1):
    T = len(y)
    var = np.maximum(np.asarray(var, float), 1e-12)
    nll_min = 0.5 * np.log(2 * np.pi * var)              # min score, at y = mu
    score = nll_min + (y - mu) ** 2 / (2 * var)          # observed score s_t(Y_t)

    lo = np.full(T, np.nan)
    hi = np.full(T, np.nan)
    covered = np.zeros(T, bool)
    size = np.zeros(T)
    past = []
    
    for t in range(T):
        q = np.quantile(past, 1-alpha, method="higher") if past else score[t]
        rad = 2 * var[t] * (q - nll_min[t])
        if rad >= 0:
            h = np.sqrt(rad)
            lo[t], hi[t] = mu[t]-h, mu[t]+h
            size[t] = 2 * h
        covered[t] = score[t] <= q
        past.append(score[t])
    return dict(lo=lo, hi=hi, covered=covered, size=size)
    
# ----------------------------------------------------------------------
# 8. Online conformal prediction with decaying and varying step size
#    (Angelopoulos, Barber & Bates, 2024; Xu, Lu & Giannakis, 2024).
#
#    Score (Gaussian NLL):
#        s_t(y) = 0.5*log(2*pi*sigma_t^2) + (y - mu_t)^2 / (2*sigma_t^2)
#    Prediction set  C_t = {y : s_t(y) <= q_t}  is the closed interval
#        mu_t +/- sqrt( 2*sigma_t^2 * (q_t - 0.5*log(2*pi*sigma_t^2)) ),
#    or EMPTY when the radicand is < 0. (Unlike a residual score, the NLL
#    diverges as |y|->inf, so the set is never infinite.)
#    Threshold update (eq. 4):
#        q_{t+1} = q_t + eta_t * ( 1{Y_t not in C_t} - alpha )
#    Step size:  
#       step = float      -> fixed eta (Gibbs & Candes 2021)
#       step = 'decaying' -> pure decay eta_t = (t+1)^(-a), no reset     (Angelopoulos, Barber & Bates, 2024)
#       step = 'varying'  -> eta_t = tau^(-a)
#                            tau = slots since last reset
#                            reset when the windowed-average set size rises for r consecutive steps  (Xu, Lu & Giannakis, 2024)
# ----------------------------------------------------------------------
def online_conformal(y, mu, var, alpha=0.1, step="varying", c=1.0, eps=0.1,
                     W=15, r=100, q1=None):
    y = np.asarray(y, float)
    T = len(y)
    var = np.maximum(np.asarray(var, float), 1e-12)
    nll_min = 0.5 * np.log(2 * np.pi * var)
    score   = nll_min + (y - mu) ** 2 / (2 * var)
    
    if q1 is None:
        q1 = float(np.median(nll_min) + chi2.ppf(1.0 - alpha, df=1) / 2)
    q = q1
    
    lo = np.full(T, np.nan)
    hi = np.full(T, np.nan)
    size = np.zeros(T)
    covered = np.zeros(T, bool)
    q_path = np.empty(T)
    
    resets, win = [], []
    tau = 1
    prev_avg = None
    rises = 0
    a = 0.5 + eps
    
    for t in range(T):
        q_path[t] = q
        rad = 2 * var[t] * (q - nll_min[t])
        if rad >= 0.0:
            h = np.sqrt(rad)
            lo[t], hi[t] = mu[t] - h, mu[t] + h
            size[t] = 2 * h
        covered[t] = score[t] <= q
        
        if step == "varying":
            win.append(size[t])
            if len(win) > W: 
                win.pop(0)
            if len(win) == W:                       # set-size change detector
                avg = np.mean(win)
                if (prev_avg is not None and avg > prev_avg):
                    rises +=1
                else: 
                    rises = 0
                prev_avg = avg
                if rises >= r:
                    tau = 1
                    rises = 0
                    resets.append(t)
            eta = c * tau ** (-a)
            tau += 1
        elif step == "decaying":
            eta = c * (t + 1) ** (-a)
        else: # fixed constant eta
            eta = float(step)
            
        q = q + eta * ((0.0 if covered[t] else 1.0) - alpha)
    return dict(lo=lo, hi=hi, size=size, covered=covered, q_path=q_path, resets=resets)

    
# ----------------------------------------------------------------------
# 9. Hyperparameter tuning
# ----------------------------------------------------------------------
def fit_hyperparameters(X, y, n0=100):
    k = (ConstantKernel(1.0,(1e-3,1e3))*RBF(1.0,(1e-2,1e2)) + WhiteKernel(0.1,(1e-6,1e1)))
    gpr = GaussianProcessRegressor(kernel=k, alpha=0.0, n_restarts_optimizer=2).fit(X[:n0], y[:n0])
    p = gpr.kernel_.get_params()
    return (float(p["k1__k2__length_scale"]),                 # length_scale
            float(np.sqrt(p["k1__k1__constant_value"])),      # sigma_f
            float(np.sqrt(max(p["k2__noise_level"], 1e-6))))  # sigma_n
