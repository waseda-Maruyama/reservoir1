"""
fast_metrics.py --- PRCpy と同一定義の NL / MC / CP を、ファイル I/O を介さず
メモリ上で計算する。パラメータ掃引用。

PRCpy 本体 (prcpy/Maths/Maths_functions.py, prcpy/RC/Pipeline_RC.py) の
実装をそのまま写し取っている:

  estimator_capacity(u, X):
      n_train = 0.75 * len(u); LinearRegression で学習
      return cov(u_test, u_pred)^2 / (var(u) * var(u_pred))

  linear_memory_curve(u, X, kmax):
      for k in 1..kmax-1: estimator_capacity(u[:-k], X[k:])

  get_non_linearity(k=25):
      u_history = sliding_window_view(pad(u, k-1), k)[:len(u)]
      NL = 1 - mean_over_channels(estimator_capacity(x_channel, u_history))

  get_linear_memory_capacity(kmax, remove_auto_correlation=True):
      MC = sum(mc_res) - sum(mc_auto)   (mc_auto は入力自身を readout とした場合)

前処理も PRCpy の Prepare_RC と同じ順序で適用する:
  smooth (Savitzky-Golay) -> cut -> normalize -> sample
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import LinearRegression
from scipy.signal import savgol_filter


def _cov(a, b):
    return np.mean((a - np.mean(a)) * (b - np.mean(b)))


def estimator_capacity(u, X):
    """PRCpy と同一。"""
    n_train = int(u.shape[0] * 0.75)
    est = LinearRegression().fit(X[:n_train], u[:n_train])
    u_pred = est.predict(X[n_train:])
    u_test = u[n_train:]
    vp = np.var(u_pred)
    if vp <= 0:
        return 0.0
    return _cov(u_test, u_pred) ** 2 / (np.var(u) * vp)


def linear_memory_curve(u, X, kmax=12):
    return [estimator_capacity(u[:-k], X[k:]) for k in range(1, kmax)]


def non_linearity(u, X, k=25):
    u_pad = np.pad(u, (k - 1), "constant", constant_values=(0, 0))
    u_hist = sliding_window_view(u_pad, k)[:len(u)]
    lin = [estimator_capacity(x, u_hist) for x in X.T]
    return 1.0 - float(np.mean(lin))


def memory_capacity(u, X, kmax=12, remove_auto=True):
    mc_res = np.array(linear_memory_curve(u, X, kmax))
    if remove_auto:
        mc_auto = np.array(linear_memory_curve(u, u.reshape(-1, 1), kmax))
        return float(np.sum(mc_res - mc_auto)), mc_res - mc_auto
    return float(np.sum(mc_res)), mc_res


def effective_rank(M):
    s = np.linalg.svd(np.asarray(M, float), compute_uv=False)
    s = s[s > 1e-12]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


def complexity(X, split="half"):
    """原論文 SM S6 の CP: readout 行列を2分割し、それぞれの有効ランクの平均。"""
    n = X.shape[0]
    if split == "half":
        h = n // 2
        A, B = X[:h], X[h:2 * h]
    else:
        A, B = X[0::2], X[1::2]
        m = min(len(A), len(B)); A, B = A[:m], B[:m]
    sq = lambda M: M[:min(M.shape), :min(M.shape)]
    return 0.5 * (effective_rank(sq(A)) + effective_rank(sq(B)))


def preprocess(S11, sample_rate=8, smooth=True, smooth_win=51, smooth_rank=4,
               normalize_global=True):
    """PRCpy の Prepare_RC と同じ順序の前処理。"""
    X = np.asarray(S11, dtype=float).copy()
    if smooth:
        X = savgol_filter(X, smooth_win, smooth_rank, axis=1)
    if normalize_global:
        mn, mx = X.min(), X.max()
        if mx > mn:
            X = (X - mn) / (mx - mn)
    if sample_rate and sample_rate > 1:
        X = X[:, ::sample_rate]
    return X


def evaluate_fast(S11, u_input, target=None, tau=10, test_size=0.3,
                  alpha=1e-1, kmax=12, nl_k=25, sample_rate=8, smooth=True,
                  top_frac=None):
    """
    NL / MC / CP と Ridge 予測の MSE をまとめて返す。

    top_frac : None 以外なら、レンジ上位 top_frac の割合のチャンネルだけを残す
        (原論文 SM: "Channels with low range are effectively noise dominated
         'dead' regions of the spectra and do not contribute meaningful
         reservoir metric information.")
    """
    from sklearn.linear_model import Ridge
    X = preprocess(S11, sample_rate=sample_rate, smooth=smooth)
    if top_frac is not None:
        rng_ = X.max(axis=0) - X.min(axis=0)
        keep = np.argsort(rng_)[::-1][:max(1, int(np.ceil(X.shape[1] * top_frac)))]
        X = X[:, np.sort(keep)]

    u = np.asarray(u_input, float)
    tgt = np.asarray(target if target is not None else u_input, float)

    nl = non_linearity(u, X, k=nl_k)
    mc, mc_curve = memory_capacity(u, X, kmax=kmax)
    cp = complexity(X)

    # tau ステップ先予測 (PRCpy の Perform_RC と同じ切り出し方)
    Xr, yr = X[:-tau], tgt[tau:]
    n_tr = int(len(yr) * (1 - test_size))
    mdl = Ridge(alpha=alpha).fit(Xr[:n_tr], yr[:n_tr])
    tr = float(np.mean((mdl.predict(Xr[:n_tr]) - yr[:n_tr]) ** 2))
    te = float(np.mean((mdl.predict(Xr[n_tr:]) - yr[n_tr:]) ** 2))
    return {"n_feat": X.shape[1], "NL": nl, "MC": mc, "CP": cp,
            "train_MSE": tr, "test_MSE": te, "mc_curve": mc_curve}