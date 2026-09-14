import numpy as np


def simulate_module1_step31(H_trajectory, params, return_all=False):
    """
    H_trajectory: shape (1500,)
    params: dict of model parameters

    return_all=False -> (n_obs, mz_obs, phi_TC_obs)
    return_all=True  -> (n_obs, mz_obs, phi_C_obs, phi_TC_obs, phi_Sk_obs)
    """
    T = len(H_trajectory)

    # 状態変数の初期化（初期状態は全てコニカル相）
    phi_C = np.ones(T)
    phi_TC = np.zeros(T)
    phi_Sk = np.zeros(T)

    m_z = np.clip(H_trajectory / params["H_c2"], 0.0, 1.0)

    # パラメータ展開
    G_C_TC_rate = params["Gamma_C_TC"]
    G_TC_C_rate = params["Gamma_TC_C"]
    gamma_0 = params["gamma_0"]
    m = params["m"]  # 非線形指数（m > 1）
    h_min, h_max = params["H_min"], params["H_max"]
    dt = params.get("dt", 1.0)

    # 3.2 アバランシェ崩壊用パラメータ（新旧両対応）
    nu_0 = params.get("nu_0", 0.0)
    delta_E0 = params.get("delta_E0", 0.0)
    kappa = params.get("kappa", 0.0)
    k_B_T = params.get("k_B_T", 1.0)
    lam_kill = params.get("lam_kill", 0.0)

    for t in range(T - 1):
        H_t = H_trajectory[t]

        # 相判定
        in_sky = 1.0 if (h_min <= H_t <= h_max) else 0.0

        # 遷移レートの計算
        G_C_TC = G_C_TC_rate * in_sky
        G_TC_C = G_TC_C_rate * (1.0 - in_sky)
        # TC相の蓄積密度に依存する自己触媒的な遷移レート
        G_TC_Sk = gamma_0 * (phi_TC[t] ** m) * in_sky

        if lam_kill > 0.0:
            G_Sk_C = lam_kill * (1.0 - in_sky)
        else:
            # KTHNY融解理論に基づく非線形緩和レート
            E_activation = delta_E0 - kappa * (phi_Sk[t] ** 2)
            E_activation = np.maximum(E_activation, 0.0)  # 負の活性化エネルギー防止
            G_Sk_C = nu_0 * np.exp(-E_activation / k_B_T) * (1.0 - in_sky)

        # 各相の差分更新
        d_phi_TC = (G_C_TC * phi_C[t] - G_TC_C * phi_TC[t] - G_TC_Sk * phi_TC[t]) * dt
        d_phi_Sk = (G_TC_Sk * phi_TC[t] - G_Sk_C * phi_Sk[t]) * dt

        phi_TC[t + 1] = np.clip(phi_TC[t] + d_phi_TC, 0.0, 1.0)
        phi_Sk[t + 1] = np.clip(phi_Sk[t] + d_phi_Sk, 0.0, 1.0)
        phi_C[t + 1] = np.clip(1.0 - phi_TC[t + 1] - phi_Sk[t + 1], 0.0, 1.0)

    # 各サイクルの終点 (H_low) を抽出
    obs_idx = np.arange(2, T, 3)
    n_obs = phi_Sk[obs_idx]
    mz_obs = m_z[obs_idx]
    phi_C_obs = phi_C[obs_idx]
    phi_TC_obs = phi_TC[obs_idx]
    phi_Sk_obs = phi_Sk[obs_idx]

    if return_all:
        return n_obs, mz_obs, phi_C_obs, phi_TC_obs, phi_Sk_obs

    return n_obs, mz_obs, phi_TC_obs