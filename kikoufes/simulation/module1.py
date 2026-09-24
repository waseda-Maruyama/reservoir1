import numpy as np


def _soft_window(H, h_min, h_max, edge_width):
    """
    スキルミオン準安定窓 [h_min, h_max] の滑らかな連続版指示関数。
    中心付近で1、端に近づくにつれて滑らかに0へ減衰し、窓の外でも連続的に0へ近づく。
    edge_width -> 0 の極限で、従来の2値 in_sky = 1[h_min<=H<=h_max] に一致する。

    Lee論文(2209.06962)本文・関連文献の物理描像（相図・核生成理論の一般論として、
    準安定領域の中心に近いほどエネルギー的に安定＝核生成に有利、端に近いほど
    隣接相へ崩壊しやすい）を反映した設計。
    """
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
    return sigmoid((H - h_min) / edge_width) * sigmoid((h_max - H) / edge_width)


def simulate_module1_step31(H_trajectory, params, return_all=False):
    """
    H_trajectory: shape (1500,)
    params: dict of model parameters

    return_all=False -> (n_obs, mz_obs, phi_TC_obs)
    return_all=True  -> (n_obs, mz_obs, phi_He_obs, phi_TC_obs, phi_Sk_obs)

    --- 変更点（この版での再設計） ---
    1. phi_C -> phi_He にリネーム（Module2側の再解釈と統一。理由は既出）。

    2. 【最重要】in_sky という2値ゲートを廃止し、連続関数 S(H) に置き換えた。
       これまでの設計（in_sky = 1[h_min<=H<=h_max] という2値）には根本的な
       欠陥があった。Hc=73mT, Hrange=90mTのような、磁場が常にスキルミオン
       窓の内側に留まる条件では in_sky が全区間で1固定となり、Module1の
       力学方程式は「窓の中か外か」という1ビットの情報しか磁場から受け取れ
       なくなる。この場合、系はH(N)（Mackey-Glass等の複雑な時系列）の実際の
       値によらない自律系（autonomous system）に退化し、単一の固定点に収束
       するだけで、周期性も履歴依存性も原理的に生まれ得ない。
       これはLee論文が明記する「予測タスクの最良性能はフィールドサイクリング
       が完全にスキルミオン相の内側に収まるときに得られる」「サイクリングに
       つれてスキルミオンは絶えず破壊・再核生成される」という記述と矛盾する。

       磁性体の相図・核生成理論の一般論（準安定領域の中心に近いほど自由エネル
       ギー的に安定=核生成に有利、境界に近いほど隣接相へ緩和しやすい）に基づき、
       in_sky を滑らかな連続関数 S(H)（_soft_window、[0,1]値）に置き換えた。
       これにより窓の内側にいる間もH(N)の実際の値が連続的に力学へ伝わる。

    3. 【今回の再設計】TC->Skの成長則を、根拠のない自己触媒項
       `gamma_0 * phi_TC^m`（TC単独の累乗）から、Avrami（JMAK）核生成成長理論
       に基づく界面成長項に置き換えた。
       旧式の構造的問題: TC->He（定数レート、常時作動）と TC->Sk（phi_TCの
       3乗、低密度では極端に弱い）が競合する構造で、phi_TCがある閾値密度
       （既存パラメータでは≈0.43）に達するまでTC->Heが常に優勢になり、
       He->TC供給をいくら強めてもphi_Skが0.3-0.4程度で頭打ちになる、という
       構造的ボトルネックがあった（フラックス収支計算で確認済み）。
       新式: growth_flux = (k_nuc + k_growth * phi_TC * phi_Sk) * S(H)
       - k_nuc: 自発核生成項（phi_Sk=0からの立ち上がりの種）
       - k_growth * phi_TC * phi_Sk: 界面成長項（TC自身の累乗ではなく、
         既存のSk"種"がTCを侵食する速度に比例）。Malsch et al. (MFM実空間
         観察論文)の「スキルミオン格子はtilted conical状態を消費して成長
         する」という記述に対応
       この形なら、わずかでもphi_Skの種があれば効率よく成長でき、TC単独の
       閾値到達を待つ必要がなくなる。崩壊側（KTHNY融解理論、Sk->Heの
       アバランシェ）は既存のまま維持し、崩壊=KTHNY融解理論／成長=Avrami
       核生成成長理論という、両側とも文献的に確立された理論の組み合わせに
       なるよう設計した。
    """
    T = len(H_trajectory)

    # 状態変数の初期化（初期状態は全てヘリカル相）
    phi_He = np.ones(T)
    phi_TC = np.zeros(T)
    phi_Sk = np.zeros(T)

    m_z = np.clip(H_trajectory / params["H_c2"], 0.0, 1.0)

    # パラメータ展開
    G_He_TC_rate = params["Gamma_He_TC"]
    G_TC_He_rate = params["Gamma_TC_He"]
    k_nuc = params.get("k_nuc", 0.001)        # 自発核生成項（Avrami理論）
    k_growth = params.get("k_growth", 0.1)    # 界面成長項の速度定数（Avrami理論）
    h_min, h_max = params["H_min"], params["H_max"]
    dt = params.get("dt", 1.0)

    # 窓の滑らかさ・崩壊ブーストの強さ
    edge_width = params.get("H_edge_width", 10.0)  # 窓端の遷移の滑らかさ [mT]
    edge_decay_boost = params.get("edge_decay_boost", 5.0)  # 端での崩壊加速の強さ

    # 3.2 アバランシェ崩壊用パラメータ（新旧両対応、KTHNY融解理論）
    nu_0 = params.get("nu_0", 0.0)
    delta_E0 = params.get("delta_E0", 0.0)
    kappa = params.get("kappa", 0.0)
    k_B_T = params.get("k_B_T", 1.0)
    lam_kill = params.get("lam_kill", 0.0)

    for t in range(T - 1):
        H_t = H_trajectory[t]

        # 窓の連続版指示関数（中心=1、端・外に向けて滑らかに0へ）
        S_t = _soft_window(H_t, h_min, h_max, edge_width)
        instability_t = 1.0 - S_t

        # --- He->TC（核生成）: S(H) に連続的に比例 ---
        G_He_TC = G_He_TC_rate * S_t

        # --- TC->Sk（成長）: Avrami（JMAK）核生成成長理論に基づく界面成長項 ---
        # 自発核生成(k_nuc) + 既存のSk"種"によるTC侵食(k_growth*phi_TC*phi_Sk)。
        # S(H)でゲート（成長も核生成同様、窓の中心に近いほど有利という想定）。
        growth_flux = (k_nuc + k_growth * phi_TC[t] * phi_Sk[t]) * S_t

        # --- TC->He（緩和）: 定数レート ---
        G_TC_He = G_TC_He_rate

        if lam_kill > 0.0:
            G_Sk_He = lam_kill * (1.0 + edge_decay_boost * instability_t)
        else:
            # KTHNY融解理論に基づく非線形緩和レート（熱活性化、常時作動）。
            # 境界ブーストはこちら（Sk->Heのアバランシェ経路）にのみ残す。
            E_activation = delta_E0 - kappa * (phi_Sk[t] ** 2)
            E_activation = np.maximum(E_activation, 0.0)  # 負の活性化エネルギー防止
            G_Sk_He = nu_0 * np.exp(-E_activation / k_B_T) * (1.0 + edge_decay_boost * instability_t)

        # growth_fluxがその場で使える phi_TC 量を超えないようにクリップ
        # （Avrami成長項が1ステップでTCを使い切ってしまう極端なパラメータでの
        #  負値化を防ぐ安全策）
        growth_flux = min(growth_flux, phi_TC[t] / dt)

        # 各相の差分更新
        d_phi_TC = (G_He_TC * phi_He[t] - G_TC_He * phi_TC[t] - growth_flux) * dt
        d_phi_Sk = (growth_flux - G_Sk_He * phi_Sk[t]) * dt

        phi_TC[t + 1] = np.clip(phi_TC[t] + d_phi_TC, 0.0, 1.0)
        phi_Sk[t + 1] = np.clip(phi_Sk[t] + d_phi_Sk, 0.0, 1.0)
        phi_He[t + 1] = np.clip(1.0 - phi_TC[t + 1] - phi_Sk[t + 1], 0.0, 1.0)

    # 各サイクルの終点 (H_low) を抽出
    obs_idx = np.arange(2, T, 3)
    n_obs = phi_Sk[obs_idx]
    mz_obs = m_z[obs_idx]
    phi_He_obs = phi_He[obs_idx]
    phi_TC_obs = phi_TC[obs_idx]
    phi_Sk_obs = phi_Sk[obs_idx]

    if return_all:
        return n_obs, mz_obs, phi_He_obs, phi_TC_obs, phi_Sk_obs

    return n_obs, mz_obs, phi_TC_obs