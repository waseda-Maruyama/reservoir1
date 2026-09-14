import numpy as np
import matplotlib.pyplot as plt


def lorentzian_dip(f, center, width, amplitude):
    """
    吸収ディップ型ローレンツ関数。
    ΔS11(f) = -amplitude * width^2 / ((f-center)^2 + width^2)
    共鳴周波数で負に落ち込む（dB表示の吸収ピークを模す）。
    """
    return -amplitude * (width ** 2) / ((f - center) ** 2 + width ** 2)


def synthesize_S11(f_axis, H, phi_C, phi_TC, phi_Sk, params):
    """
    3状態（コニカル・TC(トポロジカル欠陥/ひずみ中間状態)・スキルミオン）から
    ΔS11(f)を合成する。スキルミオンはCCW/breathingの2モードに分割。

    f_axis : (M,) 周波数軸 [GHz]
    H      : スカラー、この観測点での磁場 [mT]
    phi_C  : スカラー、コニカル相分率
    phi_TC : スカラー、TC(中間欠陥)状態分率
    phi_Sk : スカラー、スキルミオン相分率
    params : dict、モデルパラメータ
    """
    # --- コニカルモード: 中心周波数がHに対して線形シフト（ゼーマン風） ---
    f_C = params["f_C0"] + params["slope_C"] * H
    w_C = params["width_C"]
    amp_C = params["amp_C_scale"] * phi_C

    # --- TCモード: コニカルとスキルミオンの中間域。欠陥/ひずみ状態なので
    #     コヒーレントなモードより広い線幅(width_TC)を想定 ---
    f_TC = params["f_TC0"] + params["slope_TC"] * H
    w_TC = params["width_TC"]
    amp_TC = params["amp_TC_scale"] * phi_TC

    # --- スキルミオン: CCWモードとbreathingモードに分割 ---
    # phi_Sk のうち alpha_CCW の割合をCCWモード振幅に、残りをbreathingモードに配分
    amp_power = params.get("amp_Sk_power", 4.0)
    alpha_CCW = params["alpha_CCW"]

    f_Sk_ccw = params["f_Sk_ccw0"] + params["slope_Sk_ccw"] * H
    w_Sk_ccw = params["width_Sk_ccw"]
    amp_Sk_ccw = params["amp_Sk_scale"] * alpha_CCW * (phi_Sk ** amp_power)

    f_Sk_breath = params["f_Sk_breath0"] + params["slope_Sk_breath"] * H
    w_Sk_breath = params["width_Sk_breath"]
    amp_Sk_breath = params["amp_Sk_scale"] * (1.0 - alpha_CCW) * (phi_Sk ** amp_power)

    S11 = (
        lorentzian_dip(f_axis, f_C, w_C, amp_C)
        + lorentzian_dip(f_axis, f_TC, w_TC, amp_TC)
        + lorentzian_dip(f_axis, f_Sk_ccw, w_Sk_ccw, amp_Sk_ccw)
        + lorentzian_dip(f_axis, f_Sk_breath, w_Sk_breath, amp_Sk_breath)
        + params["offset_C"]
    )
    return S11


def load_module1_result(npz_path):
    """
    module1 の出力 npz を読み込んで、module2 が使う四つの配列を返す。
    期待するキー: phi_C_obs, phi_TC_obs, phi_Sk_obs, mz_obs
    """
    data = np.load(npz_path)
    required = {"phi_C_obs", "phi_TC_obs", "phi_Sk_obs", "mz_obs"}
    missing = sorted(required - set(data.files))
    if missing:
        raise KeyError(f"Missing keys in {npz_path}: {missing}")
    return (
        data["phi_C_obs"],
        data["phi_TC_obs"],
        data["phi_Sk_obs"],
        data["mz_obs"],
    )


def run_module2_fig2b_style(
    phi_C_obs,
    phi_TC_obs,
    phi_Sk_obs,
    mz_obs,
    H_c2,
    N_start=100,
    N_end=200,
    N_step=10,
    params=None,
):
    """
    Module1の出力(phi_C_obs, phi_TC_obs, phi_Sk_obs, mz_obs)から、
    N=N_start..N_end を N_step 刻みでサンプルしたΔS11(f)のセットを作る。

    mz_obs は Module1 で m_z = clip(H/H_c2, 0, 1) として計算されている前提。
    ここから H を逆算する（クリップされた場合は近似になる点に注意）。
    """
    if params is None:
        params = default_params()

    f_axis = np.linspace(params["f_min"], params["f_max"], params["n_freq"])

    N_indices = np.arange(N_start, N_end + 1, N_step)
    N_indices = [int(n) for n in N_indices if 0 <= n < len(mz_obs)]

    S11_list = []
    H_list = []
    for N in N_indices:
        phi_C = float(phi_C_obs[N])
        phi_TC = float(phi_TC_obs[N])
        phi_Sk = float(phi_Sk_obs[N])
        mz = float(mz_obs[N])

        # mzからHを逆算（クリップ域では不正確になる点に注意。
        # 可能ならModule1側でH_low_obsを直接保存して渡す方が正確）
        H = mz * H_c2

        S11 = synthesize_S11(f_axis, H, phi_C, phi_TC, phi_Sk, params)
        S11_list.append(S11)
        H_list.append(H)

    return f_axis, np.array(N_indices), np.array(H_list), np.array(S11_list)


def default_params():
    """
    初期値。実測(Fig.2b, Fig.2eなど)との比較を見ながら調整すること。

    論文からの手がかり:
    - コニカル主ピーク: 実測で3.5-5.5 GHz、特に4.1 GHz付近が最も深い
    - スキルミオンモード: 2-3 GHz帯（counter-clockwise / breathing modes、2本)
    - TCモード: コニカル-スキルミオン間の中間域と仮定。欠陥/ひずみ状態のため
      コヒーレントなモードより線幅が広いと想定（値は要フィット）
    """
    return {
        "f_min": 1.0,
        "f_max": 6.0,
        "n_freq": 1601,
        # コニカルモード
        "f_C0": 4.1,
        "slope_C": 0.002,
        "width_C": 0.15,
        "amp_C_scale": 3.0,
        # TCモード（中間欠陥状態）
        "f_TC0": 3.2,          # コニカルとスキルミオンの中間あたり（仮）
        "slope_TC": 0.0005,    # 欠陥状態はHへの応答が鈍いと仮定（仮）
        "width_TC": 0.4,       # コヒーレントなモードより広い線幅（仮）
        "amp_TC_scale": 1.5,   # 単一モードのコニカル/スキルミオンより弱め（仮）
        # スキルミオン: CCWモード
        "f_Sk_ccw0": 2.6,
        "slope_Sk_ccw": -0.003,
        "width_Sk_ccw": 0.12,
        # スキルミオン: breathingモード
        "f_Sk_breath0": 2.1,
        "slope_Sk_breath": -0.002,
        "width_Sk_breath": 0.15,
        # phi_SkのうちCCWモードに配分する割合（残りはbreathingへ）
        "alpha_CCW": 0.5,
        "amp_Sk_scale": 3.0,
        # オフセット
        "offset_C": 0.0,
    }


def plot_waterfall(f_axis, N_indices, S11_list, title="Module2 (3-state, split Sk) synthetic S11"):
    fig, ax = plt.subplots(figsize=(7, 8))
    offset_step = 1.5
    for i, N in enumerate(N_indices):
        ax.plot(f_axis, S11_list[i] - i * offset_step, lw=1.2)
        ax.text(f_axis[-1] + 0.05, -i * offset_step, f"N={N}", va="center", fontsize=8)
    ax.set_xlabel("f (GHz)")
    ax.set_ylabel(r"$\Delta S_{11}$ (offset, arb.)")
    ax.set_title(title)
    ax.set_yticks([])
    fig.tight_layout()
    return fig, ax

