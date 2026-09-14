import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


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
    # amp_Sk_power: 検出閾値を模した非線形応答（低phi_Skでは振幅がほぼゼロになる）
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


def run_module2(
    phi_C_obs,
    phi_TC_obs,
    phi_Sk_obs,
    mz_obs,
    H_c2,
    N_start=0,
    N_end=None,
    N_step=1,
    params=None,
):
    """
    Module1の出力(phi_C_obs, phi_TC_obs, phi_Sk_obs, mz_obs)から、
    N=N_start..N_end を N_step 刻みでサンプルしたΔS11(f)のセットを作る。

    N_end=None の場合は配列の末尾まで使う。
    フル分の観測(MG入力でのreservoir評価など)にはN_step=1を指定する。

    mz_obs は Module1 で m_z = clip(H/H_c2, 0, 1) として計算されている前提。
    ここから H を逆算する（クリップされた場合は近似になる点に注意）。
    """
    if params is None:
        params = default_params()

    f_axis = np.linspace(params["f_min"], params["f_max"], params["n_freq"])

    if N_end is None:
        N_end = len(mz_obs) - 1

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


# 後方互換のためのエイリアス（Fig.2b再現時の呼び出し方と同じシグネチャ）
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
    return run_module2(
        phi_C_obs, phi_TC_obs, phi_Sk_obs, mz_obs, H_c2,
        N_start=N_start, N_end=N_end, N_step=N_step, params=params,
    )


def export_prcpy_scan_files(
    f_axis,
    N_indices,
    H_arr,
    S11_arr,
    output_dir,
    scan_start_index=1,
    field_decimals=3,
):
    """
    PRCpyが読み込むCSV形式("Current","Field","Frequency","Spectra")で、
    各観測点(N)ごとに scan_<連番>_<H_lowの値>.txt を出力する。

    Current列は計算に使わないため常に0で埋める。
    Field列はその観測点でのH_low(または対応する磁場)の値を全行に定数で入れる。
    Frequency, Spectra は f_axis, S11_arr[i] をそのまま行展開する。

    output_dir: 出力先ディレクトリ(存在しなければ作成する)
    scan_start_index: ファイル名の連番の開始値(デフォルト1)
    field_decimals: ファイル名・Field列に使う磁場値の小数点以下桁数
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written_paths = []
    for i, N in enumerate(N_indices):
        scan_idx = scan_start_index + i
        H_val = float(H_arr[i])
        H_str = f"{H_val:.{field_decimals}f}"
        fname = f"scan_{scan_idx}_{H_str}.txt"
        fpath = output_dir / fname

        with open(fpath, "w", newline="") as fh:
            fh.write('"Current","Field","Frequency","Spectra"\n')
            for freq, s11 in zip(f_axis, S11_arr[i]):
                fh.write(f"0,{H_val},{freq},{s11}\n")

        written_paths.append(fpath)

    return written_paths


def default_params():
    """
    初期値。実測(Fig.2b, Fig.2eなど)との比較を見ながら調整すること。

    論文からの手がかり:
    - コニカル主ピーク: 実測で3.5-5.5 GHz、特に4.1 GHz付近が最も深い
    - スキルミオンモード: 2-3 GHz帯（counter-clockwise / breathing modes、2本)
    - TCモード: コニカル-スキルミオン間の中間域と仮定。欠陥/ひずみ状態のため
      コヒーレントなモードより線幅が広いと想定（値は要フィット）
    - amp_Sk_power: スキルミオンモードの検出閾値を模した非線形応答の指数。
      power=4でFig.2bの見た目(N=100まで無反応->急激な出現)に近い結果が得られた。
      物理的な裏付けはまだ弱く、要検討(Module1側の生成をもっと緩やかにする
      方向でも同じ見た目を作れる可能性がある)。
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
        "f_TC0": 3.2,
        "slope_TC": 0.0005,
        "width_TC": 0.4,
        "amp_TC_scale": 1.5,
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
        "amp_Sk_power": 4.0,
        # オフセット
        "offset_C": 0.0,
    }


def plot_waterfall(f_axis, N_indices, S11_list, title="Module2 synthetic S11"):
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


if __name__ == "__main__":
    # --- ダミーデータでの動作確認 ---
    n_cycles = 300
    N_full = np.arange(n_cycles)

    phi_C_obs_dummy = np.clip(1.0 - N_full / 60.0, 0, 1) * np.exp(-N_full / 300.0)
    phi_Sk_obs_dummy = np.clip((N_full % 150) / 130.0, 0, 1)
    phi_TC_obs_dummy = np.clip(1.0 - phi_C_obs_dummy - phi_Sk_obs_dummy, 0, 1) * 0.5
    mz_obs_dummy = 0.5 + 0.15 * np.sin(2 * np.pi * N_full / 100)

    f_axis, N_indices, H_arr, S11_arr = run_module2(
        phi_C_obs_dummy,
        phi_TC_obs_dummy,
        phi_Sk_obs_dummy,
        mz_obs_dummy,
        H_c2=170.0,
        N_start=0,
        N_end=n_cycles - 1,
        N_step=1,
    )

    out_dir = Path(__file__).resolve().parent / "data_full" / "Cu2OSeO3" / "skyrmion"
    written = export_prcpy_scan_files(f_axis, N_indices, H_arr, S11_arr, out_dir)

    print(f"Wrote {len(written)} scan files to: {out_dir}")
    print("Example files:")
    for p in written[:3]:
        print(" ", p.name)

    # 中身の確認（先頭ファイルの先頭数行）
    with open(written[0]) as fh:
        head = [next(fh) for _ in range(5)]
    print("\nSample file content (first 5 lines):")
    print("".join(head))