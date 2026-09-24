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

def generate_tc_fine_peaks(n_peaks=50, f_start=3.8, f_end=5.2, seed=42):
    """
    Tilted Conical (TC) 相の対称性低下に伴う多数のハイブリダイゼーション
    （マグノンモードの森）を、大小さまざまな共鳴ピークの集合として生成する。
    """
    rng = np.random.default_rng(seed)
    
    # 1. 3.8〜5.2GHz付近に多数のピークをランダム配置してソート
    f_list = np.sort(rng.uniform(f_start, f_end, n_peaks))
    
    # 2. 振幅のばらつき作成：対数正規分布で大小を散らし、ガウス型エンベロープで帯域の中心を強調
    raw_weights = rng.lognormal(mean=0.0, sigma=0.6, size=n_peaks)
    envelope = np.exp(-0.5 * ((f_list - 4.5) / 0.4) ** 2)
    amp_weights = raw_weights * envelope
    amp_weights /= np.sum(amp_weights)  # 全体の和を1に規格化
    
    # 3. 線幅：細かいギザギザを出すため主モードより細かく（0.1〜0.4）設定
    w_list = rng.uniform(0.1, 0.4, n_peaks)
    
    return f_list, amp_weights, w_list

# モジュール読み込み時に1度だけ生成しておく（計算コスト削減）
TC_F_LIST, TC_AMP_WEIGHTS, TC_W_LIST = generate_tc_fine_peaks(n_peaks=50)

def synthesize_S11(f_axis, H, phi_He, phi_TC, phi_Sk, params):
    """
    ヘリカル・スキルミオンの2状態からΔS11(f)を合成する。

    重要な訂正：旧版では phi_He を「コニカル分率」と呼んでいたが、今回のシミュ
    レーションの磁場レンジ（H_c=60・H_range=90でH自体は15〜105mTを往復）では、
    スキルミオン窓の外側で実際に出現するのは真のコニカル相（実測185mT相当）
    ではなく**ヘリカル相**である可能性が高い。そのためphi_Heは「ヘリカル分率」
    として再解釈している。変数名 phi_He 自体はModule1との互換性のため維持。

    理論的根拠のない要素は削除済み（旧版からの変更点）：
    - 旧TCモード（3.2GHz、単独Lorentzian、根拠なし）：全面削除。ただし後述の
      「核生成由来の微細構造」として、全く別の形（4GHz帯、複数の近接ピーク、
      Aqeel et al. 2021が根拠）で再導入している点に注意。同じphi_TCを使うが
      解釈・周波数帯・実装形態は別物。
    - amp_Sk_power(p)：report.pdf自身が「見た目合わせ」と認め、アブレーション
      実験でも性能に悪影響（Model3がModel0より悪化）と判明したため削除。
      スキルミオン振幅は phi_Sk に単純比例（線形）とする。
    - eta_Sk_ccw/eta_Sk_breath（履歴依存性②）：具体的な文献的根拠のない
      憶測だったため削除。
    - ヘリカルのslope（Hに対する周波数シフト）：「著しく平坦」という記述を
      素直に採用し、恣意的な小さい非ゼロ値ではなく厳密に0とした。

    ヘリカルモードの物理的根拠（Weiler et al., PRL 119, 237204 (2017)、
    Cu2OSeO3のVNA-FMR測定。ただし同文献はH0∥x,y,z((111)配向試料)での測定で
    あり、report.pdf(2209.06962)のH∥⟨100⟩とは印加方向が異なるため、モードの
    本数・相対振幅など定性的構造のみを採用し、絶対周波数・分裂幅はFig.2の
    独自測定値を用いている）:
    - ヘリカル相の捩れ構造自体が周期的な「マグノニック結晶」として働き、
      波数nQ (n=1,2,3,...)に量子化されたヘリマグノンモードを持つ
    - n=1モードは一様磁場と直接結合するため強く、±Q(右巻き/左巻き)の縮退が
      磁気双極子相互作用でわずかに解ける（未分裂のまま1本として暫定実装、
      分裂幅はピクセル解析待ち）
    - n=2モード以降は異方性による間接励起のため本質的に弱い
      （観測ベースでn=1の数%程度）。マルチドメイン状態のため2本に分裂
      （report.pdfのFig.2b/2cで観測された4.9GHz・5.1GHzの独自測定値）
    - 分散は「著しく平坦」（原論文2209.06962本文）＝Hに対する周波数シフト
      はほぼゼロ

    核生成由来の微細構造（TC / tilted conical）の根拠:
    Aqeel et al., PRL 126, 017202 (2021)。H∥⟨100⟩でreport.pdfと印加方向が
    一致（Weiler 2017とは異なりこの点は問題なし）。低温スキルミオン状態(LTS)
    の核生成が、tilted conical状態を経由する2段階プロセスであることを報告。
    - tilted conical状態は±Qモードと同じ~4GHz帯に「多数のハイブリダイゼー
      ション」として現れる（具体的な本数・間隔は文献に記載なし、目視に基づく
      粗い仮置き）
    - 実測（report.pdf Fig.2b/2c, Fig.4b）でも、4GHz帯の微細構造は
      (a) サイクル番号Nに依存して変動し、(b) ヘリカル・スキルミオンいずれが
      優勢な条件でも共通して現れ、(c) 高温（核生成が起きない条件）では消える、
      という性質が確認された。これはTC/Sk由来の起源としては矛盾しないが
      試料形状由来の固定的な磁気静的モードとしては説明できない（温度依存性
      と矛盾）ため、その可能性は棄却した
    - TC由来かSk由来かは文献・実測いずれからも判別できないが、
      「スキルミオン相が安定でなくても残留して見える」という観測から、
      両者のうちより履歴依存性の強いphi_TCに比例させることとした

    スキルミオンモードの根拠:
    - CCW/breathingへの分割は原論文本文に明記（"counter-clockwise and
      breathing modes"）
    - 振幅：ヘリカルn=1に対して「数%程度」というユーザー観測に基づき校正
    - alpha_CCW（CCWとbreathingの配分比）は根拠のない placeholder のまま
      （0.5）。分割の存在自体は文献根拠があるが、比率の数値は未検証

    アブレーション用トグル:
        split_sk_modes (bool)     : True なら スキルミオンを CCW / breathing の2モードに
                                     分割。False なら単一の統合Skモードとして扱う。
                                     （CCW/breathing分割は文献根拠のある唯一の
                                     残存アブレーション軸）
        include_helical_n2 (bool) : True なら弱いn=2ヘリマグノンモード(4.9/5.1GHz帯)
                                     を追加する。デフォルトTrue。
        include_tc_fine (bool)    : True なら核生成由来の微細構造(TC/tilted conical,
                                     4GHz帯の複数の近接ピーク)を追加する。デフォルトTrue。

    f_axis : (M,) 周波数軸 [GHz]
    H      : スカラー、この観測点での磁場 [mT]
    phi_He  : スカラー、ヘリカル相分率（旧: コニカル分率）
    phi_TC : スカラー、TC(tilted conical/核生成由来の中間状態)分率。4GHz帯の
             微細構造の振幅源として使用（Aqeel et al. 2021、詳細は下記）
    phi_Sk : スカラー、スキルミオン相分率
    params : dict、モデルパラメータ
    """
    split_sk_modes = params.get("split_sk_modes", True)
    include_helical_n2 = params.get("include_helical_n2", True)
    include_tc_fine = params.get("include_tc_fine", True)
    alpha_CCW = params["alpha_CCW"]

    # --- ヘリカル n=1 モード（±Qモード、未分裂のまま1本として暫定実装） ---
    # 分散は「著しく平坦」なのでslope=0（恣意的な非ゼロ値は置かない）
    f_He1 = params["f_He_n1"] + params["slope_He_n1"] * H
    w_He1 = params["width_He_n1"]
    amp_He1 = params["amp_He_n1_scale"] * phi_He

    S11 = lorentzian_dip(f_axis, f_He1, w_He1, amp_He1)

    # --- ヘリカル n=2 モード（弱い、異方性による間接励起、ドメインで2本に分裂） ---
    # 中心周波数はFig.2b/2cで観測された4.9GHz・5.1GHzに基づく独自測定値。
    if include_helical_n2:
        f_He2a = params["f_He_n2a"] + params["slope_He_n2"] * H
        f_He2b = params["f_He_n2b"] + params["slope_He_n2"] * H
        w_He2 = params["width_He_n2"]
        amp_He2_each = params["amp_He_n2_scale"] * phi_He * 0.5  # 2本で均等配分

        S11 = S11 + lorentzian_dip(f_axis, f_He2a, w_He2, amp_He2_each)
        S11 = S11 + lorentzian_dip(f_axis, f_He2b, w_He2, amp_He2_each)

    # --- 核生成由来の微細構造（TC / tilted conical, 4GHz帯） ---
    # 根拠: Aqeel et al., PRL 126, 017202 (2021)。H∥⟨100⟩でreport.pdfと印加方向が
    # 一致。tilted conical状態は±Qモードと同じ帯域(~4GHz)に「多数のハイブリダイ
    # ゼーション」として現れると記述されており、その振幅の由来（TC由来かSk由来か）
    # は文献からは判別できないが、以下の観測的性質から phi_TC に比例させる:
    #   - サイクル番号Nに依存して変動する（phi_TCも履歴依存の状態変数）
    #   - スキルミオン相が安定でない条件でも残留して見える
    #     （phi_TCは磁場がスキルミオン窓を外れても即座にゼロにならない）
    #   - 高温（核生成が起きない）相当の条件では消える（phi_TC=0ならゼロ）
    # 本数・周波数間隔は文献に具体的記載がなく、実測波形からの見た目に基づく
    # 粗い仮置き（振幅の精密さより、帯域・本数が正しいことを優先する方針）。
    if include_tc_fine:
        amp_fine_total = params["amp_tc_fine_scale"] * phi_TC
        
        # 事前生成した50本の細かいピーク群を加算
        for f_val, weight, w_val in zip(TC_F_LIST, TC_AMP_WEIGHTS, TC_W_LIST):
            # 各ピークの振幅は、全体のスケール(amp_fine_total) × 各ピークの相対重み(weight)
            amp_each = amp_fine_total * weight
            S11 = S11 + lorentzian_dip(f_axis, f_val + params["slope_He_n1"] * H, w_val, amp_each)

    # --- スキルミオン: 振幅は phi_Sk に単純線形比例（pは削除済み） ---
    if split_sk_modes:
        # CCWモードとbreathingモードに分割
        # phi_Sk のうち alpha_CCW の割合をCCWモード振幅に、残りをbreathingモードに配分
        f_Sk_ccw = params["f_Sk_ccw0"] + params["slope_Sk_ccw"] * H
        w_Sk_ccw = params["width_Sk_ccw"]
        amp_Sk_ccw = params["amp_Sk_scale"] * alpha_CCW * phi_Sk

        f_Sk_breath = params["f_Sk_breath0"] + params["slope_Sk_breath"] * H
        w_Sk_breath = params["width_Sk_breath"]
        amp_Sk_breath = params["amp_Sk_scale"] * (1.0 - alpha_CCW) * phi_Sk

        S11 = S11 + lorentzian_dip(f_axis, f_Sk_ccw, w_Sk_ccw, amp_Sk_ccw)
        S11 = S11 + lorentzian_dip(f_axis, f_Sk_breath, w_Sk_breath, amp_Sk_breath)
    else:
        # 統合Skモード：CCW/breathingを分割せず単一モードとして扱う。
        # 中心周波数・線幅は CCW と breathing の中間（平均）を採用し、
        # 振幅は phi_Sk の全量を単一モードに割り当てる。
        f_Sk = 0.5 * (
            (params["f_Sk_ccw0"] + params["slope_Sk_ccw"] * H)
            + (params["f_Sk_breath0"] + params["slope_Sk_breath"] * H)
        )
        w_Sk = 0.5 * (params["width_Sk_ccw"] + params["width_Sk_breath"])
        amp_Sk = params["amp_Sk_scale"] * phi_Sk

        S11 = S11 + lorentzian_dip(f_axis, f_Sk, w_Sk, amp_Sk)

    S11 = S11 + params["offset_C"]
    return S11


def load_module1_result(npz_path):
    """
    module1 の出力 npz を読み込んで、module2 が使う四つの配列を返す。
    期待するキー: phi_He_obs, phi_TC_obs, phi_Sk_obs, mz_obs
    """
    data = np.load(npz_path)
    required = {"phi_He_obs", "phi_TC_obs", "phi_Sk_obs", "mz_obs"}
    missing = sorted(required - set(data.files))
    if missing:
        raise KeyError(f"Missing keys in {npz_path}: {missing}")
    return (
        data["phi_He_obs"],
        data["phi_TC_obs"],
        data["phi_Sk_obs"],
        data["mz_obs"],
    )


def run_module2(
    phi_He_obs,
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
    Module1の出力(phi_He_obs, phi_TC_obs, phi_Sk_obs, mz_obs)から、
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
        phi_He = float(phi_He_obs[N])
        phi_TC = float(phi_TC_obs[N])
        phi_Sk = float(phi_Sk_obs[N])
        mz = float(mz_obs[N])

        # mzからHを逆算（クリップ域では不正確になる点に注意。
        # 可能ならModule1側でH_low_obsを直接保存して渡す方が正確）
        H = mz * H_c2

        S11 = synthesize_S11(f_axis, H, phi_He, phi_TC, phi_Sk, params)
        S11_list.append(S11)
        H_list.append(H)

    return f_axis, np.array(N_indices), np.array(H_list), np.array(S11_list)


# 後方互換のためのエイリアス（Fig.2b再現時の呼び出し方と同じシグネチャ）
def run_module2_fig2b_style(
    phi_He_obs,
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
        phi_He_obs, phi_TC_obs, phi_Sk_obs, mz_obs, H_c2,
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

    理論的根拠のない要素（TCモード、amp_Sk_power、eta_Sk_*）は削除済み。
    残っているパラメータも根拠レベルは一様ではないため、各項目にコメントで
    明記している。

    ヘリカルモード（旧「コニカル」）の根拠:
    - n=1(±Q, 4.13GHz): 強い主モード。分裂幅は未測定のため1本のまま暫定実装
    - n=2(4.9GHz/5.1GHz): 弱い副モード（異方性による間接励起）。マルチドメイン
      状態のため2本に分裂。中心周波数はFig.2b/2cの独自測定値
    - slope=0：分散が「著しく平坦」という原論文2209.06962本文の記述を、
      恣意的な非ゼロ値を置かずそのまま採用
    - 参考文献: Weiler et al., PRL 119, 237204 (2017)。H0∥x,y,z((111)配向
      試料)での測定であり、report.pdfのH∥⟨100⟩とは印加方向が異なるため、
      モードの本数・相対振幅など定性的構造のみを採用し、絶対周波数・分裂幅
      の数値はそのまま流用していない

    スキルミオンモードの根拠:
    - CCW/breathingへの分割は原論文本文に明記（"counter-clockwise and
      breathing modes"）
    - 振幅：ヘリカルn=1（4GHz帯）に対して「数%程度」というユーザー観測に
      基づき校正（amp_Sk_scale）。phi_Skの最大値を約0.74と見積もった上での逆算
    - alpha_CCW（CCWとbreathingへの配分比）：分割自体は文献根拠があるが、
      比率0.5は根拠のないplaceholderのまま残っている

    核生成由来の微細構造（TC/tilted conical）の根拠:
    - Aqeel et al., PRL 126, 017202 (2021)。詳細はsynthesize_S11のdocstring参照
    - 振幅：CCW/breathingモード(amp_Sk_scale=0.12)より強いというユーザー観測に
      基づき、amp_tc_fine_scaleをそれより大きい値に設定。ただし正確な倍率は
      未測定（振幅の精密さより帯域・本数を優先する方針のため）
    - 本数・周波数間隔：文献に具体的記載なし。実測波形の見た目に基づく粗い
      仮置き（4本、f_He_n1(~4.13GHz)を挟んで3.6〜4.65GHz付近に分布）
    """
    return {
        "f_min": 1.0,
        "f_max": 6.0,
        "n_freq": 1601,
        # ヘリカル n=1 モード（±Q, 主モード）
        "f_He_n1": 4.13,
        "slope_He_n1": 0.0,   # 「著しく平坦」をそのまま採用（恣意的な非ゼロ値は置かない）
        "width_He_n1": 0.15,
        "amp_He_n1_scale": 30.0,
        # ヘリカル n=2 モード（弱い副モード、4.9/5.1GHzに2分裂）
        "f_He_n2a": 4.9,
        "f_He_n2b": 5.1,
        "slope_He_n2": 0.0,
        "width_He_n2": 0.15,
        "amp_He_n2_scale": 5.4,  # n=1(30.0)の18%相当
        # スキルミオン: CCWモード
        "f_Sk_ccw0": 2.6,
        "slope_Sk_ccw": -0.003,
        "width_Sk_ccw": 0.12,
        # スキルミオン: breathingモード
        "f_Sk_breath0": 2.1,
        "slope_Sk_breath": -0.002,
        "width_Sk_breath": 0.15,
        # phi_SkのうちCCWモードに配分する割合（残りはbreathing）。根拠なしplaceholder
        "alpha_CCW": 0.5,
        # ヘリカルn=1(4GHz帯, amp_He_n1_scale=3.0)に対して「数%」という観測に基づく。
        # phi_Sk_max ≈ 0.74（これまでのシミュレーションでの実績値）としたときに
        # 3.0 * 0.03 / 0.74 ≈ 0.12 となるよう校正
        "amp_Sk_scale": 1.2,
        # 核生成由来の微細構造（TC/tilted conical, 4GHz帯）
        # 本数・間隔は実測波形の見た目に基づく粗い仮置き（f_He_n1=4.13GHzを挟む）
        "f_tc_fine_list": [3.6, 3.85, 4.4, 4.65],
        "width_tc_fine": 0.08,  # 細かいギザギザなので主モードより狭め
        # CCW/breathing(amp_Sk_scale=0.12)より強い、という観測に基づき、それより
        # 大きい値に設定。正確な倍率は未測定
        "amp_tc_fine_scale": 3,
        # オフセット
        "offset_C": 0.0,
        # --- アブレーション用トグル ---
        # split_sk_modes: CCW/breathing分割は文献根拠のある唯一の残存アブレーション軸
        "split_sk_modes": True,
        "include_helical_n2": True,
        "include_tc_fine": True,
    }


def plot_waterfall(f_axis, N_indices, S11_list, title="Module2 synthetic S11"):
    fig, ax = plt.subplots(figsize=(7, 8))
    offset_step = 1.5
    for i, N in enumerate(N_indices):
        y_offset = i * offset_step
        ax.plot(f_axis, S11_list[i] + y_offset, lw=1.2)
        ax.text(f_axis[-1] + 0.05, y_offset, f"N={N}", va="center", fontsize=8)
    ax.set_xlabel("f (GHz)")
    ax.set_ylabel(r"$\Delta S_{11}$ (offset, arb.)")
    ax.set_title(title)
    ax.set_yticks([])
    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    # --- ダミーデータでの動作確認 + split_sk_modesトグルのデモ ---
    # 旧Model 0〜3(TC有無・pの掃引)は根拠のない要素の削除に伴い廃止。
    # 現時点で文献根拠のあるアブレーション軸は split_sk_modes のみ。
    n_cycles = 300
    N_full = np.arange(n_cycles)

    phi_He_obs_dummy = np.clip(1.0 - N_full / 60.0, 0, 1) * np.exp(-N_full / 300.0)
    phi_Sk_obs_dummy = np.clip((N_full % 150) / 130.0, 0, 1)
    phi_TC_obs_dummy = np.clip(1.0 - phi_He_obs_dummy - phi_Sk_obs_dummy, 0, 1) * 0.5
    mz_obs_dummy = 0.5 + 0.15 * np.sin(2 * np.pi * N_full / 100)

    for split_sk_modes in [False, True]:
        label = "SkSplit" if split_sk_modes else "SkUnified"
        params2 = default_params()
        params2["split_sk_modes"] = split_sk_modes

        f_axis, N_indices, H_arr, S11_arr = run_module2(
            phi_He_obs_dummy,
            phi_TC_obs_dummy,
            phi_Sk_obs_dummy,
            mz_obs_dummy,
            H_c2=170.0,
            N_start=0,
            N_end=n_cycles - 1,
            N_step=1,
            params=params2,
        )

        out_dir = Path(__file__).resolve().parent / "data_full" / "Cu2OSeO3" / label
        written = export_prcpy_scan_files(f_axis, N_indices, H_arr, S11_arr, out_dir)

        print(f"[{label}] split_sk_modes={split_sk_modes}")
        print(f"  Wrote {len(written)} scan files to: {out_dir}")