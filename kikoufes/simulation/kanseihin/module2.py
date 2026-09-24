"""
module2.py --- スペクトル合成（Module 2）

Module 1 が出力する相分率 (phi_He, phi_TC, phi_Sk) とアニーリング度 psi、
および磁場 H から、マイクロ波吸収スペクトル Delta S11(f) を合成する。

============================================================================
本版の設計思想: 「振幅変調」から「周波数位置符号化」へ
============================================================================

旧版の最大の問題は、全モードの中心周波数が H の一次関数 (しかも小さな傾き)
でしか動かず、サイクルごとの違いが実質的に振幅の大小だけで表現されていた
ことである。その結果、
  - サイクル数ごとのスペクトルがほとんど同じ形になる
  - readout 行列の有効ランク (CP) が 1.3 程度と極端に低い
  - 前処理 (sample_rate, smooth) をどう変えても性能が動かない
という症状が出ていた。

Lee論文 (arXiv:2209.06962) 本文は、性能の起源を明確に「周波数位置」に帰して
いる:

  "High(low) transformation performance of the conical(helical) phase can be
   associated with the size of frequency shift by magnetic field. The
   dispersion curve of the helical phase displays a notably flat profile ...
   resulting in poor computational performance with its peak position shifting
   very weakly in response to field input. Much higher amplitude frequency
   shifts are found in the highly-performing conical and skyrmion phases,
   producing the strong nonlinearity and complexity in their reservoirs"

  (変換性能の高低は磁場による周波数シフトの大きさと結びつく。ヘリカル相の
   分散曲線は著しく平坦で、ピーク位置が入力にほとんど応答しないため性能が
   低い。高性能なコニカル相とスキルミオン相では遥かに大きな周波数シフトが
   見られ、これが強い非線形性と複雑性を生む)

  "the conical reservoir excellently encodes the input signal as shown by
   their FMR positions ... This mode yields a high nonlinearity (NL) and
   complexity (CP)"

数学的にも、共鳴の「振幅」だけを変調した場合スペクトルは少数の固定基底の
線形結合に留まり有効ランクが上がらないのに対し、「中心周波数」を動かすと
スペクトルは状態の非線形関数となり有効ランクが大きく増える。上記の実験的
記述と数学的構造が一致するため、本版では周波数位置の変調を設計の中心に据える。

----------------------------------------------------------------------------
履歴依存性の実装 (本版の中心的な新要素)
----------------------------------------------------------------------------
Lee論文 SM Fig.S4 は、同一磁場値で測ったスペクトルがサイクル履歴によって
どれだけ変わるかを定量化している:
  - Fig.S4d: ピーク周波数比 omega_N / omega_{N-25} の縦軸が 0.9 - 1.1
    -> 履歴による周波数変動は ±10% のオーダー
  - Fig.S4e: ピーク振幅比 Am_N / Am_{N-25} の縦軸が 0.0 - 3.0
    -> 振幅は履歴で最大 3 倍程度変動する
  - "Points B and C ... have the same peak position but different heights.
     This is due to the cycling-number-dependent meta-stable skyrmion
     population - the more we cycle, the more we nucleate the skyrmions.
     This intrinsic material property generates additional (long-term) memory"

そこで各モードの中心周波数を

    f = f0 + s_H * (H - H_ref)                <- 磁場による即時シフト
           + s_hist * (eta_TC*phi_TC + eta_Sk*phi_Sk + eta_psi*psi)  <- 履歴シフト

の形にした。第2項は Module 1 の状態変数 (それ自体が磁場入力の履歴の積分量)
を通じて過去の入力に依存するため、同一 H でもサイクル数によって共鳴位置が
変わる。psi (アニーリング度) は数百サイクルかけて単調進行する長期記憶変数
なので、Fig.S4d/e が示す「長期メモリ」の担い手として働く。

相ごとの周波数シフトの大小関係は文献に従って明確に差をつける:
    ヘリカル : s_H ~ 0     (著しく平坦。履歴依存もなし)
    スキルミオン : s_H 大   (大きな周波数シフト) + 履歴依存あり
Lee論文 Fig.4e/4g が「ヘリカル相・コニカル相のスペクトルは点 A-D で同一
(履歴非依存)、スキルミオン相のみ異なる」と明記しているため、ヘリカル相に
履歴依存を入れないのは単なる手抜きではなく文献的に要請される設計である。

----------------------------------------------------------------------------
TC 微細構造 (ギザギザ): 応答ベクトルを持つ「モードの森」
----------------------------------------------------------------------------
Aqeel et al. (PRL 126, 017202, H||<100>) は tilted conical 状態が ±Q モードと
同じ ~4 GHz 帯に "a multitude of hybridizations" (多数のハイブリダイゼー
ション) として現れると記述している。

旧版はこれを「乱数で決めた固定位置の多数のピーク」として実装していたが、
位置が固定だと結局サイクル間で振幅しか変わらず、ランダムに見えるだけで
情報を運ばなかった。

本版では各モード k に固有の応答ベクトル (c_H, c_TC, c_Sk, c_psi) を乱数で
一度だけ割り当て、モードごとに異なる向き・大きさで周波数と振幅が動くように
した。乱数は「モードの森の個性」を一度だけ決めるために使われ、時間発展は
完全に決定論的である。すなわち見かけ上のランダムさの背後にあるのは履歴依存
であり、同じ履歴を与えれば必ず同じスペクトルが再現される。

この設計には副次的だが重要な利点がある: 各チャンネルが異なる向きに動くため
readout 行列の有効ランク (CP) が構造的に増える。Lee論文 SM も
  "Each frequency point has unique evolution offering rich nonlinear responses
   as a whole. This large set of diverse responses to the input function
   empowers the reservoirs"
  (各周波数点が固有の時間発展を持ち、全体として豊かな非線形応答を提供する)
と、まさに「チャンネルごとに異なる応答」を性能の源泉として挙げている。

----------------------------------------------------------------------------
スキルミオンモードの分割比のサイクル依存
----------------------------------------------------------------------------
Aqeel et al. は、サイクリングによりスキルミオンが伸長 (elongated skyrmion)
すると
  "the spectral weight of the counterclockwise gyration mode is distinctly
   reduced for the elongated skyrmions"
  (伸長したスキルミオンでは反時計回りジャイレーションモードのスペクトル
   重みが明確に減少する)
と報告している。伸長・変形はサイクリングの進行に伴って蓄積するため、
CCW と breathing の配分比 alpha_CCW を psi の関数とした:

    alpha_CCW(psi) = alpha_CCW_0 + d_alpha_CCW * psi     (d_alpha_CCW < 0)

これにより分割比がサイクル数とともに移り変わる。

============================================================================
"""
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 基本形状
# ---------------------------------------------------------------------------
def lorentzian_dip(f, center, width, amplitude):
    """吸収ディップ型ローレンツ関数 (共鳴周波数で負に落ち込む)。"""
    return -amplitude * (width ** 2) / ((f - center) ** 2 + width ** 2)


def _lorentzian_bank(f_axis, centers, widths, amps):
    """複数のローレンツ関数をまとめて評価して総和を返す (ベクトル化)。"""
    centers = np.asarray(centers, dtype=float)[:, None]
    widths = np.asarray(widths, dtype=float)[:, None]
    amps = np.asarray(amps, dtype=float)[:, None]
    w2 = widths ** 2
    return -np.sum(amps * w2 / ((f_axis[None, :] - centers) ** 2 + w2), axis=0)


# ---------------------------------------------------------------------------
# TC 微細構造 (モードの森)
# ---------------------------------------------------------------------------
@dataclass
class ModeForest:
    """
    tilted conical 由来の微細構造を構成する多数の共鳴モード。

    各モードは固有の応答ベクトルを持ち、磁場と履歴変数に対して
    それぞれ異なる向き・大きさで中心周波数と振幅が変化する。
    乱数は「森の個性」を一度だけ決めるために使われ、時間発展自体は決定論的。
    """
    f0: np.ndarray          # 基準中心周波数 [GHz]
    width: np.ndarray       # 線幅 [GHz]
    weight: np.ndarray      # 相対振幅重み (総和 1)
    c_H: np.ndarray         # 磁場への周波数応答係数
    c_TC: np.ndarray        # phi_TC への周波数応答係数
    c_Sk: np.ndarray        # phi_Sk への周波数応答係数
    c_psi: np.ndarray       # psi への周波数応答係数
    b_psi: np.ndarray       # psi への振幅応答係数
    b_Sk: np.ndarray        # phi_Sk への振幅応答係数
    domain: np.ndarray      # 各モードが属するドメイン番号

    def __len__(self):
        return len(self.f0)


def build_mode_forest(n_modes=64, f_lo=3.45, f_hi=5.25, f_center=4.35,
                      f_sigma=0.45, width_lo=0.035, width_hi=0.13,
                      seed=20240517, n_domains=1):
    """
    TC 微細構造を構成するモードの森を生成する。

    n_modes を多く・線幅を狭くするほど細かいギザギザになる。
    振幅は対数正規分布で大小を散らし、ガウス型エンベロープで帯域中心を強調
    することで「大小さまざまなピークが密集する」実測の見た目に近づける。
    応答係数はいずれも標準正規分布から引き、正負両方向のシフトが混在する
    ようにしている (実測のギザギザは一様に動くのではなく、隣り合うピークが
    別々の向きに動くため)。
    """
    rng = np.random.default_rng(seed)

    f0 = np.sort(rng.uniform(f_lo, f_hi, n_modes))
    width = rng.uniform(width_lo, width_hi, n_modes)

    raw = rng.lognormal(mean=0.0, sigma=0.75, size=n_modes)
    envelope = np.exp(-0.5 * ((f0 - f_center) / f_sigma) ** 2) + 0.12
    weight = raw * envelope
    weight /= weight.sum()

    # 各モードをドメインに割り当てる。Module 1 の多ドメインアンサンブルの
    # どのサブドメインの履歴に従って動くかを決める。同一ドメインに属する
    # モードは同じ履歴を共有するが、応答係数 c_* が異なるため別々に動く。
    domain = np.arange(n_modes) % max(int(n_domains), 1)
    rng.shuffle(domain)

    return ModeForest(
        f0=f0, width=width, weight=weight,
        c_H=rng.normal(0.0, 1.0, n_modes),
        c_TC=rng.normal(0.0, 1.0, n_modes),
        c_Sk=rng.normal(0.0, 1.0, n_modes),
        c_psi=rng.normal(0.0, 1.0, n_modes),
        b_psi=rng.normal(0.0, 1.0, n_modes),
        b_Sk=rng.normal(0.0, 1.0, n_modes),
        domain=domain,
    )


# ---------------------------------------------------------------------------
# パラメータ
# ---------------------------------------------------------------------------
PARAM_PROVENANCE = {
    # --- 周波数軸・参照点 ---
    "f_min":            ("assumed",    "周波数軸下限。実測は 1-6 GHz"),
    "f_max":            ("assumed",    "周波数軸上限。同上"),
    "n_freq":           ("assumed",    "周波数点数。sample_rate と合わせて実効次元を決める"),
    "H_ref":            ("derived",    "磁場シフトの基準点 (Hlow の中央値あたり)"),
    # --- ヘリカル ---
    "f_He_n1":          ("measured",   "ヘリカル n=1 (±Q) 主モード。Fig.2/4 のピクセル解析"),
    "slope_He_n1":      ("literature", "ヘリカル分散は『著しく平坦』(Lee本文) -> 0"),
    "width_He_n1":      ("assumed",    "主モード線幅。実測の見た目に合わせた仮置き"),
    "amp_He_n1":        ("measured",   "主モード振幅スケール。他の振幅比の基準"),
    "f_He_n2a":         ("measured",   "ヘリカル n=2 高調波 (ドメイン分裂の片方)"),
    "f_He_n2b":         ("measured",   "ヘリカル n=2 高調波 (もう片方)"),
    "width_He_n2":      ("assumed",    "n=2 線幅。仮置き"),
    "amp_He_n2_rel":    ("measured",   "n=2/n=1 振幅比。実測『20%未満』観測に基づく"),
    # --- スキルミオン ---
    "f_Sk_ccw":         ("measured",   "スキルミオン CCW ジャイレーションモード"),
    "f_Sk_breath":      ("measured",   "スキルミオン breathing モード"),
    "width_Sk_ccw":     ("assumed",    "CCW 線幅。仮置き"),
    "width_Sk_breath":  ("assumed",    "breathing 線幅。仮置き"),
    "slope_Sk":         ("literature", "Sk 相は『遥かに大きな周波数シフト』(Lee本文)"),
    "eta_Sk_fast":      ("derived",    "短期履歴シフト [GHz/sigma]。NL/MC/CP/MSE から校正"),
    "eta_Sk_slow":      ("derived",    "長期履歴シフト [GHz/sigma]。同上"),
    "amp_Sk_rel":       ("measured",   "Sk/He 振幅比。実測スペクトルの観測に基づく"),
    "alpha_CCW_0":      ("assumed",    "CCW/breathing 初期配分比。数値の根拠はない"),
    "d_alpha_CCW":      ("literature", "伸長で CCW 重みが減少 (Aqeel 2021) -> 負値"),
    "amp_hist_depth":   ("literature", "履歴による振幅変調深さ。Fig.S4e の変動幅に整合"),
    # --- TC 微細構造 ---
    "amp_tc_rel":       ("derived",    "微細構造の振幅比。NL/MC/CP/MSE を同時最適化して決定"),
    "tc_shift_H":       ("literature", "TC は ±Q と同帯域で磁場応答する (Aqeel 2021)"),
    "tc_shift_hist":    ("derived",    "履歴シフト [GHz/sigma]。MC-CP トレードオフの最適点"),
    "tc_amp_hist_depth":("derived",    "履歴による振幅変調深さ。同上"),
    "n_modes":          ("assumed",    "TC モード数。文献は『多数』としか述べていない"),
    "forest_seed":      ("assumed",    "モードの森の個性を決める乱数種 (結果は決定論的)"),
    # --- トグル・その他 ---
    "split_sk_modes":     ("literature", "CCW/breathing 分割は Lee本文に明記"),
    "include_helical_n2": ("literature", "n=2 高調波の存在は Weiler 2017"),
    "include_tc_fine":    ("literature", "TC 微細構造の存在は Aqeel 2021"),
    "enable_history_shift": ("-", "アブレーション用トグル"),
    "enable_field_shift":   ("-", "アブレーション用トグル"),
    "offset":               ("-", "全体オフセット"),
}


@dataclass
class Module2Params:
    """Module 2 のパラメータ。役割ごとにグループ化して保持する。"""

    # --- 周波数軸 ---
    f_min: float = 1.0
    f_max: float = 6.0
    n_freq: int = 1601

    # --- 参照点 (磁場シフトの基準) ---
    H_ref: float = 50.5          # [mT] Hlow の中央値あたり

    # --- ヘリカル n=1 (±Q 主モード): 平坦・履歴非依存 ---
    f_He_n1: float = 4.13
    width_He_n1: float = 0.15
    amp_He_n1: float = 12.0
    slope_He_n1: float = 0.0     # 「著しく平坦」を素直に採用

    # --- ヘリカル n=2 (弱い高調波、マルチドメインで2本に分裂) ---
    f_He_n2a: float = 4.90
    f_He_n2b: float = 5.10
    width_He_n2: float = 0.15
    amp_He_n2_rel: float = 0.18  # n=1 に対する比

    # --- スキルミオン CCW / breathing ---
    f_Sk_ccw: float = 2.62
    f_Sk_breath: float = 2.10
    width_Sk_ccw: float = 0.12
    width_Sk_breath: float = 0.15
    slope_Sk: float = -0.0090    # [GHz/mT] 大きな磁場シフト (文献記述に基づく)
    # 履歴シフトは z-score (標準偏差単位) あたりの GHz で与える。
    # Fig.S4d の omega_N/omega_{N-25} in [0.9, 1.1] は 2.6 GHz で +-0.26 GHz に
    # 相当し、磁場由来シフト (slope_Sk * dH ~ 0.2 GHz) と同オーダーである。
    eta_Sk_fast: float = 0.020   # [GHz/sigma] 短期履歴 (phi_Sk の揺らぎ)
    eta_Sk_slow: float = 0.014   # [GHz/sigma] 長期履歴 (psi のアニーリング)
    amp_Sk_rel: float = 0.090    # ヘリカル n=1 に対する振幅比
    alpha_CCW_0: float = 0.62    # psi=0 での CCW 配分比
    d_alpha_CCW: float = -0.34   # psi 依存 (伸長で CCW 重みが減少)
    amp_hist_depth: float = 0.45 # 履歴による振幅変調の深さ [/sigma]

    # --- TC 微細構造 (モードの森) ---
    amp_tc_rel: float = 3.00     # ヘリカル n=1 に対する微細構造全体の振幅比
    tc_shift_H: float = 0.0035   # [GHz/mT] 磁場応答係数の標準偏差スケール
    tc_shift_hist: float = 0.030 # [GHz/sigma] 履歴応答のスケール
    tc_amp_hist_depth: float = 0.45
    n_modes: int = 192
    forest_seed: int = 20240517

    # --- トグル (アブレーション用) ---
    split_sk_modes: bool = True
    include_helical_n2: bool = True
    include_tc_fine: bool = True
    enable_history_shift: bool = True   # 履歴による周波数シフトの ON/OFF
    enable_field_shift: bool = True     # 磁場による周波数シフトの ON/OFF

    offset: float = 0.0

    def to_dict(self):
        return asdict(self)

    def describe(self):
        lines = [f"{'parameter':<18}{'value':>12}  {'provenance':<11} note", "-" * 94]
        for k, v in asdict(self).items():
            prov, note = PARAM_PROVENANCE.get(k, ("-", ""))
            try:
                vs = f"{float(v):>12.6g}"
            except (TypeError, ValueError):
                vs = f"{str(v):>12}"
            lines.append(f"{k:<18}{vs}  {prov:<11} {note}")
        return "\n".join(lines)


def default_params():
    """後方互換のため dict を返す。新規コードでは Module2Params を使うこと。"""
    return Module2Params().to_dict()


# ---------------------------------------------------------------------------
# スペクトル合成
# ---------------------------------------------------------------------------
def synthesize_S11(f_axis, H, phi_He, phi_TC, phi_Sk, psi=0.0,
                   z=(0.0, 0.0, 0.0), params=None, forest=None,
                   dom_z=None, dom_phi_TC=None):
    """
    単一観測点のスペクトル Delta S11(f) を合成する。

    Parameters
    ----------
    f_axis : (M,) ndarray   周波数軸 [GHz]
    H      : float          その観測点の磁場 [mT]
    phi_He, phi_TC, phi_Sk : float   相分率 (振幅を決める)
    psi    : float          アニーリング度 [0,1] (モード配分比を決める)
    z      : (z_TC, z_Sk, z_psi)
        履歴変数の z-score (軌道全体で標準化した偏差)。周波数シフト量を
        「標準偏差あたり何 GHz 動くか」という直接解釈できる単位で与えるために
        使う。run_module2() が軌道から自動計算して渡す。
    params : Module2Params | dict
    forest : ModeForest | None   None なら params から生成
    """
    if params is None:
        params = Module2Params()
    elif isinstance(params, dict):
        params = Module2Params(**{k: v for k, v in params.items()
                                  if k in Module2Params.__dataclass_fields__})
    p = params

    if forest is None:
        forest = build_mode_forest(n_modes=p.n_modes, seed=p.forest_seed)

    dH = (H - p.H_ref) if p.enable_field_shift else 0.0
    if p.enable_history_shift:
        z_TC, z_Sk, z_psi = float(z[0]), float(z[1]), float(z[2])
    else:
        z_TC = z_Sk = z_psi = 0.0

    S11 = np.zeros_like(f_axis)

    # ---- ヘリカル n=1 (±Q): 平坦・履歴非依存 ----
    # Lee論文 Fig.4e が「ヘリカル相のスペクトルは点 A-D で同一 = 履歴非依存」と
    # 明記しているため、ここには意図的に履歴項を入れない。
    S11 += lorentzian_dip(f_axis,
                          p.f_He_n1 + p.slope_He_n1 * dH,
                          p.width_He_n1,
                          p.amp_He_n1 * phi_He)

    # ---- ヘリカル n=2 高調波 ----
    if p.include_helical_n2:
        a = p.amp_He_n1 * p.amp_He_n2_rel * phi_He * 0.5
        S11 += lorentzian_dip(f_axis, p.f_He_n2a + p.slope_He_n1 * dH,
                              p.width_He_n2, a)
        S11 += lorentzian_dip(f_axis, p.f_He_n2b + p.slope_He_n1 * dH,
                              p.width_He_n2, a)

    # ---- TC 微細構造 (モードの森) ----
    # 各モードが固有の応答ベクトル (c_H, c_TC, c_Sk, c_psi) に従って
    # 別々の向き・大きさで動く。見かけはランダムだが完全に決定論的であり、
    # その背後にあるのは磁場と履歴変数である。
    if p.include_tc_fine and phi_TC > 0:
        # ドメイン別の履歴があればそれを使う。各モードは自分が属するドメインの
        # 履歴 (固有の時定数を持つ) に従って動くため、モードごとに異なる遅延の
        # 入力情報を担うことになる。
        if dom_z is not None:
            idx = forest.domain
            zt = np.asarray(dom_z[0])[idx]
            zs = np.asarray(dom_z[1])[idx]
            zp = np.asarray(dom_z[2])[idx]
            tc_j = (np.asarray(dom_phi_TC)[idx] if dom_phi_TC is not None
                    else np.full(len(forest), phi_TC))
        else:
            zt = np.full(len(forest), z_TC)
            zs = np.full(len(forest), z_Sk)
            zp = np.full(len(forest), z_psi)
            tc_j = np.full(len(forest), phi_TC)

        if not p.enable_history_shift:
            zt = zs = zp = np.zeros(len(forest))

        centers = (forest.f0
                   + p.tc_shift_H * forest.c_H * dH
                   + p.tc_shift_hist * (forest.c_TC * zt
                                        + forest.c_Sk * zs
                                        + forest.c_psi * zp))
        amp_scale = np.clip(
            1.0 + p.tc_amp_hist_depth * (forest.b_psi * zp + forest.b_Sk * zs),
            0.0, None)
        amps = p.amp_He_n1 * p.amp_tc_rel * tc_j * forest.weight * amp_scale
        S11 += _lorentzian_bank(f_axis, centers, forest.width, amps)

    # ---- スキルミオン CCW / breathing ----
    # 文献に従い大きな磁場シフト + 同オーダーの履歴シフトを持たせる。
    if phi_Sk > 0:
        amp_tot = p.amp_He_n1 * p.amp_Sk_rel * phi_Sk
        # 履歴による振幅変調 (Fig.S4e: Am_N/Am_{N-25} が 0-3 の範囲で変動)
        amp_tot *= max(1.0 + p.amp_hist_depth * (0.6 * z_Sk + 0.4 * z_psi), 0.0)

        shift = (p.slope_Sk * dH
                 + p.eta_Sk_fast * z_Sk
                 + p.eta_Sk_slow * z_psi)

        if p.split_sk_modes:
            # 配分比がアニーリング (伸長) とともに移り変わる (Aqeel 2021)
            alpha = float(np.clip(p.alpha_CCW_0 + p.d_alpha_CCW * psi, 0.02, 0.98))
            S11 += lorentzian_dip(f_axis, p.f_Sk_ccw + shift,
                                  p.width_Sk_ccw, amp_tot * alpha)
            # breathing は CCW と異なる磁場応答を持つ (Aqeel 2021 の異常交差)
            S11 += lorentzian_dip(f_axis, p.f_Sk_breath + shift * 0.72,
                                  p.width_Sk_breath, amp_tot * (1.0 - alpha))
        else:
            f_mid = 0.5 * (p.f_Sk_ccw + p.f_Sk_breath) + shift
            w_mid = 0.5 * (p.width_Sk_ccw + p.width_Sk_breath)
            S11 += lorentzian_dip(f_axis, f_mid, w_mid, amp_tot)

    return S11 + p.offset


# ---------------------------------------------------------------------------
# 一括実行
# ---------------------------------------------------------------------------
def run_module2(module1_result=None, params=None, *,
                phi_He_obs=None, phi_TC_obs=None, phi_Sk_obs=None,
                psi_obs=None, H_obs=None, mz_obs=None, H_c2=170.0,
                N_start=0, N_end=None, N_step=1):
    """
    Module 1 の出力から全観測点のスペクトルを合成する。

    module1_result に simulate_module1() の戻り値 dict をそのまま渡せる。
    個別配列で渡すこともできる (後方互換)。

    Returns
    -------
    f_axis, N_indices, H_arr, S11_arr
    """
    if params is None:
        params = Module2Params()
    elif isinstance(params, dict):
        params = Module2Params(**{k: v for k, v in params.items()
                                  if k in Module2Params.__dataclass_fields__})
    p = params

    if module1_result is not None:
        phi_He_obs = module1_result["phi_He_obs"]
        phi_TC_obs = module1_result["phi_TC_obs"]
        phi_Sk_obs = module1_result["phi_Sk_obs"]
        psi_obs = module1_result.get("psi_obs")
        H_obs = module1_result.get("H_obs")
        mz_obs = module1_result.get("mz_obs")

    n_total = len(phi_Sk_obs)
    if psi_obs is None:
        psi_obs = np.zeros(n_total)
    if H_obs is None:
        # mz からの逆算 (クリップ域では不正確。H_obs を直接渡す方が正確)
        H_obs = np.asarray(mz_obs, dtype=float) * H_c2

    if N_end is None:
        N_end = n_total - 1
    N_indices = np.array([int(n) for n in range(N_start, N_end + 1, N_step)
                          if 0 <= n < n_total])

    # ドメイン別の履歴 (Module 1 が多ドメインで走っている場合)
    dom_TC = module1_result.get("dom_TC_obs") if module1_result else None
    dom_Sk = module1_result.get("dom_Sk_obs") if module1_result else None
    dom_psi = module1_result.get("dom_psi_obs") if module1_result else None
    n_dom = dom_TC.shape[1] if dom_TC is not None else 1

    f_axis = np.linspace(p.f_min, p.f_max, p.n_freq)
    forest = build_mode_forest(n_modes=p.n_modes, seed=p.forest_seed,
                               n_domains=n_dom)

    # --- 履歴変数の z-score 化 ---
    # 定常状態での phi の揺らぎは +-0.03 程度と小さいため、生の偏差をそのまま
    # 使うと履歴由来の周波数シフトが磁場由来のシフトに比べて桁違いに小さく
    # なってしまう。Lee論文 Fig.S4d は履歴だけで +-10% の周波数変動 (2.6 GHz
    # なら +-0.26 GHz) を示しており、磁場由来シフトと同オーダーである。
    # そこで履歴変数を軌道全体で標準化し、eta_* を「標準偏差あたり何 GHz
    # 動くか」という直接解釈できる単位で与える。これにより文献の数値
    # (+-10%) に対してパラメータを直接校正できる。
    # 過渡期 (核生成前) を含めて標準化すると分布が歪むため、onset 後の領域で
    # 統計を取る。
    def _z(arr):
        a = np.asarray(arr, dtype=float)
        tail = a[len(a) // 4:] if len(a) > 8 else a
        mu, sd = float(np.mean(tail)), float(np.std(tail))
        return (a - mu) / sd if sd > 1e-12 else np.zeros_like(a)

    z_TC_all = _z(phi_TC_obs)
    z_Sk_all = _z(phi_Sk_obs)
    z_psi_all = _z(psi_obs)

    # ドメイン別 z-score。各ドメインは固有の時定数を持つため、異なる遅延の
    # 入力履歴を別々に保持する (フィルタバンク)。これが 4 GHz 帯の微細構造の
    # 「ランダムさの背後にある履歴依存」の実体。
    if dom_TC is not None:
        zd_TC = np.column_stack([_z(dom_TC[:, j]) for j in range(n_dom)])
        zd_Sk = np.column_stack([_z(dom_Sk[:, j]) for j in range(n_dom)])
        zd_psi = np.column_stack([_z(dom_psi[:, j]) for j in range(n_dom)])
        dom_TC_frac = dom_TC
    else:
        zd_TC = z_TC_all[:, None]; zd_Sk = z_Sk_all[:, None]
        zd_psi = z_psi_all[:, None]
        dom_TC_frac = np.asarray(phi_TC_obs)[:, None]

    S11_arr = np.empty((len(N_indices), p.n_freq))
    H_arr = np.empty(len(N_indices))
    for i, N in enumerate(N_indices):
        H = float(H_obs[N])
        H_arr[i] = H
        S11_arr[i] = synthesize_S11(
            f_axis, H,
            float(phi_He_obs[N]), float(phi_TC_obs[N]), float(phi_Sk_obs[N]),
            float(psi_obs[N]),
            z=(z_TC_all[N], z_Sk_all[N], z_psi_all[N]),
            params=p, forest=forest,
            dom_z=(zd_TC[N], zd_Sk[N], zd_psi[N]),
            dom_phi_TC=dom_TC_frac[N],
        )
    return f_axis, N_indices, H_arr, S11_arr


def load_module1_result(npz_path):
    """module1 の出力 npz を読み込む。psi_obs が無い場合はゼロで補う。"""
    data = np.load(npz_path)
    required = {"phi_He_obs", "phi_TC_obs", "phi_Sk_obs"}
    missing = sorted(required - set(data.files))
    if missing:
        raise KeyError(f"Missing keys in {npz_path}: {missing}")
    res = {k: data[k] for k in data.files}
    if "psi_obs" not in res:
        res["psi_obs"] = np.zeros_like(res["phi_Sk_obs"])
    return res


def export_prcpy_scan_files(f_axis, N_indices, H_arr, S11_arr, output_dir,
                            scan_start_index=1, field_decimals=3, clear_existing=True):
    """
    PRCpy が読み込む CSV 形式 ("Current","Field","Frequency","Spectra") で、
    観測点ごとに scan_<連番>_<磁場>.txt を出力する。

    clear_existing=True (既定) の場合、書き出し前に output_dir 内の既存の
    scan_*.txt を削除する。ファイル名には磁場の値が含まれるため、同じ
    ディレクトリに対して異なる入力系列で複数回実行すると、連番は同じでも
    磁場の値が異なり古いファイルが上書きされずに残ってしまう
    (結果、新旧2系統のデータが混在した状態でPRCpyに読み込まれ、
    時系列が意味をなさなくなる)。これを防ぐため既定でクリアする。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if clear_existing:
        stale = list(output_dir.glob("scan_*.txt"))
        for p in stale:
            p.unlink()
        if stale:
            print(f"[export_prcpy_scan_files] 既存の scan_*.txt を {len(stale)} 個削除 "
                  f"(古い実行結果の混在を防止): {output_dir}")

    written = []
    for i, _N in enumerate(N_indices):
        H_val = float(H_arr[i])
        fpath = output_dir / f"scan_{scan_start_index + i}_{H_val:.{field_decimals}f}.txt"
        block = np.column_stack([np.zeros_like(f_axis),
                                 np.full_like(f_axis, H_val),
                                 f_axis, S11_arr[i]])
        with open(fpath, "w", newline="") as fh:
            fh.write('"Current","Field","Frequency","Spectra"\n')
            np.savetxt(fh, block, delimiter=",", fmt="%.8g")
        written.append(fpath)
    return written


def plot_waterfall(f_axis, N_indices, S11_arr, title="Module2 synthetic S11",
                   offset_step=None):
    """ウォーターフォール表示 (Lee論文 Fig.2b 相当)。"""
    if offset_step is None:
        offset_step = 0.35 * (np.max(S11_arr) - np.min(S11_arr) + 1e-9)
    fig, ax = plt.subplots(figsize=(7, 9))
    for i, N in enumerate(N_indices):
        ax.plot(f_axis, S11_arr[i] + i * offset_step, lw=1.0)
        ax.text(f_axis[-1] + 0.03, i * offset_step, f"N={N}", va="center", fontsize=7)
    ax.set_xlabel("f (GHz)")
    ax.set_ylabel(r"$\Delta S_{11}$ (offset, arb.)")
    ax.set_title(title)
    ax.set_yticks([])
    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    import module1 as m1

    n_cycles = 1000
    mg = m1.generate_mackey_glass(n_cycles, tau=17)
    u = 2.0 * (mg - mg.min()) / (mg.max() - mg.min()) - 1.0
    H_traj = m1.build_field_trajectory(u, H_c=73.0, H_range=90.0)

    p1 = m1.calibrate_module1(H_traj, m1.TargetDynamics(), verbose=True)
    res1 = m1.simulate_module1(H_traj, p1)

    p2 = Module2Params()
    f_axis, N_idx, H_arr, S11 = run_module2(res1, p2)
    print(f"\nS11 shape = {S11.shape}, range = [{S11.min():.3f}, {S11.max():.3f}]")
    print(p2.describe())